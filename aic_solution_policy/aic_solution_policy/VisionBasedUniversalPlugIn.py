import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge
from ultralytics import YOLO

from aic_control_interfaces.msg import MotionUpdate
from aic_task_interfaces.msg import Task
from rclpy.time import Time
from aic_model.policy import (
    Policy,
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)

class VisionBasedUniversalPlugIn(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("UniversalVisionPlugIn initialisiert.")
        
        self._bridge = CvBridge()
        self._camera_names = ['left', 'center', 'right']
        self._cam_intrinsics = {}
        
        # --- KONFIGURATION FÜR BEIDE TYPEN ---
        # Die Keys ('sc', 'sfp') entsprechen task.cable_type
        self._configs = {
            'sc': {
                'model_path': "/home/intrinsic/ws_aic/src/aic/aic_solution/training/models/single_sc_detection.pt",
                'off_pos': [0.005327, -0.000874, -0.01194],
                'off_quat': [0.1608, -0.167181, 0.69417, -0.6814],
                'z_approach': 0.01, # 1cm über dem Port
                'z_plug': -0.04     # 4cm tief rein
            },
            'sfp': {
                'model_path': "/home/intrinsic/ws_aic/src/aic/aic_solution/training/models/best150.pt",
                'off_pos': [0.0004576051855596508, -0.00017897008773293255, -0.05107300646397306],
                'off_quat': [0.17961162465395691, 0.005559995963849536, -0.02746131717311321, -0.9833385029246792],
                'z_approach': 0.01,
                'z_plug': -0.04
            }
        }

        # Beider Modelle laden
        self._models = {}
        for c_type, cfg in self._configs.items():
            self.get_logger().info(f"Lade YOLO Modell für {c_type}: {cfg['model_path']}")
            self._models[c_type] = YOLO(cfg['model_path'])
            # Warmup
            self._models[c_type].predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

        self.get_logger().info("YOLO Warmup für alle Modelle abgeschlossen.")

    # --- PORT DETECTION LOGIK ---
    def _get_cam_frame_data(self, cam_full_name):
        try:
            target = f"{cam_full_name}/optical" 
            trans = self._parent_node._tf_buffer.lookup_transform("base_link", target, Time())
            return {
                "pos": np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]),
                "rot": R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]).as_matrix()
            }
        except Exception: return None

    def _save_cam_info(self, msg, cam_name):
        if cam_name not in self._cam_intrinsics:
            self._cam_intrinsics[cam_name] = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}

    def _triangulate_rays(self, rays):
        I = np.eye(3)
        A, b = np.zeros((3,3)), np.zeros(3)
        for r in rays:
            M = I - np.outer(r["direction"], r["direction"])
            A += M
            b += M @ r["origin"]
        return np.linalg.lstsq(A, b, rcond=None)[0]

    def _calculate_forced_pose(self, corners):
        corners = np.array(corners)
        center = np.mean(corners, axis=0)
        vec_x = corners[1] - corners[0]
        vec_x[2] = 0 
        vec_x /= np.linalg.norm(vec_x)
        vec_z = np.array([0.0, 0.0, -1.0]) 
        vec_y = np.cross(vec_z, vec_x)
        vec_y /= np.linalg.norm(vec_y)
        rot_matrix = np.stack([vec_x, vec_y, vec_z], axis=1)
        return center, R.from_matrix(rot_matrix).as_quat()

    def detect_ports(self, observation, cable_type):
        if observation is None or cable_type not in self._models: return {}
        
        for cam in self._camera_names:
            attr = f"{cam}_camera_info"
            if hasattr(observation, attr): self._save_cam_info(getattr(observation, attr), cam)

        separated = {0: {}, 1: {}}
        model = self._models[cable_type]
        
        for cam in self._camera_names:
            img_msg = getattr(observation, f"{cam}_image", None)
            if img_msg is None: continue
            
            cv_img = self._bridge.imgmsg_to_cv2(img_msg, "bgr8")
            res = model.predict(cv_img, conf=0.8, verbose=False)[0]
            
            for j, box in enumerate(res.boxes):
                cls = int(box.cls[0])
                if cls in separated:
                    separated[cls][cam] = res.keypoints.xy[j].cpu().numpy()

        found_ports = {}
        for pid, cams in separated.items():
            if len(cams) < 2: continue
            pts_3d = []
            for k in range(4):
                rays = []
                for cam_name, kpts in cams.items():
                    fdata = self._get_cam_frame_data(f"{cam_name}_camera")
                    if fdata and cam_name in self._cam_intrinsics:
                        u, v = kpts[k]
                        intr = self._cam_intrinsics[cam_name]
                        d_cam = np.array([(u - intr['cx']) / intr['fx'], (v - intr['cy']) / intr['fy'], 1.0])
                        d_world = fdata["rot"] @ d_cam
                        rays.append({"origin": fdata["pos"], "direction": d_world/np.linalg.norm(d_world)})
                if len(rays) >= 2: pts_3d.append(self._triangulate_rays(rays))

            if len(pts_3d) == 4:
                pos, quat = self._calculate_forced_pose(pts_3d)
                found_ports[pid] = {"pos": pos, "quat": quat}
        return found_ports

    # --- BEWEGUNGSLOGIK ---
    def _get_diagonal_matrix(self, values):
        mat = [0.0] * 36
        for i in range(6): mat[i * 6 + i] = values[i]
        return mat
    
    def _move_tcp_to_pose(self, pos, quat, move_robot, get_observation, stiffness_diag, damping_diag):
        motion_update = MotionUpdate()
        motion_update.header.frame_id = "base_link"
        motion_update.trajectory_generation_mode.mode = 2 
        motion_update.pose.position.x, motion_update.pose.position.y, motion_update.pose.position.z = map(float, pos)
        motion_update.pose.orientation.x, motion_update.pose.orientation.y, motion_update.pose.orientation.z, motion_update.pose.orientation.w = map(float, quat)
        motion_update.target_stiffness = self._get_diagonal_matrix(stiffness_diag)
        motion_update.target_damping = self._get_diagonal_matrix(damping_diag)

        for i in range(100):
            move_robot(motion_update=motion_update)
            obs = get_observation()
            curr = obs.controller_state.tcp_pose.position
            dist = math.sqrt((curr.x - pos[0])**2 + (curr.y - pos[1])**2 + (curr.z - pos[2])**2)
            if dist < 0.0005: return
            self.sleep_for(0.05)

    def _get_tcp_goal_pose_hardcoded(self, port_pos, port_quat, cfg):
        mat_base_to_port = np.eye(4)
        mat_base_to_port[:3, :3] = R.from_quat(port_quat).as_matrix()
        mat_base_to_port[:3, 3] = port_pos
        
        mat_cable_to_tcp = np.eye(4)
        mat_cable_to_tcp[:3, :3] = R.from_quat(cfg['off_quat']).as_matrix()
        mat_cable_to_tcp[:3, 3] = cfg['off_pos']
        
        target_matrix = mat_base_to_port @ mat_cable_to_tcp
        return target_matrix[:3, 3], R.from_matrix(target_matrix[:3, :3]).as_quat()

    # --- MAIN TASK ---
    def insert_cable(self, task: Task, get_observation: GetObservationCallback, move_robot: MoveRobotCallback, send_feedback: SendFeedbackCallback):
        self.get_logger().info(f"Starte Task für Kabel-Typ: {task.cable_type}")
        
        # 1. Konfiguration wählen
        c_type = task.plug_type.lower() # 'sfp' oder 'sc'
        if c_type not in self._configs:
            self.get_logger().error(f"Typ '{c_type}' nicht unterstützt!")
            return False
        
        cfg = self._configs[c_type]

        # 2. Ports erkennen
        obs = get_observation()
        found_ports = self.detect_ports(obs, c_type)
        if not found_ports:
            self.get_logger().error(f"Kein Port für {c_type} im Sichtfeld!")
            return False

        # 3. Ziel-Port ID aus task.port_name (z.B. 'sfp_port_0' -> 0) extrahieren
        try:
            target_id = int(task.port_name.split('_')[-1])
        except:
            target_id = list(found_ports.keys())[0]

        if target_id not in found_ports:
            self.get_logger().warning(f"Port {target_id} nicht gesehen, weiche auf {list(found_ports.keys())[0]} aus.")
            target_id = list(found_ports.keys())[0]

        # 4. Posen berechnen
        target_port = found_ports[target_id]
        tcp_pos, tcp_quat = self._get_tcp_goal_pose_hardcoded(target_port["pos"], target_port["quat"], cfg)

        # 5. Bewegungs-Sequenz ausführen
        send_feedback(f"Stecke {c_type.upper()} in Port {target_id}...")
        
        # Approach (Über dem Port)
        approach_pos = tcp_pos.copy()
        approach_pos[2] += cfg['z_approach']
        stiff_soft = [220.0] * 6
        damping = [200.0] * 6
        self._move_tcp_to_pose(approach_pos, tcp_quat, move_robot, get_observation, stiff_soft, damping)
        
        # Versteifen für präzises Einstecken
        stiff_hard = [420.0] * 6
        self._move_tcp_to_pose(approach_pos, tcp_quat, move_robot, get_observation, stiff_hard, damping)

        # Plug-In (In den Port hinein)
        plug_pos = tcp_pos.copy()
        plug_pos[2] += cfg['z_plug']
        self._move_tcp_to_pose(plug_pos, tcp_quat, move_robot, get_observation, stiff_hard, damping)

        self.get_logger().info(f"--- {c_type.upper()} Task erfolgreich beendet ---")
        return True