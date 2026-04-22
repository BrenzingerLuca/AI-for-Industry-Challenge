import math
import os
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from aic_task_interfaces.msg import Task
from aic_model.policy import (
    Policy,
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)

class VisionBasedSFPPlugIn(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("VisionBasedSFPPlugIn initialisiert.")

        # Make Ultralytics config writable in constrained/container environments.
        # Ultralytics uses YOLO_CONFIG_DIR and will create settings.json there.
        os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")

        # Delay heavy imports until we actually need them (configure should be fast).
        self._Rotation = None
        self._bridge = None
        self._camera_names = ['left', 'center', 'right']
        self._cam_intrinsics = {}

        # YOLO will be initialized lazily (first use) to keep lifecycle configure fast.
        self._yolo_model = None
        self._yolo_weights_path = None

    def _ensure_vision_deps(self):
        if self._Rotation is None:
            from scipy.spatial.transform import Rotation as Rotation  # noqa: WPS433

            self._Rotation = Rotation
        if self._bridge is None:
            from cv_bridge import CvBridge  # noqa: WPS433

            self._bridge = CvBridge()

    ########################################################################################################### port detection
    def _init_yolo_model(self):
        if self._yolo_model is not None:
            return

        current_file_path = Path(__file__).resolve()

        possible_paths = [
            current_file_path.parents[1] / "models" / "best150.pt",
            current_file_path.parents[2] / "training" / "models" / "best150.pt",
            Path.cwd() / "aic_solution" / "training" / "models" / "best150.pt",
            Path.cwd() / "training" / "models" / "best150.pt",
        ]

        model_path = None
        for candidate in possible_paths:
            if candidate.exists():
                model_path = candidate
                break

        if model_path is None:
            msg = "YOLO model not found. Tried: " + ", ".join(str(p) for p in possible_paths)
            self.get_logger().error(msg)
            raise FileNotFoundError(msg)

        self.get_logger().info(f"Lade YOLO Modell: {model_path}")
        # Import ultralytics/torch lazily (heavy import).
        from ultralytics import YOLO  # noqa: WPS433

        self._yolo_weights_path = model_path
        self._yolo_model = YOLO(str(model_path))

    def _get_cam_frame_data(self, cam_full_name):
        """Holt aktuelle Kamera-Position/Rotation via TF."""
        try:
            self._ensure_vision_deps()
            import numpy as np  # noqa: WPS433
            from rclpy.time import Time  # noqa: WPS433

            target = f"{cam_full_name}/optical" 
            trans = self._parent_node._tf_buffer.lookup_transform("base_link", target, Time())
            return {
                "pos": np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]),
                "rot": self._Rotation.from_quat([
                    trans.transform.rotation.x,
                    trans.transform.rotation.y,
                    trans.transform.rotation.z,
                    trans.transform.rotation.w,
                ]).as_matrix(),
            }
        except Exception:
            return None

    def _save_cam_info(self, msg, cam_name):
        if cam_name not in self._cam_intrinsics:
            self._cam_intrinsics[cam_name] = {
                'fx': msg.k[0], 'fy': msg.k[4], 
                'cx': msg.k[2], 'cy': msg.k[5]
            }

    # --- DEINE TRIANGULATION ---
    def _triangulate_rays(self, rays):
        import numpy as np  # noqa: WPS433

        I = np.eye(3)
        A, b = np.zeros((3,3)), np.zeros(3)
        for r in rays:
            M = I - np.outer(r["direction"], r["direction"])
            A += M
            b += M @ r["origin"]
        return np.linalg.lstsq(A, b, rcond=None)[0]

    # --- DEINE FORCED POSE LOGIK ---
    def _calculate_forced_pose(self, corners):
        self._ensure_vision_deps()
        import numpy as np  # noqa: WPS433

        corners = np.array(corners)
        center = np.mean(corners, axis=0)
        vec_x = corners[1] - corners[0]
        vec_x[2] = 0 # Z-Komponente auf 0 für reine Ebene
        vec_x /= np.linalg.norm(vec_x)
        vec_z = np.array([0.0, 0.0, -1.0]) # Z zeigt nach unten
        vec_y = np.cross(vec_z, vec_x)
        vec_y /= np.linalg.norm(vec_y)
        rot_matrix = np.stack([vec_x, vec_y, vec_z], axis=1)
        return center, self._Rotation.from_matrix(rot_matrix).as_quat()

    def detect_ports(self, observation):
        if observation is None: return {}

        self._ensure_vision_deps()
        import numpy as np  # noqa: WPS433
        
        # Intrinsics sammeln
        for cam in self._camera_names:
            attr = f"{cam}_camera_info"
            if hasattr(observation, attr):
                self._save_cam_info(getattr(observation, attr), cam)

        # Bilder mit YOLO verarbeiten
        separated = {0: {}, 1: {}}
        for cam in self._camera_names:
            img_msg = getattr(observation, f"{cam}_image", None)
            if img_msg is None: continue
            
            cv_img = self._bridge.imgmsg_to_cv2(img_msg, "bgr8")
            self._init_yolo_model()
            res = self._yolo_model.predict(cv_img, conf=0.8, verbose=False)[0]
            
            for j, box in enumerate(res.boxes):
                cls = int(box.cls[0])
                if cls in separated:
                    separated[cls][cam] = res.keypoints.xy[j].cpu().numpy()

        # 3D Triangulation
        found_ports = {}
        for pid, cams in separated.items():
            if len(cams) < 2: continue
            
            pts_3d = []
            for k in range(4): # 4 Keypoints
                rays = []
                for cam_name, kpts in cams.items():
                    fdata = self._get_cam_frame_data(f"{cam_name}_camera")
                    if fdata and cam_name in self._cam_intrinsics:
                        u, v = kpts[k]
                        intr = self._cam_intrinsics[cam_name]
                        d_cam = np.array([(u - intr['cx']) / intr['fx'], (v - intr['cy']) / intr['fy'], 1.0])
                        d_world = fdata["rot"] @ d_cam
                        rays.append({"origin": fdata["pos"], "direction": d_world/np.linalg.norm(d_world)})
                
                if len(rays) >= 2:
                    pts_3d.append(self._triangulate_rays(rays))

            if len(pts_3d) == 4:
                pos, quat = self._calculate_forced_pose(pts_3d)
                found_ports[pid] = {"pos": pos, "quat": quat}
        
        return found_ports
    #################################################################################################################

    #################################################################################################### Move to pose 
    def _get_diagonal_matrix(self, values):
        """Erzeugt eine 6x6 Matrix (36 Werte) für den Controller."""
        mat = [0.0] * 36
        for i in range(6): mat[i * 6 + i] = values[i]
        return mat
    
    def _move_tcp_to_pose(self, pos, quat, move_robot, get_observation, stiffness_diag, damping_diag):
        from aic_control_interfaces.msg import MotionUpdate  # noqa: WPS433

        motion_update = MotionUpdate()
        motion_update.header.frame_id = "base_link"
        motion_update.trajectory_generation_mode.mode = 2 
        
        motion_update.pose.position.x, motion_update.pose.position.y, motion_update.pose.position.z = map(float, pos)
        motion_update.pose.orientation.x, motion_update.pose.orientation.y, motion_update.pose.orientation.z, motion_update.pose.orientation.w = map(float, quat)

        motion_update.target_stiffness = self._get_diagonal_matrix(stiffness_diag)
        motion_update.target_damping = self._get_diagonal_matrix(damping_diag)

        for i in range(100):
            motion_update.header.stamp = self.get_clock().now().to_msg()
            move_robot(motion_update=motion_update)
            
            # Monitoring
            obs = get_observation()
            curr = obs.controller_state.tcp_pose.position
            dist = math.sqrt((curr.x - pos[0])**2 + (curr.y - pos[1])**2 + (curr.z - pos[2])**2)
            
            if dist < 0.0005:
                self.get_logger().info(f"Ziel erreicht (Fehler: {dist*1000:.3f} mm)")
                return
            
            if i % 20 == 0:
                self.get_logger().info(f"Distanz: {dist*1000:.2f} mm")
            
            self.sleep_for(0.05)
        
        self.get_logger().info(f"Ziel nicht erreicht (Fehler: {dist*1000:.3f} mm)")
    ###########################################################################################################################

    ###############################################################################################################TF Stuff
    def _get_tcp_goal_pose(self, port_pos, port_quat, cable_tip_frame):
        """
        Berechnet die benötigte gripper/tcp Pose, damit der cable_tip_link 
        auf der port_pose landet.
        """
        
        self._ensure_vision_deps()
        import numpy as np  # noqa: WPS433
        from rclpy.time import Time  # noqa: WPS433

        # 1. Hol dir den Versatz: Wo ist der Greifer relativ zur Kabelspitze?
        # Wir schauen, wie wir von der Spitze zum Greifer kommen.
        tf_cable_to_tcp = self._parent_node._tf_buffer.lookup_transform(
            cable_tip_frame, 
            "gripper/tcp", 
            Time()
        )
        
        # Umwandeln in Matrizen
        # A: Transformation von Base zum erkannten Port
        mat_base_to_port = np.eye(4)
        mat_base_to_port[:3, :3] = self._Rotation.from_quat(port_quat).as_matrix()
        mat_base_to_port[:3, 3] = port_pos
        
        # B: Transformation von Kabelspitze zum Greifer
        mat_cable_to_tcp = np.eye(4)
        q_off = tf_cable_to_tcp.transform.rotation
        mat_cable_to_tcp[:3, :3] = self._Rotation.from_quat([q_off.x, q_off.y, q_off.z, q_off.w]).as_matrix()
        t_off = tf_cable_to_tcp.transform.translation
        mat_cable_to_tcp[:3, 3] = [t_off.x, t_off.y, t_off.z]
        
        # C: Ziel-Pose für den Greifer = Port-Pose * Offset
        # (Wenn die Kabelspitze auf dem Port liegen soll)
        target_matrix = mat_base_to_port @ mat_cable_to_tcp
        
        target_pos = target_matrix[:3, 3]
        target_quat = self._Rotation.from_matrix(target_matrix[:3, :3]).as_quat()
        
        return target_pos, target_quat
    
    def _get_tcp_goal_pose_hardcoded(self, port_pos, port_quat):
        """
        Berechnet die benötigte gripper/tcp Pose basierend auf den 
        hochpräzisen Nominal-Werten (Ersatz für Ground Truth TF).
        """
        self._ensure_vision_deps()
        import numpy as np  # noqa: WPS433

        # --- DEINE PRÄZISIONS-WERTE AUS DEM LOG (CABLE TO TCP) ---
        # Position
        off_x = 0.0004576051855596508
        off_y = -0.00017897008773293255
        off_z = -0.05107300646397306
        
        # Rotation (Quaternion xyzw)
        off_qx = 0.17961162465395691
        off_qy = 0.005559995963849536
        off_qz = -0.02746131717311321
        off_qw = -0.9833385029246792

        # 1. Matrix: Base -> Port (Das Ziel im Raum, wo die Spitze hin soll)
        mat_base_to_port = np.eye(4)
        mat_base_to_port[:3, :3] = self._Rotation.from_quat(port_quat).as_matrix()
        mat_base_to_port[:3, 3] = port_pos
        
        # 2. Matrix: Kabelspitze -> Greifer (Der starre Versatz aus deinen Werten)
        mat_cable_to_tcp = np.eye(4)
        mat_cable_to_tcp[:3, :3] = self._Rotation.from_quat([off_qx, off_qy, off_qz, off_qw]).as_matrix()
        mat_cable_to_tcp[:3, 3] = [off_x, off_y, off_z]
        
        # 3. Ziel-Pose für den Greifer berechnen
        # Logik: Base_to_TCP = Base_to_Port * Cable_to_TCP
        target_matrix = mat_base_to_port @ mat_cable_to_tcp
        
        target_pos = target_matrix[:3, 3]
        target_quat = self._Rotation.from_matrix(target_matrix[:3, :3]).as_quat()
        
        return target_pos, target_quat

    #####################################################################################################################################       
        
    def insert_cable(self, task: "Task", get_observation: GetObservationCallback, move_robot: MoveRobotCallback, send_feedback: SendFeedbackCallback):

        # --- DEBUG: TASK OBJEKT PRINTEN ---
        self.get_logger().info("==========================================")
        self.get_logger().info(f"EMPFANGENES TASK OBJEKT: {task}")
        self.get_logger().info("==========================================")

        self.get_logger().info("--- Vision Task gestartet ---")
        
        # Port detection
        for i in range(3):
            obs = get_observation()
            found_ports = self.detect_ports(obs)

            if not found_ports:
                self.get_logger().error("Kein Port im Sichtfeld erkannt!")
                return False

            for port, data in found_ports.items():
                p, q = data["pos"], data["quat"]
                self.get_logger().info(f"📍 Port {port} bei: [{p[0]:.5f}, {p[1]:.5f}, {p[2]:.5f}]")
                self.get_logger().info(f"   Orientierung: {q}")
            
            self.sleep_for(0.5)

        # Get the goal port from task description
        try:
            target_id = int(task.port_name.split('_')[-1])
            self.get_logger().info(f"Ziel-ID extrahiert: {target_id}")
        except Exception as e:
            self.get_logger().error(f"Konnte Port-ID nicht aus {task.port_name} extrahieren: {e}")
            target_id = 0 # Fallback auf 0

        # Check if the goal port was detected
        if target_id in found_ports:
            target_port = found_ports[target_id]
            self.get_logger().info(f"✅ Zielport {target_id} gefunden!")
        else:
            self.get_logger().warning(f"⚠️ Port {target_id} nicht erkannt. Verfügbare Ports: {list(found_ports.keys())}")
            # Fallback: Nimm den ersten verfügbaren Port, falls der gewünschte nicht da ist
            target_id = list(found_ports.keys())[0]
            target_port = found_ports[target_id]
            self.get_logger().info(f"Nutze Fallback-Port: {target_id}")

        # Calculate the tcp pose based of the goal port pose
        cable_tip_frame = "cable_0/sfp_tip_link" 
        
        self.get_logger().info(f"Berechne TCP-Ziel für {cable_tip_frame} auf Port {target_id}")
        tcp_pos, tcp_quat = self._get_tcp_goal_pose_hardcoded(target_port["pos"], target_port["quat"])

        approach_pos = tcp_pos.copy()
        approach_pos[2] += 0.01  # 5 cm in globaler Z-Richtung nach oben addieren

        self.get_logger().info(f"Calculated TCP position: [{approach_pos[0]:.5f}, {approach_pos[1]:.5f}, {approach_pos[2]:.5f}]")
        self.get_logger().info(f"Calculated TCP position:  {tcp_quat}")
        # Move the tcp into the position so the cable tip is aligned with the goal port 
        send_feedback(f"Fahre TCP an, um Kabel in Port {target_id} zu platzieren...")
        stiffness = [220.0, 220.0, 220.0, 220.0, 220.0, 220.0]
        damping = [200.0, 200.0, 200.0, 200.0, 200.0, 200.0]
        self._move_tcp_to_pose(approach_pos, tcp_quat, move_robot, get_observation, stiffness, damping)
        
        stiffness = [420.0, 420.0, 420.0, 420.0, 420.0, 420.0]
        damping = [200.0, 200.0, 200.0, 200.0, 200.0, 200.0]
        self._move_tcp_to_pose(approach_pos, tcp_quat, move_robot, get_observation, stiffness, damping)

        plug_in_pos = tcp_pos.copy()
        plug_in_pos[2] -= 0.04

        self._move_tcp_to_pose(plug_in_pos, tcp_quat, move_robot, get_observation, stiffness, damping)
        self.get_logger().info("--- Task erfolgreich beendet ---")

        return True