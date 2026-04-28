import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge
from ultralytics import YOLO
from rclpy.duration import Duration
from aic_control_interfaces.msg import MotionUpdate
from aic_task_interfaces.msg import Task
from rclpy.time import Time
from aic_model.policy import (
    Policy,
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from std_srvs.srv import Trigger

class SpiralSearch(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("UniversalVisionPlugIn mit erweiterten Debug-Infos initialisiert.")
        #self._tare_client = self._parent_node.create_client(Trigger, '/aic_controller/tare_force_torque_sensor')
        self._bridge = CvBridge()
        self._camera_names = ['left', 'center', 'right']
        self._cam_intrinsics = {}
        
        self._configs = {
            'sc': {
                'model_path': "/models/single_sc_detection.pt",
                #'model_path': "/home/lucab/ws_aic/src/aic/aic_solution/training/models/single_sc_detection.pt",
                'off_pos': [0.0, -0.015385, -0.04045],
                'off_quat': [0.1608, -0.167181, 0.69417, -0.6814],
                'z_approach': 0.01,
                'z_plug': -0.04,
                'cable_tip_frame': "cable_0/sc_tip_link"
            },
            'sfp': {
                'model_path': "/models/best150.pt",
                #'model_path': "/home/lucab/ws_aic/src/aic/aic_solution/training/models/best150.pt",
                'off_pos': [0.0, -0.015385, -0.04245],
                'off_quat': [0.179611, 0.005559, -0.027461, -0.983338],
                'z_approach': 0.01,
                'z_plug': -0.045,
                'cable_tip_frame': "cable_0/sfp_tip_link"
            }
        }

        self._models = {}
        for c_type, cfg in self._configs.items():
            self.get_logger().info(f"Lade YOLO Modell [{c_type}]: {cfg['model_path']}")
            self._models[c_type] = YOLO(cfg['model_path'])
            self._models[c_type].predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)

    # --- Triangulation & Detection ---
    def _get_cam_frame_data(self, cam_full_name):
        try:
            target = f"{cam_full_name}/optical" 
            trans = self._parent_node._tf_buffer.lookup_transform("base_link", target, Time())
            return {
                "pos": np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]),
                "rot": R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]).as_matrix()
            }
        except Exception as e:
            return None

    def detect_ports(self, observation, cable_type):
        if observation is None: return {}
        
        # Intrinsics
        for cam in self._camera_names:
            attr = f"{cam}_camera_info"
            if hasattr(observation, attr):
                msg = getattr(observation, attr)
                self._cam_intrinsics[cam] = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}

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
            if len(cams) < 2: continue # Brauche mind. 2 Kameras
            
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
                
                if len(rays) >= 2:
                    # Triangulation Kern
                    I = np.eye(3)
                    A, b = np.zeros((3,3)), np.zeros(3)
                    for r in rays:
                        M = I - np.outer(r["direction"], r["direction"])
                        A += M
                        b += M @ r["origin"]
                    pts_3d.append(np.linalg.lstsq(A, b, rcond=None)[0])

            if len(pts_3d) == 4:
                # Forced Pose Berechnung
                corners = np.array(pts_3d)
                center = np.mean(corners, axis=0)
                vec_x = corners[1] - corners[0]
                vec_x[2] = 0 
                vec_x /= np.linalg.norm(vec_x)
                vec_z = np.array([0.0, 0.0, -1.0]) 
                vec_y = np.cross(vec_z, vec_x)
                vec_y /= np.linalg.norm(vec_y)
                rot_matrix = np.stack([vec_x, vec_y, vec_z], axis=1)
                found_ports[pid] = {"pos": center, "quat": R.from_matrix(rot_matrix).as_quat()}
        
        return found_ports

    ######################################################################## force stuff
    def _check_force_threshold(self, observation):
        """
        Überprüft die aktuellen Kräfte am Handgelenk.
        Pfad laut Debug: obs.wrist_wrench
        """
        try:
            if hasattr(observation, 'wrist_wrench') and observation.wrist_wrench is not None:
                # wrist_wrench ist ein WrenchStamped -> .wrench -> .force
                force = observation.wrist_wrench.wrench.force
                fx, fy, fz = force.x, force.y, force.z
                
                # Berechnung der Gesamtkraft (Vektorlänge) oder Einzelachsen
                # Hier prüfen wir jede Achse einzeln auf 20N (wie gewünscht)
                if abs(fx) > 20.0 or abs(fy) > 20.0 or abs(fz) > 20.0:
                    self.get_logger().warning(
                        f"⚠️ HOHE KRAFT! FX: {fx:6.2f} N | FY: {fy:6.2f} N | FZ: {fz:6.2f} N"
                    )
                    return True
        except Exception as e:
            # Falls doch mal ein Attribut fehlt, keine Unterbrechung des Programms
            pass
        return False

    def _call_tare_service(self):
        """Ruft den externen Tare-Service auf und wartet auf Vollzug."""
        self.get_logger().info("Warte auf Tare-Service...")
        if not self._tare_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Tare-Service '/aic_controller/tare_force_torque_sensor' nicht erreichbar!")
            return False

        future = self._tare_client.call_async(Trigger.Request())
        
        # Warten auf die Antwort (in Policy-Methoden ist das meist unbedenklich)
        # Da wir in einem rclpy-Kontext sind, nutzen wir eine kleine Schleife:
        import time
        start_time = time.time()
        while not future.done():
            if time.time() - start_time > 5.0:
                self.get_logger().error("Timeout beim Warten auf Tare-Antwort!")
                return False
            time.sleep(0.1)

        result = future.result()
        if result.success:
            self.get_logger().info(f"Tare erfolgreich: {result.message}")
            self.sleep_for(0.5) # Kurze Pause, damit die Werte sich stabilisieren
            return True
        else:
            self.get_logger().error(f"Tare fehlgeschlagen: {result.message}")
            return False
    ######################################################################################

    # --- Bewegungssteuerung mit Debugging ---
    def _move_tcp_smooth_cartesian(self, pos, quat, move_robot, get_observation, stiffness, damping, n_steps=80, label="Target"):
        """
        Sendet konstant die Zielpose mit niedriger Steifigkeit.
        Der Impedanzregler konvergiert selbst sanft → wenig Jerk.
        """
        motion_update = MotionUpdate()
        motion_update.header.frame_id = "base_link"
        motion_update.trajectory_generation_mode.mode = 2

        motion_update.pose.position.x = float(pos[0])
        motion_update.pose.position.y = float(pos[1])
        motion_update.pose.position.z = float(pos[2])
        motion_update.pose.orientation.x = float(quat[0])
        motion_update.pose.orientation.y = float(quat[1])
        motion_update.pose.orientation.z = float(quat[2])
        motion_update.pose.orientation.w = float(quat[3])

        mat_stiff = [0.0] * 36
        mat_damp  = [0.0] * 36

        for j in range(6):
            mat_stiff[j*6+j] = float(stiffness[j])
            mat_damp[j*6+j]  = float(damping[j])

        motion_update.target_stiffness = mat_stiff
        motion_update.target_damping   = mat_damp

        self.get_logger().info(f"==> Fahre {label} an (smooth): P=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

        dist = float('inf')
        for i in range(n_steps):
            move_robot(motion_update=motion_update)
            obs = get_observation()
            self._check_force_threshold(obs)

            curr = obs.controller_state.tcp_pose.position
            dist = math.sqrt(
                (curr.x - pos[0])**2 +
                (curr.y - pos[1])**2 +
                (curr.z - pos[2])**2
            )

            if i % 25 == 0:
                self.get_logger().info(f"    [{i}] Distanz zu {label}: {dist*1000:.2f} mm")
                if dist < 0.001: # 0.5mm Schwellwert
                    self.get_logger().info(f"    Ziel erreicht! Restfehler: {dist*1000:.3f} mm")
                    return dist

            self.sleep_for(0.1)
        self.get_logger().info(f"    [{label}] Fertig. Restfehler: {dist*1000:.3f} mm")
        return dist

    def _get_tcp_goal_pose(self, port_pos, port_quat, cable_tip_frame):
        """
        Berechnet die benötigte gripper/tcp Pose, damit der cable_tip_link 
        auf der port_pose landet.
        """
        
        # 1. Hol dir den Versatz: Wo ist der Greifer relativ zur Kabelspitze?
        # Wir schauen, wie wir von der Spitze zum Greifer kommen.
        timeout = Duration(seconds=1.0)
        tf_cable_to_tcp = self._parent_node._tf_buffer.lookup_transform(
            cable_tip_frame, 
            "gripper/tcp", 
            Time(),
            timeout=timeout
        )
        
        # Umwandeln in Matrizen
        # A: Transformation von Base zum erkannten Port
        mat_base_to_port = np.eye(4)
        mat_base_to_port[:3, :3] = R.from_quat(port_quat).as_matrix()
        mat_base_to_port[:3, 3] = port_pos
        
        # B: Transformation von Kabelspitze zum Greifer
        mat_cable_to_tcp = np.eye(4)
        q_off = tf_cable_to_tcp.transform.rotation
        mat_cable_to_tcp[:3, :3] = R.from_quat([q_off.x, q_off.y, q_off.z, q_off.w]).as_matrix()
        t_off = tf_cable_to_tcp.transform.translation
        mat_cable_to_tcp[:3, 3] = [t_off.x, t_off.y, t_off.z]

        self.get_logger().info(f"TCP zu PLUG TIP TRANSLATION: X: {t_off.x}, Y: {t_off.y}, Z: {t_off.z}")
        self.get_logger().info(f"TCP zu PLUG TIP ORIENTATION: qX: {q_off.x}, qY: {q_off.y}, qZ: {q_off.z}, qW: {q_off.w}")
        
        # C: Ziel-Pose für den Greifer = Port-Pose * Offset
        # (Wenn die Kabelspitze auf dem Port liegen soll)
        target_matrix = mat_base_to_port @ mat_cable_to_tcp
        
        target_pos = target_matrix[:3, 3]
        target_quat = R.from_matrix(target_matrix[:3, :3]).as_quat()
        
        return target_pos, target_quat
    
    def _get_tcp_goal_pose_hardcoded(self, port_pos, port_quat, port_type):
        """
        Berechnet die benötigte gripper/tcp Pose basierend auf den 
        hochpräzisen Nominal-Werten (Ersatz für Ground Truth TF).
        """
        # --- DEINE PRÄZISIONS-WERTE AUS DEM LOG (CABLE TO TCP) ---
        # Position
        if port_type == 'sc':

            # Position
            off_x = 0.005327
            off_y = -0.000874
            off_z = -0.01194
            
            # Rotation (Quaternion xyzw)
            off_qx = 0.1608
            off_qy = -0.167181
            off_qz = 0.69417
            off_qw = -0.6814

        # Gripper Offset-Werte aus trial 1
        elif port_type == 'sfp':
            off_x = 0.0
            off_y = 0.0004
            off_z = -0.05795

            off_qx = 0.17785
            off_qy = 0.00505
            off_qz = -0.02738
            off_qw = -0.98366


        # 1. Matrix: Base -> Port (Das Ziel im Raum, wo die Spitze hin soll)
        mat_base_to_port = np.eye(4)
        mat_base_to_port[:3, :3] = R.from_quat(port_quat).as_matrix()
        mat_base_to_port[:3, 3] = port_pos
        
        # 2. Matrix: Kabelspitze -> Greifer (Der starre Versatz aus deinen Werten)
        mat_cable_to_tcp = np.eye(4)
        mat_cable_to_tcp[:3, :3] = R.from_quat([off_qx, off_qy, off_qz, off_qw]).as_matrix()
        mat_cable_to_tcp[:3, 3] = [off_x, off_y, off_z]
        
        # 3. Ziel-Pose für den Greifer berechnen
        # Logik: Base_to_TCP = Base_to_Port * Cable_to_TCP
        target_matrix = mat_base_to_port @ mat_cable_to_tcp
        
        target_pos = target_matrix[:3, 3]
        target_quat = R.from_matrix(target_matrix[:3, :3]).as_quat()
        
        return target_pos, target_quat

    def _spiral_search_and_insert(self, center_pos, quat, move_robot, get_observation,
                               max_radius=0.003,
                               n_turns=3,
                               steps=120,
                               label="Spiral"):
        """
        Fährt eine Spirale in XY auf fixer Z-Höhe (center_pos).
        Niedrige Z-Steifigkeit sorgt dafür dass der Stecker automatisch
        reinrutscht sobald XY über der Öffnung ist.
        """
        stiff_spiral = [300.0, 300.0, 80.0, 200.0, 200.0, 200.0]
        damp_spiral  = [ 40.0,  40.0, 15.0,  30.0,  30.0,  30.0]

        self.get_logger().info(f"==> Starte Spiralsuche für {label} | max_radius={max_radius*1000:.1f}mm | turns={n_turns} | steps={steps}")

        t_vals = np.linspace(0, n_turns * 2 * np.pi, steps)

        for idx, t in enumerate(t_vals):
            r = (t / (n_turns * 2 * np.pi)) * max_radius
            dx = r * np.cos(t)
            dy = r * np.sin(t)

            search_pos = center_pos.copy()
            search_pos[0] += dx
            search_pos[1] += dy
            # Z bleibt fix auf center_pos[2]

            motion_update = self._build_motion_update(search_pos, quat, stiff_spiral, damp_spiral)
            move_robot(motion_update=motion_update)

            obs = get_observation()
            self._check_force_threshold(obs)

            curr = obs.controller_state.tcp_pose.position
            dist = math.sqrt(
                (curr.x - search_pos[0])**2 +
                (curr.y - search_pos[1])**2 +
                (curr.z - center_pos[2])**2  # Z immer gegen center_pos messen
            )

            if idx % 20 == 0:
                self.get_logger().info(f"    [{idx}/{steps}] r={r*1000:.2f}mm | dx={dx*1000:.1f}mm dy={dy*1000:.1f}mm | dist={dist*1000:.2f}mm")

            self.sleep_for(0.05)

        # Am Ende Restfehler gegen originale center_pos messen
        obs = get_observation()
        curr = obs.controller_state.tcp_pose.position
        final_dist = math.sqrt(
            (curr.x - center_pos[0])**2 +
            (curr.y - center_pos[1])**2 +
            (curr.z - center_pos[2])**2
        )

        self.get_logger().info(f"    [{label}] Spiralsuche fertig. Restfehler zu center: {final_dist*1000:.2f}mm")
        return final_dist

    def _build_motion_update(self, pos, quat, stiffness, damping):
        """Helper: MotionUpdate aus pos/quat/stiffness/damping bauen."""
        motion_update = MotionUpdate()
        motion_update.header.frame_id = "base_link"
        motion_update.trajectory_generation_mode.mode = 2

        motion_update.pose.position.x = float(pos[0])
        motion_update.pose.position.y = float(pos[1])
        motion_update.pose.position.z = float(pos[2])
        motion_update.pose.orientation.x = float(quat[0])
        motion_update.pose.orientation.y = float(quat[1])
        motion_update.pose.orientation.z = float(quat[2])
        motion_update.pose.orientation.w = float(quat[3])

        mat_stiff = [0.0] * 36
        mat_damp  = [0.0] * 36
        for j in range(6):
            mat_stiff[j*6+j] = float(stiffness[j])
            mat_damp[j*6+j]  = float(damping[j])
        motion_update.target_stiffness = mat_stiff
        motion_update.target_damping   = mat_damp

        return motion_update

    # --- Main ---
    def insert_cable(self, task: Task, get_observation: GetObservationCallback, move_robot: MoveRobotCallback, send_feedback: SendFeedbackCallback):
        # 1. Task Debug Info
        self.get_logger().info("============================================================")
        self.get_logger().info(f"NEUER TASK START: {task.cable_type.upper()}")
        self.get_logger().info(f"Port: {task.port_name} | Plug: {task.plug_name}")
        self.get_logger().info("============================================================")

        send_feedback("Taring force sensor...")
        #self._call_tare_service()

        c_type = task.port_type.lower()
        if c_type not in self._configs:
            self.get_logger().error(f"Kabeltyp '{c_type}' ist nicht konfiguriert!")
            return False
        
        cfg = self._configs[c_type]

        # 2. Erkennung
        obs = get_observation()
        found_ports = self.detect_ports(obs, c_type)

        if not found_ports:
            self.get_logger().error("DETECTION FAILED: Kein passender Port gefunden!")
            return False

        # Port Auswahl Logik
        try:
            target_id = int(task.port_name.split('_')[-1])
        except:
            target_id = list(found_ports.keys())[0]

        if target_id not in found_ports:
            self.get_logger().info(f"DEBUG: Port {target_id} nicht gesehen. Nehme verfügbaren: {list(found_ports.keys())}")
            target_id = list(found_ports.keys())[0]

        target_port = found_ports[target_id]
        pp, pq = target_port["pos"], target_port["quat"]
        self.get_logger().info(f"ERKANNT: Port {target_id} bei Base-Link: Pos={pp}, Quat={pq}")


        cable_tip_frame = cfg['cable_tip_frame']

        # self.get_logger().info(f"Berechne TCP-Ziel für {cable_tip_frame} auf Port {target_id}")
        # tcp_pos, tcp_quat = self._get_tcp_goal_pose(target_port["pos"], target_port["quat"], cable_tip_frame)

        # self.get_logger().info(f"BERECHNET: TCP Ziel-Pose mit Ground truth: Pos={tcp_pos}, Quat={tcp_quat}")
        tcp_pos, tcp_quat = self._get_tcp_goal_pose_hardcoded(target_port["pos"], target_port["quat"], c_type)

        self.get_logger().info(f"BERECHNET: TCP Ziel-Pose ohne Ground truth: Pos={tcp_pos}, Quat={tcp_quat}")

        # 4. Ausführung
        send_feedback(f"Starte Einsteckvorgang für {c_type}...")

        # Schritt A: Approach (Über dem Port)
        approach_pos = tcp_pos.copy()
        approach_pos[2] += cfg['z_approach']

        plug_pos = tcp_pos.copy()
        plug_pos[2] += cfg['z_plug']

        # Phase 1: Sanft zum Approach (wie GentleGiant - niedrig + gedämpft)
        self._move_tcp_smooth_cartesian(
            approach_pos, tcp_quat, move_robot, get_observation,
            stiffness=[150.0, 150.0, 150.0, 80.0, 80.0, 80.0],
            damping=[40.0,  40.0,  40.0,  20.0, 20.0, 20.0],
            n_steps=60,
            label="Approach"
        )

        # Phase 2: Versteifen OHNE Bewegung (einregeln lassen)
        self._move_tcp_smooth_cartesian(
            approach_pos, tcp_quat, move_robot, get_observation,
            stiffness=[400.0]*6,
            damping=[50.0]*6,
            n_steps=20,          # kurz, nur zum Einregeln
            label="Stiffening"
        )

        # if dist > 0.020:  # > 2cm → Spiralsuche
        # self.get_logger().info(f"Restfehler {dist*1000:.1f}mm > 20mm → Starte Spiralsuche")
        final_dist = self._spiral_search_and_insert(
            center_pos=plug_pos,
            quat=tcp_quat,
            move_robot=move_robot,
            get_observation=get_observation,
            max_radius=0.003,
            label=c_type
        )
        
        if final_dist < 1.0:
            self.get_logger().info("============================================================")
            self.get_logger().info(f"TASK ERFOLGREICH BEENDET ({c_type})")
            self.get_logger().info("============================================================")
            return True

        else:
            final_plug_position = plug_pos.copy()
            final_plug_position[2] -= 0.01 
            self._move_tcp_smooth_cartesian(
            final_plug_position, tcp_quat, move_robot, get_observation,
            stiffness=[300.0, 300.0,  80.0, 300.0, 300.0, 300.0],
            damping=[ 40.0,  40.0,  15.0,  40.0,  40.0,  40.0],
            n_steps=80,
            label="Plug-In"
        )