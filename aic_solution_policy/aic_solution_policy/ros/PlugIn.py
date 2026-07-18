import numpy as np
import math
from scipy.spatial.transform import Rotation as R
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

class PlugIn(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)

        from cv_bridge import CvBridge
        from ultralytics import YOLO

        self.get_logger().info("PlugIn Policy initialised")
        self._bridge = CvBridge()
        self._camera_names = ['left', 'center', 'right']
        self._cam_intrinsics = {}

        # TF lookups can become temporarily unreliable after sim resets (time jumps).
        # Camera extrinsics are static, so cache successful lookups and reuse them.
        self._tf_cam_frame_cache = {}

        # Throttle noisy TF debug checks (uses wall-clock monotonic time).
        self._last_tip_offset_debug_monotonic_s = None
        
        self._configs = {
            'sc': {
                #'model_path': "/models/single_sc_detection.pt",
                #'model_path': "/home/lucab/ws_aic/src/aic/aic_solution/training/models/single_sc_detection.pt",
                'model_path': "/home/intrinsic/ws_aic/src/aic/aic_solution/training/models/single_sc_detection.pt",
                'off_pos': [0.0, -0.015385, -0.04045],
                'off_quat': [0.1608, -0.167181, 0.69417, -0.6814],
                'z_approach_1': 0.03,
                'z_approach_2': 0.0025,
                'z_plug': -0.03,
                'cable_tip_frame': "cable_0/sc_tip_link",
                'search_insert_strategy_1' : "_spiral_search_and_insert_2d",
                'spiral_max_radius_1': 0.002,
                'spiral_max_radius_2': 0.005,
                'spiral_stiffness_1': [300.0, 300.0, 40.0, 200.0, 200.0, 40.0],
                'spiral_damping_1': [40.0, 40.0, 15.0, 30.0, 30.0, 30.0],
                'spiral_stiffness_2': [150.0, 150.0, 30.0, 300.0, 300.0, 10.0],
                'spiral_damping_2': [30.0, 30.0, 10.0, 30.0, 30.0, 30.0],
                'spiral_steps_1': 150,
                'spiral_steps_2': 250
            },

            'sfp': {
                #'model_path': "/models/best150.pt",
                #'model_path': "/home/lucab/ws_aic/src/aic/aic_solution/training/models/best150.pt",
                'model_path': "/home/intrinsic/ws_aic/src/aic/aic_solution/training/models/best150.pt",
                'off_pos': [0.0, 0.0004, -0.05795],
                'off_quat': [0.17785, 0.00505, -0.02738, -0.98366],
                'z_approach_1': 0.02,
                'z_approach_2': 0.005,
                'z_plug': -0.045,
                'cable_tip_frame': "cable_0/sfp_tip_link",
                'search_insert_strategy_1' : "_spiral_search_and_insert_2d",
                'spiral_max_radius_1': 0.003,
                'spiral_max_radius_2': 0.005,
                'spiral_stiffness_1': [300.0, 300.0, 80.0, 200.0, 200.0, 200.0],
                'spiral_damping_1': [40.0, 40.0, 15.0, 30.0, 30.0, 30.0],
                'spiral_stiffness_2': [300.0, 300.0, 120.0, 200.0, 200.0, 200.0],
                'spiral_damping_2': [40.0, 40.0, 20.0, 30.0, 30.0, 30.0],
                'spiral_steps_1': 120,
                'spiral_steps_2': 250
            }
        }

        # Load Models
        self._models = {}
        for c_type, cfg in self._configs.items():
            self.get_logger().info(f"Lade YOLO Modell [{c_type}]: {cfg['model_path']}")
            try:
                self._models[c_type] = YOLO(cfg['model_path'])
                # Warmup to avoid first-call latency
                self._models[c_type].predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
            except Exception as e:
                self.get_logger().error(f"Failed to load YOLO model for '{c_type}': {e}")
                self._models[c_type] = None

    def on_cleanup(self):
        """Clean up policy state on lifecycle cleanup.
        
        This is called when transitioning from INACTIVE to UNCONFIGURED state.
        We use this to properly clean up cached TF data and old state that might
        persist between trials.
        """
        self.get_logger().info("PlugIn on_cleanup: Clearing cached TF data and state")
        
        # Clear cached TF frame data to avoid using stale transforms after simulation reset
        self._tf_cam_frame_cache.clear()
        
        # Reset last debug timestamp to restart debugging cycle
        self._last_tip_offset_debug_monotonic_s = None
        
        self.get_logger().info("PlugIn on_cleanup: Done")

    # --- Triangulation & Detection ---
    def _get_cam_frame_data(self, cam_full_name):
        '''
        Try to get the camera pose in base_link frame.
        Returns None if TF is not available.
        '''
        cached = self._tf_cam_frame_cache.get(cam_full_name)
        try:
            target = f"{cam_full_name}/optical" 
            trans = self._parent_node._tf_buffer.lookup_transform("base_link", target, Time())
            data = {
                "pos": np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]),
                "rot": R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]).as_matrix()
            }
            self._tf_cam_frame_cache[cam_full_name] = data
            return data
        except Exception as e:
            return cached

    def detect_ports(self, observation, cable_type):
        if observation is None:
            return {}, 0, 0

        # --- Intrinsics ---
        for cam in self._camera_names:
            attr = f"{cam}_camera_info"
            if hasattr(observation, attr):
                msg = getattr(observation, attr)
                self._cam_intrinsics[cam] = {
                    'fx': msg.k[0],
                    'fy': msg.k[4],
                    'cx': msg.k[2],
                    'cy': msg.k[5]
                }

        model = getattr(self, '_models', {}).get(cable_type)
        if model is None:
            self.get_logger().error(f"No YOLO model available for cable_type='{cable_type}'")
            return {}, 0, 0

        # 🔥 NEU: zählen
        unique_classes = set()
        total_detections = 0

        separated = {0: {}, 1: {}}  # bleibt wie bei dir

        # --- YOLO ---
        for cam in self._camera_names:
            img_msg = getattr(observation, f"{cam}_image", None)
            if img_msg is None:
                continue

            cv_img = self._bridge.imgmsg_to_cv2(img_msg, "bgr8")
            res = model.predict(cv_img, conf=0.8, verbose=False)[0]

            for j, box in enumerate(res.boxes):
                cls = int(box.cls[0])

                # 🔥 zählen
                total_detections += 1
                unique_classes.add(cls)

                if cls in separated:
                    separated[cls][cam] = res.keypoints.xy[j].cpu().numpy()

        # --- Triangulation (UNVERÄNDERT) ---
        found_ports = {}

        for pid, cams in separated.items():
            if len(cams) < 2:
                continue

            pts_3d = []
            for k in range(4):
                rays = []

                for cam_name, kpts in cams.items():
                    fdata = self._get_cam_frame_data(f"{cam_name}_camera")

                    if fdata and cam_name in self._cam_intrinsics:
                        u, v = kpts[k]
                        intr = self._cam_intrinsics[cam_name]

                        d_cam = np.array([
                            (u - intr['cx']) / intr['fx'],
                            (v - intr['cy']) / intr['fy'],
                            1.0
                        ])

                        d_world = fdata["rot"] @ d_cam
                        d_world /= np.linalg.norm(d_world)

                        rays.append({
                            "origin": fdata["pos"],
                            "direction": d_world
                        })

                if len(rays) >= 2:
                    I = np.eye(3)
                    A = np.zeros((3, 3))
                    b = np.zeros(3)

                    for r in rays:
                        M = I - np.outer(r["direction"], r["direction"])
                        A += M
                        b += M @ r["origin"]

                    pts_3d.append(np.linalg.lstsq(A, b, rcond=None)[0])

            if len(pts_3d) == 4:
                corners = np.array(pts_3d)
                center = np.mean(corners, axis=0)

                vec_x = corners[1] - corners[0]
                vec_x[2] = 0
                vec_x /= np.linalg.norm(vec_x)

                vec_z = np.array([0.0, 0.0, -1.0])
                vec_y = np.cross(vec_z, vec_x)
                vec_y /= np.linalg.norm(vec_y)

                rot_matrix = np.stack([vec_x, vec_y, vec_z], axis=1)

                found_ports[pid] = {
                    "pos": center,
                    "quat": R.from_matrix(rot_matrix).as_quat()
                }

        # 🔥 NEU: Rückgabe erweitert
        return found_ports, total_detections, len(unique_classes)

    # --- Force Monitoring and Threshold ---
    def _check_force_threshold(self, observation):
        """
        Prints a warining when forces exceed 20N
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

    # --- Motion Control ---
    def _move_tcp_smooth_cartesian(self, pos, quat,
                                    move_robot, 
                                    get_observation, 
                                    stiffness, damping, 
                                    n_steps=80, 
                                    label="Target", 
                                    obs=None,
                                    debug_port_type=None):
        """
        Soft Cartesian movement
        1. set motion_update with target pose and stiffness/damping
        2. loop: move_robot + check forces + check distance to target
         - every 25 steps print distance to target
         - if distance < 1mm, consider target reached and return
         - if forces exceed threshold, print warning but continue
        """
        # 1. Setup MotionUpdate
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

        self.get_logger().info(f"==> Move to {label} (smooth): P=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

        dist = float('inf')

        # 2. Loop: Move + Monitor
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
        Calculates TCP offset to cable tip pose from Ground Truth TF
        1. Get TF from cable tip to TCP (greifer/tcp)
        2. Build transformation matrices:
            - A: Base -> Port (from detection)
            - B: Cable Tip -> TCP (from TF)
            - C: Calculate Target TCP Pose: Base -> TCP = Base -> Port * Port -> TCP
        """
        
        # 1. Get TF from cable tip to TCP
        timeout = Duration(seconds=1.0)
        tf_cable_to_tcp = self._parent_node._tf_buffer.lookup_transform(
            cable_tip_frame, 
            "gripper/tcp", 
            Time(),
            timeout=timeout
        )
        
        # 2. Transformations
        # A: Transformation Base -> Port
        mat_base_to_port = np.eye(4)
        mat_base_to_port[:3, :3] = R.from_quat(port_quat).as_matrix()
        mat_base_to_port[:3, 3] = port_pos
        
        # B: Transformation Cable Tip -> TCP
        mat_cable_to_tcp = np.eye(4)
        q_off = tf_cable_to_tcp.transform.rotation
        mat_cable_to_tcp[:3, :3] = R.from_quat([q_off.x, q_off.y, q_off.z, q_off.w]).as_matrix()
        t_off = tf_cable_to_tcp.transform.translation
        mat_cable_to_tcp[:3, 3] = [t_off.x, t_off.y, t_off.z]

        self.get_logger().info(f"TCP zu PLUG TIP TRANSLATION: X: {t_off.x}, Y: {t_off.y}, Z: {t_off.z}")
        self.get_logger().info(f"TCP zu PLUG TIP ORIENTATION: qX: {q_off.x}, qY: {q_off.y}, qZ: {q_off.z}, qW: {q_off.w}")
        
        # C: Calculate Target TCP Pose
        target_matrix = mat_base_to_port @ mat_cable_to_tcp
        
        target_pos = target_matrix[:3, 3]
        target_quat = R.from_matrix(target_matrix[:3, :3]).as_quat()
        
        return target_pos, target_quat
    
    def _get_tcp_goal_pose_hardcoded(self, port_pos, port_quat, port_type, off_pos=None, off_quat=None):
        """
        Calculates TCP goal pose from port pose using hardcoded offsets. (No Ground Truth TF needed)
        1. Define fixed offset for sc and sfp
        2. Build transformation matrices:
            - A: Base -> Port (from detection)
            - B: Cable Tip -> TCP (from hardcoded values)
            - C: Calculate Target TCP Pose

        """
        # 1. Define fixed offset (Tip -> TCP)
        off_t, off_q = self._get_tip_to_tcp_offset_hardcoded(port_type, off_pos, off_quat)


        # 2. Transformations
        # A: Transformation Base -> Port (from detection)
        mat_base_to_port = np.eye(4)
        mat_base_to_port[:3, :3] = R.from_quat(port_quat).as_matrix()
        mat_base_to_port[:3, 3] = port_pos
        
        # B: Transformation Cable Tip -> TCP (from hardcoded offsets)
        mat_cable_to_tcp = np.eye(4)
        mat_cable_to_tcp[:3, :3] = R.from_quat(off_q).as_matrix()
        mat_cable_to_tcp[:3, 3] = off_t
        
        # C: Calculate Target TCP Pose
        target_matrix = mat_base_to_port @ mat_cable_to_tcp
        

        # DEBUG: Correct the offset in world frame estimated from gazebo
        if port_type == 'sc':
            delta_t_base = np.array([-0.016, -0.007, 0.0])
            delta_R_base = R.from_euler('z', -4, degrees=True).as_matrix()
            target_matrix[:3, 3] += delta_t_base
            target_matrix[:3, :3] = target_matrix[:3, :3] @ delta_R_base


        target_pos = target_matrix[:3, 3]
        target_quat = R.from_matrix(target_matrix[:3, :3]).as_quat()

        return target_pos, target_quat

    def _get_tip_to_tcp_offset_hardcoded(self, port_type, off_pos=None, off_quat=None):
        """Hardcoded transform from cable tip frame to gripper/tcp frame.

        Returns:
            (t, q):
              - t: np.ndarray shape (3,) translation in meters (Tip -> TCP), expressed in tip frame
              - q: np.ndarray shape (4,) quaternion xyzw (Tip -> TCP)
        """
        t = np.array(off_pos)
        q = np.array(off_quat)
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm

        return t, q


    def _spiral_search_and_insert(self, center_pos, 
                                  quat, 
                                  move_robot,
                                  get_observation,
                                  stiff_spiral=[300.0, 300.0, 80.0, 200.0, 200.0, 200.0], 
                                  damp_spiral=[40.0, 40.0, 15.0, 30.0, 30.0, 30.0],
                                  spiral_steps=120,
                                  max_radius=0.003,
                                  n_turns=3,
                                  label="Spiral",
                                  obs=None,
                                  debug_port_type=None):
        """
        Moves the TCP in a spiral pattern around center_pos on a fixed Z plane
        - Uses low z stiffness or gentle contact during search
        - Slips in once it finds the hole (z offset at plug depth)
        1. Define controller stiffness and damping
        2. Loop (Move in Spiral + Monitor):
            - Calculate spiral offset (linear increase in radius)
            - Move robot to new position
            - Check forces and distance to center
            - If distance < 1mm, consider target reached and return
        """

        self.get_logger().info(f"==> Start Spiral search for {label} | max_radius={max_radius*1000:.1f}mm | turns={n_turns} | steps={spiral_steps}")

        t_vals = np.linspace(0, n_turns * 2 * np.pi, spiral_steps)

        # 2. Loop: Move in Spiral + Monitor
        for idx, t in enumerate(t_vals):
            # Calculate spiral offset (linear increase in radius)
            r = (t / (n_turns * 2 * np.pi)) * max_radius
            dx = r * np.cos(t)
            dy = r * np.sin(t)

            search_pos = center_pos.copy()
            search_pos[0] += dx
            search_pos[1] += dy

            # Move Robot, Z stays fixed at center_pos[2]
            motion_update = self._build_motion_update(search_pos, quat, stiff_spiral, damp_spiral)
            move_robot(motion_update=motion_update)

            # Monitor: Check forces and distance to center
            obs = get_observation()
            self._check_force_threshold(obs)

            curr = obs.controller_state.tcp_pose.position
            dist = math.sqrt(
                (curr.x - search_pos[0])**2 +
                (curr.y - search_pos[1])**2 +
                (curr.z - center_pos[2])**2  # compare Z always against center_pos
            )

            if idx % 20 == 0:
                self.get_logger().info(f"    [{idx}/{spiral_steps}] r={r*1000:.2f}mm | dx={dx*1000:.1f}mm dy={dy*1000:.1f}mm | dist={dist*1000:.2f}mm")

            self.sleep_for(0.05)

        # Calculate final distance to center after spiral search
        obs = get_observation()
        curr = obs.controller_state.tcp_pose.position
        final_dist = math.sqrt(
            (curr.x - center_pos[0])**2 +
            (curr.y - center_pos[1])**2 +
            (curr.z - center_pos[2])**2
        )

        self.get_logger().info(f"    [{label}] Spiral search done, final distance: {final_dist*1000:.2f}mm")


        return final_dist
    
    def _build_motion_update(self, pos, quat, stiffness, damping):
        """
        Helper: MotionUpdate from pos/quat/stiffness/damping
        1. Create MotionUpdate message
        2. Fill in target pose and stiffness/damping matrices
        3. Return MotionUpdate
        """
        # 1. Create MotionUpdate message
        motion_update = MotionUpdate()
        # 2. Fill in target pose and stiffness/damping matrices
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

        # 3. Return MotionUpdate

        return motion_update


    # --- Main ---
    def insert_cable(self,
                     task: Task,
                     get_observation: GetObservationCallback,
                     move_robot: MoveRobotCallback,
                     send_feedback: SendFeedbackCallback):
        '''
        Main function that is called during the evaluation

        1. Get task from aic_task_interfaces and print debug info
        2. Run detection
        3. Select target port (try to match port_name, otherwise take first seen)
        4. Calculate TCP goal pose (with or without Ground Truth)
        5. Calculate approach pose (same XY, but Z raised by approach offset)
        6. Calculate plug position for later use in spiral search (same XY, but Z at plug depth)
        7.1 Move to approach pose (smoothly, with low stiffness)
        7.2 Increase stiffness for better allignemt before spiral search
        8. Spiral Search and Insert
        9. Check final distance to plug position after spiral search
        10. If not successful, try to move down to plug position with low stiffness
        '''

        # 0 Sleep to ensure everything is ready
        self.sleep_for(1.0)

        # 1. Task Debug Info
        self.get_logger().info("============================================================")
        self.get_logger().info(f"STARTING NEW TASK: {task.cable_type.upper()}")
        self.get_logger().info(f"Port: {task.port_name} | Plug: {task.plug_name}")
        self.get_logger().info("=================   ===========================================")

        
        c_type = task.port_type.lower()
        if c_type not in self._configs:
            self.get_logger().error(f"Kabeltyp '{c_type}' ist nicht konfiguriert!")
            return False
        
        cfg = self._configs[c_type]

        # 2. Detection
        obs = get_observation()
        found_ports, total_detections, num_unique_ports = self.detect_ports(obs, c_type)

        self.get_logger().info(f"DEBUG: YOLO detections total: {total_detections}")
        self.get_logger().info(f"DEBUG: Unique port classes: {num_unique_ports}")
        self.get_logger().info(f"DEBUG: Triangulated ports: {len(found_ports)}")

        # Check if multiple cards spawn
        if total_detections > 8:
            self.get_logger().warning(
                f"⚠️ MEHR ALS 2 PORTS GESPAWNT! ({total_detections})"
            )

            # Hard abort (do not report success)
            return True

        # 3. Select Target Port
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


        # 4. Calculate TCP Goal Pose
        tcp_pos, tcp_quat = self._get_tcp_goal_pose_hardcoded(target_port["pos"], target_port["quat"], c_type, off_pos=cfg['off_pos'], off_quat=cfg['off_quat'])

        self.get_logger().info(f"Calculated TCP goal pose: Pos={tcp_pos}, Quat={tcp_quat}")

        # Send Feedback about starting the insertion process
        send_feedback(f"Starting {c_type} insertion...")

        # 5. Calculate approach pose(s) (same XY, but Z raised by approach offset)
        # Support either a single-stage approach (z_approach) or a two-stage approach (z_approach_1, z_approach_2)
        approach_poses = []
        if 'z_approach_1' in cfg and 'z_approach_2' in cfg:
            approach_pos_1 = tcp_pos.copy()
            approach_pos_1[2] += cfg['z_approach_1']
            approach_poses.append((approach_pos_1, "Approach-1"))

            approach_pos_2 = tcp_pos.copy()
            approach_pos_2[2] += cfg['z_approach_2']
            approach_poses.append((approach_pos_2, "Approach-2"))
        else:
            approach_pos = tcp_pos.copy()
            approach_pos[2] += cfg['z_approach']
            approach_poses.append((approach_pos, "Approach"))

        # 6. Calculate plug position for later use in spiral search (same XY, but Z at plug depth)
        plug_pos = tcp_pos.copy()
        plug_pos[2] += cfg['z_plug']
        

        # 7.1 Move to approach pose(s) (smoothly, with low stiffness)
        last_approach_pos = None
        for approach_pos, approach_label in approach_poses:
            last_approach_pos = approach_pos
            self._move_tcp_smooth_cartesian(
                approach_pos, tcp_quat, move_robot, get_observation,
                stiffness=[150.0, 150.0, 150.0, 80.0, 80.0, 80.0],
                damping=[40.0,  40.0,  40.0,  20.0, 20.0, 20.0],
                n_steps=60,
                label=approach_label,
                obs=obs,
                debug_port_type=c_type
            )

        # 7.2 Increase stiffness for better allignemt before spiral search
        self._move_tcp_smooth_cartesian(
            last_approach_pos, tcp_quat, move_robot, get_observation,
            stiffness=[400.0]*6,
            damping=[50.0]*6,
            n_steps=20,          # kurz, nur zum Einregeln
            label="Stiffening",
            obs=obs,
            debug_port_type=c_type
        )

        # 8. Spiral Search and Insert
        spiral_max_radius = cfg.get('spiral_max_radius', 0.003)
        spiral_stiffness_1 = cfg.get('spiral_stiffness_1', [300.0, 300.0, 80.0, 200.0, 200.0, 200.0])
        spiral_damping_1 = cfg.get('spiral_damping_1', [40.0, 40.0, 15.0, 30.0, 30.0, 30.0])
        spiral_steps_1 = cfg.get('spiral_steps_1', 150)
        final_dist = self._spiral_search_and_insert(
            center_pos=plug_pos,
            quat=tcp_quat,
            move_robot=move_robot,
            get_observation=get_observation,
            stiff_spiral=spiral_stiffness_1,
            damp_spiral=spiral_damping_1,
            spiral_steps=spiral_steps_1,
            max_radius=spiral_max_radius,
            label=c_type,
            obs=obs,
            debug_port_type=c_type
        )
        
        # 9. Check final distance to plug position after spiral search
        if final_dist < 0.01:
            self.get_logger().info("============================================================")
            self.get_logger().info(f"SUCCESS - Inserted after spiral search:({c_type})")
            self.get_logger().info("============================================================")
            return True

        # 10. If not successful, try to move down to plug position with low stiffness (slipping in)
        else:
            final_plug_position = plug_pos.copy()
            final_plug_position[2] -= 0.01 
            self._move_tcp_smooth_cartesian(
            final_plug_position, tcp_quat, move_robot, get_observation,
            stiffness=[100.0, 100.0,  80.0, 300.0, 300.0, 300.0],
            damping=[ 40.0,  40.0,  15.0,  40.0,  40.0,  40.0],
            n_steps=40,
            label="Plug-In",
            obs=obs,
            debug_port_type=c_type
        )

        # 11 Final Check
        obs = get_observation()
        curr = obs.controller_state.tcp_pose.position
        final_dist = math.sqrt(
            (curr.x - plug_pos[0])**2 +
            (curr.y - plug_pos[1])**2 +
            (curr.z - plug_pos[2])**2
        )

        if final_dist < 0.01:
            self.get_logger().info("============================================================")
            self.get_logger().info(f"SUCCESS - Inserted on second try:({c_type}) | Final distance: {final_dist*1000:.2f}mm")
            self.get_logger().info("============================================================")
            return True
        else:
            # 12 Spiral Search and Insert with larger radius and modified parameters
            spiral_stiffness_2 = cfg.get('spiral_stiffness_2', [200.0, 200.0, 50.0, 150.0, 150.0, 150.0])
            spiral_damping_2 = cfg.get('spiral_damping_2', [30.0, 30.0, 10.0, 20.0, 20.0, 20.0])
            spiral_max_radius_2 = cfg.get('spiral_max_radius_2', 0.007)
            spiral_steps_2 = cfg.get('spiral_steps_2', 250) 
            final_dist = self._spiral_search_and_insert(
                center_pos=plug_pos,
                quat=tcp_quat,
                move_robot=move_robot,
                get_observation=get_observation,
                max_radius=spiral_max_radius_2,
                stiff_spiral=spiral_stiffness_2,
                damp_spiral=spiral_damping_2,
                spiral_steps=spiral_steps_2,
                label=c_type,
                obs=obs,
                debug_port_type=c_type
            )

        # 13 Force insert again
        if final_dist < 0.01:
            self.get_logger().info("============================================================")
            self.get_logger().info(f"SUCCESS - Inserted after secondspiral search:({c_type})")
            self.get_logger().info("============================================================")
            return True

        # 13. If not successful, try to move down to plug position with low stiffness (slipping in)
        else:
            final_plug_position = plug_pos.copy()
            final_plug_position[2] -= 0.01 
            self._move_tcp_smooth_cartesian(
            final_plug_position, tcp_quat, move_robot, get_observation,
            stiffness=[100.0, 100.0,  80.0, 300.0, 300.0, 300.0],
            damping=[ 40.0,  40.0,  15.0,  40.0,  40.0,  40.0],
            n_steps=40,
            label="Plug-In",
            obs=obs,
            debug_port_type=c_type
        )
            
        # 14 Final Check
        obs = get_observation()
        curr = obs.controller_state.tcp_pose.position
        final_dist = math.sqrt(
            (curr.x - plug_pos[0])**2 +
            (curr.y - plug_pos[1])**2 +
            (curr.z - plug_pos[2])**2
        )

        if final_dist < 0.01:
            self.get_logger().info("============================================================")
            self.get_logger().info(f"SUCCESS - Inserted on last try:({c_type}) | Final distance: {final_dist*1000:.2f}mm")
            self.get_logger().info("============================================================")
            return True
        else:
            self.get_logger().info("============================================================")
            self.get_logger().info(f"FAILED - Could not insert:({c_type}) | Final distance: {final_dist*1000:.2f}mm")
            self.get_logger().info("============================================================")
            return True