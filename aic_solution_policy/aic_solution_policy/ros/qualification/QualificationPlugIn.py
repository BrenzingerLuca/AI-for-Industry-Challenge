"""Qualification-round insertion policy.

Detects the SFP/SC port opening with a custom-trained YOLO keypoint model
and multi-camera triangulation (see vision.PortDetector), derives the TCP
goal pose from a hardcoded tip-to-TCP offset, and inserts with a two-stage
spiral search. The shared vision-based residual regressor refines the
approach pose right before the spiral search.

This is what qualified the team for Phase 1 (27/160 teams); Phase 1 itself
hands port detection off to Intrinsic FlowState instead (see Phase1PlugIn).
"""

import math

import numpy as np
from aic_task_interfaces.msg import Task
from aic_model.policy import (
    Policy,
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from rclpy.duration import Duration
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

from ..residual_offset_model import ResidualOffsetCorrector, apply_predicted_correction
from .config import CONNECTOR_CONFIGS
from .motion import move_tcp_smooth_cartesian, spiral_search_and_insert
from .vision import PortDetector


class QualificationPlugIn(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)

        from cv_bridge import CvBridge
        from ultralytics import YOLO

        self.get_logger().info("QualificationPlugIn Policy initialised")
        self._bridge = CvBridge()
        self._camera_names = ['left', 'center', 'right']
        self._configs = CONNECTOR_CONFIGS

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

        self._port_detector = PortDetector(
            tf_buffer=self._parent_node._tf_buffer,
            bridge=self._bridge,
            models=self._models,
            camera_names=self._camera_names,
            logger=self.get_logger(),
        )

        # ROS param so the correction can be A/B-tested without rebuilding the package.
        self._parent_node.declare_parameter('residual_correction.enabled', True)

        # Trained offset-correction model(s) from residual_policy.ipynb, one per
        # connector type.
        self._offset_corrector = ResidualOffsetCorrector(
            bridge=self._bridge,
            logger=self.get_logger(),
            checkpoint_paths={c: cfg.get('residual_model_path') for c, cfg in self._configs.items()},
            skip_description="approach-pose correction",
        )

    def on_cleanup(self):
        """Drop cached TF data on lifecycle cleanup (INACTIVE <- UNCONFIGURED) so a
        simulation reset can't leave stale camera transforms for the next trial.
        """
        self.get_logger().info("QualificationPlugIn on_cleanup: Clearing cached TF data")
        self._port_detector.reset()
        self.get_logger().info("QualificationPlugIn on_cleanup: Done")

    # --- TCP goal pose from the detected port ---

    def _get_tcp_goal_pose(self, port_pos, port_quat, cable_tip_frame):
        """Ground-truth variant of the goal-pose calculation: reads the actual
        cable-tip-to-TCP transform from TF instead of the hardcoded offset
        below. Not on the current call path (kept for ground_truth:=true
        debugging) -- see _get_tcp_goal_pose_hardcoded for what insert_cable
        actually uses.
        """
        timeout = Duration(seconds=1.0)
        tf_cable_to_tcp = self._parent_node._tf_buffer.lookup_transform(
            cable_tip_frame,
            "gripper/tcp",
            Time(),
            timeout=timeout
        )

        mat_base_to_port = np.eye(4)
        mat_base_to_port[:3, :3] = R.from_quat(port_quat).as_matrix()
        mat_base_to_port[:3, 3] = port_pos

        mat_cable_to_tcp = np.eye(4)
        q_off = tf_cable_to_tcp.transform.rotation
        mat_cable_to_tcp[:3, :3] = R.from_quat([q_off.x, q_off.y, q_off.z, q_off.w]).as_matrix()
        t_off = tf_cable_to_tcp.transform.translation
        mat_cable_to_tcp[:3, 3] = [t_off.x, t_off.y, t_off.z]

        self.get_logger().info(f"TCP zu PLUG TIP TRANSLATION: X: {t_off.x}, Y: {t_off.y}, Z: {t_off.z}")
        self.get_logger().info(f"TCP zu PLUG TIP ORIENTATION: qX: {q_off.x}, qY: {q_off.y}, qZ: {q_off.z}, qW: {q_off.w}")

        target_matrix = mat_base_to_port @ mat_cable_to_tcp
        target_pos = target_matrix[:3, 3]
        target_quat = R.from_matrix(target_matrix[:3, :3]).as_quat()

        return target_pos, target_quat

    def _get_tcp_goal_pose_hardcoded(self, port_pos, port_quat, port_type, off_pos=None, off_quat=None):
        """TCP goal pose from the detected port pose plus a hardcoded tip-to-TCP
        offset (no ground-truth TF needed) -- this is the version insert_cable uses.
        """
        off_t, off_q = self._get_tip_to_tcp_offset_hardcoded(port_type, off_pos, off_quat)

        mat_base_to_port = np.eye(4)
        mat_base_to_port[:3, :3] = R.from_quat(port_quat).as_matrix()
        mat_base_to_port[:3, 3] = port_pos

        mat_cable_to_tcp = np.eye(4)
        mat_cable_to_tcp[:3, :3] = R.from_quat(off_q).as_matrix()
        mat_cable_to_tcp[:3, 3] = off_t

        target_matrix = mat_base_to_port @ mat_cable_to_tcp

        # Small empirical correction for the 'sc' port pose as estimated in Gazebo.
        if port_type == 'sc':
            delta_t_base = np.array([-0.016, -0.007, 0.0])
            delta_R_base = R.from_euler('z', -4, degrees=True).as_matrix()
            target_matrix[:3, 3] += delta_t_base
            target_matrix[:3, :3] = target_matrix[:3, :3] @ delta_R_base

        target_pos = target_matrix[:3, 3]
        target_quat = R.from_matrix(target_matrix[:3, :3]).as_quat()

        return target_pos, target_quat

    def _get_tip_to_tcp_offset_hardcoded(self, port_type, off_pos=None, off_quat=None):
        """Hardcoded transform from the cable tip frame to gripper/tcp.

        Returns (t, q): translation in meters and quaternion xyzw, both
        Tip -> TCP, expressed in the tip frame.
        """
        t = np.array(off_pos)
        q = np.array(off_quat)
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0:
            q = q / q_norm
        return t, q

    # --- Main ---

    def insert_cable(self,
                      task: Task,
                      get_observation: GetObservationCallback,
                      move_robot: MoveRobotCallback,
                      send_feedback: SendFeedbackCallback):
        """
        1. Get task from aic_task_interfaces and log debug info
        2. Run detection
        3. Select target port (try to match port_name, otherwise take first seen)
        4. Calculate TCP goal pose (hardcoded tip-to-TCP offset)
        5. Calculate approach pose(s) (same XY, but Z raised by approach offset)
        6. Calculate plug position for later use in spiral search (same XY, but Z at plug depth)
        7.1 Move to approach pose (smoothly, with low stiffness)
        7.15 Run the trained offset-correction model once at the approach pose
             and shift tcp_pos/tcp_quat/plug_pos to cancel out its predicted
             tip-to-port offset
        7.2 Increase stiffness for better alignment before spiral search
        8. Spiral search and insert
        9. Check final distance to plug position after spiral search
        10. If not successful, try to move down to plug position with low stiffness
        """
        self.sleep_for(1.0)

        self.get_logger().info("============================================================")
        self.get_logger().info(f"STARTING NEW TASK: {task.cable_type.upper()}")
        self.get_logger().info(f"Port: {task.port_name} | Plug: {task.plug_name}")
        self.get_logger().info("============================================================")

        c_type = task.port_type.lower()
        if c_type not in self._configs:
            self.get_logger().error(f"Kabeltyp '{c_type}' ist nicht konfiguriert!")
            return False

        cfg = self._configs[c_type]

        # 2. Detection
        obs = get_observation()
        found_ports, total_detections, num_unique_ports = self._port_detector.detect(obs, c_type)

        self.get_logger().info(f"DEBUG: YOLO detections total: {total_detections}")
        self.get_logger().info(f"DEBUG: Unique port classes: {num_unique_ports}")
        self.get_logger().info(f"DEBUG: Triangulated ports: {len(found_ports)}")

        # Check if multiple cards spawned
        if total_detections > 8:
            self.get_logger().warning(f"⚠️ MEHR ALS 2 PORTS GESPAWNT! ({total_detections})")
            # Hard abort (do not report success)
            return True

        # 3. Select target port
        try:
            target_id = int(task.port_name.split('_')[-1])
        except Exception:
            target_id = list(found_ports.keys())[0]

        if target_id not in found_ports:
            self.get_logger().info(f"DEBUG: Port {target_id} nicht gesehen. Nehme verfügbaren: {list(found_ports.keys())}")
            target_id = list(found_ports.keys())[0]

        target_port = found_ports[target_id]
        pp, pq = target_port["pos"], target_port["quat"]
        self.get_logger().info(f"ERKANNT: Port {target_id} bei Base-Link: Pos={pp}, Quat={pq}")

        # 4. Calculate TCP goal pose
        tcp_pos, tcp_quat = self._get_tcp_goal_pose_hardcoded(
            target_port["pos"], target_port["quat"], c_type, off_pos=cfg['off_pos'], off_quat=cfg['off_quat']
        )
        self.get_logger().info(f"Calculated TCP goal pose: Pos={tcp_pos}, Quat={tcp_quat}")

        send_feedback(f"Starting {c_type} insertion...")

        # 5. Approach pose(s): same XY, but Z raised. Either a single-stage
        # approach (z_approach) or a two-stage approach (z_approach_1/2).
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

        # 6. Plug position for the spiral search (same XY, Z at plug depth)
        plug_pos = tcp_pos.copy()
        plug_pos[2] += cfg['z_plug']

        # 7.1 Move to approach pose(s), smoothly, with low stiffness
        last_approach_pos = None
        for approach_pos, approach_label in approach_poses:
            last_approach_pos = approach_pos
            move_tcp_smooth_cartesian(
                approach_pos, tcp_quat, move_robot, get_observation,
                stiffness=[150.0, 150.0, 150.0, 80.0, 80.0, 80.0],
                damping=[40.0, 40.0, 40.0, 20.0, 20.0, 20.0],
                sleep_for=self.sleep_for, logger=self.get_logger(),
                n_steps=60, label=approach_label,
            )

        # 7.15 Vision-based residual correction: run the trained offset model on
        # the camera images at the approach pose (ground truth is NOT used here)
        # and shift the target pose to cancel out its predicted tip-to-port
        # offset, so the subsequent stiffening/spiral-search steps start from a
        # better-aligned pose. Z is left untouched.
        residual_enabled = self._parent_node.get_parameter('residual_correction.enabled').value
        z_approach_last = cfg['z_approach_2'] if 'z_approach_2' in cfg else cfg['z_approach']
        if residual_enabled:
            obs = get_observation()
            predicted_offset = self._offset_corrector.predict(obs, c_type)
            if predicted_offset is not None:
                dx, dy, dz, droll, dpitch, dyaw = predicted_offset
                self.get_logger().info(
                    f"Residual model prediction: dxyz=({dx*1e3:.2f},{dy*1e3:.2f},{dz*1e3:.2f})mm "
                    f"rpy=({droll:.1f},{dpitch:.1f},{dyaw:.1f})deg"
                )
                corrected_approach_pos, tcp_quat = apply_predicted_correction(
                    last_approach_pos, tcp_quat, cfg['off_pos'], cfg['off_quat'], predicted_offset
                )
                last_approach_pos = corrected_approach_pos
                tcp_pos = corrected_approach_pos.copy()
                tcp_pos[2] -= z_approach_last
                plug_pos = tcp_pos.copy()
                plug_pos[2] += cfg['z_plug']
            else:
                self.get_logger().warning(f"No residual correction available for '{c_type}', skipping")

        # 7.2 Increase stiffness for better alignment before spiral search --
        # also serves as the move to the (possibly corrected) last_approach_pos.
        move_tcp_smooth_cartesian(
            last_approach_pos, tcp_quat, move_robot, get_observation,
            stiffness=[400.0] * 6,
            damping=[50.0] * 6,
            sleep_for=self.sleep_for, logger=self.get_logger(),
            n_steps=20, label="Stiffening",
        )

        # 8. Spiral search and insert
        spiral_max_radius = cfg.get('spiral_max_radius', 0.003)
        spiral_stiffness_1 = cfg.get('spiral_stiffness_1', [300.0, 300.0, 80.0, 200.0, 200.0, 200.0])
        spiral_damping_1 = cfg.get('spiral_damping_1', [40.0, 40.0, 15.0, 30.0, 30.0, 30.0])
        spiral_steps_1 = cfg.get('spiral_steps_1', 150)
        final_dist = spiral_search_and_insert(
            center_pos=plug_pos, quat=tcp_quat,
            move_robot=move_robot, get_observation=get_observation,
            sleep_for=self.sleep_for, logger=self.get_logger(),
            stiff_spiral=spiral_stiffness_1, damp_spiral=spiral_damping_1,
            spiral_steps=spiral_steps_1, max_radius=spiral_max_radius,
            label=c_type,
        )

        # 9. Check final distance to plug position after spiral search
        if final_dist < 0.01:
            self.get_logger().info("============================================================")
            self.get_logger().info(f"SUCCESS - Inserted after spiral search:({c_type})")
            self.get_logger().info("============================================================")
            return True

        # 10. If not successful, slip in by moving down with low stiffness
        else:
            final_plug_position = plug_pos.copy()
            final_plug_position[2] -= 0.01
            move_tcp_smooth_cartesian(
                final_plug_position, tcp_quat, move_robot, get_observation,
                stiffness=[100.0, 100.0, 80.0, 300.0, 300.0, 300.0],
                damping=[40.0, 40.0, 15.0, 40.0, 40.0, 40.0],
                sleep_for=self.sleep_for, logger=self.get_logger(),
                n_steps=40, label="Plug-In",
            )

        # 11. Final check
        obs = get_observation()
        curr = obs.controller_state.tcp_pose.position
        final_dist = math.sqrt(
            (curr.x - plug_pos[0]) ** 2 +
            (curr.y - plug_pos[1]) ** 2 +
            (curr.z - plug_pos[2]) ** 2
        )

        if final_dist < 0.01:
            self.get_logger().info("============================================================")
            self.get_logger().info(f"SUCCESS - Inserted on second try:({c_type}) | Final distance: {final_dist*1000:.2f}mm")
            self.get_logger().info("============================================================")
            return True
        else:
            # 12. Spiral search and insert again, with larger radius/modified parameters
            spiral_stiffness_2 = cfg.get('spiral_stiffness_2', [200.0, 200.0, 50.0, 150.0, 150.0, 150.0])
            spiral_damping_2 = cfg.get('spiral_damping_2', [30.0, 30.0, 10.0, 20.0, 20.0, 20.0])
            spiral_max_radius_2 = cfg.get('spiral_max_radius_2', 0.007)
            spiral_steps_2 = cfg.get('spiral_steps_2', 250)
            final_dist = spiral_search_and_insert(
                center_pos=plug_pos, quat=tcp_quat,
                move_robot=move_robot, get_observation=get_observation,
                sleep_for=self.sleep_for, logger=self.get_logger(),
                max_radius=spiral_max_radius_2,
                stiff_spiral=spiral_stiffness_2, damp_spiral=spiral_damping_2,
                spiral_steps=spiral_steps_2, label=c_type,
            )

        # 13. Force insert again
        if final_dist < 0.01:
            self.get_logger().info("============================================================")
            self.get_logger().info(f"SUCCESS - Inserted after secondspiral search:({c_type})")
            self.get_logger().info("============================================================")
            return True

        # If still not successful, slip in by moving down with low stiffness
        else:
            final_plug_position = plug_pos.copy()
            final_plug_position[2] -= 0.01
            move_tcp_smooth_cartesian(
                final_plug_position, tcp_quat, move_robot, get_observation,
                stiffness=[100.0, 100.0, 80.0, 300.0, 300.0, 300.0],
                damping=[40.0, 40.0, 15.0, 40.0, 40.0, 40.0],
                sleep_for=self.sleep_for, logger=self.get_logger(),
                n_steps=40, label="Plug-In",
            )

        # 14. Final check
        obs = get_observation()
        curr = obs.controller_state.tcp_pose.position
        final_dist = math.sqrt(
            (curr.x - plug_pos[0]) ** 2 +
            (curr.y - plug_pos[1]) ** 2 +
            (curr.z - plug_pos[2]) ** 2
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
