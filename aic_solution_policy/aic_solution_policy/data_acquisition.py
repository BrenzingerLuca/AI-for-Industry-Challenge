import math
import os
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation as R
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException

from aic_control_interfaces.msg import MotionUpdate
from aic_task_interfaces.msg import Task
from aic_model.policy import (
    Policy,
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)


# NOTE: aic_model's dynamic policy loader imports the module named by the
# "policy" ROS parameter and then looks for a class whose name matches the
# LAST dotted component of that module path exactly (see aic_model.py,
# `expected_policy_class_name = policy_module_name.split(".")[-1]`). To be
# loadable as `-p policy:=aic_solution_policy.data_acquisition` the class
# living in this file must therefore be named `data_acquisition`, not
# `DataAcquisition` -- this mirrors how PlugIn.py must contain `class PlugIn`.
class data_acquisition(Policy):
    """Generates training data for an offset-correcting insertion policy.

    For each configured port:
      1. Look up the ground-truth port pose and the ground-truth cable-tip
         pose (published on /tf when the sim is launched with
         `ground_truth:=true`) and move the TCP so the tip is perfectly
         aligned with the port, `approach_height_m` above it.
      2. Repeatedly perturb that aligned pose by a random tip-frame offset
         (translation within +/-offset_{x,y,z}_range_m, rotation within
         +/-offset_{roll,pitch,yaw}_range_deg, each axis independent) under
         high stiffness, and at each resulting pose record the three camera
         images plus the *actual* (measured/TF) tip, TCP and port poses --
         not the commanded target -- into an HDF5 dataset.

    Unlike PlugIn.py this policy does not run YOLO detection or attempt
    insertion: since ground truth TF is required anyway, exact poses are
    read directly from /tf instead of being triangulated from camera
    detections, which also avoids the heavy ultralytics/torch import.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)

        from cv_bridge import CvBridge

        self._bridge = CvBridge()
        self._camera_names = ['left', 'center', 'right']

        # Ground-truth cable-tip TF frame per connector type. There is only
        # ever one cable spawned in the current sim setup (attached to the
        # gripper), and it always shows up as "cable_0" regardless of the
        # task's cable_name -- this matches the frame already validated by
        # PlugIn.py's ground-truth path (_get_tcp_goal_pose).
        self._configs = {
            'sc': {'cable_tip_frame': "cable_0/sc_tip_link"},
            'sfp': {'cable_tip_frame': "cable_0/sfp_tip_link"},
        }

        node = self._parent_node
        node.declare_parameter(
            'data_acquisition.output_dir',
            "/home/intrinsic/ws_aic/src/aic/aic_solution/dataset/hdf5",
        )
        node.declare_parameter('data_acquisition.cable_type', 'sfp')
        # Each entry is "target_module_name:port_name", e.g. one NIC card mount
        # actually has *two* SFP ports (sfp_port_0 and sfp_port_1) -- pass
        # exactly the (card, port) pairs you want scanned, e.g.:
        #   -p 'data_acquisition.ports:=[nic_card_mount_0:sfp_port_0,nic_card_mount_0:sfp_port_1]'
        # Default below assumes 2 ports/card across all 5 cards; if a port
        # name here doesn't exist on the board, that pair just gets skipped
        # (with a warning) rather than aborting the whole run.
        node.declare_parameter('data_acquisition.ports', [
            'nic_card_mount_0:sfp_port_0', 'nic_card_mount_0:sfp_port_1',
            'nic_card_mount_1:sfp_port_0', 'nic_card_mount_1:sfp_port_1',
            'nic_card_mount_2:sfp_port_0', 'nic_card_mount_2:sfp_port_1',
            'nic_card_mount_3:sfp_port_0', 'nic_card_mount_3:sfp_port_1',
            'nic_card_mount_4:sfp_port_0', 'nic_card_mount_4:sfp_port_1',
        ])
        node.declare_parameter('data_acquisition.num_samples_per_port', 25)
        node.declare_parameter('data_acquisition.approach_height_m', 0.001)
        # Per-axis perturbation ranges, all expressed in the cable tip's own
        # (local) frame -- see _apply_local_offset for how they're applied.
        # Each sample draws independently and uniformly from +/-range on
        # every axis, so e.g. offset_z_range_m can be tightened relative to
        # x/y without touching the others. Axes (tip-local, right-handed):
        #   x: sideways, along the port opening's width
        #   y: sideways, along the port opening's height
        #   z: forward/back, along the insertion direction
        #   roll:  rotation about the tip-local X axis
        #   pitch: rotation about the tip-local Y axis
        #   yaw:   rotation about the tip-local Z axis (insertion axis)
        node.declare_parameter('data_acquisition.offset_x_range_m', 0.0035)
        node.declare_parameter('data_acquisition.offset_y_range_m', 0.0035)
        node.declare_parameter('data_acquisition.offset_z_range_m', 0.0)
        node.declare_parameter('data_acquisition.offset_roll_range_deg', 5.0)
        node.declare_parameter('data_acquisition.offset_pitch_range_deg', 5.0)
        node.declare_parameter('data_acquisition.offset_yaw_range_deg', 5.0)
        node.declare_parameter('data_acquisition.stiffness', [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        node.declare_parameter('data_acquisition.damping', [45.0, 45.0, 45.0, 30.0, 30.0, 30.0])
        node.declare_parameter('data_acquisition.approach_move_steps', 60)
        node.declare_parameter('data_acquisition.sample_move_steps', 25)
        node.declare_parameter('data_acquisition.tf_wait_timeout_s', 5.0)
        node.declare_parameter('data_acquisition.random_seed', -1)

        # _move_tcp_smooth_cartesian only checks *position* error (<1mm) --
        # that can be satisfied while the impedance controller is still
        # oscillating/settling, so capturing right away can label an image
        # with a tip/port pose that's a few control steps stale by the time
        # it's looked up. Before every capture we instead poll the measured
        # tcp_velocity and wait for it to actually stop, up to a timeout (see
        # _wait_until_settled) -- capturing anyway once the timeout is hit
        # rather than stalling the whole run on one bad settle.
        node.declare_parameter('data_acquisition.settle_max_linear_speed_mps', 0.001)
        node.declare_parameter('data_acquisition.settle_max_angular_speed_dps', 1.0)
        node.declare_parameter('data_acquisition.settle_timeout_s', 3.0)
        node.declare_parameter('data_acquisition.settle_poll_interval_s', 0.05)

        self.get_logger().info("data_acquisition Policy initialised")

    def _param(self, name):
        return self._parent_node.get_parameter(name).value

    # --- TF helpers ---
    def _wait_for_tf(self, frame_id, timeout_sec):
        """Poll until `base_link` -> frame_id is available (ground_truth:=true)."""
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform("base_link", frame_id, Time())
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for TF '{frame_id}'... (requires ground_truth:=true)"
                    )
                attempt += 1
                self.sleep_for(0.1)
        return False

    @staticmethod
    def _tf_to_pos_quat(tf_stamped):
        t = tf_stamped.transform.translation
        q = tf_stamped.transform.rotation
        return np.array([t.x, t.y, t.z]), np.array([q.x, q.y, q.z, q.w])

    def _lookup_pos_quat(self, frame_id):
        tf_stamped = self._parent_node._tf_buffer.lookup_transform("base_link", frame_id, Time())
        return self._tf_to_pos_quat(tf_stamped)

    def _lookup_pos_quat_with_stamp(self, frame_id):
        """Like _lookup_pos_quat, but also returns the *actual* timestamp of
        the transform used (tf2 always stamps its response with the real
        transform's time, even when "latest" was requested) -- so the caller
        can measure how stale it is relative to e.g. an image, without having
        to ask tf2 for an exact past/future stamp itself.

        Ground-truth TF publishing was observed lagging camera timestamps by
        up to ~0.75s under real sim load (PosePublisher not keeping pace with
        camera rendering) -- looking up "at" the image's own stamp reliably
        hit "extrapolation into the future" and fell back to latest anyway,
        so this measures the drift instead of trying to force it to zero."""
        tf_stamped = self._parent_node._tf_buffer.lookup_transform("base_link", frame_id, Time())
        return self._tf_to_pos_quat(tf_stamped), Time.from_msg(tf_stamped.header.stamp)

    # --- Settle check ---
    @staticmethod
    def _get_tcp_speed(observation):
        """Linear (m/s) and angular (deg/s) speed magnitude from the
        controller's measured tcp_velocity."""
        twist = observation.controller_state.tcp_velocity
        linear = math.sqrt(twist.linear.x ** 2 + twist.linear.y ** 2 + twist.linear.z ** 2)
        angular = math.sqrt(twist.angular.x ** 2 + twist.angular.y ** 2 + twist.angular.z ** 2)
        return linear, math.degrees(angular)

    def _wait_until_settled(self, get_observation, max_linear_mps, max_angular_dps,
                             timeout_s, poll_interval_s, label=""):
        """Poll tcp_velocity until it drops below both thresholds, or
        timeout_s elapses -- capturing anyway once timed out rather than
        stalling the run. Returns (obs, settled, linear_mps, angular_dps):
        `obs` is the last observation fetched (the one the caller should
        capture from, to avoid an extra get_observation() call re-opening the
        race this is meant to close), or None if no observation ever came
        through. `linear_mps`/`angular_dps` are NaN if `obs` is None."""
        start = self.time_now()
        timeout = Duration(seconds=timeout_s)
        obs = None
        linear_mps = angular_dps = float('nan')
        while (self.time_now() - start) < timeout:
            obs = get_observation()
            if obs is None:
                self.sleep_for(poll_interval_s)
                continue
            linear_mps, angular_dps = self._get_tcp_speed(obs)
            if linear_mps <= max_linear_mps and angular_dps <= max_angular_dps:
                return obs, True, linear_mps, angular_dps
            self.sleep_for(poll_interval_s)

        if obs is not None:
            self.get_logger().warning(
                f"[{label}] did not settle within {timeout_s:.1f}s (last speed "
                f"{linear_mps * 1e3:.2f}mm/s, {angular_dps:.2f}deg/s) -- capturing anyway"
            )
        return obs, False, linear_mps, angular_dps

    # --- Force Monitoring ---
    def _check_force_threshold(self, observation):
        try:
            if hasattr(observation, 'wrist_wrench') and observation.wrist_wrench is not None:
                force = observation.wrist_wrench.wrench.force
                fx, fy, fz = force.x, force.y, force.z
                if abs(fx) > 20.0 or abs(fy) > 20.0 or abs(fz) > 20.0:
                    self.get_logger().warning(
                        f"HIGH FORCE! FX: {fx:6.2f} N | FY: {fy:6.2f} N | FZ: {fz:6.2f} N"
                    )
                    return True
        except Exception:
            pass
        return False

    # --- Motion Control ---
    def _build_motion_update(self, pos, quat, stiffness, damping):
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
        mat_damp = [0.0] * 36
        for j in range(6):
            mat_stiff[j * 6 + j] = float(stiffness[j])
            mat_damp[j * 6 + j] = float(damping[j])
        motion_update.target_stiffness = mat_stiff
        motion_update.target_damping = mat_damp
        return motion_update

    def _move_tcp_smooth_cartesian(self, pos, quat, move_robot, get_observation,
                                    stiffness, damping, n_steps, label="Target"):
        """Soft Cartesian move: stream the target pose while monitoring forces
        and distance, returning early once within 1mm of the target."""
        motion_update = self._build_motion_update(pos, quat, stiffness, damping)
        self.get_logger().info(f"==> Move to {label}: P=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

        dist = float('inf')
        for i in range(n_steps):
            move_robot(motion_update=motion_update)
            obs = get_observation()
            self._check_force_threshold(obs)

            curr = obs.controller_state.tcp_pose.position
            dist = math.sqrt(
                (curr.x - pos[0]) ** 2 + (curr.y - pos[1]) ** 2 + (curr.z - pos[2]) ** 2
            )

            if i % 25 == 0:
                self.get_logger().info(f"    [{i}] distance to {label}: {dist * 1000:.2f} mm")
                if dist < 0.001:
                    return dist

            self.sleep_for(0.1)
        self.get_logger().info(f"    [{label}] done. Remaining error: {dist * 1000:.3f} mm")
        return dist

    # --- Ground-truth alignment ---
    def _get_aligned_tcp_pose(self, port_pos, port_quat, cable_tip_frame):
        """TCP pose that puts the cable tip exactly at the port pose, using the
        rigid cable_tip_frame -> gripper/tcp offset (ground truth, no vision)."""
        timeout = Duration(seconds=1.0)
        tf_tip_to_tcp = self._parent_node._tf_buffer.lookup_transform(
            cable_tip_frame, "gripper/tcp", Time(), timeout=timeout
        )

        mat_base_to_port = np.eye(4)
        mat_base_to_port[:3, :3] = R.from_quat(port_quat).as_matrix()
        mat_base_to_port[:3, 3] = port_pos

        mat_tip_to_tcp = np.eye(4)
        q = tf_tip_to_tcp.transform.rotation
        t = tf_tip_to_tcp.transform.translation
        mat_tip_to_tcp[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        mat_tip_to_tcp[:3, 3] = [t.x, t.y, t.z]

        target = mat_base_to_port @ mat_tip_to_tcp
        return target[:3, 3], R.from_matrix(target[:3, :3]).as_quat()

    # --- Random offset sampling ---
    @staticmethod
    def _sample_offset(rng, x_range, y_range, z_range, roll_range_deg, pitch_range_deg, yaw_range_deg):
        """Draw one random offset per axis, each independently uniform over
        +/-its own range (all tip-local, see _apply_local_offset)."""
        dx = rng.uniform(-x_range, x_range)
        dy = rng.uniform(-y_range, y_range)
        dz = rng.uniform(-z_range, z_range)
        droll = rng.uniform(-roll_range_deg, roll_range_deg)
        dpitch = rng.uniform(-pitch_range_deg, pitch_range_deg)
        dyaw = rng.uniform(-yaw_range_deg, yaw_range_deg)
        return dx, dy, dz, droll, dpitch, dyaw

    @staticmethod
    def _apply_local_offset(pos, quat, dx, dy, dz, droll, dpitch, dyaw):
        """Perturb pos/quat by a small delta expressed in the pose's own
        (tip-local) frame, so xyz/rpy ranges mean the same thing regardless
        of the port's orientation in the world.

        Rotation axes (tip-local, right-handed, applied in x-y-z order):
          roll  -- rotation about the tip-local X axis (sideways, port width)
          pitch -- rotation about the tip-local Y axis (sideways, port height)
          yaw   -- rotation about the tip-local Z axis (the insertion direction)
        """
        rot = R.from_quat(quat)
        delta_rot = R.from_euler('xyz', [droll, dpitch, dyaw], degrees=True)
        new_rot = rot * delta_rot
        new_pos = pos + rot.apply([dx, dy, dz])
        return new_pos, new_rot.as_quat()

    # --- Sample capture ---
    def _try_capture_sample(self, obs, cable_tip_frame, port_frame):
        """Returns (images, tip_pose, tcp_pose, port_pose, diagnostics) from
        `obs` (an already-fetched observation, see _wait_until_settled -- not
        fetched here, so there's no extra gap between "confirmed settled" and
        "captured"), using the *actual* measured/TF state (not the commanded
        target). Returns None if a camera image or TF frame isn't available
        yet (transient, worth retrying).

        tip_pose/port_pose use "latest" TF (see _lookup_pos_quat_with_stamp --
        looking them up *at* the images' own timestamp instead was tried and
        reliably failed with "extrapolation into the future": ground-truth TF
        publishing was observed lagging camera timestamps by up to ~0.75s
        under real sim load, well past anything worth blocking on). Instead
        `diagnostics` records how far each lookup's actual timestamp drifted
        from the image timestamp, plus the TCP speed at capture (see
        _wait_until_settled), so alignment quality can be audited/filtered
        post-hoc rather than enforced at collection time."""
        if obs is None:
            return None
        self._check_force_threshold(obs)

        images = {}
        image_stamps = []
        for cam in self._camera_names:
            img_msg = getattr(obs, f"{cam}_image", None)
            if img_msg is None:
                return None
            images[cam] = self._bridge.imgmsg_to_cv2(img_msg, "bgr8")
            image_stamps.append(Time.from_msg(img_msg.header.stamp))

        # Cameras are expected to publish on the same sim step (same stamp);
        # take the newest defensively in case they don't.
        image_stamp = max(image_stamps, key=lambda t: t.nanoseconds)
        try:
            tip_pose, tip_pose_stamp = self._lookup_pos_quat_with_stamp(cable_tip_frame)
            port_pose, port_pose_stamp = self._lookup_pos_quat_with_stamp(port_frame)
        except TransformException as ex:
            self.get_logger().warning(f"TF lookup failed during capture: {ex}")
            return None

        tcp = obs.controller_state.tcp_pose
        tcp_pose = (
            np.array([tcp.position.x, tcp.position.y, tcp.position.z]),
            np.array([tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w]),
        )
        diagnostics = {
            "image_stamp_s": image_stamp.nanoseconds * 1e-9,
            "tip_pose_dt_s": (tip_pose_stamp.nanoseconds - image_stamp.nanoseconds) * 1e-9,
            "port_pose_dt_s": (port_pose_stamp.nanoseconds - image_stamp.nanoseconds) * 1e-9,
        }
        return images, tip_pose, tcp_pose, port_pose, diagnostics

    # --- HDF5 dataset writer ---
    # Per-sample scalar validation fields, in addition to the fixed-width
    # pose/offset arrays -- see _try_capture_sample and _wait_until_settled
    # for what each one means. Stored so alignment/settle quality can be
    # audited (e.g. filtering or weighting samples by these) without having
    # to re-run data collection.
    _VALIDATION_FIELDS = (
        'image_stamp_s', 'tip_pose_dt_s', 'port_pose_dt_s',
        'tcp_linear_speed_mps', 'tcp_angular_speed_dps', 'settled',
    )

    @classmethod
    def _get_or_create_port_group(cls, h5file, port_id, image_shapes, attrs):
        if port_id in h5file:
            return h5file[port_id]
        grp = h5file.create_group(port_id)
        grp.attrs.update(attrs)
        img_grp = grp.create_group('images')
        for cam, shape in image_shapes.items():
            img_grp.create_dataset(
                cam, shape=(0, *shape), maxshape=(None, *shape),
                dtype=np.uint8, chunks=(1, *shape),
            )
        for name, width in (('tip_pose', 7), ('tcp_pose', 7), ('port_pose', 7), ('offset', 6)):
            grp.create_dataset(name, shape=(0, width), maxshape=(None, width), dtype=np.float64)
        grp.create_dataset('timestamp', shape=(0,), maxshape=(None,), dtype=np.float64)
        for name in cls._VALIDATION_FIELDS:
            grp.create_dataset(name, shape=(0,), maxshape=(None,), dtype=np.float64)
        return grp

    @classmethod
    def _append_sample_to_h5(cls, grp, images, tip_pose, tcp_pose, port_pose, offset, timestamp, diagnostics):
        def append(ds, row):
            ds.resize(ds.shape[0] + 1, axis=0)
            ds[-1] = row

        for cam, img in images.items():
            append(grp['images'][cam], img)
        append(grp['tip_pose'], np.concatenate(tip_pose))
        append(grp['tcp_pose'], np.concatenate(tcp_pose))
        append(grp['port_pose'], np.concatenate(port_pose))
        append(grp['offset'], np.array(offset))
        append(grp['timestamp'], np.array([timestamp]))
        for name in cls._VALIDATION_FIELDS:
            append(grp[name], np.array([float(diagnostics[name])]))

    # --- Main ---
    def insert_cable(self,
                      task: Task,
                      get_observation: GetObservationCallback,
                      move_robot: MoveRobotCallback,
                      send_feedback: SendFeedbackCallback):
        """Entry point invoked via the /insert_cable action (see aic_model);
        the Task fields are only logged here, all behavior is driven by the
        `data_acquisition.*` ROS parameters set at launch.

        1. Read parameters and open one HDF5 file for this run.
        2. For each configured target_module_name (card/rail):
           a. Wait for its ground-truth port TF frame, skip if not present.
           b. Move the TCP to the port-aligned pose, `approach_height_m` above it.
           c. Repeatedly sample a random tip-local offset, move there under
              high stiffness, and record camera images + actual tip/TCP/port
              poses into that port's HDF5 group.
        3. Close the file and report success.
        """
        import h5py

        self.sleep_for(1.0)
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"STARTING DATA ACQUISITION RUN (task id={task.id})")
        self.get_logger().info("=" * 60)

        cable_type = self._param('data_acquisition.cable_type')
        if cable_type not in self._configs:
            self.get_logger().error(f"cable_type '{cable_type}' is not configured!")
            return False
        cable_tip_frame = self._configs[cable_type]['cable_tip_frame']

        port_specs_raw = list(self._param('data_acquisition.ports'))
        try:
            port_specs = [tuple(spec.split(':', 1)) for spec in port_specs_raw]
            if any(len(spec) != 2 for spec in port_specs):
                raise ValueError
        except ValueError:
            self.get_logger().error(
                f"data_acquisition.ports entries must look like 'target_module_name:port_name', got: {port_specs_raw}"
            )
            return False
        num_samples_per_port = int(self._param('data_acquisition.num_samples_per_port'))
        approach_height_m = float(self._param('data_acquisition.approach_height_m'))
        offset_x_range_m = float(self._param('data_acquisition.offset_x_range_m'))
        offset_y_range_m = float(self._param('data_acquisition.offset_y_range_m'))
        offset_z_range_m = float(self._param('data_acquisition.offset_z_range_m'))
        offset_roll_range_deg = float(self._param('data_acquisition.offset_roll_range_deg'))
        offset_pitch_range_deg = float(self._param('data_acquisition.offset_pitch_range_deg'))
        offset_yaw_range_deg = float(self._param('data_acquisition.offset_yaw_range_deg'))
        stiffness = list(self._param('data_acquisition.stiffness'))
        damping = list(self._param('data_acquisition.damping'))
        approach_move_steps = int(self._param('data_acquisition.approach_move_steps'))
        sample_move_steps = int(self._param('data_acquisition.sample_move_steps'))
        tf_wait_timeout_s = float(self._param('data_acquisition.tf_wait_timeout_s'))
        settle_max_linear_mps = float(self._param('data_acquisition.settle_max_linear_speed_mps'))
        settle_max_angular_dps = float(self._param('data_acquisition.settle_max_angular_speed_dps'))
        settle_timeout_s = float(self._param('data_acquisition.settle_timeout_s'))
        settle_poll_interval_s = float(self._param('data_acquisition.settle_poll_interval_s'))
        seed = int(self._param('data_acquisition.random_seed'))
        rng = np.random.default_rng(seed if seed >= 0 else None)

        output_dir = self._param('data_acquisition.output_dir')
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        h5_path = os.path.join(output_dir, f"{cable_type}_dataset_{timestamp_str}.hdf5")
        self.get_logger().info(f"Writing dataset to: {h5_path}")

        h5file = h5py.File(h5_path, 'a')
        h5file.attrs['cable_type'] = cable_type
        h5file.attrs['pose_format'] = 'xyz_m + quat_xyzw, base_link frame'
        h5file.attrs['image_encoding'] = 'bgr8'
        h5file.attrs['offset_format'] = 'dx,dy,dz [m] (tip-local) + droll,dpitch,dyaw [deg] (tip-local)'
        h5file.attrs['validation_fields'] = (
            'image_stamp_s: sim time the captured images were rendered. '
            'tip_pose_dt_s/port_pose_dt_s: (TF lookup timestamp - image_stamp_s), from '
            '"latest" TF at capture time; large |dt| means the recorded pose is stale '
            'relative to the images (ground-truth TF publishing can lag camera '
            'timestamps under real sim load) -- filter/weight samples by this. '
            'tcp_linear_speed_mps/tcp_angular_speed_dps: measured TCP speed at capture. '
            'settled: 1 if speed was below the settle_max_* thresholds before timeout, '
            '0 if capture proceeded anyway after settle_timeout_s.'
        )

        offset_ranges = (
            offset_x_range_m, offset_y_range_m, offset_z_range_m,
            offset_roll_range_deg, offset_pitch_range_deg, offset_yaw_range_deg,
        )
        move_cfg = (stiffness, damping, approach_move_steps, sample_move_steps)
        settle_cfg = (settle_max_linear_mps, settle_max_angular_dps, settle_timeout_s, settle_poll_interval_s)

        total_collected = 0
        try:
            for target_module_name, port_name in port_specs:
                port_id = f"{target_module_name}_{port_name}"
                # One port's unexpected failure (e.g. a TF frame disappearing
                # mid-run) shouldn't abort the whole multi-port sweep --
                # samples already flushed to disk for earlier ports stay
                # valid regardless, so just log and move on to the next one.
                try:
                    total_collected += self._collect_port_samples(
                        h5file, target_module_name, port_name, cable_type, cable_tip_frame,
                        get_observation, move_robot, send_feedback,
                        num_samples_per_port, approach_height_m, offset_ranges,
                        move_cfg, tf_wait_timeout_s, settle_cfg, rng,
                    )
                except Exception as ex:
                    self.get_logger().error(f"[{port_id}] unexpected error, skipping port: {ex}")
        finally:
            h5file.close()

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"DATA ACQUISITION RUN COMPLETE: {total_collected} samples -> {h5_path}")
        self.get_logger().info("=" * 60)
        return True

    def _collect_port_samples(self, h5file, target_module_name, port_name, cable_type, cable_tip_frame,
                               get_observation, move_robot, send_feedback,
                               num_samples_per_port, approach_height_m, offset_ranges,
                               move_cfg, tf_wait_timeout_s, settle_cfg, rng):
        """Aligns the TCP to one port and collects up to num_samples_per_port
        randomly-offset samples into its HDF5 group. Returns the number of
        samples actually collected (0 if the port's TF frame never showed up)."""
        (offset_x_range_m, offset_y_range_m, offset_z_range_m,
         offset_roll_range_deg, offset_pitch_range_deg, offset_yaw_range_deg) = offset_ranges
        stiffness, damping, approach_move_steps, sample_move_steps = move_cfg
        settle_max_linear_mps, settle_max_angular_dps, settle_timeout_s, settle_poll_interval_s = settle_cfg

        # Use the entrance frame, not the port's back/bottom "_link" frame
        # (see aic_scoring's ScoringTier2::PortEntranceTfName / this
        # package's README) -- otherwise the approach pose ends up too
        # deep, at the back of the port instead of its opening.
        port_frame = f"task_board/{target_module_name}/{port_name}_link_entrance"
        port_id = f"{target_module_name}_{port_name}"

        if not self._wait_for_tf(port_frame, tf_wait_timeout_s):
            self.get_logger().warning(f"Port TF '{port_frame}' not found, skipping {port_id}")
            return 0

        port_pos, port_quat = self._lookup_pos_quat(port_frame)
        aligned_pos, aligned_quat = self._get_aligned_tcp_pose(port_pos, port_quat, cable_tip_frame)

        approach_pos = aligned_pos.copy()
        approach_pos[2] += approach_height_m

        send_feedback(f"Approaching {port_id}")
        self._move_tcp_smooth_cartesian(
            approach_pos, aligned_quat, move_robot, get_observation,
            stiffness=stiffness, damping=damping, n_steps=approach_move_steps,
            label=f"Approach-{port_id}",
        )

        grp = None
        port_attrs = {
            'cable_type': cable_type,
            'port_name': port_name,
            'target_module_name': target_module_name,
            'approach_height_m': approach_height_m,
            'offset_x_range_m': offset_x_range_m,
            'offset_y_range_m': offset_y_range_m,
            'offset_z_range_m': offset_z_range_m,
            'offset_roll_range_deg': offset_roll_range_deg,
            'offset_pitch_range_deg': offset_pitch_range_deg,
            'offset_yaw_range_deg': offset_yaw_range_deg,
        }

        samples_collected = 0
        max_attempts = num_samples_per_port * 3
        attempt = 0
        while samples_collected < num_samples_per_port and attempt < max_attempts:
            attempt += 1
            dx, dy, dz, droll, dpitch, dyaw = self._sample_offset(
                rng, offset_x_range_m, offset_y_range_m, offset_z_range_m,
                offset_roll_range_deg, offset_pitch_range_deg, offset_yaw_range_deg,
            )
            target_pos, target_quat = self._apply_local_offset(
                approach_pos, aligned_quat, dx, dy, dz, droll, dpitch, dyaw
            )

            self._move_tcp_smooth_cartesian(
                target_pos, target_quat, move_robot, get_observation,
                stiffness=stiffness, damping=damping, n_steps=sample_move_steps,
                label=f"{port_id} sample {samples_collected}",
            )

            sample_label = f"{port_id} sample {samples_collected}"
            obs, settled, linear_mps, angular_dps = self._wait_until_settled(
                get_observation, settle_max_linear_mps, settle_max_angular_dps,
                settle_timeout_s, settle_poll_interval_s, label=sample_label,
            )

            capture = self._try_capture_sample(obs, cable_tip_frame, port_frame)
            if capture is None:
                self.get_logger().warning(f"[{port_id}] capture failed, retrying (attempt {attempt})")
                continue

            images, tip_pose, tcp_pose, port_pose, diagnostics = capture
            diagnostics = {
                **diagnostics,
                'tcp_linear_speed_mps': linear_mps,
                'tcp_angular_speed_dps': angular_dps,
                'settled': settled,
            }
            if grp is None:
                image_shapes = {cam: img.shape for cam, img in images.items()}
                grp = self._get_or_create_port_group(h5file, port_id, image_shapes, port_attrs)

            self._append_sample_to_h5(
                grp, images, tip_pose, tcp_pose, port_pose,
                offset=(dx, dy, dz, droll, dpitch, dyaw),
                timestamp=self.time_now().nanoseconds * 1e-9,
                diagnostics=diagnostics,
            )
            h5file.flush()
            samples_collected += 1

            if samples_collected % 10 == 0:
                self.get_logger().info(f"[{port_id}] collected {samples_collected}/{num_samples_per_port}")
                send_feedback(f"[{port_id}] {samples_collected}/{num_samples_per_port} samples")

        self.get_logger().info(
            f"[{port_id}] done: {samples_collected}/{num_samples_per_port} samples collected"
        )
        return samples_collected
