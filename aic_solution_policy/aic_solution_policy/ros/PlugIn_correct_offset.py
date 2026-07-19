import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from aic_control_interfaces.msg import MotionUpdate
from aic_task_interfaces.msg import Task
from aic_model.policy import (
    Policy,
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from std_srvs.srv import Trigger

import torch
import torch.nn as nn
import torchvision

# Must exactly match residual_policy.ipynb's preprocessing/architecture --
# we load that notebook's saved state_dict directly, so any mismatch here
# (image size, camera order, crop box, layer shapes) silently produces
# garbage predictions instead of an error. Copied unchanged from PlugIn.py.
_RESIDUAL_IMAGE_SIZE = 128
_RESIDUAL_CAMS = ['left', 'center', 'right']
_RESIDUAL_CROP = {
    'sfp': {
        'left': (560, 600, 740, 760),
        'center': (480, 560, 660, 720),
        'right': (420, 600, 600, 760),
    },
    'sc': {
        'left': (570, 660, 710, 800),
        'center': (500, 660, 640, 800),
        'right': (430, 660, 570, 800),
    },
}
_RESIDUAL_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_RESIDUAL_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _resolve_aic_solution_dir():
    """
    Locate aic_solution/ (parent of dataset/checkpoints). pixi-build-ros
    installs this package via a real copy, not a symlink, so at runtime
    __file__ points into .pixi/envs/.../site-packages rather than the
    source tree -- prefer PIXI_PROJECT_ROOT (set by `pixi shell`/`pixi run`,
    see SFP_PLUGIN_PHASE1_TESTING.md) and fall back to the source-tree-relative
    guess for direct/dev execution outside a pixi environment.
    """
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    pixi_root = os.environ.get('PIXI_PROJECT_ROOT')
    if pixi_root:
        candidates.append(os.path.join(pixi_root, 'aic_solution'))
    candidates.append(os.path.normpath(os.path.join(this_file_dir, '..', '..', '..')))
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, 'dataset', 'checkpoints')):
            return candidate
    return candidates[0]


_AIC_SOLUTION_DIR = _resolve_aic_solution_dir()
_CHECKPOINT_DIR = os.path.join(_AIC_SOLUTION_DIR, 'dataset', 'checkpoints')


class _SharedViewEncoder(nn.Module):
    """Same architecture as residual_policy.ipynb's SharedViewEncoder."""

    def __init__(self, out_dim, pretrained=False):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x):
        return self.proj(self.backbone(x))


class _MultiViewRegressor(nn.Module):
    """Same architecture as residual_policy.ipynb's MultiViewRegressor."""

    def __init__(self, num_cams, feat_dim, hidden, out_dim=6):
        super().__init__()
        self.num_cams = num_cams
        self.encoder = _SharedViewEncoder(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * num_cams, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, images):
        b, n, c, h, w = images.shape
        feats = self.encoder(images.reshape(b * n, c, h, w)).reshape(b, n * self.encoder.proj.out_features)
        return self.mlp(feats)


class PlugIn_correct_offset(Policy):
    """
    Perception-free (no port detection) insertion policy with real,
    force-threshold-based contact detection -- see insert_cable() steps 1-3 --
    plus a vision-based residual-regressor offset correction run right after
    contact is detected (step 3.5) and before the spiral search (step 4).

    Assumes the robot is already positioned above the target port with the
    plug grasped and aligned, so the pre-contact target pose is derived
    purely from the TCP pose measured at task start (no vision, no port
    detection) -- only the post-contact correction step uses the cameras.
    Supports both 'sfp' and 'sc' via task.port_type, using the same trained
    regressor checkpoints / tip-to-TCP offsets as PlugIn.py.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)

        from cv_bridge import CvBridge

        self.get_logger().info("PlugIn_correct_offset Policy initialised")
        self._bridge = CvBridge()
        self._camera_names = ['left', 'center', 'right']

        # Per-cable-type settings. Every parameter that describes the
        # approach/insertion behavior (not just connector geometry) lives
        # here now, split by port_type, so sfp and sc can be tuned
        # independently. The two entries currently carry identical values
        # for the parameters that used to be shared/untuned-per-type
        # (descent_stiffness/damping, force thresholds, velocities,
        # durations, stall/snag-recovery settings) -- that's a starting
        # point copied from the previous shared defaults, not a claim that
        # sfp and sc should behave the same; tune each independently from
        # here.
        self._configs = {
            'sfp': {
                'off_pos': [0.0, 0.0004, -0.05795],
                'off_quat': [0.17785, 0.00505, -0.02738, -0.98366],
                'residual_model_path': os.path.join(_CHECKPOINT_DIR, 'regressor_best_sfp.pt'),
                'insertion_offset_z': 0.01,
                'spiral_stiffness': [300.0, 300.0, 120.0, 200.0, 200.0, 200.0],
                'spiral_damping': [40.0, 40.0, 15.0, 30.0, 30.0, 30.0],
                'spiral_steps': 120,
                'spiral_max_radius': 0.004,
                'spiral_n_turns': 4,

                # Free-space descent-to-contact primitive (step 3).
                'descent_stiffness': [300.0, 300.0, 300.0, 200.0, 200.0, 200.0],
                'descent_damping': [40.0, 40.0, 35.0, 30.0, 30.0, 30.0],
                'contact_force_threshold_n': 10.0,   # Fz to call initial descent "contact"
                'press_force_n': 10.0,               # constant press force target during spiral (same as contact threshold)
                'press_margin_m': 0.007,             # commanded penetration bias used to realize the press force via the soft z-stiffness above
                'max_descent_margin_m': 0.01,        # safety floor: abort if no contact within target_z - 10mm; also the "in > 10mm" inside-port margin
                'entry_depth_threshold_m': 0.004,    # TCP-z below port-entrance point => tip is inside
                'additional_insert_depth_m': 0.05,   # safety ceiling for the final press (stops earlier via z-stall once actually seated)

                # Descent-to-contact commanded velocity / independent wall-clock timeout.
                'descent_velocity_m_s': 0.03,        # 30 mm/s
                'descent_max_duration_s': 20.0,

                # Final insertion press velocity / timeout.
                'final_insert_velocity_m_s': 0.04,   # 40 mm/s
                'final_insert_max_duration_s': 30.0,

                # z-stall fallback: if the commanded descent keeps going but the
                # measured TCP-z stops moving for this many consecutive steps,
                # treat it as contact even if the (tared) force reading hasn't
                # tripped yet.
                'stall_window_steps': 15,
                'stall_epsilon_m': 0.0003,           # 0.3mm
                'stall_grace_steps': 40,             # ignore stall check during initial settle-in

                # Number of small ramped waypoints used to move to the
                # vision-corrected pose after contact (see _smooth_move_to),
                # same style as the spiral search's incremental steps.
                'correction_move_steps': 40,

                # Extra clearance ABOVE start_pos to retreat to before
                # applying the XY correction (see insert_cable step 3.5).
                # insertion_offset_z + max_descent_margin_m together are only
                # ~20mm, so an edge-catch can happen just a few mm below
                # start_pos - retreating only up to start_pos in that case
                # barely clears the snag at all. This margin guarantees real
                # separation regardless of how shallow the catch was.
                'retreat_clearance_m': 0.01,

                # Final-insert snag recovery: the connector needs ~4.6cm total
                # travel (from the initial contact point) to be fully seated,
                # but sometimes catches mechanically before that and the
                # z-stall fires early. Only trust a z-stall as a real seat
                # once past this depth (buffer under the known 4.6cm); a
                # stall short of that is treated as a snag.
                'min_seat_depth_from_contact_m': 0.043,
                'snag_recovery_stiffness': [300.0, 300.0, 300.0, 40.0, 40.0, 40.0],
                'snag_recovery_damping': [40.0, 40.0, 15.0, 12.0, 12.0, 12.0],
                'snag_recovery_max_attempts': 5,
                'snag_recovery_max_duration_s': 5.0,
                'snag_recovery_attempts_before_spiral_search': 3,
                'snag_recovery_unstick_margin_m': 0.001,
                'snag_recovery_retract_m': 0.012,
            },
            'sc': {
                'off_pos': [0.0, -0.015385, -0.04045],
                'off_quat': [0.1608, -0.167181, 0.69417, -0.6814],
                'residual_model_path': os.path.join(_CHECKPOINT_DIR, 'regressor_best_sc.pt'),
                'insertion_offset_z': 0.01,
                'spiral_stiffness': [300.0, 300.0, 40.0, 200.0, 200.0, 40.0],
                'spiral_damping': [40.0, 40.0, 15.0, 30.0, 30.0, 30.0],
                'spiral_steps': 150,
                'spiral_max_radius': 0.004,
                'spiral_n_turns': 4,

                # Free-space descent-to-contact primitive (step 3). Not yet
                # tuned for 'sc' -- copied from 'sfp' as a starting point.
                'descent_stiffness': [300.0, 300.0, 300.0, 200.0, 200.0, 200.0],
                'descent_damping': [40.0, 40.0, 35.0, 30.0, 30.0, 30.0],
                'contact_force_threshold_n': 10.0,
                'press_force_n': 10.0,
                'press_margin_m': 0.007,
                'max_descent_margin_m': 0.011,
                'entry_depth_threshold_m': 0.004,
                'additional_insert_depth_m': 0.02,

                'descent_velocity_m_s': 0.03,
                'descent_max_duration_s': 20.0,

                'final_insert_velocity_m_s': 0.04,
                'final_insert_max_duration_s': 30.0,

                'stall_window_steps': 15,
                'stall_epsilon_m': 0.0003,
                'stall_grace_steps': 40,

                'correction_move_steps': 40,
                'retreat_clearance_m': 0.01,

                'min_seat_depth_from_contact_m': 0.012,
                'snag_recovery_stiffness': [300.0, 300.0, 300.0, 40.0, 40.0, 40.0],
                'snag_recovery_damping': [40.0, 40.0, 15.0, 12.0, 12.0, 12.0],
                'snag_recovery_max_attempts': 5,
                'snag_recovery_max_duration_s': 5.0,
                'snag_recovery_attempts_before_spiral_search': 3,
                'snag_recovery_unstick_margin_m': 0.001,
                'snag_recovery_retract_m': 0.012,
            },
        }

        # Control-loop period used by _ramp_descend/_smooth_move_to for every
        # commanded update. A controller-loop timing constant, not a
        # connector/approach parameter, so it stays shared across port types.
        self._ramp_step_dt = 0.05

        # TEMP DEBUG probe (see _debug_force_probe_only below) is unaffected
        # by the above - it just logs every step over a fixed step count.
        self._debug_probe_steps = 400

        # Raw wrist_wrench is untared (fed straight from /fts_broadcaster/wrench,
        # see aic_adapter.cpp) - subtract a baseline measured at task start so a
        # static bias (e.g. tool/plug weight) doesn't get mistaken for contact.
        self._force_baseline = np.zeros(3)

        # TEMP DEBUG: descend the full insertion_offset_z with no
        # contact/stall early-exit at all, logging Fz every step, so the raw
        # force signal can be inspected to see whether contact is even
        # distinguishable before tuning _contact_force_threshold_n. Set back
        # to False once a good threshold is known.
        self._debug_force_probe_only = False

        # ROS param so the correction can be A/B-tested without rebuilding the package.
        self._parent_node.declare_parameter('residual_correction.enabled', True)

        # Load the trained offset-correction model(s) from residual_policy.ipynb, one
        # per connector type. Runs on CPU: it's a single forward pass per insert_cable
        # call, not worth risking GPU contention with the sim for.
        self._residual_device = torch.device('cpu')
        self._residual_models = {}
        for c_type, cfg in self._configs.items():
            model_path = cfg.get('residual_model_path')
            if not model_path or not os.path.isfile(model_path):
                self.get_logger().warning(
                    f"No residual correction checkpoint for '{c_type}' at {model_path!r}; "
                    f"post-contact correction will be skipped for this connector type."
                )
                continue
            try:
                self._residual_models[c_type] = self._load_residual_model(model_path)
                self.get_logger().info(f"Loaded residual correction model [{c_type}]: {model_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load residual correction model for '{c_type}': {e}")

    # --- Residual offset-correction model (trained in residual_policy.ipynb, unchanged from PlugIn.py) ---
    def _load_residual_model(self, checkpoint_path):
        # weights_only=False: trusted, self-produced checkpoint (contains numpy
        # target_mean/std alongside the state_dict, which torch's default
        # weights_only=True safe-unpickler may reject).
        checkpoint = torch.load(checkpoint_path, map_location=self._residual_device, weights_only=False)
        model = _MultiViewRegressor(num_cams=len(_RESIDUAL_CAMS), feat_dim=256, hidden=256)
        model.load_state_dict(checkpoint['model'])
        model.to(self._residual_device)
        model.eval()
        return {
            'model': model,
            'target_mean': np.asarray(checkpoint['target_mean'], dtype=np.float32),
            'target_std': np.asarray(checkpoint['target_std'], dtype=np.float32),
        }

    def _preprocess_image_for_residual_model(self, img_bgr, cam, cable_type):
        """Must mirror residual_policy.ipynb's OffsetDataset._load_image exactly
        (crop to the fixed per-camera ROI for this cable_type, then resize --
        no train-time augmentation here, this is eval-mode preprocessing)."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        box = _RESIDUAL_CROP.get(cable_type, {}).get(cam)
        if box is not None:
            x0, y0, x1, y1 = box
            img_rgb = img_rgb[y0:y1, x0:x1]
        img_resized = cv2.resize(img_rgb, (_RESIDUAL_IMAGE_SIZE, _RESIDUAL_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        img_norm = img_resized.astype(np.float32) / 255.0
        img_norm = (img_norm - _RESIDUAL_IMAGENET_MEAN) / _RESIDUAL_IMAGENET_STD
        return torch.from_numpy(img_norm.transpose(2, 0, 1))

    def _predict_offset_correction(self, observation, cable_type):
        """Runs the trained offset-correction model on the current camera images.

        Returns [dx,dy,dz] (meters) + [droll,dpitch,dyaw] (degrees): the
        model's estimate of the cable tip's current pose relative to the
        port, in the port's frame -- or None if no model/images are available.
        """
        bundle = self._residual_models.get(cable_type)
        if bundle is None or observation is None:
            return None

        images = []
        for cam in _RESIDUAL_CAMS:
            img_msg = getattr(observation, f"{cam}_image", None)
            if img_msg is None:
                self.get_logger().warning(f"Residual model: missing '{cam}' image, skipping correction")
                return None
            cv_img = self._bridge.imgmsg_to_cv2(img_msg, "bgr8")
            images.append(self._preprocess_image_for_residual_model(cv_img, cam, cable_type))
        images_tensor = torch.stack(images, dim=0).unsqueeze(0).to(self._residual_device)

        with torch.no_grad():
            pred_norm = bundle['model'](images_tensor)[0].cpu().numpy()

        return pred_norm * bundle['target_std'] + bundle['target_mean']

    def _apply_predicted_correction(self, tcp_pos, tcp_quat, off_pos, off_quat,
                                     predicted_offset, correct_z=False):
        """Shifts a TCP pose so the cable tip cancels out `predicted_offset`
        (the tip's currently-predicted pose relative to the port, in the
        port's frame -- same convention as compute_relative_offset() in
        residual_policy.ipynb: [dx,dy,dz] meters + [droll,dpitch,dyaw] degrees).

        Depth (dz) is ignored by default: the training data randomized dz
        deliberately (for view diversity), it doesn't represent an alignment
        error to fix, and Z is already governed by the force-controlled
        descent/press steps -- this correction only fixes lateral/rotational
        alignment before the spiral search.

        The cable tip and gripper/tcp differ by a fixed, non-identity
        rotation (off_quat), so a correction expressed in the tip's own
        frame has to be conjugated by that offset before it can be composed
        onto the TCP pose directly. Unchanged from PlugIn.py.

        Zeroing dz above only zeroes it in the tip's own (tilted) frame --
        the correction vector is then rotated by r_tip_old (the tip's actual
        orientation, tilted by off_quat) before being added to tcp_pos, and a
        vector that's flat in a tilted frame is generally NOT flat once
        rotated into world/base frame. Left uncorrected, that silently
        introduces a real world-frame Z shift into what's supposed to be a
        Z-neutral lateral move (surfaced as the arm sinking during the
        correction move). So when correct_z is False, the world-frame Z is
        clamped back to the original tcp_pos[2] explicitly below, rather than
        relying on dz=0 pre-rotation to keep it neutral.
        """
        dx, dy, dz, droll, dpitch, dyaw = predicted_offset
        if not correct_z:
            dz = 0.0

        droll = 0.0
        dpitch = 0.0
        dyaw = 0.0

        r_pred = R.from_euler('xyz', [droll, dpitch, dyaw], degrees=True)
        delta_pos_tip = -np.array([dx, dy, dz])
        delta_rot_tip = r_pred.inv()

        r_k = R.from_quat(off_quat)   # tip -> TCP fixed rotation
        t_k = np.array(off_pos)       # tip -> TCP fixed translation, in tip frame

        r_tcp_old = R.from_quat(tcp_quat)
        r_tip_old = r_tcp_old * r_k.inv()

        pos_correction = r_tip_old.apply(delta_pos_tip + delta_rot_tip.apply(t_k) - t_k)
        tcp_pos_new = np.array(tcp_pos) + pos_correction
        if not correct_z:
            tcp_pos_new[2] = tcp_pos[2]

        rot_correction_tcp_frame = r_k.inv() * delta_rot_tip * r_k
        tcp_quat_new = (r_tcp_old * rot_correction_tcp_frame).as_quat()

        return tcp_pos_new, tcp_quat_new

    # --- Force Monitoring and Threshold (unchanged from PlugIn) ---
    def _check_force_threshold(self, observation):
        """
        Prints a warning when forces exceed 20N
        """
        try:
            if hasattr(observation, 'wrist_wrench') and observation.wrist_wrench is not None:
                force = observation.wrist_wrench.wrench.force
                fx, fy, fz = force.x, force.y, force.z

                if abs(fx) > 20.0 or abs(fy) > 20.0 or abs(fz) > 20.0:
                    self.get_logger().warning(
                        f"⚠️ HOHE KRAFT! FX: {fx:6.2f} N | FY: {fy:6.2f} N | FZ: {fz:6.2f} N"
                    )
                    return True
        except Exception:
            pass
        return False

    def _get_force_xyz(self, observation):
        try:
            if hasattr(observation, 'wrist_wrench') and observation.wrist_wrench is not None:
                f = observation.wrist_wrench.wrench.force
                return np.array([f.x, f.y, f.z])
        except Exception:
            pass
        return np.zeros(3)

    def _get_force_z(self, observation):
        """Tared Fz: raw wrist_wrench.z minus the baseline measured at task start."""
        return float(self._get_force_xyz(observation)[2] - self._force_baseline[2])

    def _try_tare_ft_sensor(self):
        """
        Best-effort hardware tare via /aic_controller/tare_force_torque_sensor.
        Note: this only zeroes the controller's *internal* force-feedback
        offset, not the raw /fts_broadcaster/wrench topic our wrist_wrench
        observation reads from - so it does not replace the software baseline
        measured in _measure_force_baseline, just complements it.
        """
        try:
            client = self._parent_node.create_client(Trigger, "/aic_controller/tare_force_torque_sensor")
            if not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warning(
                    "tare_force_torque_sensor Service nicht erreichbar - überspringe "
                    "(Software-Baseline-Tare läuft trotzdem)."
                )
                return
            response = client.call(Trigger.Request())
            if response.success:
                self.get_logger().info(f"F/T-Sensor (Controller-intern) tariert: {response.message}")
            else:
                self.get_logger().warning(f"Tarieren (Service) fehlgeschlagen: {response.message}")
        except Exception as e:
            self.get_logger().warning(f"Tarieren (Service) nicht möglich: {e}")

    def _measure_force_baseline(self, get_observation, n_samples=20, sleep_s=0.02):
        """
        Software self-tare: average the raw wrist_wrench force over a short,
        stationary window and use it as the zero-offset for all subsequent
        Fz threshold checks this task.
        """
        samples = []
        for _ in range(n_samples):
            obs = get_observation()
            samples.append(self._get_force_xyz(obs))
            self.sleep_for(sleep_s)
        baseline = np.mean(samples, axis=0)
        self.get_logger().info(
            f"F/T-Sensor Baseline (Software-Tare) gemessen: {baseline} N - wird von allen "
            f"folgenden Fz-Messungen abgezogen."
        )
        return baseline

    def _pos_to_array(self, position):
        return np.array([position.x, position.y, position.z])

    def _get_current_tcp_pose(self, obs):
        pos = self._pos_to_array(obs.controller_state.tcp_pose.position)
        q = obs.controller_state.tcp_pose.orientation
        quat = np.array([q.x, q.y, q.z, q.w])
        return pos, quat

    # --- Motion helpers (unchanged from PlugIn) ---
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

    # --- New flow: shared ramped descent with force- and stall-based stop ---
    def _ramp_descend(self, start_pos, target_pos, quat,
                       move_robot, get_observation, cfg,
                       stiffness, damping, velocity_m_s, max_duration_s,
                       label="Descent",
                       force_threshold_n=None,
                       extra_depth_below_target_m=0.0):
        """
        Ramps the commanded Z linearly, at velocity_m_s, from start_pos[2]
        down to target_pos[2] - extra_depth_below_target_m (XY held fixed at
        start_pos XY throughout - target_pos only supplies the Z floor, so a
        start_pos shifted laterally, e.g. after a residual correction, stays
        shifted instead of snapping back to target_pos's XY), stopping as
        soon as any of:
          - force_threshold_n is given and measured |Fz| (tared) exceeds it,
          - measured TCP-z stalls (no real movement) for _stall_window_steps
            in a row (past _stall_grace_steps) - i.e. a hard mechanical stop
            was hit (real contact, or fully seated - can't go further),
            regardless of whether the force reading caught it, or
          - max_duration_s of wall-clock time has elapsed, with the commanded
            ramp target having reached the floor and held there throughout
            (see below - running out of *distance* alone is no longer a
            separate exit condition).
        velocity_m_s and max_duration_s are independent: max_duration_s is a
        safety backstop for cases where a real stop signal was missed, not a
        function of speed, so tuning velocity_m_s no longer silently changes
        the timeout.

        The loop always runs for the full max_duration_s (n_steps =
        timeout_steps) - it used to also cap n_steps at distance_steps (time
        to nominally cover the commanded distance at velocity_m_s), which
        seemed like a harmless optimization but silently shrank the number of
        steps as velocity_m_s went up. At high enough velocity that pushed
        n_steps below _stall_grace_steps, so the stall check could never
        even run once before the loop gave up - the ramp target still clamps
        at the floor via `min(..., total_travel)` below once distance is
        covered, so running the full timeout just means holding there and
        continuing to watch for a stall/force stop instead of exiting early.
        Returns (reached_pos, stopped_early). stopped_early is True only for
        the force/stall cases above; running out of time without one returns
        False.
        """
        step_dt = self._ramp_step_dt
        floor_z = target_pos[2] - extra_depth_below_target_m
        total_travel = start_pos[2] - floor_z
        step_distance = velocity_m_s * step_dt

        distance_steps = max(1, int(np.ceil(total_travel / step_distance)))
        timeout_steps = max(1, int(np.ceil(max_duration_s / step_dt)))
        n_steps = timeout_steps

        self.get_logger().info(
            f"==> {label}: von z={start_pos[2]:.4f} Richtung z={floor_z:.4f} "
            f"mit v={velocity_m_s * 1000:.1f}mm/s (max. {max_duration_s:.1f}s) "
            f"({f'Kraftschwelle |Fz|>={force_threshold_n:.1f}N oder ' if force_threshold_n is not None else ''}"
            f"z-Stillstand ueber {cfg['stall_window_steps']} Schritte (<{cfg['stall_epsilon_m'] * 1000:.1f}mm) als Stopp-Kriterium)"
        )

        curr_pos = start_pos.copy()
        fz = 0.0
        z_history = []
        for i in range(n_steps):
            traveled = min((i + 1) * step_distance, total_travel)
            cmd_pos = np.array([start_pos[0], start_pos[1], start_pos[2] - traveled])

            motion_update = self._build_motion_update(cmd_pos, quat, stiffness, damping)
            move_robot(motion_update=motion_update)

            obs = get_observation()
            self._check_force_threshold(obs)
            fz = self._get_force_z(obs)
            curr_pos = self._pos_to_array(obs.controller_state.tcp_pose.position)

            z_history.append(curr_pos[2])
            if len(z_history) > cfg['stall_window_steps']:
                z_history.pop(0)

            if i % 20 == 0:
                self.get_logger().info(f"    [{i}/{n_steps}] t={i * step_dt:.1f}s z={curr_pos[2]:.4f} Fz(tariert)={fz:.2f}N")

            if force_threshold_n is not None and abs(fz) >= force_threshold_n:
                self.get_logger().info(f"    {label}: Kraft-Stopp bei z={curr_pos[2]:.4f} (Fz={fz:.2f}N)")
                return curr_pos, True

            if (i >= cfg['stall_grace_steps']
                    and len(z_history) == cfg['stall_window_steps']
                    and (max(z_history) - min(z_history)) < cfg['stall_epsilon_m']):
                self.get_logger().info(
                    f"    {label}: z-Stillstand bei z={curr_pos[2]:.4f} - keine weitere Bewegung "
                    f"trotz tieferem Kommando (Fz={fz:.2f}N)."
                )
                return curr_pos, True

            self.sleep_for(step_dt)

        reason = (
            f"Boden erreicht (kommandierte Distanz {total_travel * 1000:.0f}mm), "
            f"{max_duration_s:.1f}s Zeitlimit ohne weiteren Stopp ausgeschoepft"
            if timeout_steps >= distance_steps else
            f"Zeitlimit ({max_duration_s:.1f}s) erreicht, bevor die volle Distanz "
            f"({total_travel * 1000:.0f}mm) kommandiert war"
        )
        self.get_logger().warning(
            f"    {label}: Ende ({reason}) erreicht ohne Stopp-Kriterium (letzte z={curr_pos[2]:.4f}, Fz={fz:.2f}N)."
        )
        return curr_pos, False

    def _retract_up(self, start_pos, quat, move_robot, get_observation,
                     stiffness, damping, retract_m, velocity_m_s,
                     label="Retract"):
        """
        Ramps the commanded Z straight up by retract_m (XY held fixed at
        start_pos XY) - the inverse of _ramp_descend's direction, used to
        back a snag off before the rescue spiral search instead of trying
        to search laterally while still jammed against whatever it caught on.
        No stall/force stop check - it's a short, fixed-distance move.
        """
        step_dt = self._ramp_step_dt
        step_distance = velocity_m_s * step_dt
        n_steps = max(1, int(np.ceil(retract_m / step_distance)))

        self.get_logger().info(
            f"    {label}: fahre {retract_m * 1000:.1f}mm hoch von z={start_pos[2]:.4f}."
        )

        curr_pos = start_pos.copy()
        for i in range(n_steps):
            traveled = min((i + 1) * step_distance, retract_m)
            cmd_pos = np.array([start_pos[0], start_pos[1], start_pos[2] + traveled])

            motion_update = self._build_motion_update(cmd_pos, quat, stiffness, damping)
            move_robot(motion_update=motion_update)

            obs = get_observation()
            self._check_force_threshold(obs)
            fz = self._get_force_z(obs)
            curr_pos = self._pos_to_array(obs.controller_state.tcp_pose.position)

            # Logged every step (not just every 20th, unlike _ramp_descend) --
            # this move is short, and if it's still sinking back down after
            # a retreat, per-step z-vs-commanded and Fz is what shows whether
            # that's a real force ceiling (Fz pinned near the controller's
            # maximum_wrench while z stalls short of cmd_pos) or something else.
            self.get_logger().info(
                f"    [{i}/{n_steps}] z_cmd={cmd_pos[2]:.4f} z_ist={curr_pos[2]:.4f} Fz(tariert)={fz:.2f}N"
            )

            self.sleep_for(step_dt)

        self.get_logger().info(f"    {label}: jetzt bei z={curr_pos[2]:.4f}.")
        return curr_pos

    def _move_down_until_contact(self, start_pos, target_pos, quat,
                                  move_robot, get_observation, cfg,
                                  label="Descent-To-Contact"):
        """Free-space descent to first contact - see _ramp_descend. All
        descent parameters (stiffness, velocity, force threshold, ...) come
        from the given per-port-type cfg."""
        return self._ramp_descend(
            start_pos, target_pos, quat, move_robot, get_observation, cfg,
            stiffness=cfg['descent_stiffness'], damping=cfg['descent_damping'],
            velocity_m_s=cfg['descent_velocity_m_s'], max_duration_s=cfg['descent_max_duration_s'],
            label=label,
            force_threshold_n=cfg['contact_force_threshold_n'],
            extra_depth_below_target_m=cfg['max_descent_margin_m'],
        )

    def _press_insert_until_seated(self, entry_pos, quat,
                                    move_robot, get_observation,
                                    max_insert_depth_m, velocity_m_s, max_duration_s,
                                    contact_pos, spiral_cfg,
                                    label="Final-Insert"):
        """
        Presses further in from entry_pos (soft spiral Z-stiffness, same as
        used for the port search) until TCP-z stalls, i.e. the connector is
        fully seated and physically can't go any further - instead of
        aiming for one fixed extra depth and waiting out the full time
        budget regardless of whether it was already fully inserted.
        max_insert_depth_m is only a safety ceiling on how far it will try.

        A z-stall is only accepted as "fully seated" once the TCP has
        travelled at least min_seat_depth_from_contact_m below contact_pos
        (the connector needs ~4.6cm total below the port entrance, so this is
        a buffered floor). contact_pos is the caller's best estimate of the
        port ENTRANCE depth, not necessarily the exact point contact was
        detected at - on the direct-insert path (see insert_cable) the
        detected contact is already past the entrance, so the caller passes
        the assumed entrance (target_pos) here instead, crediting that
        already-covered depth rather than resetting the seat-depth budget to
        zero from wherever contact happened to be found. A stall short of
        that is treated as a mechanical snag: retry the same
        push with a softened rotational stiffness (_snag_recovery_stiffness)
        so the connector can self-align, up to _snag_recovery_max_attempts
        times, before giving up. Every _snag_recovery_attempts_before_spiral_search
        straight pushes in a row that haven't freed it, it backs off
        _snag_recovery_retract_m upward first (releases the jam instead of
        searching laterally while still wedged against it), then tries one
        lateral spiral search (the same primitive used to find the port
        entry) from there, on the theory that a snag is usually a local
        lateral catch that a small sideways search can slip past -
        this repeats (not just once) for as long as attempts remain, so a
        push-push-push-rescue cycle keeps recurring until it's either freed
        or the attempt budget runs out.
        """
        curr_pos = entry_pos.copy()
        stiffness, damping = spiral_cfg['spiral_stiffness'], spiral_cfg['spiral_damping']
        attempt_duration = max_duration_s
        attempts_since_rescue = 0
        min_seat_depth_from_contact_m = spiral_cfg['min_seat_depth_from_contact_m']
        snag_recovery_stiffness = spiral_cfg['snag_recovery_stiffness']
        snag_recovery_damping = spiral_cfg['snag_recovery_damping']
        snag_recovery_max_attempts = spiral_cfg['snag_recovery_max_attempts']
        snag_recovery_max_duration_s = spiral_cfg['snag_recovery_max_duration_s']
        snag_recovery_attempts_before_spiral_search = spiral_cfg['snag_recovery_attempts_before_spiral_search']
        snag_recovery_unstick_margin_m = spiral_cfg['snag_recovery_unstick_margin_m']
        snag_recovery_retract_m = spiral_cfg['snag_recovery_retract_m']

        for attempt in range(snag_recovery_max_attempts + 1):
            curr_pos, stopped_early = self._ramp_descend(
                curr_pos, entry_pos, quat, move_robot, get_observation, spiral_cfg,
                stiffness=stiffness, damping=damping,
                velocity_m_s=velocity_m_s, max_duration_s=attempt_duration, label=label,
                force_threshold_n=None,
                extra_depth_below_target_m=max_insert_depth_m,
            )
            depth_from_contact = contact_pos[2] - curr_pos[2]

            if not stopped_early:
                self.get_logger().info(
                    f"    {label}: Sicherheitsgrenze ({max_insert_depth_m * 1000:.0f}mm) erreicht ohne klaren "
                    f"Stillstand, letzte Position z={curr_pos[2]:.4f} (Tiefe={depth_from_contact * 1000:.1f}mm)."
                )
                return curr_pos

            if depth_from_contact >= min_seat_depth_from_contact_m:
                self.get_logger().info(
                    f"    {label}: vollstaendig eingesteckt (z-Stillstand) bei z={curr_pos[2]:.4f} "
                    f"(Tiefe={depth_from_contact * 1000:.1f}mm)."
                )
                return curr_pos

            if attempt == snag_recovery_max_attempts:
                self.get_logger().warning(
                    f"    {label}: Stillstand bei Tiefe={depth_from_contact * 1000:.1f}mm "
                    f"(< {min_seat_depth_from_contact_m * 1000:.0f}mm Soll) nach {attempt} "
                    f"Recovery-Versuchen - gebe auf."
                )
                return curr_pos

            attempts_since_rescue += 1
            if attempts_since_rescue >= snag_recovery_attempts_before_spiral_search:
                attempts_since_rescue = 0
                stuck_z = curr_pos[2]
                self.get_logger().warning(
                    f"    {label}: nach {snag_recovery_attempts_before_spiral_search} weiteren "
                    f"Versuchen (insgesamt {attempt + 1}) weiterhin fest bei Tiefe="
                    f"{depth_from_contact * 1000:.1f}mm - versuche laterale Spiralsuche zum Loesen."
                )
                curr_pos = self._retract_up(
                    curr_pos, quat, move_robot, get_observation,
                    stiffness=stiffness, damping=damping,
                    retract_m=snag_recovery_retract_m,
                    velocity_m_s=velocity_m_s,
                    label=f"{label}-Rescue-Retract",
                )
                spiral_center = curr_pos.copy()
                spiral_center[2] -= spiral_cfg['press_margin_m']
                unstuck, curr_pos = self._spiral_search_until_entry(
                    spiral_center, quat, move_robot, get_observation,
                    entry_z=stuck_z - snag_recovery_unstick_margin_m,
                    spiral_cfg=spiral_cfg,
                    label=f"{label}-Rescue-Spiral",
                )
                if unstuck:
                    self.get_logger().info(
                        f"    {label}: durch Spiralsuche wieder in Bewegung bei z={curr_pos[2]:.4f} "
                        f"- setze Einstecken fort."
                    )
                else:
                    self.get_logger().warning(
                        f"    {label}: Spiralsuche zum Loesen ohne erkannte Bewegung beendet "
                        f"(letzte z={curr_pos[2]:.4f}) - versuche trotzdem mit reduzierter "
                        f"Rotations-Steifigkeit weiter."
                    )
                stiffness, damping = snag_recovery_stiffness, snag_recovery_damping
                attempt_duration = snag_recovery_max_duration_s
                continue

            self.get_logger().warning(
                f"    {label}: verfrueher Stillstand bei Tiefe={depth_from_contact * 1000:.1f}mm "
                f"(< {min_seat_depth_from_contact_m * 1000:.0f}mm) - vermutlich verhakt. "
                f"Reduziere Rotations-Steifigkeit und versuche erneut "
                f"(Versuch {attempt + 1}/{snag_recovery_max_attempts})."
            )
            stiffness, damping = snag_recovery_stiffness, snag_recovery_damping
            attempt_duration = snag_recovery_max_duration_s

    # --- TEMP DEBUG: full descent, no contact stop, just log the force signal ---
    def _debug_descent_force_probe(self, start_pos, target_pos, quat,
                                    move_robot, get_observation, cfg,
                                    n_steps, label="Force-Probe-Descent"):
        """
        Descends all the way to target_pos (no early exit on force or stall),
        logging Fz (raw and tared) every step, then prints a min/max/mean
        summary so we can see from the log whether real contact produces a
        distinguishable peak at all, before trusting any threshold.
        """
        total_travel = start_pos[2] - target_pos[2]

        self.get_logger().info(
            f"==> {label}: volle Absenkung {total_travel * 1000:.1f}mm OHNE Kontakt-Stopp "
            f"(nur zum Beobachten der Kraftspitzen)."
        )

        z_log, raw_log, tared_log = [], [], []
        for i in range(n_steps):
            frac = (i + 1) / n_steps
            cmd_pos = np.array([target_pos[0], target_pos[1], start_pos[2] - frac * total_travel])

            motion_update = self._build_motion_update(cmd_pos, quat, cfg['descent_stiffness'], cfg['descent_damping'])
            move_robot(motion_update=motion_update)

            obs = get_observation()
            self._check_force_threshold(obs)
            raw_fz = float(self._get_force_xyz(obs)[2])
            tared_fz = self._get_force_z(obs)
            curr_pos = self._pos_to_array(obs.controller_state.tcp_pose.position)

            z_log.append(curr_pos[2])
            raw_log.append(raw_fz)
            tared_log.append(tared_fz)

            self.get_logger().info(
                f"    [{i}/{n_steps}] z={curr_pos[2]:.4f} Fz_raw={raw_fz:.2f}N Fz_tariert={tared_fz:.2f}N"
            )

            self.sleep_for(0.05)

        zs = np.array(z_log)
        raws = np.array(raw_log)
        tareds = np.array(tared_log)
        peak_idx = int(np.argmax(np.abs(tareds)))

        self.get_logger().info("==== Force-Probe Zusammenfassung ====")
        self.get_logger().info(
            f"Fz tariert: min={tareds.min():.2f}N max={tareds.max():.2f}N "
            f"mean={tareds.mean():.2f}N std={tareds.std():.2f}N"
        )
        self.get_logger().info(
            f"Fz raw:     min={raws.min():.2f}N max={raws.max():.2f}N "
            f"mean={raws.mean():.2f}N std={raws.std():.2f}N"
        )
        self.get_logger().info(
            f"Groesster |Fz tariert| = {tareds[peak_idx]:.2f}N bei z={zs[peak_idx]:.4f} "
            f"(Schritt {peak_idx}/{n_steps})"
        )
        self.get_logger().info("======================================")

        return self._pos_to_array(get_observation().controller_state.tcp_pose.position)

    # --- New flow: spiral search with constant press, exits on port entry ---
    def _spiral_search_until_entry(self, center_pos, quat,
                                    move_robot, get_observation,
                                    entry_z, spiral_cfg, label="Spiral-Search"):
        """
        Spiral search identical to PlugIn's _spiral_search_and_insert
        (same stiffness/damping/steps/max_radius, taken unchanged), but exits
        as soon as the TCP sinks below entry_z (port entrance detected),
        instead of only reporting a final distance.

        Returns (True, curr_pos) on detected entry. If the whole spiral runs
        out without a detected entry, still returns the last TCP position
        (instead of None) as (False, curr_pos), so the caller can fall back
        to pressing straight down from there rather than aborting outright -
        the entry threshold is a detection heuristic, not proof the plug
        isn't already resting over the hole.
        """
        max_radius = spiral_cfg['spiral_max_radius']
        n_turns = spiral_cfg['spiral_n_turns']
        steps = spiral_cfg['spiral_steps']

        self.get_logger().info(
            f"==> {label}: max_radius={max_radius * 1000:.1f}mm turns={n_turns} steps={steps} "
            f"entry bei z<{entry_z:.4f}"
        )

        t_vals = np.linspace(0, n_turns * 2 * np.pi, steps)

        for idx, t in enumerate(t_vals):
            r = (t / (n_turns * 2 * np.pi)) * max_radius
            dx = r * np.cos(t)
            dy = r * np.sin(t)

            search_pos = center_pos.copy()
            search_pos[0] += dx
            search_pos[1] += dy

            motion_update = self._build_motion_update(search_pos, quat, spiral_cfg['spiral_stiffness'], spiral_cfg['spiral_damping'])
            move_robot(motion_update=motion_update)

            obs = get_observation()
            self._check_force_threshold(obs)
            curr_pos = self._pos_to_array(obs.controller_state.tcp_pose.position)

            if idx % 20 == 0:
                self.get_logger().info(f"    [{idx}/{steps}] r={r * 1000:.2f}mm z={curr_pos[2]:.4f}")

            if curr_pos[2] < entry_z:
                self.get_logger().info(f"    Eintritt erkannt: TCP-z={curr_pos[2]:.4f} < {entry_z:.4f}")
                return True, curr_pos

            self.sleep_for(0.05)

        self.get_logger().warning(
            f"    Kein Eintritt erkannt: max. Radius ({max_radius * 1000:.1f}mm) ohne Erfolg durchsucht."
        )
        return False, curr_pos

    # --- New flow: distinguish "already inside the port" from "stuck on the edge" ---
    def _check_inside_port(self, start_pos, contact_pos, cfg):
        """
        After the initial descent-to-contact stops (force or z-stall), decide
        whether the tip is already inside the port (descended straight in, no
        edge catch) or stopped early on the port's rim.

        delta_z is how far the TCP actually travelled down from the very
        first approach pose (start_pos, queried once at task start - not the
        current attempt's start, so this stays comparable across the retry
        after a correction). distance is how much of the assumed
        insertion_offset_z + a max_descent_margin_m safety margin is still
        "left" below the actual contact depth:
          distance = insertion_offset_z - delta_z + max_descent_margin_m
        distance <= 0 means the tip already went more than max_descent_margin_m
        past the assumed port-entrance depth without an earlier stop -> it's
        in. distance > 0 means it stopped well short of that -> likely caught
        on the edge, needs correction.
        """
        delta_z = start_pos[2] - contact_pos[2]
        distance = cfg['insertion_offset_z'] - delta_z + cfg['max_descent_margin_m']
        inside = distance <= 0.0
        self.get_logger().info(
            f"    Inside-Port-Check: delta_z={delta_z * 1000:.1f}mm, distance={distance * 1000:.1f}mm "
            f"-> {'innen (direkt eingesteckt)' if inside else 'nicht innen (vermutlich Kante)'}"
        )
        return inside

    # --- New flow: smooth ramped move to the vision-corrected pose ---
    def _smooth_move_to(self, start_pos, start_quat, target_pos, target_quat,
                         move_robot, get_observation,
                         stiffness, damping, steps, label="Smooth-Move"):
        """
        Ramps linearly (position lerp + orientation slerp) from start_pos/
        start_quat to target_pos/target_quat over `steps` small waypoints -
        the same incremental, small-step style _spiral_search_until_entry
        already uses to move the TCP - instead of commanding the corrected
        pose in a single jump right after contact, which would ask the
        impedance controller for a large instantaneous step while still
        pressed against the part.
        """
        self.get_logger().info(
            f"==> {label}: von xy=({start_pos[0]:.4f},{start_pos[1]:.4f}) nach "
            f"xy=({target_pos[0]:.4f},{target_pos[1]:.4f}) in {steps} Schritten."
        )

        slerp = Slerp([0.0, 1.0], R.from_quat([start_quat, target_quat]))

        curr_pos, curr_quat = start_pos.copy(), np.asarray(target_quat)
        for i in range(steps):
            frac = (i + 1) / steps
            cmd_pos = start_pos + frac * (target_pos - start_pos)
            cmd_quat = slerp([frac])[0].as_quat()

            motion_update = self._build_motion_update(cmd_pos, cmd_quat, stiffness, damping)
            move_robot(motion_update=motion_update)

            obs = get_observation()
            self._check_force_threshold(obs)
            curr_pos = self._pos_to_array(obs.controller_state.tcp_pose.position)
            curr_quat = cmd_quat

            self.sleep_for(self._ramp_step_dt)

        self.get_logger().info(f"    {label}: fertig bei z={curr_pos[2]:.4f}.")
        return curr_pos, curr_quat

    # --- New flow: predict the residual offset correction (no motion) ---
    def _get_predicted_correction(self, c_type, get_observation):
        """
        Runs the trained offset-correction model on the CURRENT camera images
        and returns the raw predicted_offset (or None if correction is
        disabled, no model is loaded for this connector type, or the images
        weren't available - callers should treat that as "nothing to correct
        with"). Deliberately does not move the robot: call this before any
        pre-correction lift, while the TCP is still at the pose the model was
        trained to see (right at contact) - predicting from an already-lifted
        vantage would feed it an out-of-distribution view.
        """
        residual_enabled = self._parent_node.get_parameter('residual_correction.enabled').value
        if not residual_enabled:
            return None

        obs = get_observation()
        predicted_offset = self._predict_offset_correction(obs, c_type)
        if predicted_offset is None:
            self.get_logger().warning(f"No residual correction available for '{c_type}', skipping")
            return None

        dx, dy, dz, droll, dpitch, dyaw = predicted_offset
        self.get_logger().info(
            f"Residual model prediction: dxyz=({dx*1e3:.2f},{dy*1e3:.2f},{dz*1e3:.2f})mm "
            f"rpy=({droll:.1f},{dpitch:.1f},{dyaw:.1f})deg"
        )
        return predicted_offset

    # --- New flow: smoothly move to the pose implied by an already-predicted correction ---
    def _move_to_corrected_pose(self, pos, quat, predicted_offset, cfg,
                                 move_robot, get_observation):
        """
        Takes a predicted_offset already computed by _get_predicted_correction
        (from the pre-lift contact pose) and the pose to move FROM (e.g. the
        post-lift pose) - applies the correction relative to that pose and
        smoothly ramps the TCP there (see _smooth_move_to) rather than
        jumping in one commanded step. The correction itself (a lateral/
        rotational shift derived from the model's predicted offset and the
        fixed tip/TCP geometry) doesn't depend on which pose it's anchored
        to, so anchoring it post-lift keeps the lifted Z rather than
        collapsing back down to the pre-lift contact depth.

        Uses descent_stiffness/damping (full-strength Z), not
        spiral_stiffness/damping - the spiral config is deliberately soft in
        Z to realize a controlled press force while inside the port, which
        would let the TCP sag/sink back down under gravity or residual
        contact during this move instead of holding the lift.
        """
        corrected_pos, corrected_quat = self._apply_predicted_correction(
            pos, quat, cfg['off_pos'], cfg['off_quat'], predicted_offset
        )
        return self._smooth_move_to(
            pos, quat, corrected_pos, corrected_quat,
            move_robot, get_observation,
            stiffness=cfg['descent_stiffness'], damping=cfg['descent_damping'],
            steps=cfg['correction_move_steps'],
            label="Offset-Correction-Move",
        )

    # --- Main ---
    def insert_cable(self,
                      task: Task,
                      get_observation: GetObservationCallback,
                      move_robot: MoveRobotCallback,
                      send_feedback: SendFeedbackCallback):
        '''
        Thin wrapper around _run_insertion(): always returns True, whether
        the insertion actually completed (fully seated) or a step (or the
        whole attempt) only got as far as its time/attempt budget allowed,
        or something unexpected raised - a stuck/timed-out/errored attempt
        should still report "done" rather than fail or crash the task.
        Downstream evaluation judges actual success from the resulting
        port/robot state, not this return value, so it's not lost by always
        returning True here.
        '''
        try:
            self._run_insertion(task, get_observation, move_robot, send_feedback)
        except Exception as e:
            self.get_logger().error(f"insert_cable: unerwarteter Fehler, breche Task ab: {e}")
        return True

    def _run_insertion(self,
                        task: Task,
                        get_observation: GetObservationCallback,
                        move_robot: MoveRobotCallback,
                        send_feedback: SendFeedbackCallback):
        '''
        1. Query & store the current TCP pose once (robot already positioned
           above the target port, plug grasped and aligned).
        2. Target (port entrance) = start pose, cfg['insertion_offset_z'] down
           along -z. Tare the F/T sensor (hardware best-effort + software
           baseline) before any force-based decision.
        3. Force-controlled descent until contact is detected (tared force
           threshold, or a z-stall fallback if the tare isn't perfect).
        3.1 Inside-port check (_check_inside_port): did the tip already travel
            more than max_descent_margin_m past the assumed port-entrance
            depth without an earlier stop? If so it went straight in - skip
            straight to the final press (step 5).
        3.5 Otherwise, it stopped early - likely caught on the port's edge.
            Vision-based residual correction: run the trained offset model
            (regressor_best_sfp.pt / regressor_best_sc.pt) on the camera
            images now that the TCP is in physical contact, and smoothly move
            (no single-step jump) to cancel out its predicted tip-to-port
            offset. Z is left untouched.
        3.6 Retry the descent-to-contact from the corrected pose, exactly
            like step 3, and re-run the inside-port check on the new contact
            point. If that also lands inside, skip to the final press too.
        4. If it's still not inside after the retry, fall back to the spiral
           search (unchanged from before) with constant press force, exiting
           as soon as port entry is detected.
        5. Force-controlled final insertion by the configured extra depth.
        6. Report success/failure. Every loop has a fixed step budget, so a
           missing contact/entry cleanly aborts instead of hanging.
        '''
        self.sleep_for(1.0)

        c_type = task.port_type.lower()
        if c_type not in self._configs:
            self.get_logger().error(f"Kabeltyp '{c_type}' ist nicht konfiguriert!")
            return False
        cfg = self._configs[c_type]

        self.get_logger().info("============================================================")
        self.get_logger().info(f"STARTING NEW TASK (PlugIn_correct_offset): {c_type.upper()} id={task.id}")
        self.get_logger().info("============================================================")

        # 1. Query & store current TCP pose once
        obs = get_observation()
        start_pos, start_quat = self._get_current_tcp_pose(obs)
        self.get_logger().info(f"Startpose TCP: Pos={start_pos}, Quat={start_quat}")

        # 2. Target pose (port entrance) = start pose, insertion_offset_z down (-z)
        target_pos = start_pos.copy()
        target_pos[2] -= cfg['insertion_offset_z']

        # Tare the F/T sensor before doing anything force-based: the raw
        # wrist_wrench reading is untared and can carry a static bias (e.g.
        # tool/plug weight), which would otherwise be mistaken for contact
        # the instant we start descending.
        send_feedback("Tarieren des F/T-Sensors...")
        self._try_tare_ft_sensor()
        self.sleep_for(0.3)
        self._force_baseline = self._measure_force_baseline(get_observation)

        send_feedback(f"Starting {c_type} insertion (force-controlled, perception at contact)...")

        # TEMP DEBUG: just descend the full distance and log the force signal,
        # skip contact detection / spiral / insertion entirely.
        if self._debug_force_probe_only:
            self._debug_descent_force_probe(
                start_pos, target_pos, start_quat,
                move_robot, get_observation, cfg,
                n_steps=self._debug_probe_steps,
            )
            self.get_logger().info("Force-Probe abgeschlossen - siehe Log fuer Fz-Verlauf. Kein Insert versucht.")
            return True

        # 3. Force-controlled descent until contact
        contact_pos, contact_detected = self._move_down_until_contact(
            start_pos, target_pos, start_quat, move_robot, get_observation, cfg,
        )
        entry_quat = start_quat

        # 3.1 Did we already go straight in, or did we stop early (edge catch)?
        inside_port = self._check_inside_port(start_pos, contact_pos, cfg)

        if not inside_port:
            # 3.5 Stopped early - likely caught on the port edge. Predict the
            # residual offset correction FIRST, while the TCP is still at the
            # pose the model was trained to see (right at contact) - moving
            # away before predicting would feed it an out-of-distribution view.
            predicted_offset = self._get_predicted_correction(c_type, get_observation)

            corrected = False
            if predicted_offset is not None:
                # Retreat back up past the original approach pose - not just
                # up to it - by retreat_clearance_m (same stiffness/damping/
                # velocity as the original approach descent). insertion_offset_z
                # + max_descent_margin_m are only ~20mm total, so an edge-catch
                # can happen just a few mm below start_pos - retreating only
                # up to start_pos in that case barely clears the snag at all.
                retract_m = (start_pos[2] - contact_pos[2]) + cfg['retreat_clearance_m']
                approach_pos = self._retract_up(
                    contact_pos, start_quat, move_robot, get_observation,
                    stiffness=cfg['descent_stiffness'], damping=cfg['descent_damping'],
                    retract_m=retract_m,
                    velocity_m_s=cfg['descent_velocity_m_s'],
                    label="Retract-To-Approach",
                )

                # Apply the XY correction at the approach pose, in free space
                # (see _move_to_corrected_pose).
                corrected_pos, corrected_quat = self._move_to_corrected_pose(
                    approach_pos, start_quat, predicted_offset, cfg,
                    move_robot, get_observation,
                )
                corrected = True

            if corrected:
                entry_quat = corrected_quat
                # 3.6 Try the insert again from the corrected pose, exactly
                # like the initial attempt in step 3.
                contact_pos, _ = self._move_down_until_contact(
                    corrected_pos, target_pos, entry_quat, move_robot, get_observation, cfg,
                    label="Descent-To-Contact-Retry",
                )
                inside_port = self._check_inside_port(start_pos, contact_pos, cfg)

        if inside_port:
            self.get_logger().info(
                f"Bereits direkt eingesteckt (Kontakt bei z={contact_pos[2]:.4f}) - "
                f"ueberspringe Korrektur/Spiralsuche, gehe direkt zum finalen Einstecken."
            )
            entry_pos = contact_pos
            # min_seat_depth_from_contact_m is calibrated as "depth needed
            # below the port ENTRANCE to be fully seated". In this branch
            # contact_pos is already >= max_descent_margin_m past the
            # *assumed* entrance (target_pos) - that's the whole reason
            # _check_inside_port tripped - so measuring remaining-seat-depth
            # from contact_pos itself would silently throw away the depth
            # already covered and could never recognize "already at/near the
            # bottom" (any stall right after entry_pos would read as
            # depth_from_contact~=0, misclassified as a premature snag no
            # matter how deep contact_pos actually already is). Anchor to
            # target_pos instead so that head start is credited correctly.
            seat_reference_pos = target_pos
        else:
            # 4. Still not inside after the retry - fall back to the spiral
            # search (unchanged), anchored to the latest contact point.
            spiral_center = contact_pos.copy()
            spiral_center[2] -= cfg['press_margin_m']  # commanded penetration bias -> press force via soft z-stiffness

            # Anchored to the actually measured (and possibly corrected) contact
            # point, not the a-priori assumed target_pos - the real descent
            # distance can differ from that assumption, so anchoring to
            # target_pos made entry_z unreachably deep.
            entry_z = contact_pos[2] - cfg['entry_depth_threshold_m']
            entered, entry_pos = self._spiral_search_until_entry(
                spiral_center, entry_quat,
                move_robot, get_observation,
                entry_z=entry_z,
                spiral_cfg=cfg,
            )

            if not entered:
                self.get_logger().warning(
                    "Kein Eintritt in den Port erkannt (Spiral-Search Timeout / max. Radius) - "
                    "versuche trotzdem den finalen Einsteck-Schritt von der letzten Spiral-Position aus."
                )

            # Edge-catch path: contact_pos here is the empirically detected
            # rim/entrance contact, which is exactly what
            # min_seat_depth_from_contact_m is calibrated against - unchanged
            # from before.
            seat_reference_pos = contact_pos

        # 5. Force-controlled final insertion - press until it stalls (fully
        # seated), not to one fixed extra depth.
        self._press_insert_until_seated(
            entry_pos, entry_quat, move_robot, get_observation,
            max_insert_depth_m=cfg['additional_insert_depth_m'],
            velocity_m_s=cfg['final_insert_velocity_m_s'],
            max_duration_s=cfg['final_insert_max_duration_s'],
            contact_pos=seat_reference_pos,
            spiral_cfg=cfg,
        )

        self.get_logger().info("============================================================")
        self.get_logger().info(f"SUCCESS - {c_type.upper()} Kabel eingesteckt (PlugIn_correct_offset).")
        self.get_logger().info("============================================================")
        return True
