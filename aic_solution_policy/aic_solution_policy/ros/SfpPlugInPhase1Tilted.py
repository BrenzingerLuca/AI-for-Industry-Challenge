import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from rclpy.duration import Duration
from rclpy.time import Time
from aic_control_interfaces.msg import MotionUpdate
from aic_task_interfaces.msg import Task
from aic_model.policy import (
    Policy,
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from std_srvs.srv import Trigger


class SfpPlugInPhase1Tilted(Policy):
    """
    Experimental variant of SfpPlugInPhase1: instead of holding the plug
    vertical over the port, tilts it ~20 degrees about the *tip* frame's own
    Y axis (RViz "green" axis) first, then drives straight down until
    contact - same perception-free, TF-free approach otherwise (target pose
    derived purely from the TCP pose measured at task start).

    Scope of this experiment (see conversation): tilt-in-place, then
    move-down-until-contact. Spiral search / final press-to-seat are copied
    over unchanged from SfpPlugInPhase1 for possible reuse in a later phase,
    but are not invoked here.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)

        self.get_logger().info("SfpPlugInPhase1Tilted Policy initialised")

        # Controller / impedance settings taken unchanged from PlugIn's
        # 'sfp' config (spiral_stiffness_1 / spiral_damping_1 / spiral_steps_1 /
        # spiral_max_radius_1) - these are what bound the contact force during
        # search and insertion.
        self._cfg = {
            'spiral_stiffness': [300.0, 300.0, 80.0, 200.0, 200.0, 200.0],
            'spiral_damping': [40.0, 40.0, 15.0, 30.0, 30.0, 30.0],
            'spiral_steps': 120,
            'spiral_max_radius': 0.004,   # 3 mm
            'spiral_n_turns': 4,          # -> ~1 mm/turn radius increase
            # Separate, stiffer Z config used only for the free-space descent
            # (start -> port entrance). The spiral's soft Z (80 N/m) is needed
            # to bound contact force once actually touching the port, but
            # during the plain descent it wasn't enough to push through the
            # cable's own resistance/drag, so the TCP lagged well behind the
            # commanded ramp. Spiral/insert stiffness above stays unchanged.
            'descent_stiffness': [300.0, 300.0, 200.0, 200.0, 200.0, 200.0],
            'descent_damping': [40.0, 40.0, 35.0, 30.0, 30.0, 30.0],
        }

        # New-flow parameters
        self._insertion_offset_z = 0.05          # target = tilted pose - 5cm (-z)
        self._contact_force_threshold_n = 10.0    # Fz to call initial descent "contact"
        self._max_descent_margin_m = 0.01         # safety floor: abort if no contact within target_z - 10mm

        self._descent_steps = 400          # @0.05s/step -> up to 20s

        # z-stall fallback: if the commanded descent keeps going but the
        # measured TCP-z stops moving for this many consecutive steps, treat
        # it as contact even if the (tared) force reading hasn't tripped yet.
        self._stall_window_steps = 15
        self._stall_epsilon_m = 0.0003     # 0.3mm
        self._stall_grace_steps = 40       # ignore stall check during initial settle-in

        # Raw wrist_wrench is untared (fed straight from /fts_broadcaster/wrench,
        # see aic_adapter.cpp) - subtract a baseline measured after tilting (not
        # before) so the changed static load/torque from holding the plug
        # tilted doesn't get mistaken for contact.
        self._force_baseline = np.zeros(3)

        # --- Tilt parameters ---
        # Tilt the plug this many degrees about the *tip* frame's own local Y
        # axis (RViz "green" axis), pivoting about the tip origin (tip
        # position held fixed). Positive = right-hand rule about tip-local +Y.
        self._tilt_deg = 20.0
        self._tilt_steps = 60

        # Rigid Tip -> TCP grasp offset for the SFP plug, taken unchanged from
        # PlugIn.py's _configs['sfp']['off_pos']/['off_quat'] (originally
        # measured once from ground-truth TF: cable_0/sfp_tip_link ->
        # gripper/tcp, in sim). Translation in tip frame, quaternion xyzw
        # rotation tip->tcp.
        #
        # This is only used here (once, at init) to derive the fixed
        # TCP-local pose delta below that realizes the desired tip tilt -
        # NOT looked up live via TF, since cable_0/sfp_tip_link won't be
        # available outside sim / with a real plug.
        self._tip_to_tcp_offset_pos = np.array([0.0, 0.0004, -0.05795])
        self._tip_to_tcp_offset_quat = np.array([0.17785, 0.00505, -0.02738, -0.98366])

        self._tilt_delta_pos, self._tilt_delta_quat = self._compute_tilt_delta(self._tilt_deg)
        self.get_logger().info(
            f"Tilt-Delta (TCP-lokal, fuer {self._tilt_deg:.1f} deg um tip-lokale Y-Achse): "
            f"dPos={self._tilt_delta_pos}, dQuat={self._tilt_delta_quat}"
        )

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
        """Tared Fz: raw wrist_wrench.z minus the baseline measured after tilting."""
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

    # --- Tilt: hardcoded gripper-local move that tilts the plug about the
    # tip frame's own Y axis, without needing the tip frame at runtime ---
    def _compute_tilt_delta(self, tilt_deg):
        """
        Derives the fixed TCP-local pose offset (delta_pos, delta_quat) that,
        when composed onto *any* current TCP pose (new_pos = pos + R(quat) @
        delta_pos, new_quat = quat * delta_quat), tilts the rigidly-grasped
        plug by tilt_deg about the tip frame's own local Y axis while
        pivoting about the tip origin (tip position unchanged).

        Derivation: let T_tip_tcp be the fixed rigid grasp offset (tip ->
        TCP). Rotating the tip in place by Rot_y(theta) about its own local
        axes gives T_base_tip_new = T_base_tip_old @ Rot_y(theta). Since
        T_base_tcp = T_base_tip @ T_tip_tcp, this is equivalent to
        right-multiplying the *TCP* pose by the constant, pose-independent
        delta:
            Delta = T_tip_tcp^-1 @ Rot_y(theta) @ T_tip_tcp
        which is exactly what's returned here - computed once from the
        hardcoded grasp offset, no live TF needed.
        """
        q_off = self._tip_to_tcp_offset_quat / np.linalg.norm(self._tip_to_tcp_offset_quat)
        t_tip_tcp = np.eye(4)
        t_tip_tcp[:3, :3] = R.from_quat(q_off).as_matrix()
        t_tip_tcp[:3, 3] = self._tip_to_tcp_offset_pos
        t_tip_tcp_inv = np.linalg.inv(t_tip_tcp)

        rot_y = np.eye(4)
        rot_y[:3, :3] = R.from_euler('y', tilt_deg, degrees=True).as_matrix()

        delta = t_tip_tcp_inv @ rot_y @ t_tip_tcp
        delta_pos = delta[:3, 3]
        delta_quat = R.from_matrix(delta[:3, :3]).as_quat()
        return delta_pos, delta_quat

    def _apply_local_delta(self, pos, quat, delta_pos, delta_quat):
        """Composes a TCP-local pose delta onto a base_link-frame TCP pose."""
        new_pos = pos + R.from_quat(quat).apply(delta_pos)
        new_quat = (R.from_quat(quat) * R.from_quat(delta_quat)).as_quat()
        return new_pos, new_quat

    def _debug_try_get_tip_quat(self):
        """
        Best-effort ground-truth check (sim only): looks up the actual
        cable_0/sfp_tip_link orientation in base_link, purely for sanity-
        checking the hardcoded tilt math against RViz/TF. Not used for any
        control decision, and never blocks - returns None if unavailable
        (e.g. on real hardware, where this frame doesn't exist).
        """
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", "cable_0/sfp_tip_link", Time(), timeout=Duration(seconds=0.5)
            )
            q = tf.transform.rotation
            return np.array([q.x, q.y, q.z, q.w])
        except Exception:
            return None

    def _tilt_plug_in_place(self, start_pos, start_quat, move_robot, get_observation,
                             n_steps=None, label="Tilt"):
        """
        Smoothly ramps the TCP from start_pos/start_quat to the tilted pose
        given by composing the hardcoded _tilt_delta_pos/_tilt_delta_quat
        (see _compute_tilt_delta) onto it - a pure free-space reorientation,
        no force/contact stop condition.
        """
        n_steps = n_steps or self._tilt_steps
        target_pos, target_quat = self._apply_local_delta(
            start_pos, start_quat, self._tilt_delta_pos, self._tilt_delta_quat
        )

        self.get_logger().info(
            f"==> {label}: kippe Stecker um {self._tilt_deg:.1f} deg um tip-lokale Y-Achse "
            f"(TCP {start_pos} -> {target_pos})"
        )

        slerp = Slerp([0.0, 1.0], R.from_quat(np.stack([start_quat, target_quat])))

        stiffness = self._cfg['descent_stiffness']
        damping = self._cfg['descent_damping']

        for i in range(1, n_steps + 1):
            frac = i / n_steps
            interp_pos = start_pos + frac * (target_pos - start_pos)
            interp_quat = slerp(frac).as_quat()

            motion_update = self._build_motion_update(interp_pos, interp_quat, stiffness, damping)
            move_robot(motion_update=motion_update)

            obs = get_observation()
            self._check_force_threshold(obs)

            self.sleep_for(0.05)

        # Hold final tilted pose briefly so the controller settles.
        motion_update = self._build_motion_update(target_pos, target_quat, stiffness, damping)
        for _ in range(10):
            move_robot(motion_update=motion_update)
            self.sleep_for(0.05)

        obs = get_observation()
        final_pos, final_quat = self._get_current_tcp_pose(obs)
        self.get_logger().info(f"    {label}: gekippte TCP-Pose erreicht: pos={final_pos}, quat={final_quat}")
        return final_pos, final_quat

    # --- New flow: shared ramped descent with force- and stall-based stop ---
    def _ramp_descend(self, start_pos, target_pos, quat,
                       move_robot, get_observation,
                       stiffness, damping, n_steps,
                       label="Descent",
                       force_threshold_n=None,
                       extra_depth_below_target_m=0.0):
        """
        Ramps the commanded Z linearly from start_pos[2] down to
        target_pos[2] - extra_depth_below_target_m over n_steps (XY held at
        target_pos XY), stopping as soon as either:
          - force_threshold_n is given and measured |Fz| (tared) exceeds it, or
          - measured TCP-z stalls (no real movement) for _stall_window_steps
            in a row (past _stall_grace_steps) - i.e. a hard mechanical stop
            was hit (real contact, or fully seated - can't go further),
            regardless of whether the force reading caught it.
        Returns (reached_pos, stopped_early). If stopped_early is False, the
        step budget ran out without either signal firing.
        """
        floor_z = target_pos[2] - extra_depth_below_target_m
        total_travel = start_pos[2] - floor_z

        self.get_logger().info(
            f"==> {label}: von z={start_pos[2]:.4f} Richtung z={floor_z:.4f} "
            f"({f'Kraftschwelle |Fz|>={force_threshold_n:.1f}N oder ' if force_threshold_n is not None else ''}"
            f"z-Stillstand ueber {self._stall_window_steps} Schritte (<{self._stall_epsilon_m * 1000:.1f}mm) als Stopp-Kriterium)"
        )

        curr_pos = start_pos.copy()
        fz = 0.0
        z_history = []
        for i in range(n_steps):
            frac = (i + 1) / n_steps
            cmd_pos = np.array([target_pos[0], target_pos[1], start_pos[2] - frac * total_travel])

            motion_update = self._build_motion_update(cmd_pos, quat, stiffness, damping)
            move_robot(motion_update=motion_update)

            obs = get_observation()
            self._check_force_threshold(obs)
            fz = self._get_force_z(obs)
            curr_pos = self._pos_to_array(obs.controller_state.tcp_pose.position)

            z_history.append(curr_pos[2])
            if len(z_history) > self._stall_window_steps:
                z_history.pop(0)

            if i % 20 == 0:
                self.get_logger().info(f"    [{i}/{n_steps}] z={curr_pos[2]:.4f} Fz(tariert)={fz:.2f}N")

            if force_threshold_n is not None and abs(fz) >= force_threshold_n:
                self.get_logger().info(f"    {label}: Kraft-Stopp bei z={curr_pos[2]:.4f} (Fz={fz:.2f}N)")
                return curr_pos, True

            if (i >= self._stall_grace_steps
                    and len(z_history) == self._stall_window_steps
                    and (max(z_history) - min(z_history)) < self._stall_epsilon_m):
                self.get_logger().info(
                    f"    {label}: z-Stillstand bei z={curr_pos[2]:.4f} - keine weitere Bewegung "
                    f"trotz tieferem Kommando (Fz={fz:.2f}N)."
                )
                return curr_pos, True

            self.sleep_for(0.05)

        self.get_logger().warning(
            f"    {label}: Schrittbudget ({n_steps}) erreicht ohne Stopp-Kriterium (letzte z={curr_pos[2]:.4f}, Fz={fz:.2f}N)."
        )
        return curr_pos, False

    def _move_down_until_contact(self, start_pos, target_pos, quat,
                                  move_robot, get_observation,
                                  force_threshold_n, max_extra_depth_m,
                                  n_steps, label="Descent-To-Contact"):
        """Free-space descent to first contact - see _ramp_descend."""
        cfg = self._cfg
        return self._ramp_descend(
            start_pos, target_pos, quat, move_robot, get_observation,
            stiffness=cfg['descent_stiffness'], damping=cfg['descent_damping'],
            n_steps=n_steps, label=label,
            force_threshold_n=force_threshold_n,
            extra_depth_below_target_m=max_extra_depth_m,
        )

    def _press_insert_until_seated(self, entry_pos, quat,
                                    move_robot, get_observation,
                                    max_insert_depth_m, n_steps,
                                    label="Final-Insert"):
        """
        Retained from SfpPlugInPhase1 for a possible later phase - not
        currently invoked by insert_cable() below. Presses further in from
        entry_pos (soft spiral Z-stiffness) until TCP-z stalls.
        """
        curr_pos, stopped_early = self._ramp_descend(
            entry_pos, entry_pos, quat, move_robot, get_observation,
            stiffness=self._cfg['spiral_stiffness'], damping=self._cfg['spiral_damping'],
            n_steps=n_steps, label=label,
            force_threshold_n=None,
            extra_depth_below_target_m=max_insert_depth_m,
        )
        if stopped_early:
            self.get_logger().info(f"    {label}: vollstaendig eingesteckt (z-Stillstand) bei z={curr_pos[2]:.4f}.")
        else:
            self.get_logger().info(
                f"    {label}: Sicherheitsgrenze ({max_insert_depth_m * 1000:.0f}mm) erreicht ohne klaren "
                f"Stillstand, letzte Position z={curr_pos[2]:.4f}."
            )
        return curr_pos

    # --- New flow: spiral search with constant press, exits on port entry ---
    def _spiral_search_until_entry(self, center_pos, quat,
                                    move_robot, get_observation,
                                    entry_z, label="Spiral-Search"):
        """
        Retained from SfpPlugInPhase1 for a possible later phase - not
        currently invoked by insert_cable() below (spiral search assumes a
        vertical plug; re-check applicability before wiring it in after a
        tilted descent). Exits as soon as the TCP sinks below entry_z.
        """
        cfg = self._cfg
        max_radius = cfg['spiral_max_radius']
        n_turns = cfg['spiral_n_turns']
        steps = cfg['spiral_steps']

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

            motion_update = self._build_motion_update(search_pos, quat, cfg['spiral_stiffness'], cfg['spiral_damping'])
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
        return False, None

    # --- Main ---
    def insert_cable(self,
                      task: Task,
                      get_observation: GetObservationCallback,
                      move_robot: MoveRobotCallback,
                      send_feedback: SendFeedbackCallback):
        '''
        1. Query & store the current TCP pose once (robot already positioned
           above the target port, plug grasped and vertical).
        2. Tilt the plug in place by _tilt_deg about the tip frame's own
           local Y axis, using the hardcoded TCP-local delta (no tip-frame
           TF needed at runtime) - pivoting about the tip origin.
        3. Tare the F/T sensor (hardware best-effort + software baseline)
           *after* tilting, since the static load/torque on the sensor
           changes once the plug is held tilted.
        4. Force-controlled descent (straight down in base_link Z, tilted
           orientation held fixed) until contact is detected (tared force
           threshold, or a z-stall fallback if the tare isn't perfect).
        5. Report success/failure. Step budget bounds the run so a missing
           contact cleanly aborts instead of hanging.
        '''
        self.sleep_for(1.0)

        self.get_logger().info("============================================================")
        self.get_logger().info(f"STARTING NEW TASK (SfpPlugInPhase1Tilted): id={task.id}")
        self.get_logger().info("============================================================")

        # 1. Query & store current TCP pose once
        obs = get_observation()
        start_pos, start_quat = self._get_current_tcp_pose(obs)
        self.get_logger().info(f"Startpose TCP: Pos={start_pos}, Quat={start_quat}")

        # 2. Tilt the plug in place about the tip's local Y axis.
        send_feedback(f"Tilting plug {self._tilt_deg:.0f} deg about tip-local Y axis...")
        pre_tilt_tip_quat = self._debug_try_get_tip_quat()

        tilted_pos, tilted_quat = self._tilt_plug_in_place(
            start_pos, start_quat, move_robot, get_observation,
        )

        # Best-effort sim-only sanity check against ground-truth TF - never
        # blocks, just logs whether the hardcoded math matches what actually
        # happened to cable_0/sfp_tip_link.
        post_tilt_tip_quat = self._debug_try_get_tip_quat()
        if pre_tilt_tip_quat is not None and post_tilt_tip_quat is not None:
            rel = R.from_quat(pre_tilt_tip_quat).inv() * R.from_quat(post_tilt_tip_quat)
            rotvec = rel.as_rotvec()
            angle_deg = float(np.degrees(np.linalg.norm(rotvec)))
            axis = rotvec / np.linalg.norm(rotvec) if angle_deg > 1e-6 else rotvec
            self.get_logger().info(
                f"[Debug/Sim] Tatsaechliche Kippung von cable_0/sfp_tip_link: {angle_deg:.2f} deg "
                f"um Achse {axis} (tip-lokal, erwartet: {self._tilt_deg:.1f} deg um [0,1,0])."
            )
        else:
            self.get_logger().info(
                "[Debug/Sim] cable_0/sfp_tip_link TF nicht verfuegbar - Kipp-Sanity-Check uebersprungen "
                "(erwartet auf echter Hardware / ohne Ground-Truth-Plugin)."
            )

        # 3. Tare the F/T sensor after tilting: the raw wrist_wrench reading
        # is untared and its static bias changes once the plug is held
        # tilted (different gravity torque), so tare only once settled there.
        send_feedback("Tarieren des F/T-Sensors...")
        self._try_tare_ft_sensor()
        self.sleep_for(0.3)
        self._force_baseline = self._measure_force_baseline(get_observation)

        send_feedback("Starting SFP insertion (Phase 1, tilted, perception-free)...")

        # 4. Force-controlled descent until contact, straight down in
        # base_link Z with the tilted orientation held fixed.
        target_pos = tilted_pos.copy()
        target_pos[2] -= self._insertion_offset_z

        contact_pos, contact_detected = self._move_down_until_contact(
            tilted_pos, target_pos, tilted_quat,
            move_robot, get_observation,
            force_threshold_n=self._contact_force_threshold_n,
            max_extra_depth_m=self._max_descent_margin_m,
            n_steps=self._descent_steps,
        )

        if not contact_detected:
            self.get_logger().error("FAILED - Kein Kontakt beim Herunterfahren erkannt.")
            return False

        self.get_logger().info("============================================================")
        self.get_logger().info(f"SUCCESS - Kontakt erkannt bei tilted descent, z={contact_pos[2]:.4f}.")
        self.get_logger().info("============================================================")
        return True
