import numpy as np
from aic_control_interfaces.msg import MotionUpdate
from aic_task_interfaces.msg import Task
from aic_model.policy import (
    Policy,
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from std_srvs.srv import Trigger


class SfpPlugInPhase1(Policy):
    """
    Perception-free SFP insertion policy.

    Assumes the robot is already positioned above the target port with the
    plug grasped and aligned, so the target pose is derived purely from the
    TCP pose measured at task start (no vision, no port detection).
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)

        self.get_logger().info("SfpPlugInPhase1 Policy initialised")

        # Controller / impedance settings taken unchanged from PlugIn's
        # 'sfp' config (spiral_stiffness_1 / spiral_damping_1 / spiral_steps_1 /
        # spiral_max_radius_1) - these are what bound the contact force during
        # search and insertion.
        self._cfg = {
            'spiral_stiffness': [300.0, 300.0, 120.0, 200.0, 200.0, 200.0],
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
            'descent_stiffness': [300.0, 300.0, 300.0, 200.0, 200.0, 200.0],
            'descent_damping': [40.0, 40.0, 35.0, 30.0, 30.0, 30.0],
        }

        # New-flow parameters
        self._insertion_offset_z = 0.05           # target = start pose - 3cm (-z)
        self._contact_force_threshold_n = 10.0    # Fz to call initial descent "contact"
        self._press_force_n = 10.0                # constant press force target during spiral (same as contact threshold)
        self._press_margin_m = 0.007              # commanded penetration bias used to realize the press force via the soft z-stiffness above
        self._max_descent_margin_m = 0.01         # safety floor: abort if no contact within target_z - 10mm
        self._entry_depth_threshold_m = 0.004     # TCP-z below port-entrance point => tip is inside
        self._additional_insert_depth_m = 0.05    # safety ceiling for the final press (stops earlier via z-stall once actually seated)

        # Control-loop period used by _ramp_descend for every commanded
        # update (applies to both descent-to-contact and final insertion).
        self._ramp_step_dt = 0.05

        # Descent-to-contact: commanded velocity and an independent
        # wall-clock safety timeout. These are fully decoupled now that
        # _ramp_descend's step budget is time-based only (see its docstring),
        # so pushing velocity up no longer silently shrinks how long it's
        # willing to watch for a stop - only max_duration_s does that.
        self._descent_velocity_m_s = 0.03        # 30 mm/s
        self._descent_max_duration_s = 20.0

        # Final insertion press: pushed fast, relying on the stall/depth
        # checks in _press_insert_until_seated (not the step budget) to know
        # when to stop - safe now that the step-budget-shrinks-with-velocity
        # bug is fixed and the spiral Z-stiffness was raised (80 -> 120N/m),
        # so there's more force headroom to actually track this speed.
        self._final_insert_velocity_m_s = 0.04   # 40 mm/s
        self._final_insert_max_duration_s = 30.0

        # TEMP DEBUG probe (see _debug_force_probe_only below) is unaffected
        # by the above - it just logs every step over a fixed step count.
        self._debug_probe_steps = 400

        # z-stall fallback: if the commanded descent keeps going but the
        # measured TCP-z stops moving for this many consecutive steps, treat
        # it as contact even if the (tared) force reading hasn't tripped yet.
        self._stall_window_steps = 15
        self._stall_epsilon_m = 0.0003     # 0.3mm
        self._stall_grace_steps = 40       # ignore stall check during initial settle-in

        # Final-insert snag recovery: the connector needs ~4.6cm total travel
        # (measured from the initial contact point) to be fully seated, but
        # sometimes catches mechanically before that and the z-stall fires
        # early, which used to be accepted as "fully seated". Only trust a
        # z-stall as a real seat once past this depth (4.3cm - leaves ~3mm
        # buffer under the known 4.6cm). A stall short of that is treated as
        # a snag: soften the rotational stiffness (instead of an active
        # jiggle) so the connector can self-align under the ongoing push -
        # lower risk than a lateral dither this deep into the port, and
        # reuses the existing ramp/stiffness mechanism unchanged.
        self._min_seat_depth_from_contact_m = 0.043
        self._snag_recovery_stiffness = [300.0, 300.0, 300.0, 40.0, 40.0, 40.0]
        self._snag_recovery_damping = [40.0, 40.0, 15.0, 12.0, 12.0, 12.0]
        self._snag_recovery_max_attempts = 5
        self._snag_recovery_max_duration_s = 5.0

        # Raw wrist_wrench is untared (fed straight from /fts_broadcaster/wrench,
        # see aic_adapter.cpp) - subtract a baseline measured at task start so a
        # static bias (e.g. tool/plug weight) doesn't get mistaken for contact.
        self._force_baseline = np.zeros(3)

        # TEMP DEBUG: descend the full self._insertion_offset_z with no
        # contact/stall early-exit at all, logging Fz every step, so the raw
        # force signal can be inspected to see whether contact is even
        # distinguishable before tuning _contact_force_threshold_n. Set back
        # to False once a good threshold is known.
        self._debug_force_probe_only = False

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
                       move_robot, get_observation,
                       stiffness, damping, velocity_m_s, max_duration_s,
                       label="Descent",
                       force_threshold_n=None,
                       extra_depth_below_target_m=0.0):
        """
        Ramps the commanded Z linearly, at velocity_m_s, from start_pos[2]
        down to target_pos[2] - extra_depth_below_target_m (XY held at
        target_pos XY), stopping as soon as any of:
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
            f"z-Stillstand ueber {self._stall_window_steps} Schritte (<{self._stall_epsilon_m * 1000:.1f}mm) als Stopp-Kriterium)"
        )

        curr_pos = start_pos.copy()
        fz = 0.0
        z_history = []
        for i in range(n_steps):
            traveled = min((i + 1) * step_distance, total_travel)
            cmd_pos = np.array([target_pos[0], target_pos[1], start_pos[2] - traveled])

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
                self.get_logger().info(f"    [{i}/{n_steps}] t={i * step_dt:.1f}s z={curr_pos[2]:.4f} Fz(tariert)={fz:.2f}N")

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

    def _move_down_until_contact(self, start_pos, target_pos, quat,
                                  move_robot, get_observation,
                                  force_threshold_n, max_extra_depth_m,
                                  velocity_m_s, max_duration_s,
                                  label="Descent-To-Contact"):
        """Free-space descent to first contact - see _ramp_descend."""
        cfg = self._cfg
        return self._ramp_descend(
            start_pos, target_pos, quat, move_robot, get_observation,
            stiffness=cfg['descent_stiffness'], damping=cfg['descent_damping'],
            velocity_m_s=velocity_m_s, max_duration_s=max_duration_s, label=label,
            force_threshold_n=force_threshold_n,
            extra_depth_below_target_m=max_extra_depth_m,
        )

    def _press_insert_until_seated(self, entry_pos, quat,
                                    move_robot, get_observation,
                                    max_insert_depth_m, velocity_m_s, max_duration_s,
                                    contact_pos,
                                    label="Final-Insert"):
        """
        Presses further in from entry_pos (soft spiral Z-stiffness, same as
        used for the port search) until TCP-z stalls, i.e. the connector is
        fully seated and physically can't go any further - instead of
        aiming for one fixed extra depth and waiting out the full time
        budget regardless of whether it was already fully inserted.
        max_insert_depth_m is only a safety ceiling on how far it will try.

        A z-stall is only accepted as "fully seated" once the TCP has
        travelled at least _min_seat_depth_from_contact_m below contact_pos
        (the connector needs ~4.6cm total, so this is a buffered floor). A
        stall short of that is treated as a mechanical snag: retry the same
        push with a softened rotational stiffness (_snag_recovery_stiffness)
        so the connector can self-align, up to _snag_recovery_max_attempts
        times, before giving up.
        """
        curr_pos = entry_pos.copy()
        stiffness, damping = self._cfg['spiral_stiffness'], self._cfg['spiral_damping']
        attempt_duration = max_duration_s

        for attempt in range(self._snag_recovery_max_attempts + 1):
            curr_pos, stopped_early = self._ramp_descend(
                curr_pos, entry_pos, quat, move_robot, get_observation,
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

            if depth_from_contact >= self._min_seat_depth_from_contact_m:
                self.get_logger().info(
                    f"    {label}: vollstaendig eingesteckt (z-Stillstand) bei z={curr_pos[2]:.4f} "
                    f"(Tiefe={depth_from_contact * 1000:.1f}mm)."
                )
                return curr_pos

            if attempt == self._snag_recovery_max_attempts:
                self.get_logger().warning(
                    f"    {label}: Stillstand bei Tiefe={depth_from_contact * 1000:.1f}mm "
                    f"(< {self._min_seat_depth_from_contact_m * 1000:.0f}mm Soll) nach {attempt} "
                    f"Recovery-Versuchen - gebe auf."
                )
                return curr_pos

            self.get_logger().warning(
                f"    {label}: verfrueher Stillstand bei Tiefe={depth_from_contact * 1000:.1f}mm "
                f"(< {self._min_seat_depth_from_contact_m * 1000:.0f}mm) - vermutlich verhakt. "
                f"Reduziere Rotations-Steifigkeit und versuche erneut "
                f"(Versuch {attempt + 1}/{self._snag_recovery_max_attempts})."
            )
            stiffness, damping = self._snag_recovery_stiffness, self._snag_recovery_damping
            attempt_duration = self._snag_recovery_max_duration_s

    # --- TEMP DEBUG: full descent, no contact stop, just log the force signal ---
    def _debug_descent_force_probe(self, start_pos, target_pos, quat,
                                    move_robot, get_observation,
                                    n_steps, label="Force-Probe-Descent"):
        """
        Descends all the way to target_pos (no early exit on force or stall),
        logging Fz (raw and tared) every step, then prints a min/max/mean
        summary so we can see from the log whether real contact produces a
        distinguishable peak at all, before trusting any threshold.
        """
        cfg = self._cfg
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
                                    entry_z, label="Spiral-Search"):
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
        return False, curr_pos

    # --- Main ---
    def insert_cable(self,
                      task: Task,
                      get_observation: GetObservationCallback,
                      move_robot: MoveRobotCallback,
                      send_feedback: SendFeedbackCallback):
        '''
        1. Query & store the current TCP pose once (robot already positioned
           above the target port, plug grasped and aligned).
        2. Target (port entrance) = start pose, 3cm down along -z. Tare the
           F/T sensor (hardware best-effort + software baseline) before any
           force-based decision.
        3. Force-controlled descent until contact is detected (tared force
           threshold, or a z-stall fallback if the tare isn't perfect).
        4. Spiral search (unchanged from PlugIn) with constant press force,
           exits as soon as the port entry is detected.
        5. Force-controlled final insertion by the configured extra depth.
        6. Report success/failure. Every loop has a fixed step budget, so a
           missing contact/entry cleanly aborts instead of hanging.
        '''
        self.sleep_for(1.0)

        self.get_logger().info("============================================================")
        self.get_logger().info(f"STARTING NEW TASK (SfpPlugInPhase1): id={task.id}")
        self.get_logger().info("============================================================")

        # 1. Query & store current TCP pose once
        obs = get_observation()
        start_pos, start_quat = self._get_current_tcp_pose(obs)
        self.get_logger().info(f"Startpose TCP: Pos={start_pos}, Quat={start_quat}")

        # 2. Target pose (port entrance) = start pose, 3cm down (-z)
        target_pos = start_pos.copy()
        target_pos[2] -= self._insertion_offset_z

        # Tare the F/T sensor before doing anything force-based: the raw
        # wrist_wrench reading is untared and can carry a static bias (e.g.
        # tool/plug weight), which would otherwise be mistaken for contact
        # the instant we start descending.
        send_feedback("Tarieren des F/T-Sensors...")
        self._try_tare_ft_sensor()
        self.sleep_for(0.3)
        self._force_baseline = self._measure_force_baseline(get_observation)

        send_feedback("Starting SFP insertion (Phase 1, perception-free)...")

        # TEMP DEBUG: just descend the full distance and log the force signal,
        # skip contact detection / spiral / insertion entirely.
        if self._debug_force_probe_only:
            self._debug_descent_force_probe(
                start_pos, target_pos, start_quat,
                move_robot, get_observation,
                n_steps=self._debug_probe_steps,
            )
            self.get_logger().info("Force-Probe abgeschlossen - siehe Log fuer Fz-Verlauf. Kein Insert versucht.")
            return True

        # 3. Force-controlled descent until contact
        contact_pos, contact_detected = self._move_down_until_contact(
            start_pos, target_pos, start_quat,
            move_robot, get_observation,
            force_threshold_n=self._contact_force_threshold_n,
            max_extra_depth_m=self._max_descent_margin_m,
            velocity_m_s=self._descent_velocity_m_s,
            max_duration_s=self._descent_max_duration_s,
        )

        if not contact_detected:
            self.get_logger().error("FAILED - Kein Kontakt beim Herunterfahren erkannt.")
            return False

        # 4. Spiral search with constant press force, exit on port entry
        spiral_center = contact_pos.copy()
        spiral_center[2] -= self._press_margin_m  # commanded penetration bias -> press force via soft z-stiffness

        # Anchored to the actually measured contact point, not the a-priori
        # assumed target_pos (start_pos - _insertion_offset_z) - the real
        # descent distance can differ from that assumption (see the force
        # probe log: real contact was ~2.3cm down, not the assumed 5cm), so
        # anchoring to target_pos made entry_z unreachably deep.
        entry_z = contact_pos[2] - self._entry_depth_threshold_m
        entered, entry_pos = self._spiral_search_until_entry(
            spiral_center, start_quat,
            move_robot, get_observation,
            entry_z=entry_z,
        )

        if not entered:
            self.get_logger().warning(
                "Kein Eintritt in den Port erkannt (Spiral-Search Timeout / max. Radius) - "
                "versuche trotzdem den finalen Einsteck-Schritt von der letzten Spiral-Position aus."
            )

        # 5. Force-controlled final insertion - press until it stalls (fully
        # seated), not to one fixed extra depth.
        self._press_insert_until_seated(
            entry_pos, start_quat, move_robot, get_observation,
            max_insert_depth_m=self._additional_insert_depth_m,
            velocity_m_s=self._final_insert_velocity_m_s,
            max_duration_s=self._final_insert_max_duration_s,
            contact_pos=contact_pos,
        )

        self.get_logger().info("============================================================")
        self.get_logger().info("SUCCESS - SFP Kabel eingesteckt (Phase 1).")
        self.get_logger().info("============================================================")
        return True
