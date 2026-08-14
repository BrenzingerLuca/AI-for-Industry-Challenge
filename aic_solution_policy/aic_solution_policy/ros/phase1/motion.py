"""Force/stall-aware Cartesian motion primitives for the phase-1 policy."""

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from std_srvs.srv import Trigger

from ..common import build_motion_update, check_force_threshold


class ForceControlledMotion:
    """Ramped descent-to-contact, retraction, spiral search that exits on
    port entry, and smooth ramped moves -- all sharing the same tared force
    reading and step timing. Used by Phase1PlugIn's insertion sequence.
    """

    def __init__(self, parent_node, logger, sleep_for, step_dt=0.05):
        self._parent_node = parent_node
        self._logger = logger
        self._sleep_for = sleep_for
        self._step_dt = step_dt
        # Raw wrist_wrench is untared (fed straight from /fts_broadcaster/wrench,
        # see aic_adapter.cpp) -- subtract a baseline measured at task start so a
        # static bias (e.g. tool/plug weight) doesn't get mistaken for contact.
        self.force_baseline = np.zeros(3)

    def get_force_xyz(self, observation):
        try:
            if hasattr(observation, 'wrist_wrench') and observation.wrist_wrench is not None:
                f = observation.wrist_wrench.wrench.force
                return np.array([f.x, f.y, f.z])
        except Exception:
            pass
        return np.zeros(3)

    def get_force_z(self, observation):
        """Tared Fz: raw wrist_wrench.z minus the baseline measured at task start."""
        return float(self.get_force_xyz(observation)[2] - self.force_baseline[2])

    def try_tare_ft_sensor(self):
        """Best-effort hardware tare via /aic_controller/tare_force_torque_sensor.

        Only zeroes the controller's internal force-feedback offset, not the
        raw /fts_broadcaster/wrench topic wrist_wrench reads from -- this
        complements, not replaces, the software baseline from measure_force_baseline.
        """
        try:
            client = self._parent_node.create_client(Trigger, "/aic_controller/tare_force_torque_sensor")
            if not client.wait_for_service(timeout_sec=2.0):
                self._logger.warning(
                    "tare_force_torque_sensor Service nicht erreichbar - überspringe "
                    "(Software-Baseline-Tare läuft trotzdem)."
                )
                return
            response = client.call(Trigger.Request())
            if response.success:
                self._logger.info(f"F/T-Sensor (Controller-intern) tariert: {response.message}")
            else:
                self._logger.warning(f"Tarieren (Service) fehlgeschlagen: {response.message}")
        except Exception as e:
            self._logger.warning(f"Tarieren (Service) nicht möglich: {e}")

    def measure_force_baseline(self, get_observation, n_samples=20, sleep_s=0.02):
        """Software self-tare: average the raw wrist_wrench force over a short,
        stationary window and use it as the zero-offset for all subsequent Fz
        threshold checks this task.
        """
        samples = []
        for _ in range(n_samples):
            obs = get_observation()
            samples.append(self.get_force_xyz(obs))
            self._sleep_for(sleep_s)
        baseline = np.mean(samples, axis=0)
        self._logger.info(
            f"F/T-Sensor Baseline (Software-Tare) gemessen: {baseline} N - wird von allen "
            f"folgenden Fz-Messungen abgezogen."
        )
        return baseline

    def pos_to_array(self, position):
        return np.array([position.x, position.y, position.z])

    def get_current_tcp_pose(self, obs):
        pos = self.pos_to_array(obs.controller_state.tcp_pose.position)
        q = obs.controller_state.tcp_pose.orientation
        quat = np.array([q.x, q.y, q.z, q.w])
        return pos, quat

    def ramp_descend(self, start_pos, target_pos, quat,
                      move_robot, get_observation, cfg,
                      stiffness, damping, velocity_m_s, max_duration_s,
                      label="Descent",
                      force_threshold_n=None,
                      extra_depth_below_target_m=0.0):
        """Ramps the commanded Z linearly, at velocity_m_s, from start_pos[2]
        down to target_pos[2] - extra_depth_below_target_m (XY held fixed at
        start_pos XY throughout; target_pos only supplies the Z floor).
        Stops as soon as any of:
          - force_threshold_n is given and measured |Fz| (tared) exceeds it,
          - measured TCP-z stalls (no real movement) for stall_window_steps in
            a row (past stall_grace_steps) -- a hard mechanical stop was hit
            (real contact, or fully seated), regardless of whether the force
            reading caught it, or
          - max_duration_s of wall-clock time has elapsed.
        velocity_m_s and max_duration_s are independent: max_duration_s is a
        safety backstop for cases where a real stop signal was missed, not a
        function of speed.

        The loop always runs for the full max_duration_s -- the commanded ramp
        target clamps at the floor via `min(..., total_travel)` once distance
        is covered, so running the full timeout just means holding there and
        continuing to watch for a stall/force stop instead of exiting early
        (this matters because at high velocity, capping steps at the nominal
        travel time could end the loop before stall_grace_steps even elapsed).

        Returns (reached_pos, stopped_early). stopped_early is True only for
        the force/stall cases above; running out of time without one returns False.
        """
        step_dt = self._step_dt
        floor_z = target_pos[2] - extra_depth_below_target_m
        total_travel = start_pos[2] - floor_z
        step_distance = velocity_m_s * step_dt

        distance_steps = max(1, int(np.ceil(total_travel / step_distance)))
        timeout_steps = max(1, int(np.ceil(max_duration_s / step_dt)))
        n_steps = timeout_steps

        self._logger.info(
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

            motion_update = build_motion_update(cmd_pos, quat, stiffness, damping)
            move_robot(motion_update=motion_update)

            obs = get_observation()
            check_force_threshold(obs, self._logger)
            fz = self.get_force_z(obs)
            curr_pos = self.pos_to_array(obs.controller_state.tcp_pose.position)

            z_history.append(curr_pos[2])
            if len(z_history) > cfg['stall_window_steps']:
                z_history.pop(0)

            if i % 20 == 0:
                self._logger.info(f"    [{i}/{n_steps}] t={i * step_dt:.1f}s z={curr_pos[2]:.4f} Fz(tariert)={fz:.2f}N")

            if force_threshold_n is not None and abs(fz) >= force_threshold_n:
                self._logger.info(f"    {label}: Kraft-Stopp bei z={curr_pos[2]:.4f} (Fz={fz:.2f}N)")
                return curr_pos, True

            if (i >= cfg['stall_grace_steps']
                    and len(z_history) == cfg['stall_window_steps']
                    and (max(z_history) - min(z_history)) < cfg['stall_epsilon_m']):
                self._logger.info(
                    f"    {label}: z-Stillstand bei z={curr_pos[2]:.4f} - keine weitere Bewegung "
                    f"trotz tieferem Kommando (Fz={fz:.2f}N)."
                )
                return curr_pos, True

            self._sleep_for(step_dt)

        reason = (
            f"Boden erreicht (kommandierte Distanz {total_travel * 1000:.0f}mm), "
            f"{max_duration_s:.1f}s Zeitlimit ohne weiteren Stopp ausgeschoepft"
            if timeout_steps >= distance_steps else
            f"Zeitlimit ({max_duration_s:.1f}s) erreicht, bevor die volle Distanz "
            f"({total_travel * 1000:.0f}mm) kommandiert war"
        )
        self._logger.warning(
            f"    {label}: Ende ({reason}) erreicht ohne Stopp-Kriterium (letzte z={curr_pos[2]:.4f}, Fz={fz:.2f}N)."
        )
        return curr_pos, False

    def retract_up(self, start_pos, quat, move_robot, get_observation,
                    stiffness, damping, retract_m, velocity_m_s,
                    label="Retract"):
        """Ramps the commanded Z straight up by retract_m (XY held fixed) --
        the inverse of ramp_descend's direction, used to back a snag off
        before the rescue spiral search instead of searching laterally while
        still jammed against whatever it caught on. No stall/force stop
        check; it's a short, fixed-distance move.
        """
        step_dt = self._step_dt
        step_distance = velocity_m_s * step_dt
        n_steps = max(1, int(np.ceil(retract_m / step_distance)))

        self._logger.info(f"    {label}: fahre {retract_m * 1000:.1f}mm hoch von z={start_pos[2]:.4f}.")

        curr_pos = start_pos.copy()
        for i in range(n_steps):
            traveled = min((i + 1) * step_distance, retract_m)
            cmd_pos = np.array([start_pos[0], start_pos[1], start_pos[2] + traveled])

            motion_update = build_motion_update(cmd_pos, quat, stiffness, damping)
            move_robot(motion_update=motion_update)

            obs = get_observation()
            check_force_threshold(obs, self._logger)
            fz = self.get_force_z(obs)
            curr_pos = self.pos_to_array(obs.controller_state.tcp_pose.position)

            # Logged every step (unlike ramp_descend's every-20th) -- this move
            # is short, and per-step z-vs-commanded and Fz shows whether a
            # stuck retreat is a real force ceiling or something else.
            self._logger.info(
                f"    [{i}/{n_steps}] z_cmd={cmd_pos[2]:.4f} z_ist={curr_pos[2]:.4f} Fz(tariert)={fz:.2f}N"
            )

            self._sleep_for(step_dt)

        self._logger.info(f"    {label}: jetzt bei z={curr_pos[2]:.4f}.")
        return curr_pos

    def move_down_until_contact(self, start_pos, target_pos, quat,
                                 move_robot, get_observation, cfg,
                                 label="Descent-To-Contact"):
        """Free-space descent to first contact (see ramp_descend). All descent
        parameters (stiffness, velocity, force threshold, ...) come from cfg.
        """
        return self.ramp_descend(
            start_pos, target_pos, quat, move_robot, get_observation, cfg,
            stiffness=cfg['descent_stiffness'], damping=cfg['descent_damping'],
            velocity_m_s=cfg['descent_velocity_m_s'], max_duration_s=cfg['descent_max_duration_s'],
            label=label,
            force_threshold_n=cfg['contact_force_threshold_n'],
            extra_depth_below_target_m=cfg['max_descent_margin_m'],
        )

    def press_insert_until_seated(self, entry_pos, quat,
                                   move_robot, get_observation,
                                   max_insert_depth_m, velocity_m_s, max_duration_s,
                                   contact_pos, spiral_cfg,
                                   label="Final-Insert"):
        """Presses further in from entry_pos (soft spiral Z-stiffness) until
        TCP-z stalls, i.e. the connector is fully seated. max_insert_depth_m
        is only a safety ceiling.

        A z-stall only counts as "fully seated" once the TCP has travelled at
        least min_seat_depth_from_contact_m below contact_pos (the caller's
        best estimate of the port entrance depth -- on the direct-insert path
        that's the assumed entrance target_pos rather than the exact contact
        point, so the already-covered depth gets credited instead of resetting
        the seat-depth budget to zero). A stall short of that is treated as a
        mechanical snag: retry with softened rotational stiffness
        (snag_recovery_stiffness) so the connector can self-align, up to
        snag_recovery_max_attempts times. Every
        snag_recovery_attempts_before_spiral_search straight pushes that
        haven't freed it, back off snag_recovery_retract_m upward first
        (releases the jam instead of searching laterally while still wedged),
        then try one lateral spiral search from there -- this repeats for as
        long as attempts remain.
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
            curr_pos, stopped_early = self.ramp_descend(
                curr_pos, entry_pos, quat, move_robot, get_observation, spiral_cfg,
                stiffness=stiffness, damping=damping,
                velocity_m_s=velocity_m_s, max_duration_s=attempt_duration, label=label,
                force_threshold_n=None,
                extra_depth_below_target_m=max_insert_depth_m,
            )
            depth_from_contact = contact_pos[2] - curr_pos[2]

            if not stopped_early:
                self._logger.info(
                    f"    {label}: Sicherheitsgrenze ({max_insert_depth_m * 1000:.0f}mm) erreicht ohne klaren "
                    f"Stillstand, letzte Position z={curr_pos[2]:.4f} (Tiefe={depth_from_contact * 1000:.1f}mm)."
                )
                return curr_pos

            if depth_from_contact >= min_seat_depth_from_contact_m:
                self._logger.info(
                    f"    {label}: vollstaendig eingesteckt (z-Stillstand) bei z={curr_pos[2]:.4f} "
                    f"(Tiefe={depth_from_contact * 1000:.1f}mm)."
                )
                return curr_pos

            if attempt == snag_recovery_max_attempts:
                self._logger.warning(
                    f"    {label}: Stillstand bei Tiefe={depth_from_contact * 1000:.1f}mm "
                    f"(< {min_seat_depth_from_contact_m * 1000:.0f}mm Soll) nach {attempt} "
                    f"Recovery-Versuchen - gebe auf."
                )
                return curr_pos

            attempts_since_rescue += 1
            if attempts_since_rescue >= snag_recovery_attempts_before_spiral_search:
                attempts_since_rescue = 0
                stuck_z = curr_pos[2]
                self._logger.warning(
                    f"    {label}: nach {snag_recovery_attempts_before_spiral_search} weiteren "
                    f"Versuchen (insgesamt {attempt + 1}) weiterhin fest bei Tiefe="
                    f"{depth_from_contact * 1000:.1f}mm - versuche laterale Spiralsuche zum Loesen."
                )
                curr_pos = self.retract_up(
                    curr_pos, quat, move_robot, get_observation,
                    stiffness=stiffness, damping=damping,
                    retract_m=snag_recovery_retract_m,
                    velocity_m_s=velocity_m_s,
                    label=f"{label}-Rescue-Retract",
                )
                spiral_center = curr_pos.copy()
                spiral_center[2] -= spiral_cfg['press_margin_m']
                unstuck, curr_pos = self.spiral_search_until_entry(
                    spiral_center, quat, move_robot, get_observation,
                    entry_z=stuck_z - snag_recovery_unstick_margin_m,
                    spiral_cfg=spiral_cfg,
                    label=f"{label}-Rescue-Spiral",
                )
                if unstuck:
                    self._logger.info(
                        f"    {label}: durch Spiralsuche wieder in Bewegung bei z={curr_pos[2]:.4f} "
                        f"- setze Einstecken fort."
                    )
                else:
                    self._logger.warning(
                        f"    {label}: Spiralsuche zum Loesen ohne erkannte Bewegung beendet "
                        f"(letzte z={curr_pos[2]:.4f}) - versuche trotzdem mit reduzierter "
                        f"Rotations-Steifigkeit weiter."
                    )
                stiffness, damping = snag_recovery_stiffness, snag_recovery_damping
                attempt_duration = snag_recovery_max_duration_s
                continue

            self._logger.warning(
                f"    {label}: verfrueher Stillstand bei Tiefe={depth_from_contact * 1000:.1f}mm "
                f"(< {min_seat_depth_from_contact_m * 1000:.0f}mm) - vermutlich verhakt. "
                f"Reduziere Rotations-Steifigkeit und versuche erneut "
                f"(Versuch {attempt + 1}/{snag_recovery_max_attempts})."
            )
            stiffness, damping = snag_recovery_stiffness, snag_recovery_damping
            attempt_duration = snag_recovery_max_duration_s

    def debug_descent_force_probe(self, start_pos, target_pos, quat,
                                   move_robot, get_observation, cfg,
                                   n_steps, label="Force-Probe-Descent"):
        """Descends all the way to target_pos (no early exit on force or
        stall), logging Fz (raw and tared) every step, then prints a min/max/
        mean summary -- to see whether real contact produces a distinguishable
        peak at all, before trusting any threshold.
        """
        total_travel = start_pos[2] - target_pos[2]

        self._logger.info(
            f"==> {label}: volle Absenkung {total_travel * 1000:.1f}mm OHNE Kontakt-Stopp "
            f"(nur zum Beobachten der Kraftspitzen)."
        )

        z_log, raw_log, tared_log = [], [], []
        for i in range(n_steps):
            frac = (i + 1) / n_steps
            cmd_pos = np.array([target_pos[0], target_pos[1], start_pos[2] - frac * total_travel])

            motion_update = build_motion_update(cmd_pos, quat, cfg['descent_stiffness'], cfg['descent_damping'])
            move_robot(motion_update=motion_update)

            obs = get_observation()
            check_force_threshold(obs, self._logger)
            raw_fz = float(self.get_force_xyz(obs)[2])
            tared_fz = self.get_force_z(obs)
            curr_pos = self.pos_to_array(obs.controller_state.tcp_pose.position)

            z_log.append(curr_pos[2])
            raw_log.append(raw_fz)
            tared_log.append(tared_fz)

            self._logger.info(
                f"    [{i}/{n_steps}] z={curr_pos[2]:.4f} Fz_raw={raw_fz:.2f}N Fz_tariert={tared_fz:.2f}N"
            )

            self._sleep_for(0.05)

        zs = np.array(z_log)
        raws = np.array(raw_log)
        tareds = np.array(tared_log)
        peak_idx = int(np.argmax(np.abs(tareds)))

        self._logger.info("==== Force-Probe Zusammenfassung ====")
        self._logger.info(
            f"Fz tariert: min={tareds.min():.2f}N max={tareds.max():.2f}N "
            f"mean={tareds.mean():.2f}N std={tareds.std():.2f}N"
        )
        self._logger.info(
            f"Fz raw:     min={raws.min():.2f}N max={raws.max():.2f}N "
            f"mean={raws.mean():.2f}N std={raws.std():.2f}N"
        )
        self._logger.info(
            f"Groesster |Fz tariert| = {tareds[peak_idx]:.2f}N bei z={zs[peak_idx]:.4f} "
            f"(Schritt {peak_idx}/{n_steps})"
        )
        self._logger.info("======================================")

        return self.pos_to_array(get_observation().controller_state.tcp_pose.position)

    def spiral_search_until_entry(self, center_pos, quat,
                                   move_robot, get_observation,
                                   entry_z, spiral_cfg, label="Spiral-Search"):
        """Spiral search identical to the qualification policy's (same
        stiffness/damping/steps/max_radius shape), but exits as soon as the
        TCP sinks below entry_z (port entrance detected), instead of only
        reporting a final distance.

        Returns (True, curr_pos) on detected entry. If the whole spiral runs
        out without a detected entry, still returns the last TCP position as
        (False, curr_pos) so the caller can fall back to pressing straight
        down from there -- entry_z is a detection heuristic, not proof the
        plug isn't already resting over the hole.
        """
        max_radius = spiral_cfg['spiral_max_radius']
        n_turns = spiral_cfg['spiral_n_turns']
        steps = spiral_cfg['spiral_steps']

        self._logger.info(
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

            motion_update = build_motion_update(search_pos, quat, spiral_cfg['spiral_stiffness'], spiral_cfg['spiral_damping'])
            move_robot(motion_update=motion_update)

            obs = get_observation()
            check_force_threshold(obs, self._logger)
            curr_pos = self.pos_to_array(obs.controller_state.tcp_pose.position)

            if idx % 20 == 0:
                self._logger.info(f"    [{idx}/{steps}] r={r * 1000:.2f}mm z={curr_pos[2]:.4f}")

            if curr_pos[2] < entry_z:
                self._logger.info(f"    Eintritt erkannt: TCP-z={curr_pos[2]:.4f} < {entry_z:.4f}")
                return True, curr_pos

            self._sleep_for(0.05)

        self._logger.warning(
            f"    Kein Eintritt erkannt: max. Radius ({max_radius * 1000:.1f}mm) ohne Erfolg durchsucht."
        )
        return False, curr_pos

    def check_inside_port(self, start_pos, contact_pos, cfg):
        """After the initial descent-to-contact stops, decide whether the tip
        is already inside the port (descended straight in) or stopped early
        on the port's rim.

        delta_z is how far the TCP travelled down from the very first
        approach pose (start_pos, queried once at task start -- stays
        comparable across the retry after a correction). distance is how much
        of the assumed insertion_offset_z + max_descent_margin_m safety
        margin is still "left" below the actual contact depth. distance <= 0
        means the tip already went past the assumed port-entrance depth
        without an earlier stop -> it's in; distance > 0 means it stopped
        well short -> likely caught on the edge.
        """
        delta_z = start_pos[2] - contact_pos[2]
        distance = cfg['insertion_offset_z'] - delta_z + cfg['max_descent_margin_m']
        inside = distance <= 0.0
        self._logger.info(
            f"    Inside-Port-Check: delta_z={delta_z * 1000:.1f}mm, distance={distance * 1000:.1f}mm "
            f"-> {'innen (direkt eingesteckt)' if inside else 'nicht innen (vermutlich Kante)'}"
        )
        return inside

    def smooth_move_to(self, start_pos, start_quat, target_pos, target_quat,
                        move_robot, get_observation,
                        stiffness, damping, steps, label="Smooth-Move"):
        """Ramps linearly (position lerp + orientation slerp) from start to
        target over `steps` small waypoints, instead of commanding the target
        pose in a single jump -- which would ask the impedance controller for
        a large instantaneous step while still pressed against the part.
        """
        self._logger.info(
            f"==> {label}: von xy=({start_pos[0]:.4f},{start_pos[1]:.4f}) nach "
            f"xy=({target_pos[0]:.4f},{target_pos[1]:.4f}) in {steps} Schritten."
        )

        slerp = Slerp([0.0, 1.0], R.from_quat([start_quat, target_quat]))

        curr_pos, curr_quat = start_pos.copy(), np.asarray(target_quat)
        for i in range(steps):
            frac = (i + 1) / steps
            cmd_pos = start_pos + frac * (target_pos - start_pos)
            cmd_quat = slerp([frac])[0].as_quat()

            motion_update = build_motion_update(cmd_pos, cmd_quat, stiffness, damping)
            move_robot(motion_update=motion_update)

            obs = get_observation()
            check_force_threshold(obs, self._logger)
            curr_pos = self.pos_to_array(obs.controller_state.tcp_pose.position)
            curr_quat = cmd_quat

            self._sleep_for(self._step_dt)

        self._logger.info(f"    {label}: fertig bei z={curr_pos[2]:.4f}.")
        return curr_pos, curr_quat
