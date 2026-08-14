"""Phase-1 insertion policy.

Port detection is handled upstream by Intrinsic FlowState, so this policy is
perception-free (no cameras/YOLO for localization) and purely force-
controlled: descend until contact, decide whether the tip already went
straight in or got caught on the port's rim, and if caught, apply the same
shared vision-based residual regressor used in the qualification round --
now evaluated at the contact pose rather than at an approach distance -- to
correct laterally before retrying or falling back to a spiral search. This is
the policy that reached 14/160 in Phase 1.

Assumes the robot is already positioned above the target port with the plug
grasped and aligned, so the pre-contact target pose comes purely from the TCP
pose measured at task start.
"""

from aic_task_interfaces.msg import Task
from aic_model.policy import (
    Policy,
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)

from ..residual_offset_model import ResidualOffsetCorrector, apply_predicted_correction
from .config import CONNECTOR_CONFIGS
from .motion import ForceControlledMotion


class Phase1PlugIn(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)

        from cv_bridge import CvBridge

        self.get_logger().info("Phase1PlugIn Policy initialised")
        self._bridge = CvBridge()
        self._camera_names = ['left', 'center', 'right']
        self._configs = CONNECTOR_CONFIGS

        self._motion = ForceControlledMotion(
            parent_node=self._parent_node,
            logger=self.get_logger(),
            sleep_for=self.sleep_for,
        )

        # TEMP DEBUG probe (see ForceControlledMotion.debug_descent_force_probe)
        # just logs every step over a fixed step count, independent of the
        # normal insertion flow below.
        self._debug_probe_steps = 400
        # TEMP DEBUG: descend the full insertion_offset_z with no contact/stall
        # early-exit at all, logging Fz every step, so the raw force signal can
        # be inspected to see whether contact is distinguishable at all before
        # tuning contact_force_threshold_n. Set back to False once a good
        # threshold is known.
        self._debug_force_probe_only = False

        # ROS param so the correction can be A/B-tested without rebuilding the package.
        self._parent_node.declare_parameter('residual_correction.enabled', True)

        # Trained offset-correction model(s) from residual_policy.ipynb, one per
        # connector type -- same checkpoints as the qualification-round policy.
        self._offset_corrector = ResidualOffsetCorrector(
            bridge=self._bridge,
            logger=self.get_logger(),
            checkpoint_paths={c: cfg.get('residual_model_path') for c, cfg in self._configs.items()},
            skip_description="post-contact correction",
        )

    # --- Residual correction wiring (predict, then move to the corrected pose) ---

    def _get_predicted_correction(self, c_type, get_observation):
        """Runs the offset-correction model on the CURRENT camera images and
        returns the raw predicted_offset (or None if correction is disabled,
        no model is loaded for this connector type, or images weren't
        available). Deliberately does not move the robot: call this before
        any pre-correction lift, while the TCP is still at the pose the model
        was trained to see (right at contact) -- predicting from an
        already-lifted vantage would feed it an out-of-distribution view.
        """
        residual_enabled = self._parent_node.get_parameter('residual_correction.enabled').value
        if not residual_enabled:
            return None

        obs = get_observation()
        predicted_offset = self._offset_corrector.predict(obs, c_type)
        if predicted_offset is None:
            self.get_logger().warning(f"No residual correction available for '{c_type}', skipping")
            return None

        dx, dy, dz, droll, dpitch, dyaw = predicted_offset
        self.get_logger().info(
            f"Residual model prediction: dxyz=({dx*1e3:.2f},{dy*1e3:.2f},{dz*1e3:.2f})mm "
            f"rpy=({droll:.1f},{dpitch:.1f},{dyaw:.1f})deg"
        )
        return predicted_offset

    def _move_to_corrected_pose(self, pos, quat, predicted_offset, cfg, move_robot, get_observation):
        """Applies an already-predicted correction (from _get_predicted_correction,
        computed at the pre-lift contact pose) relative to `pos` (e.g. the
        post-lift pose) and smoothly ramps the TCP there. Uses
        descent_stiffness/damping (full-strength Z), not spiral_stiffness/
        damping -- the spiral config is deliberately soft in Z to realize a
        controlled press force while inside the port, which would let the TCP
        sag under gravity during this move instead of holding the lift.
        """
        corrected_pos, corrected_quat = apply_predicted_correction(
            pos, quat, cfg['off_pos'], cfg['off_quat'], predicted_offset
        )
        return self._motion.smooth_move_to(
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
        """Thin wrapper around _run_insertion(): always returns True, whether
        the insertion actually completed or a step (or the whole attempt)
        only got as far as its time/attempt budget allowed, or something
        unexpected raised -- a stuck/timed-out/errored attempt should still
        report "done" rather than fail or crash the task. Downstream
        evaluation judges actual success from the resulting port/robot state,
        not this return value.
        """
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
        """
        1. Query & store the current TCP pose once (robot already positioned
           above the target port, plug grasped and aligned).
        2. Target (port entrance) = start pose, insertion_offset_z down along
           -z. Tare the F/T sensor before any force-based decision.
        3. Force-controlled descent until contact is detected.
        3.1 Inside-port check: did the tip already travel past the assumed
            port-entrance depth without an earlier stop? If so, skip straight
            to the final press (step 5).
        3.5 Otherwise it stopped early -- likely caught on the port's edge.
            Run the residual offset model now, at the contact pose, and
            smoothly move to cancel out its predicted tip-to-port offset.
        3.6 Retry the descent-to-contact from the corrected pose and re-run
            the inside-port check. If that also lands inside, skip to step 5.
        4. If still not inside after the retry, fall back to the spiral
           search with constant press force, exiting as soon as port entry is
           detected.
        5. Force-controlled final insertion by the configured extra depth.
        6. Report success/failure. Every loop has a fixed step budget, so a
           missing contact/entry cleanly aborts instead of hanging.
        """
        self.sleep_for(1.0)

        c_type = task.port_type.lower()
        if c_type not in self._configs:
            self.get_logger().error(f"Kabeltyp '{c_type}' ist nicht konfiguriert!")
            return False
        cfg = self._configs[c_type]

        self.get_logger().info("============================================================")
        self.get_logger().info(f"STARTING NEW TASK (Phase1PlugIn): {c_type.upper()} id={task.id}")
        self.get_logger().info("============================================================")

        # 1. Query & store current TCP pose once
        obs = get_observation()
        start_pos, start_quat = self._motion.get_current_tcp_pose(obs)
        self.get_logger().info(f"Startpose TCP: Pos={start_pos}, Quat={start_quat}")

        # 2. Target pose (port entrance) = start pose, insertion_offset_z down (-z)
        target_pos = start_pos.copy()
        target_pos[2] -= cfg['insertion_offset_z']

        # Tare the F/T sensor before doing anything force-based: the raw
        # wrist_wrench reading is untared and can carry a static bias (e.g.
        # tool/plug weight), which would otherwise be mistaken for contact
        # the instant we start descending.
        send_feedback("Tarieren des F/T-Sensors...")
        self._motion.try_tare_ft_sensor()
        self.sleep_for(0.3)
        self._motion.force_baseline = self._motion.measure_force_baseline(get_observation)

        send_feedback(f"Starting {c_type} insertion (force-controlled, perception at contact)...")

        # TEMP DEBUG: just descend the full distance and log the force signal,
        # skip contact detection / spiral / insertion entirely.
        if self._debug_force_probe_only:
            self._motion.debug_descent_force_probe(
                start_pos, target_pos, start_quat,
                move_robot, get_observation, cfg,
                n_steps=self._debug_probe_steps,
            )
            self.get_logger().info("Force-Probe abgeschlossen - siehe Log fuer Fz-Verlauf. Kein Insert versucht.")
            return True

        # 3. Force-controlled descent until contact
        contact_pos, contact_detected = self._motion.move_down_until_contact(
            start_pos, target_pos, start_quat, move_robot, get_observation, cfg,
        )
        entry_quat = start_quat

        # 3.1 Did we already go straight in, or did we stop early (edge catch)?
        inside_port = self._motion.check_inside_port(start_pos, contact_pos, cfg)

        if not inside_port:
            # 3.5 Stopped early -- likely caught on the port edge. Predict the
            # residual offset correction FIRST, while the TCP is still at the
            # pose the model was trained to see (right at contact) -- moving
            # away before predicting would feed it an out-of-distribution view.
            self.sleep_for(1.0)
            predicted_offset = self._get_predicted_correction(c_type, get_observation)

            corrected = False
            if predicted_offset is not None:
                # Retreat back up past the original approach pose -- not just
                # up to it -- by retreat_clearance_m. insertion_offset_z +
                # max_descent_margin_m are only ~20mm total, so an edge-catch
                # can happen just a few mm below start_pos; retreating only up
                # to start_pos in that case barely clears the snag at all.
                retract_m = (start_pos[2] - contact_pos[2]) + cfg['retreat_clearance_m']
                approach_pos = self._motion.retract_up(
                    contact_pos, start_quat, move_robot, get_observation,
                    stiffness=cfg['descent_stiffness'], damping=cfg['descent_damping'],
                    retract_m=retract_m,
                    velocity_m_s=cfg['descent_velocity_m_s'],
                    label="Retract-To-Approach",
                )

                # Apply the XY correction at the approach pose, in free space.
                corrected_pos, corrected_quat = self._move_to_corrected_pose(
                    approach_pos, start_quat, predicted_offset, cfg,
                    move_robot, get_observation,
                )
                corrected = True

            if corrected:
                entry_quat = corrected_quat
                # 3.6 Try the insert again from the corrected pose, exactly
                # like the initial attempt in step 3.
                contact_pos, _ = self._motion.move_down_until_contact(
                    corrected_pos, target_pos, entry_quat, move_robot, get_observation, cfg,
                    label="Descent-To-Contact-Retry",
                )
                inside_port = self._motion.check_inside_port(start_pos, contact_pos, cfg)

        if inside_port:
            self.get_logger().info(
                f"Bereits direkt eingesteckt (Kontakt bei z={contact_pos[2]:.4f}) - "
                f"ueberspringe Korrektur/Spiralsuche, gehe direkt zum finalen Einstecken."
            )
            entry_pos = contact_pos
            # min_seat_depth_from_contact_m is calibrated as "depth needed
            # below the port ENTRANCE to be fully seated". In this branch
            # contact_pos is already >= max_descent_margin_m past the assumed
            # entrance (target_pos) -- that's why check_inside_port tripped --
            # so measuring remaining-seat-depth from contact_pos itself would
            # throw away the depth already covered. Anchor to target_pos
            # instead so that head start is credited correctly.
            seat_reference_pos = target_pos
        else:
            # 4. Still not inside after the retry -- fall back to the spiral
            # search, anchored to the latest contact point.
            spiral_center = contact_pos.copy()
            spiral_center[2] -= cfg['press_margin_m']  # commanded penetration bias -> press force via soft z-stiffness

            # Anchored to the actually measured (and possibly corrected)
            # contact point, not the a-priori assumed target_pos -- the real
            # descent distance can differ from that assumption.
            entry_z = contact_pos[2] - cfg['entry_depth_threshold_m']
            entered, entry_pos = self._motion.spiral_search_until_entry(
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
            # min_seat_depth_from_contact_m is calibrated against.
            seat_reference_pos = contact_pos

        # 5. Force-controlled final insertion -- press until it stalls (fully
        # seated), not to one fixed extra depth.
        self._motion.press_insert_until_seated(
            entry_pos, entry_quat, move_robot, get_observation,
            max_insert_depth_m=cfg['additional_insert_depth_m'],
            velocity_m_s=cfg['final_insert_velocity_m_s'],
            max_duration_s=cfg['final_insert_max_duration_s'],
            contact_pos=seat_reference_pos,
            spiral_cfg=cfg,
        )

        self.get_logger().info("============================================================")
        self.get_logger().info(f"SUCCESS - {c_type.upper()} Kabel eingesteckt (Phase1PlugIn).")
        self.get_logger().info("============================================================")
        return True
