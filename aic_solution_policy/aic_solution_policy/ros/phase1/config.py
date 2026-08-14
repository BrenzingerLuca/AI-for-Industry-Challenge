"""Per-connector-type parameters for the phase-1 (force-controlled) policy.

Every parameter describing the approach/insertion behavior lives here, split
by port_type, so sfp and sc can be tuned independently. Both entries started
from the same shared defaults (descent/press/force-threshold/stall/snag
settings) copied from sfp -- that's a starting point, not a claim that the
two connectors should behave identically; tune each independently here.
"""

from ..paths import CHECKPOINT_DIR
import os

CONNECTOR_CONFIGS = {
    'sfp': {
        'off_pos': [0.0, 0.0004, -0.05795],
        'off_quat': [0.17785, 0.00505, -0.02738, -0.98366],
        'residual_model_path': os.path.join(CHECKPOINT_DIR, 'regressor_best_sfp.pt'),
        'insertion_offset_z': 0.01,
        'spiral_stiffness': [300.0, 300.0, 120.0, 200.0, 200.0, 200.0],
        'spiral_damping': [40.0, 40.0, 15.0, 30.0, 30.0, 30.0],
        'spiral_steps': 120,
        'spiral_max_radius': 0.004,
        'spiral_n_turns': 4,

        # Free-space descent-to-contact primitive.
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
        # measured TCP-z stops moving for this many consecutive steps, treat
        # it as contact even if the (tared) force reading hasn't tripped yet.
        'stall_window_steps': 15,
        'stall_epsilon_m': 0.0003,           # 0.3mm
        'stall_grace_steps': 40,             # ignore stall check during initial settle-in

        # Number of small ramped waypoints used to move to the vision-corrected
        # pose after contact, same style as the spiral search's incremental steps.
        'correction_move_steps': 40,

        # Extra clearance ABOVE start_pos to retreat to before applying the XY
        # correction. insertion_offset_z + max_descent_margin_m together are
        # only ~20mm, so an edge-catch can happen just a few mm below
        # start_pos -- retreating only up to start_pos in that case barely
        # clears the snag at all. This margin guarantees real separation
        # regardless of how shallow the catch was.
        'retreat_clearance_m': 0.00,

        # Final-insert snag recovery: the connector needs ~4.6cm total travel
        # (from the initial contact point) to be fully seated, but sometimes
        # catches mechanically before that and the z-stall fires early. Only
        # trust a z-stall as a real seat once past this depth (buffer under
        # the known 4.6cm); a stall short of that is treated as a snag.
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
        'residual_model_path': os.path.join(CHECKPOINT_DIR, 'regressor_best_sc.pt'),
        'insertion_offset_z': 0.01,
        'spiral_stiffness': [600.0, 600.0, 120.0, 300.0, 300.0, 40.0],
        'spiral_damping': [40.0, 40.0, 15.0, 30.0, 30.0, 30.0],
        'spiral_steps': 150,
        'spiral_max_radius': 0.004,
        'spiral_n_turns': 4,

        # Free-space descent-to-contact primitive. Not yet tuned for 'sc' --
        # copied from 'sfp' as a starting point.
        'descent_stiffness': [600.0, 600.0, 600.0, 300.0, 300.0, 300.0],
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
