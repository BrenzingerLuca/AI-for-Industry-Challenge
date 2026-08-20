"""Per-connector-type parameters for the qualification-round policy."""

import os

from ..paths import CHECKPOINT_DIR, DETECTION_MODEL_DIR

CONNECTOR_CONFIGS = {
    'sc': {
        'model_path': os.path.join(DETECTION_MODEL_DIR, 'single_sc_detection.pt'),
        'off_pos': [0.0, -0.015385, -0.04045],
        'off_quat': [0.1608, -0.167181, 0.69417, -0.6814],
        'z_approach_1': 0.03,
        'z_approach_2': -0.03,
        'z_plug': -0.03,
        'cable_tip_frame': "cable_0/sc_tip_link",
        'search_insert_strategy_1': "_spiral_search_and_insert_2d",
        'spiral_max_radius_1': 0.002,
        'spiral_max_radius_2': 0.005,
        'spiral_stiffness_1': [300.0, 300.0, 40.0, 200.0, 200.0, 40.0],
        'spiral_damping_1': [40.0, 40.0, 15.0, 30.0, 30.0, 30.0],
        'spiral_stiffness_2': [150.0, 150.0, 30.0, 300.0, 300.0, 10.0],
        'spiral_damping_2': [30.0, 30.0, 10.0, 30.0, 30.0, 30.0],
        'spiral_steps_1': 150,
        'spiral_steps_2': 250,
        'residual_model_path': os.path.join(CHECKPOINT_DIR, 'regressor_best_sc.pt'),
        # Not loaded/used yet -- only the regressor runs at inference time.
        # Kept here so the path is ready if diffusion inference gets wired up later.
        'residual_diffusion_model_path': os.path.join(CHECKPOINT_DIR, 'diffusion_best_sc.pt'),
    },
    'sfp': {
        'model_path': os.path.join(DETECTION_MODEL_DIR, 'best150.pt'),
        'off_pos': [0.0, 0.0004, -0.05795],
        'off_quat': [0.17785, 0.00505, -0.02738, -0.98366],
        'z_approach_1': 0.02,
        'z_approach_2': -0.003,
        'z_plug': -0.045,
        'cable_tip_frame': "cable_0/sfp_tip_link",
        'search_insert_strategy_1': "_spiral_search_and_insert_2d",
        'spiral_max_radius_1': 0.003,
        'spiral_max_radius_2': 0.005,
        'spiral_stiffness_1': [300.0, 300.0, 80.0, 200.0, 200.0, 200.0],
        'spiral_damping_1': [40.0, 40.0, 15.0, 30.0, 30.0, 30.0],
        'spiral_stiffness_2': [300.0, 300.0, 120.0, 200.0, 200.0, 200.0],
        'spiral_damping_2': [40.0, 40.0, 20.0, 30.0, 30.0, 30.0],
        'spiral_steps_1': 120,
        'spiral_steps_2': 250,
        'residual_model_path': os.path.join(CHECKPOINT_DIR, 'regressor_best_sfp.pt'),
        'residual_diffusion_model_path': os.path.join(CHECKPOINT_DIR, 'diffusion_best_sfp.pt'),
    },
}
