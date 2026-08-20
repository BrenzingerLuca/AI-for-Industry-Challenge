"""Locates aic_solution/ on disk so the policies can find trained model checkpoints.

Shared by both aic_solution_policy.ros.qualification and
aic_solution_policy.ros.phase1.
"""

import os


def resolve_aic_solution_dir():
    """Find aic_solution/ (parent of dataset/checkpoints and training/models).

    pixi-build-ros installs this package via a real copy, not a symlink, so
    at runtime __file__ points into .pixi/envs/.../site-packages rather than
    the source tree -- prefer PIXI_PROJECT_ROOT (set by `pixi shell`/`pixi
    run`, see docs/running-and-testing.md) and fall back to the
    source-tree-relative guess for direct/dev execution outside a pixi
    environment.
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


AIC_SOLUTION_DIR = resolve_aic_solution_dir()
CHECKPOINT_DIR = os.path.join(AIC_SOLUTION_DIR, 'dataset', 'checkpoints')
DETECTION_MODEL_DIR = os.path.join(AIC_SOLUTION_DIR, 'training', 'models')
