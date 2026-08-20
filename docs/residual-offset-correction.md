# Residual Offset-Correction Model

Both the [qualification](qualification-phase.md) and [phase 1](phase1-flowstate.md)
policies use the same trained model to nudge their approach pose from camera
images. It's the one piece of the vision stack that survived the switch to
Intrinsic FlowState, because it solves a different problem than port
detection: not "where is the port", but "how far off is the plug tip from
where it should be, right now".

Code: [`residual_offset_model.py`](../aic_solution_policy/aic_solution_policy/ros/residual_offset_model.py).
Training notebook: [`residual_policy.ipynb`](../aic_solution_policy/residual_policy.ipynb).

## What it does

Given the three camera images at the current pose, a ResNet18 encoder (shared
across views) feeds an MLP head that regresses the 6-DoF offset between the
cable tip and the target port: `dx, dy, dz` in meters and `droll, dpitch,
dyaw` in degrees, expressed in the port's own frame. Separate checkpoints are
trained per connector type (`regressor_best_sfp.pt`, `regressor_best_sc.pt`).

In both policies, only the lateral position (`dx`, `dy`) is actually used —
depth and rotation are dropped, since Z is already governed by the
approach/plug depths (qualification) or the force-controlled descent (phase
1), and the connector orientation doesn't need correcting.

## Training data

[`data_acquisition.py`](../aic_solution_policy/aic_solution_policy/data_acquisition.py)
collects the training set: it aligns the cable tip to a port using
ground-truth TF, then repeatedly perturbs that pose by a random offset and
records the three camera images plus the actual tip/TCP/port poses. The
model is trained to recover that offset from the images alone, so at
inference time — with no ground truth available — it can estimate the same
thing from what the cameras see.

## Why it's needed at all

Neither policy's port-pose estimate is perfect: the qualification round's
triangulation has its own noise, and FlowState's placement in phase 1 is good
but not exact. Rather than trying to eliminate that error at the source, this
model estimates it directly from the cameras and folds a correction into the
approach pose before the spiral search — cheaper to train than to chase
perfect calibration.
