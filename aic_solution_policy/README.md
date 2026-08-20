# aic_solution_policy

ROS 2 package (ament_python) holding our two insertion policies for the AI
for Industry Challenge, plus the training-data collection policy.

- `aic_solution_policy/ros/qualification/` — `QualificationPlugIn`, the
  vision-based qualification-round policy.
- `aic_solution_policy/ros/phase1/` — `Phase1PlugIn`, the force-controlled
  phase-1 policy (FlowState positions the robot).
- `aic_solution_policy/ros/residual_offset_model.py` — the offset-correction
  regressor shared by both.
- `aic_solution_policy/data_acquisition.py` — collects the training data for
  that regressor.
- `residual_policy.ipynb` — trains it.

For the story behind the two policies, how they work, and how to run/test
them, see [`aic_solution/docs/`](../docs/):

- [qualification-phase.md](../docs/qualification-phase.md)
- [phase1-flowstate.md](../docs/phase1-flowstate.md)
- [residual-offset-correction.md](../docs/residual-offset-correction.md)
- [running-and-testing.md](../docs/running-and-testing.md)

## Development

After changing a policy file, reinstall before testing — `pixi-build-ros`
copies files into the pixi env rather than symlinking them:
```bash
pixi reinstall ros-kilted-aic-solution-policy
```

If the reinstall itself fails, wipe the pixi build cache and reinstall:
```bash
cd ~/ws_aic/src/aic
rm -rf .pixi
pixi install
```
