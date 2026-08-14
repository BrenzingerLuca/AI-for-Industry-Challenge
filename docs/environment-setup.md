# Environment Setup

General pixi/distrobox setup for this workspace. See
[running-and-testing.md](running-and-testing.md) for how to actually launch
the qualification or phase-1 policy end to end.

## Setup the aic_solution_policy package

1. Add the following line to `/ws_aic/src/aic/pixi.toml` under `[dependencies]`:
```bash
ros-kilted-aic-solution-policy = { path = "aic_solution/aic_solution_policy" }
```

2. Install:
```bash
cd ~/ws_aic/src/aic
pixi reinstall ros-kilted-aic-solution-policy
```

3. Smoke-test with the example WaveArm policy in a second terminal:
```bash
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_solution_policy.WaveArm
```

### Bugfixes

If the reinstall fails, delete `/ws_aic/src/aic/.pixi` and reinstall everything:
```bash
cd ~/ws_aic/src/aic
rm -rf .pixi
pixi install
```

## Helpful commands

### Start the evaluation container (from the official getting-started guide)
```bash
# Tell distrobox to use Docker as the container manager
export DBX_CONTAINER_MANAGER=docker

# Create and enter the eval container
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
# If you do *not* have an NVIDIA GPU, remove the --nvidia flag
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
distrobox enter -r aic_eval

# Inside the container, start the environment
/entrypoint.sh ground_truth:=false start_aic_engine:=true
```

### Enter the evaluation container
```bash
distrobox enter -r aic_eval
```

### Start the environment

Basic setup (no ground truth, with engine):
```bash
/entrypoint.sh ground_truth:=false start_aic_engine:=true
```

Full scene with all NIC card mounts and cables:
```bash
/entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=false \
  spawn_task_board:=true \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.08 \
  nic_card_mount_1_present:=true nic_card_mount_1_translation:=-0.04 \
  nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.0 \
  nic_card_mount_3_present:=true nic_card_mount_3_translation:=0.04 \
  nic_card_mount_4_present:=true nic_card_mount_4_translation:=0.08 \
  spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true
```

For more scene configuration options see
[scene_description.md](https://github.com/intrinsic-dev/aic/blob/main/docs/scene_description.md).

### Control the robot with the keyboard

```bash
cd ~/ws_aic/src/aic/aic_utils/aic_teleoperation/aic_teleoperation
pixi shell
python3 <script_name>.py
```

### Run a policy

```bash
cd ~/ws_aic/src/aic
pixi shell
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm
```

Replace `aic_example_policies.ros.WaveArm` with your own policy's dotted
module path, e.g. `aic_solution_policy.ros.qualification.QualificationPlugIn`
or `aic_solution_policy.ros.phase1.Phase1PlugIn` (see
[running-and-testing.md](running-and-testing.md)).

### General note on ROS commands

Always enter the pixi environment before running any ROS commands:
```bash
pixi shell
```
