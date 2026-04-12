# AIC Solution

Solution repository for the [AI for Industry Challenge](https://github.com/intrinsic-dev/aic) by Intrinsic.

> **Note:** This repo contains only our solution code. For the full toolkit and setup instructions, follow the [official getting started guide](https://github.com/intrinsic-dev/aic/blob/main/docs/getting_started.md) first.

---

## Helpful Commands

### Enter the Evaluation Container

```bash
distrobox enter -r aic_eval
```

### Start the Environment

Basic setup (no ground truth, with engine):
```bash
/entrypoint.sh ground_truth:=false start_aic_engine:=true
```

Full scene with NIC card mounts and cables:
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

For more scene configuration options see [scene_description.md](https://github.com/intrinsic-dev/aic/blob/main/docs/scene_description.md).

---

### Control the Robot with Keyboard

Navigate to the teleoperation scripts inside the pixi environment:

```bash
cd ~/ws_aic/src/aic/aic_utils/aic_teleoperation/aic_teleoperation
pixi shell
python3 <script_name>.py
```

---

### Run a Policy

```bash
cd ~/ws_aic/src/aic
pixi shell
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm
```

Replace `aic_example_policies.ros.WaveArm` with your own policy class.

---

### General Note on ROS Commands

Always enter the pixi environment before running any ROS commands:

```bash
pixi shell
```