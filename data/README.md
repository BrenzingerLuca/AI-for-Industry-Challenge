# Data Collection

This folder contains scripts to collect and label image datasets from the simulation,
as well as the collected datasets themselves.

The goal is to train a model that can detect SFP ports in camera images — specifically:
- **Which port is which** (class 0 = port 0, class 1 = port 1)
- **The orientation of the port** (needed to insert the cable the right way around)
- **The position of the port** via bounding box and keypoints (used to localize the correct port across all three cameras)

Labels are generated automatically by projecting the ground truth TF-Frames
into the camera image, so no manual annotation is needed.

---

## Scripts

| Script | Description |
|--------|-------------|
| `dataset_collector_multiple_cameras.py` | Subscribes to all 3 cameras simultaneously and saves images + YOLO labels on keypress |
| `dataset_collector_single_cam.py` | Same as above but for the center camera only — simpler reference implementation |
| `exploration/tf_point_projection.py` | Early prototype: projects a single TF frame origin onto the image as a point |
| `exploration/tf_keypoints_projection.py` | Extends the above: projects 4 port corners and prints the YOLO label string to the terminal |

---

## Label Format

Labels follow the **YOLO Pose** format:

```
<class> <cx> <cy> <bw> <bh> <kp1x> <kp1y> <v1> <kp2x> <kp2y> <v2> <kp3x> <kp3y> <v3> <kp4x> <kp4y> <v4>
```
| Field | Description |
|-------|-------------|
| `class` | Class ID: `0` = sfp_port_0, `1` = sfp_port_1 |
| `cx`, `cy` | Bounding box center, normalized to `[0, 1]` |
| `bw`, `bh` | Bounding box width and height, normalized to `[0, 1]` |
| `kp1x`, `kp1y` | Keypoint 1 (Top-Left corner), normalized to `[0, 1]` |
| `kp2x`, `kp2y` | Keypoint 2 (Top-Right corner), normalized to `[0, 1]` |
| `kp3x`, `kp3y` | Keypoint 3 (Bottom-Right corner), normalized to `[0, 1]` |
| `kp4x`, `kp4y` | Keypoint 4 (Bottom-Left corner), normalized to `[0, 1]` |
| `v1`–`v4` | Visibility flag: `2` = visible |

The keypoint order encodes orientation — corner 0 → corner 1 is always the top edge of the port,
which tells the model which way is up for cable insertion.

---

## How to Label

You need **3 terminals** running in parallel.

### Terminal 1 — Start the Simulation

```bash
distrobox enter -r aic_eval
```

Then launch the environment. Example to spawn NIC card 0 without cable:

```bash
/entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=false \
  spawn_task_board:=true \
  nic_card_mount_0_present:=true
```

For more scene configuration options see [scene_description.md](https://github.com/intrinsic-dev/aic/blob/main/docs/scene_description.md).

> **Important:** Make sure the NIC card you spawn matches the `TARGET_FRAMES` set in the collector script.

### Terminal 2 — Run the Collector

Before running, open the script and edit the `CONFIG` section at the top:
- Set `OUTPUT_PATH` to your desired save location
- Set `TARGET_FRAMES` to match the NIC card mounts you spawned

```bash
cd ~/ws_aic/src/aic/aic_solution/data
pixi shell

# Label with all 3 cameras:
python3 dataset_collector_multiple_cameras.py

# Or label with center camera only:
python3 dataset_collector_single_cam.py
```

In **RViz**, select one of the `/left_camera/debug_image`, `/center_camera/debug_image`,
or `/right_camera/debug_image` topics under Image to verify the projected bounding boxes
look correct before saving.

### Terminal 3 — Move the Robot

```bash
cd ~/ws_aic/src/aic/aic_utils/aic_teleoperation/aic_teleoperation
pixi shell
python3 cartesian_keyboard_teleop.py
```

Use the keyboard to move the robot to different viewpoints. Once the bounding boxes
in RViz look good, switch to Terminal 2 and **press ENTER** to save the current frame
from all cameras simultaneously.

Repeat — move, check, save.
