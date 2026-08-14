# Running & Testing the Policies

Assumes the pixi/ROS environment from
[environment-setup.md](environment-setup.md) is already set up.

State published (relevant TF frames / topics):
- Force/torque wrench: `/fts_broadcaster/wrench`
- Port TF: `task_board/nic_card_mount_0/sfp_port_0_link_entrance`
- Plug tip TF: `cable_0/sfp_tip_link`
- TCP TF: `gripper/tcp`

Target poses for `move_robot` are expressed in `base_link` or the TCP frame.

## Qualification policy (`QualificationPlugIn`)

1. Start the simulation:
```bash
/entrypoint.sh ground_truth:=true start_aic_engine:=false spawn_task_board:=true \
  nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.08 \
  spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true

# or for the SC port:
/entrypoint.sh spawn_task_board:=true sc_port_0_present:=true sc_mount_rail_0_present:=true \
  spawn_cable:=true cable_type:=sfp_sc_cable_reversed attach_cable_to_gripper:=true \
  ground_truth:=true start_aic_engine:=false
```

2. Start the policy node:
```bash
cd ~/ws_aic/src/aic
pixi shell
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true \
  -p policy:=aic_solution_policy.ros.qualification.QualificationPlugIn
```

3. Activate and send a task:
```bash
cd ~/ws_aic/src/aic
pixi shell
ros2 lifecycle set /aic_model configure
ros2 lifecycle set /aic_model activate

ros2 action send_goal /insert_cable aic_task_interfaces/action/InsertCable \
    "{task: {
        id: 'cable_1',
        cable_type: 'sfp',
        cable_name: 'sfp_cable',
        plug_type: 'sfp',
        plug_name: 'sfp_plug',
        port_type: 'sfp',
        port_name: 'sfp_port_0',
        target_module_name: 'nic_card_0',
        time_limit: 60
            }
    }"
```

For end-to-end evaluation instead of a single manual goal, run the policy
node as above, then in a second terminal:
```bash
distrobox enter -r aic_eval
/entrypoint.sh ground_truth:=true start_aic_engine:=true
```

## Phase-1 policy (`Phase1PlugIn`)

Perception-free: assumes the robot is already positioned above the target
port (plug grasped and aligned) — normally done by FlowState, but for local
testing without FlowState, [`position_over_port_debug.py`](../../aic_bringup/scripts/position_over_port_debug.py)
fills in for it.

1. Start the simulation with ground truth (needed for the debug positioning script):
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  ground_truth:=true \
  spawn_task_board:=true \
  spawn_cable:=true \
  nic_card_mount_0_present:=true
```
Leave `start_aic_engine` at its default (`false`) — otherwise the engine
takes over lifecycle transitions/goals and can kill the model node on a rule
violation while you're debugging.

2. Start the policy node (after any code change, reinstall first — no
   symlink install, `pixi-build-ros` copies the file):
```bash
pixi reinstall ros-kilted-aic-solution-policy
ros2 run aic_model aic_model --ros-args -p use_sim_time:=true \
  -p policy:=aic_solution_policy.ros.phase1.Phase1PlugIn
```

3. Activate the lifecycle:
```bash
ros2 lifecycle set /aic_model configure
ros2 lifecycle set /aic_model activate
ros2 lifecycle get /aic_model
```

4. Position the robot above the port (standoff must match
   `insertion_offset_z` in `phase1/config.py`, currently 5cm for SFP):
```bash
python3 aic_bringup/scripts/position_over_port_debug.py --ros-args -p standoff_z:=0.05
```

5. Send the goal (task fields besides `id` are ignored by this policy, but a
   valid `Task` message is still required):
```bash
ros2 action send_goal /insert_cable aic_task_interfaces/action/InsertCable \
  "{task: {id: 'test_1', cable_type: 'sfp_sc', cable_name: 'cable_0', plug_type: 'sfp', plug_name: 'sfp_tip', port_type: 'sfp', port_name: 'sfp_port_1', target_module_name: 'nic_card_mount_0', time_limit: 180}}" \
  --feedback
```

Progress/result are visible live in the policy node's log.

### `position_over_port_debug.py` parameters

All parameters via `--ros-args -p name:=value`, combinable.

| Parameter | Default | Meaning |
|---|---|---|
| `standoff_z` | `0.03` | Distance from plug tip to port entrance in Z (base_link/world) — must match `insertion_offset_z` in the policy config |
| `port_entrance_frame` | `task_board/nic_card_mount_0/sfp_port_1_link_entrance` | TF frame of the target port |
| `cable_tip_frame` | `cable_0/sfp_tip_link` | TF frame of the plug tip |
| `tcp_frame` | `gripper/tcp` | TF frame of the TCP |
| `base_frame` | `base_link` | Reference frame for all calculations/commands |
| `n_steps` | `100` | Interpolation steps for the move |
| `controller_namespace` | `aic_controller` | Namespace for `pose_commands`/`change_target_mode` |
| `simple_mode` | `false` | Debug: translation only, no orientation/tip-offset math, keeps current TCP orientation (isolates position testing) |
| `orientation_correction_rpy_deg` | `[0,0,0]` | Extra rotation (degrees, xyz) applied to the port orientation, in case its Z axis isn't pointing where expected |
| `xy_offset_min_m` / `xy_offset_max_m` | `0.0` / `0.0` | Simulates perception inaccuracy: a random XY offset sampled from `[min,max]` (independently per axis) is added to the target pose and logged |
| `pose_backup_file` | `/tmp/position_over_port_debug_last_pose.json` | Where the pose is backed up before each move |
| `reset` | `false` | Move back to the last backed-up pose instead of recomputing (undo) |

### Known pitfalls

- **No symlink install**: code changes only take effect after
  `pixi reinstall ros-kilted-aic-solution-policy`.
- **The F/T sensor is untared**: `observation.wrist_wrench` comes raw from
  `/fts_broadcaster/wrench` (see `aic_adapter.cpp`), not the controller's
  internal tare. `Phase1PlugIn` tares itself (best-effort service call +
  a 20-sample software baseline at task start).
- **`task.time_limit`** isn't enforced by `aic_model` itself (only
  `aic_engine` uses it as its own abort timeout) — a manual
  `ros2 action send_goal` is only bounded by the policy's own internal step
  budgets (descent/press/spiral step counts in `phase1/config.py`).
- **`standoff_z`** in the debug script must match `insertion_offset_z` in
  the policy config, or the assumed target pose will be far from the real port.

## Data acquisition (offset-correction training data)

`data_acquisition.py` implements a `data_acquisition` policy (lowercase,
matching the filename — `aic_model`'s dynamic loader looks up a class named
after the last component of the `policy` parameter). For each configured
port it aligns the cable tip to the port using ground-truth TF, then
repeatedly perturbs that pose and records the three camera images plus the
actual tip/TCP/port poses to an HDF5 file — one file per run, one group per
port. See [residual-offset-correction.md](residual-offset-correction.md) for
what this data trains.

Requires `ground_truth:=true`. All behavior is controlled via
`data_acquisition.*` ROS parameters (see `__init__` in `data_acquisition.py`
for defaults) — the `Task` sent to the action only triggers the run, its
fields are otherwise ignored.

1. Start the simulation (all 5 NIC card mounts + sfp/sc cable attached):
```bash
/entrypoint.sh   ground_truth:=true   start_aic_engine:=false   spawn_task_board:=true   nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.08   nic_card_mount_1_present:=true nic_card_mount_1_translation:=-0.04   nic_card_mount_2_present:=true nic_card_mount_2_translation:=0.0   nic_card_mount_3_present:=true nic_card_mount_3_translation:=0.04   nic_card_mount_4_present:=true nic_card_mount_4_translation:=0.08   spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true
```

2. Start the policy node:
```bash
cd ~/ws_aic/src/aic
pixi shell
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_solution_policy.data_acquisition \
    -p data_acquisition.num_samples_per_port:=100 \
    -p 'data_acquisition.ports:=[nic_card_mount_0:sfp_port_0,nic_card_mount_0:sfp_port_1,nic_card_mount_1:sfp_port_0,nic_card_mount_1:sfp_port_1]'
```

`data_acquisition.output_dir` defaults to `src/aic/aic_solution/dataset/hdf5`
(shared with `residual_policy.ipynb`'s `CFG["data_glob"]`/`ckpt_dir`/`log_dir`
and both policies' `residual_model_path` entries) — override with
`-p data_acquisition.output_dir:=...` for somewhere else.

`data_acquisition.ports` is a list of `target_module_name:port_name` pairs
(default: both SFP ports on all 5 NIC card mounts) — each card has two SFP
ports, so pass whichever combinations you want scanned. A pair whose TF frame
doesn't exist is skipped with a warning, not aborted.

**SC ports**: the `aic_eval` image actually running in this env
(`ghcr.io/intrinsic-dev/aic/aic_eval:latest`, baked at build time — not the
same as the `aic_description` source in this repo, which has since been
extended to 5 slots) only wires up **2** SC ports: `sc_port_0` (rail 0,
Y=0.0295) and `sc_port_1` (rail 1, Y=0.0705), one per rail. `sc_port_2`/`3`/`4`
are no-ops in this build. Each present port's TF name is the fixed
`sc_port_base` (not an index), and `data_acquisition.cable_type` must be set
to `sc` (not `sfp`) so the ground-truth `sc_tip_link` frame is used. This is a
*different* system from `sc_mount_rail_0`/`1` — don't confuse the two.

```bash
/entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=false \
  spawn_task_board:=true \
  sc_port_0_present:=true sc_port_0_translation:=0.0 \
  sc_port_1_present:=true sc_port_1_translation:=0.0 \
  spawn_cable:=true cable_type:=sfp_sc_cable_reversed attach_cable_to_gripper:=true
```

```bash
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_solution_policy.data_acquisition \
    -p data_acquisition.cable_type:=sc \
    -p data_acquisition.num_samples_per_port:=200 \
    -p 'data_acquisition.ports:=[sc_port_0:sc_port_base,sc_port_1:sc_port_base]'
```

3. Configure/activate and trigger the run (Task fields are ignored, any
   valid Task will do):
```bash
cd ~/ws_aic/src/aic
pixi shell
ros2 lifecycle set /aic_model configure
ros2 lifecycle set /aic_model activate

ros2 action send_goal /insert_cable aic_task_interfaces/action/InsertCable \
"{task: {id: 'data_acq_1', cable_type: 'sfp', port_type: 'sfp', time_limit: 3600}}"
```

Output: `<output_dir>/sfp_dataset_<timestamp>.hdf5`, with one group per
`<target_module_name>_<port_name>` containing `images/{left,center,right}`,
`tip_pose`, `tcp_pose`, `port_pose` (all `[x,y,z,qx,qy,qz,qw]`), `offset`
(`[dx,dy,dz,droll,dpitch,dyaw]`) and `timestamp` datasets.

Requires the `h5py` conda dependency in the workspace `pixi.toml` — run
`pixi install` (or your usual lock/update command) after pulling this change.
