# Plug_In Documentation

## Status:

- Force and Torque Wrench: /fts_broadcaster/wrench
- TF Frames for Port: task_board/nic_card_mount_0/sfp_port_0_link_entrance
- TF Frames for Plug Tip: cable_0/sfp_tip_link 
- TF Frames for TCP: gripper/tcp

To send Target Pose we need it from base_link or TCP

Bugs to fix next:
- Orientation alignment works but position is off
- Think again how to transorm from tcp to plug tip -> base link and back to entrence...


## Getting startet

1. Terminal with (start simulation):

```bash
/entrypoint.sh   ground_truth:=true   start_aic_engine:=false   spawn_task_board:=true   nic_card_mount_0_present:=true nic_card_mount_0_translation:=-0.08 spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true
```

2. Terminal with (plugin package):
```bash
cd ~/ws_aic/src/aic
pixi shell
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_solution_policy.Plug_in
```

3. Terminal with (plugin package):
```bash
cd ~/ws_aic/src/aic
pixi shell
ros2 lifecycle get /aic_model
ros2 lifecycle set /aic_model configure
ros2 lifecycle set /aic_model activate

ros2 action send_goal /insert_cable aic_task_interfaces/action/InsertCable "{task: {id: 'cable_1', cable_type: 'sfp', cable_name: 'sfp_cable', plug_type: 'sfp', plug_name: 'sfp_plug', port_type: 'sfp', port_name: 'sfp_port_0', target_module_name: 'nic_card_0', time_limit: 60}}"
```

## Developement Guide
To debug new package versionens 2. Terminal:
```bash
exit
pixi reinstall ros-kilted-aic-solution-policy
```
than again all steps from Terminal 3.
