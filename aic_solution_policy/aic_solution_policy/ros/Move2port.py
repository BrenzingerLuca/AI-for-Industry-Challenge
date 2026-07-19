#!/usr/bin/env python3

#
#  Copyright (C) 2025 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
Debug/test helper: positions the TCP so that the grasped plug tip sits a
fixed standoff above a given port entrance, pointing straight down (insertion
axis vertical), using ground-truth TF (no vision). Intended to bring the
robot into the starting state a policy like SfpPlugInPhase1 or
PlugIn_correct_offset assumes ("robot already above the port, plug grasped
and aligned") so that policy can be tested/debugged in isolation.

Supports both "sfp" and "sc" ports via the port_type parameter (default
"sfp") - see PORT_ENTRANCE_FRAMES/CABLE_TIP_LINKS below. port_frame and
cable_name are escape hatches to override the looked-up port TF frame /
cable spawn-name directly, for scenes that don't match the defaults (a
different port index, or a cable spawned under a different name).

Besides port_type/port_frame/cable_name, the other things meant to be tuned
per run are how far above the port to stop (standoff_z) and a horizontal
offset from the port entrance (offset_x/offset_y), e.g. to test the
policy's tolerance to a mis-positioned start. Everything else (impedance
gains, ...) is a fixed constant below.

Not part of the submitted policy - TF ground truth is only available in
simulation with the ground-truth plugin enabled.
"""

import json
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode, TargetMode
from aic_control_interfaces.srv import ChangeTargetMode
from geometry_msgs.msg import Pose, Point, Quaternion


# --- Fixed setup, not exposed as parameters ---------------------------------
BASE_FRAME = "base_link"
TCP_FRAME = "gripper/tcp"
CONTROLLER_NAMESPACE = "aic_controller"
POSE_BACKUP_FILE = "/tmp/position_over_port_debug_last_pose.json"

# Port-entrance TF frame per port_type. SC ports are spawned as their own
# top-level models (task_board/sc_port_<N>/...), not nested under
# nic_card_mount_0 like SFP is, so this isn't just a string substitution.
# sc_port_1 is the default here because that's what aic_engine's
# sample_config.yaml / eval_config.yaml SC trial actually spawns (sc_port_0
# is the other populated index; sc_port_2..4 are unused xacro args - see
# aic_engine.cpp / task_board.urdf.xacro). Override via the port_frame
# parameter if your scene spawns a different port index.
PORT_ENTRANCE_FRAMES = {
    "sfp": "task_board/nic_card_mount_0/sfp_port_1_link_entrance",
    "sc": "task_board/sc_port_1/sc_port_base_link_entrance",
}
# Cable-tip link name per port_type. The cable's spawn-name prefix
# (cable_0, cable_1, ...) is configured separately via the cable_name
# parameter, since it depends on how the cable was spawned, not on
# port_type: spawn_cable.launch.py always names it "cable_0" regardless of
# cable type, while aic_engine's sample/eval configs spawn the SC trial's
# cable as "cable_1".
CABLE_TIP_LINKS = {
    "sfp": "sfp_tip_link",
    "sc": "sc_tip_link",
}

# Cartesian impedance used for this move. Matches SfpPlugInPhase1's
# free-space descent_stiffness/damping (known to hold a grasped plug
# without drooping).
# Note: the controller also hard-clamps the resulting wrench via
# aic_controller's impedance.maximum_wrench (currently [10,10,10,10,10,10]
# N/Nm in aic_ros2_controllers.yaml) - raising stiffness here cannot exceed
# that ceiling, so a single fixed target pose accelerates gently regardless
# of how far away it is; there is no need to ramp the commanded pose
# ourselves (see docs/aic_controller.md - aic_controller does its own
# smoothing via the impedance spring, its internal waypoint interpolation
# is not actually active for MODE_POSITION targets).
STIFFNESS = [300.0, 300.0, 200.0, 200.0, 200.0, 200.0]
DAMPING = [40.0, 40.0, 35.0, 30.0, 30.0, 30.0]

# How long to keep streaming the (constant) target before giving up, and how
# close TCP position/orientation needs to get to call it "reached".
MOVE_TIMEOUT_S = 20.0
MOVE_PUBLISH_RATE_HZ = 20.0
POSITION_TOLERANCE_M = 0.003
ANGLE_TOLERANCE_DEG = 2.0


class PositionOverPortDebugNode(Node):
    def __init__(self):
        super().__init__("position_over_port_debug")

        # The only parameters meant to be tuned per run.
        self.offset_x = self.declare_parameter("offset_x", 0.0).value
        self.offset_y = self.declare_parameter("offset_y", 0.0).value
        self.standoff_z = self.declare_parameter("standoff_z", 0.03).value
        # Undo the last move (moves back to the pose backed up just before
        # it), instead of driving to the port.
        self.reset = self.declare_parameter("reset", False).value

        # Which port to move to: "sfp" or "sc" (see PORT_ENTRANCE_FRAMES /
        # CABLE_TIP_LINKS above). port_frame/cable_name let you override the
        # looked-up frame/cable-spawn-name directly for scenes that don't
        # match the defaults (e.g. a different port index, or a cable
        # spawned as "cable_1" via aic_engine's sample/eval configs).
        port_type = self.declare_parameter("port_type", "sfp").value.lower()
        if port_type not in PORT_ENTRANCE_FRAMES:
            raise ValueError(
                f"Unknown port_type '{port_type}' - expected one of {sorted(PORT_ENTRANCE_FRAMES)}"
            )
        self.port_type = port_type
        self.cable_name = self.declare_parameter("cable_name", "cable_0").value
        self.port_frame = self.declare_parameter("port_frame", "").value or PORT_ENTRANCE_FRAMES[port_type]
        self.cable_tip_frame = f"{self.cable_name}/{CABLE_TIP_LINKS[port_type]}"
        self.get_logger().info(
            f"port_type='{self.port_type}': port_frame='{self.port_frame}', "
            f"cable_tip_frame='{self.cable_tip_frame}'"
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(buffer=self._tf_buffer, node=self, spin_thread=True)

        self.motion_update_pub = self.create_publisher(
            MotionUpdate, f"/{CONTROLLER_NAMESPACE}/pose_commands", 10
        )
        while self.motion_update_pub.get_subscription_count() == 0:
            self.get_logger().info(
                f"Waiting for subscriber to '{CONTROLLER_NAMESPACE}/pose_commands'..."
            )
            time.sleep(1.0)

        self._change_target_mode_client = self.create_client(
            ChangeTargetMode, f"/{CONTROLLER_NAMESPACE}/change_target_mode"
        )
        while not self._change_target_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for change_target_mode service...")

    def set_cartesian_mode(self):
        req = ChangeTargetMode.Request()
        req.target_mode.mode = TargetMode.MODE_CARTESIAN
        response = self._change_target_mode_client.call(req)
        if not response.success:
            raise RuntimeError("Unable to set Cartesian target mode")
        self.get_logger().info("Set target mode to CARTESIAN")

    def _lookup(self, target_frame, source_frame, timeout_sec=15.0):
        deadline = time.time() + timeout_sec
        last_exc = None
        while time.time() < deadline:
            try:
                return self._tf_buffer.lookup_transform(target_frame, source_frame, Time())
            except Exception as e:
                last_exc = e
                time.sleep(0.2)
        raise RuntimeError(f"TF lookup '{target_frame}' <- '{source_frame}' timed out: {last_exc}")

    @staticmethod
    def _nadir_rotation_preserving_yaw(port_rot: R) -> R:
        """
        Build an orientation whose local Z axis points straight down (-Z in
        base_link/world) - the plug's insertion axis - regardless of which
        way the port_entrance TF frame's axes actually happen to point.
        The horizontal heading (yaw) of the port frame's X axis is kept, so
        the plug still roughly follows the port's rotation about the
        vertical axis (e.g. connector keying), instead of a fixed, guessed
        correction that only worked for one particular port orientation.
        """
        port_mat = port_rot.as_matrix()
        heading = port_mat[:, 0].copy()
        heading[2] = 0.0
        if np.linalg.norm(heading) < 1e-6:
            # Port X axis is (near) vertical - fall back to its Y axis.
            heading = port_mat[:, 1].copy()
            heading[2] = 0.0
        heading /= np.linalg.norm(heading)

        z_axis = np.array([0.0, 0.0, -1.0])
        y_axis = np.cross(z_axis, heading)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)

        return R.from_matrix(np.column_stack([x_axis, y_axis, z_axis]))

    def compute_target_tcp_pose(self):
        # Ground-truth port entrance pose in base_link.
        tf_port = self._lookup(BASE_FRAME, self.port_frame)
        port_pos = np.array([
            tf_port.transform.translation.x,
            tf_port.transform.translation.y,
            tf_port.transform.translation.z,
        ])
        port_quat = np.array([
            tf_port.transform.rotation.x,
            tf_port.transform.rotation.y,
            tf_port.transform.rotation.z,
            tf_port.transform.rotation.w,
        ])

        # Ground-truth cable tip -> TCP offset (same TF pair PlugIn's
        # _get_tcp_goal_pose uses on its ground-truth path).
        tf_tip_to_tcp = self._lookup(self.cable_tip_frame, TCP_FRAME)
        t_off = tf_tip_to_tcp.transform.translation
        q_off = tf_tip_to_tcp.transform.rotation

        # Desired tip pose: standoff_z above the port entrance (+z,
        # base_link/world), plus the configured horizontal offset.
        target_tip_pos = port_pos.copy()
        target_tip_pos[0] += self.offset_x
        target_tip_pos[1] += self.offset_y
        target_tip_pos[2] += self.standoff_z

        target_tip_rot = self._nadir_rotation_preserving_yaw(R.from_quat(port_quat))
        target_tip_quat = target_tip_rot.as_quat()

        mat_base_to_tip = np.eye(4)
        mat_base_to_tip[:3, :3] = target_tip_rot.as_matrix()
        mat_base_to_tip[:3, 3] = target_tip_pos

        mat_tip_to_tcp = np.eye(4)
        mat_tip_to_tcp[:3, :3] = R.from_quat([q_off.x, q_off.y, q_off.z, q_off.w]).as_matrix()
        mat_tip_to_tcp[:3, 3] = [t_off.x, t_off.y, t_off.z]

        target_matrix = mat_base_to_tip @ mat_tip_to_tcp
        target_tcp_pos = target_matrix[:3, 3]
        target_tcp_quat = R.from_matrix(target_matrix[:3, :3]).as_quat()

        self.get_logger().info(f"Port entrance ({BASE_FRAME}): pos={port_pos}, quat={port_quat}")
        self.get_logger().info(
            f"Target tip pose (+{self.standoff_z * 1000:.0f}mm z, "
            f"dx={self.offset_x * 1000:+.1f}mm dy={self.offset_y * 1000:+.1f}mm, pointing straight down): "
            f"pos={target_tip_pos}, quat={target_tip_quat}"
        )
        self.get_logger().info(f"Target TCP pose: pos={target_tcp_pos}, quat={target_tcp_quat}")
        return target_tcp_pos, target_tcp_quat

    def backup_current_pose(self):
        pos, quat = self.get_current_tcp_pose()
        with open(POSE_BACKUP_FILE, "w") as f:
            json.dump({"pos": pos.tolist(), "quat": quat.tolist()}, f)
        self.get_logger().info(
            f"Backed up current TCP pose to {POSE_BACKUP_FILE} "
            f"(pos={pos}, quat={quat}) - rerun with -p reset:=true to return here."
        )

    def load_backed_up_pose(self):
        if not os.path.exists(POSE_BACKUP_FILE):
            raise RuntimeError(
                f"No backup pose found at {POSE_BACKUP_FILE} - run once without "
                f"reset:=true first so there is something to reset to."
            )
        with open(POSE_BACKUP_FILE) as f:
            data = json.load(f)
        return np.array(data["pos"]), np.array(data["quat"])

    def get_current_tcp_pose(self):
        tf_tcp = self._lookup(BASE_FRAME, TCP_FRAME)
        pos = np.array([
            tf_tcp.transform.translation.x,
            tf_tcp.transform.translation.y,
            tf_tcp.transform.translation.z,
        ])
        quat = np.array([
            tf_tcp.transform.rotation.x,
            tf_tcp.transform.rotation.y,
            tf_tcp.transform.rotation.z,
            tf_tcp.transform.rotation.w,
        ])
        return pos, quat

    def _build_motion_update(self, pos, quat):
        msg = MotionUpdate()
        msg.header.frame_id = BASE_FRAME
        msg.pose = Pose(
            position=Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
            orientation=Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])),
        )
        mat_stiff = [0.0] * 36
        mat_damp = [0.0] * 36
        for j in range(6):
            mat_stiff[j * 6 + j] = float(STIFFNESS[j])
            mat_damp[j * 6 + j] = float(DAMPING[j])
        msg.target_stiffness = mat_stiff
        msg.target_damping = mat_damp
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION
        return msg

    def move_to_pose(self, target_pos, target_quat):
        """
        Streams a single, constant target pose to the controller (as
        documented in docs/aic_controller.md) until the measured TCP pose
        (via TF) reaches it, or MOVE_TIMEOUT_S elapses. aic_controller's
        impedance control does its own accel/decel smoothing - the target
        does not need to be ramped from this side.
        """
        start_pos, _ = self.get_current_tcp_pose()
        self.get_logger().info(f"Moving TCP from {start_pos} to {target_pos}...")

        target_rot = R.from_quat(target_quat)
        period = 1.0 / MOVE_PUBLISH_RATE_HZ
        deadline = time.time() + MOVE_TIMEOUT_S
        reached = False

        while time.time() < deadline:
            msg = self._build_motion_update(target_pos, target_quat)
            msg.header.stamp = self.get_clock().now().to_msg()
            self.motion_update_pub.publish(msg)

            curr_pos, curr_quat = self.get_current_tcp_pose()
            pos_err = float(np.linalg.norm(curr_pos - target_pos))
            ang_err_deg = float(np.degrees((R.from_quat(curr_quat).inv() * target_rot).magnitude()))

            if pos_err < POSITION_TOLERANCE_M and ang_err_deg < ANGLE_TOLERANCE_DEG:
                reached = True
                break

            time.sleep(period)

        if reached:
            self.get_logger().info(
                f"Reached target pose (pos_err={pos_err * 1000:.1f}mm, ang_err={ang_err_deg:.1f}deg)."
            )
        else:
            self.get_logger().warning(
                f"Timed out after {MOVE_TIMEOUT_S:.0f}s without reaching target "
                f"(pos_err={pos_err * 1000:.1f}mm, ang_err={ang_err_deg:.1f}deg). "
                f"If this persists, check impedance stiffness / gravity compensation / "
                f"maximum_wrench clamp in aic_ros2_controllers.yaml, not this script's pose math."
            )

        # Hold the final pose briefly so the controller settles.
        msg = self._build_motion_update(target_pos, target_quat)
        for _ in range(10):
            msg.header.stamp = self.get_clock().now().to_msg()
            self.motion_update_pub.publish(msg)
            time.sleep(0.1)


def main(args=None):
    try:
        with rclpy.init(args=args):
            node = PositionOverPortDebugNode()
            node.set_cartesian_mode()

            if node.reset:
                target_pos, target_quat = node.load_backed_up_pose()
                node.get_logger().info(f"Resetting to backed-up pose: pos={target_pos}, quat={target_quat}")
            else:
                node.backup_current_pose()
                target_pos, target_quat = node.compute_target_tcp_pose()

            node.move_to_pose(target_pos, target_quat)
            rclpy.shutdown()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main(sys.argv)