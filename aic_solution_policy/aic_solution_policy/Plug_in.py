#
#  Copyright (C) 2026 Intrinsic Innovation LLC
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

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, WrenchStamped
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time


class Plug_in(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("Plug_in.__init__(): subscribe to /fts_broadcaster/wrench")
        self._last_wrench = None
        self._num_wrench_msgs = 0
        self._last_constant_error_base = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )
        self._wrench_sub = self._parent_node.create_subscription(
            WrenchStamped,
            "/fts_broadcaster/wrench",
            self._on_wrench,
            qos,
        )
        self._display_timer = self._parent_node.create_timer(0.25, self._display_tick)

    def _on_wrench(self, msg: WrenchStamped):
        """Silently store the latest wrench message."""
        self._last_wrench = msg
        self._num_wrench_msgs += 1

    def _display_tick(self):
        """Display wrench and constant entrance-tip error at 4 Hz."""
        if self._last_wrench is not None:
            self.get_logger().info(
                "wrench | force: "
                f"x={self._last_wrench.wrench.force.x:.3f}, y={self._last_wrench.wrench.force.y:.3f}, z={self._last_wrench.wrench.force.z:.3f} "
                "| torque: "
                f"x={self._last_wrench.wrench.torque.x:.3f}, y={self._last_wrench.wrench.torque.y:.3f}, z={self._last_wrench.wrench.torque.z:.3f}"
            )
        elif self._num_wrench_msgs == 0:
            count = self._parent_node.count_publishers("/fts_broadcaster/wrench")
            self.get_logger().warn(
                f"No wrench messages received yet. Publishers on /fts_broadcaster/wrench: {count}"
            )

        if self._last_constant_error_base is not None:
            dx, dy, dz = self._last_constant_error_base
            self.get_logger().info(
                "constant_error(base_link) | "
                f"x={dx:.4f}, y={dy:.4f}, z={dz:.4f}"
            )

    def _lookup_base_pose(self, frame_id: str):
        transform = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            frame_id,
            Time(),
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return translation, rotation

    def _rotate_vector_by_quaternion(self, quaternion, vector):
        qx = quaternion.x
        qy = quaternion.y
        qz = quaternion.z
        qw = quaternion.w
        vx, vy, vz = vector

        ix = qw * vx + qy * vz - qz * vy
        iy = qw * vy + qz * vx - qx * vz
        iz = qw * vz + qx * vy - qy * vx
        iw = -qx * vx - qy * vy - qz * vz

        ox = ix * qw + iw * -qx + iy * -qz - iz * -qy
        oy = iy * qw + iw * -qy + iz * -qx - ix * -qz
        oz = iz * qw + iw * -qz + ix * -qy - iy * -qx
        return ox, oy, oz

    def _quaternion_multiply(self, q1, q2):
        return Quaternion(
            x=q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
            y=q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
            z=q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
            w=q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        del task

        self.get_logger().info("Plug_in.insert_cable() enter: aligning tip to entrance")
        send_feedback("Aligning cable tip to socket entrance")

        try:
            while True:
                self.sleep_for(0.25)
                entrance_pos, entrance_rot = self._lookup_base_pose(
                    "task_board/nic_card_mount_0/sfp_port_0_link_entrance"
                )
                tip_pos, _ = self._lookup_base_pose("cable_0/sfp_tip_link")
                tcp_in_tip = self._parent_node._tf_buffer.lookup_transform(
                    "cable_0/sfp_tip_link",
                    "gripper/tcp",
                    Time(),
                )

                # Constant error between entrance and tip in base_link.
                err_x = entrance_pos.x - tip_pos.x
                err_y = entrance_pos.y - tip_pos.y
                err_z = entrance_pos.z - tip_pos.z
                self._last_constant_error_base = (err_x, err_y, err_z)

                # Use the entrance orientation as the desired tip orientation.
                desired_tip_rotation = entrance_rot

                # lookup_transform("cable_0/sfp_tip_link", "gripper/tcp") returns T_tip_tcp.
                tip_from_tcp_t = tcp_in_tip.transform.translation
                tip_from_tcp_q = tcp_in_tip.transform.rotation

                offset_base_x, offset_base_y, offset_base_z = self._rotate_vector_by_quaternion(
                    desired_tip_rotation,
                    (tip_from_tcp_t.x, tip_from_tcp_t.y, tip_from_tcp_t.z),
                )
                target_rotation = self._quaternion_multiply(desired_tip_rotation, tip_from_tcp_q)

                target_pose = Pose(
                    position=Point(
                        x=entrance_pos.x + offset_base_x,
                        y=entrance_pos.y + offset_base_y,
                        z=entrance_pos.z + offset_base_z,
                    ),
                    orientation=Quaternion(
                        x=target_rotation.x,
                        y=target_rotation.y,
                        z=target_rotation.z,
                        w=target_rotation.w,
                    ),
                )

                # Command the TCP pose so that the tip sits exactly on the entrance pose.
                self.set_pose_target(move_robot=move_robot, pose=target_pose, frame_id="base_link")

                self.get_logger().info(
                    "entrance(base_link) | "
                    f"x={entrance_pos.x:.4f}, y={entrance_pos.y:.4f}, z={entrance_pos.z:.4f} | "
                    "tip(base_link) | "
                    f"x={tip_pos.x:.4f}, y={tip_pos.y:.4f}, z={tip_pos.z:.4f} | "
                    "tip_error(base_link) | "
                    f"x={err_x:.4f}, y={err_y:.4f}, z={err_z:.4f} | "
                    "tcp_target(base_link) | "
                    f"x={target_pose.position.x:.4f}, y={target_pose.position.y:.4f}, z={target_pose.position.z:.4f}"
                )

                observation = get_observation()
                if observation is None:
                    self.get_logger().info("No observation received.")
                    continue

        except Exception as exc:
            self.get_logger().error(f"Failed to get socket entrance transform: {exc}")
            send_feedback(f"Error: {exc}")
            return False

        self.get_logger().info("Plug_in.insert_cable() exiting...")
        return True
