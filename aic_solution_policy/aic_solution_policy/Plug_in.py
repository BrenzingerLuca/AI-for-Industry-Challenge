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
from aic_solution_policy.VectorHelpers import (
    compute_tcp_target_pose,
    lookup_pose_in_base,
)
from aic_solution_policy.RVizHelpers import rviz_vector
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, WrenchStamped
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time


class Plug_in(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)

        # Laufzeitstatus fuer Monitoring/Debug-Ausgaben.
        self.get_logger().info("Plug_in.__init__(): subscribe to /fts_broadcaster/wrench")
        self._last_wrench = None
        self._num_wrench_msgs = 0
        self._last_constant_error_base = None

        # Zuverlaessige QoS fuer Kraftsensor-Nachrichten.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )

        # Subscriber fuer aktuelle Kraft-/Momentdaten am Tool.
        self._wrench_sub = self._parent_node.create_subscription(
            WrenchStamped,
            "/fts_broadcaster/wrench",
            self._on_wrench,
            qos,
        )

        # Periodische Diagnoseausgabe fuer Wrench und Positionsfehler.
        #self._display_timer = self._parent_node.create_timer(0.5, self._display_tick)

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
                # Zyklisch arbeiten, um TF und Sensorik ruhig nachzufuehren.
                self.sleep_for(0.25)

                # Entrance-Pose in base_link fuer Sollausrichtung laden.
                entrance_pos, entrance_rot = lookup_pose_in_base(
                    self._parent_node._tf_buffer,
                    "task_board/nic_card_mount_0/sfp_port_0_link_entrance",
                )

                tip_pos, tip_rot = lookup_pose_in_base(
                    self._parent_node._tf_buffer,
                    "cable_0/sfp_tip_link",
                )

                tip_from_tcp = self._parent_node._tf_buffer.lookup_transform(
                    "cable_0/sfp_tip_link",
                    "gripper/tcp",
                    Time(),
                )

                target_pose_in_base = compute_tcp_target_pose(
                    entrance_pos,
                    entrance_rot,
                    tip_from_tcp,
                )

                rviz_vector(self._parent_node, target_pose_in_base, color="green")
                rviz_vector(
                    self._parent_node,
                    Pose(
                        position=Point(x=tip_pos.x, y=tip_pos.y, z=tip_pos.z),
                        orientation=Quaternion(
                            x=tip_rot.x,
                            y=tip_rot.y,
                            z=tip_rot.z,
                            w=tip_rot.w,
                        ),
                    ),
                    color="cyan",
                )

                tcp_pos, tcp_rot = lookup_pose_in_base(
                    self._parent_node._tf_buffer,
                    "gripper/tcp",
                )

                tcp_in_base = Pose(
                    position=Point(
                        x=tcp_pos.x,
                        y=tcp_pos.y,
                        z=tcp_pos.z,
                    ),
                    orientation=Quaternion(
                        x=tcp_rot.x,
                        y=tcp_rot.y,
                        z=tcp_rot.z,
                        w=tcp_rot.w,
                    ),
                )

                # Error between desired entrance pose and actual tip pose in base_link.
                err_x = entrance_pos.x - tip_pos.x
                err_y = entrance_pos.y - tip_pos.y
                err_z = entrance_pos.z - tip_pos.z
                self._last_constant_error_base = (err_x, err_y, err_z)

                # Correct Target Pose with TCP Offset x=-0.0016, y=0.0010, z=-0.1073 (Not sure from where the offset is coming
                target_pose_in_base.position.x -= 0.0016
                target_pose_in_base.position.y += 0.0010
                target_pose_in_base.position.z -= 0.1073

                # Command the TCP pose so that the tip sits exactly on the entrance pose.
                self.set_pose_target(move_robot=move_robot, pose=target_pose_in_base, frame_id="base_link")

                self.get_logger().info(
                    "entrance_pose(base_link) | "
                    f"x={entrance_pos.x:.4f}, y={entrance_pos.y:.4f}, z={entrance_pos.z:.4f} | "
                    "tip error(base_link) | "
                    f"x={err_x:.4f}, y={err_y:.4f}, z={err_z:.4f} | "
                    "target_tcp_pose(base_link) | "
                    f"x={target_pose_in_base.position.x:.4f}, y={target_pose_in_base.position.y:.4f}, z={target_pose_in_base.position.z:.4f} | "
                    "tcp(base_link) | "
                    f"x={tcp_in_base.position.x:.4f}, y={tcp_in_base.position.y:.4f}, z={tcp_in_base.position.z:.4f} | "
                    "tip(base_link) | "
                    f"x={tip_pos.x:.4f}, y={tip_pos.y:.4f}, z={tip_pos.z:.4f}"
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
