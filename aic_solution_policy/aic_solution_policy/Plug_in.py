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

    def get_force_feedback(self, time_window=0.1):
        """Returns the force and torque delta over last 0.1s as a tuple: (force: (x, y, z), torque: (x, y, z)). 
        If no wrench message has been received yet, returns ((0, 0, 0), (0, 0, 0))."""
        if self._last_wrench is not None:
            current_forces = (
                (self._last_wrench.wrench.force.x,
                self._last_wrench.wrench.force.y, 
                self._last_wrench.wrench.force.z),
                (self._last_wrench.wrench.torque.x,
                self._last_wrench.wrench.torque.y, 
                self._last_wrench.wrench.torque.z)
            )
            
            # Store forces with timestamp for comparison over the configured window.
            stamp = self._last_wrench.header.stamp
            current_time_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            if not hasattr(self, '_force_history'):
                self._force_history = []
            
            self._force_history.append((current_time_s, current_forces))
            
            # Keep only forces from the last configured time window.
            cutoff_time_s = current_time_s - time_window
            self._force_history = [(t, f) for t, f in self._force_history if t >= cutoff_time_s]
            
            if len(self._force_history) > 1:
                earlier_forces = self._force_history[0][1]
                delta_forces = (
                    (current_forces[0][0] - earlier_forces[0][0],
                    current_forces[0][1] - earlier_forces[0][1],
                    current_forces[0][2] - earlier_forces[0][2]),
                    (current_forces[1][0] - earlier_forces[1][0],
                    current_forces[1][1] - earlier_forces[1][1],
                    current_forces[1][2] - earlier_forces[1][2])
                )
            
                # Calculate the gradient from current forces to the earlier_forces
                forces_gradient = (
                    (delta_forces[0][0] / time_window, delta_forces[0][1] / time_window, delta_forces[0][2] / time_window),
                    (delta_forces[1][0] / time_window, delta_forces[1][1] / time_window, delta_forces[1][2] / time_window)
                )
                return delta_forces, forces_gradient
        
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    
    def set_my_target_pose(self, move_robot: MoveRobotCallback,
                            pose: Pose,
                            offset_x= 0.0,
                            offset_y= 0.0,
                            offset_z= 0.0,
                            frame_id="base_link"):
        """Set the target pose for the robot, with an optional offset."""

        # Correct Target Pose with TCP Offset x=-0.0016, y=0.0010, z=-0.1073 (Not sure from where the offset is coming
        pose.position.x -= 0.0016 + offset_x
        pose.position.y += 0.0010 + offset_y
        pose.position.z -= 0.1073 + offset_z

        self.set_pose_target(move_robot=move_robot, pose=pose, frame_id=frame_id)
        
    def allign_connector(self, target_pos_in_base_link, target_rot_in_base_link, move_robot):
        """Bewegt den TCP so, dass die Spitze auf der Zielpose sitzt.

        Args:
            target_pos_in_base_link: Gewuenschte Position der Spitze in base_link.
            target_rot_in_base_link: Gewuenschte Rotation der Spitze in base_link.
        """
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
            target_pos_in_base_link,
            target_rot_in_base_link,
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

        entrance_pos = target_pos_in_base_link

        # Error between desired entrance pose and actual tip pose in base_link.
        err_x = entrance_pos.x - tip_pos.x
        err_y = entrance_pos.y - tip_pos.y
        err_z = entrance_pos.z - tip_pos.z
        self._last_constant_error_base = (err_x, err_y, err_z)

        # Command the TCP pose so that the tip sits exactly on the entrance pose.
        self.set_my_target_pose(move_robot=move_robot,
                            pose=target_pose_in_base,
                            offset_x=0.0,
                            offset_y=0.0,
                            offset_z=0.0,
                            frame_id="base_link")

        # If the error is small enough, we can consider the cable to be successfully aligned with the entrance.
        if abs(err_x) < 0.005 and abs(err_y) < 0.005 and abs(err_z) < 0.005:
            self.get_logger().info("Cable tip is well aligned with entrance (error < 5mm).")
            return True
        else:
            self.get_logger().warn(
                f"Cable tip is not well aligned with entrance: error_x={err_x:.4f}, error_y={err_y:.4f}, error_z={err_z:.4f}"
            )
            return False

    def pluig_in(self, target_pos_in_base_link, target_rot_in_base_link, move_robot):
        """Insert the cable by moving the TCP so that the tip sits on the target pose.

        Args:
            target_pos_in_base_link: Wanted position of the tip in base_link.
            target_rot_in_base_link: Wanted rotation of the tip in base_link.
        """
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
            target_pos_in_base_link,
            target_rot_in_base_link,
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
        err_x = target_pos_in_base_link.x - tip_pos.x
        err_y = target_pos_in_base_link.y - tip_pos.y
        err_z = target_pos_in_base_link.z - tip_pos.z
        self._last_constant_error_base = (err_x, err_y, err_z)

        # Command the TCP pose so that the tip sits exactly on the entrance pose.
        self.set_my_target_pose(move_robot=move_robot,
                            pose=target_pose_in_base,
                            offset_x=0.0,
                            offset_y=0.0,
                            offset_z=0.0,
                            frame_id="base_link")

        # If the error is small enough, we can consider the cable to be successfully aligned with the entrance.
        if abs(err_x) < 0.005 and abs(err_y) < 0.005 and abs(err_z) < 0.005:
            return True
        else:
            return False

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

                '''
                Workflow:
                1) KeyPoint Prediction
                2) Triangulation -> Returns TF in base_link
                3) Call plug_in(target_pose_in_base_link)
                '''

                # Print force delta and gradient for debugging/monitoring.
                delta_forces, forces_gradient = self.get_force_feedback(time_window=0.5)
                self.get_logger().info(
                    f"Force delta over last 0.5s: "
                    f"force: x={delta_forces[0][0]:.3f}, y={delta_forces[0][1]:.3f}, z={delta_forces[0][2]:.3f} | "
                    f"torque: x={delta_forces[1][0]:.3f}, y={delta_forces[1][1]:.3f}, z={delta_forces[1][2]:.3f}"
                )
                self.get_logger().info(
                    f"Force gradient over last 0.5s: "
                    f"force: x={forces_gradient[0][0]:.3f}, y={forces_gradient[0][1]:.3f}, z={forces_gradient[0][2]:.3f} | "
                    f"torque: x={forces_gradient[1][0]:.3f}, y={forces_gradient[1][1]:.3f}, z={forces_gradient[1][2]:.3f}"
                )

                # Entrance-Pose in base_link fuer Sollausrichtung laden.
                entrance_pos, entrance_rot = lookup_pose_in_base(
                    self._parent_node._tf_buffer,
                    "task_board/nic_card_mount_0/sfp_port_0_link_entrance",
                )

                aligned = self.allign_connector(entrance_pos, entrance_rot, move_robot)


                # Calculatet 7-cm z to entrance 
                target_pos_in_base_link = Point(
                    x=entrance_pos.x,
                    y=entrance_pos.y,
                    z=entrance_pos.z - 0.07
                )
                target_rot_in_base_link = entrance_rot

                if aligned:
                    self.get_logger().info("Cable tip is aligned with entrance, proceeding to insert.")
                    send_feedback("Cable tip aligned, inserting cable...")
                    inserted = self.pluig_in(target_pos_in_base_link, target_rot_in_base_link, move_robot)
                    if inserted:
                        self.get_logger().info("Cable successfully inserted.")
                        send_feedback("Cable successfully inserted.")
                        break
                    else:
                        self.get_logger().warn("Cable insertion failed, retrying alignment.")
                        send_feedback("Cable insertion failed, retrying alignment...")
                else:
                    self.get_logger().warn("Cable tip is not aligned with entrance, retrying.")
                    send_feedback("Cable tip not aligned, retrying...")

                

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
