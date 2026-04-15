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

from pathlib import Path

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
from aic_solution_policy.ForceFeedbackHelpers import ForceFeedbackHelper
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
        self._last_constant_error_base = None
        self._force_feedback = ForceFeedbackHelper(self.get_logger())

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
            self._force_feedback.on_wrench,
            qos,
        )

        # Periodische Diagnoseausgabe fuer Wrench und Positionsfehler.
        self._display_timer = self._parent_node.create_timer(0.5, self._display_tick)
        self._force_stream_timer = self._parent_node.create_timer(0.1, self._force_stream_tick)

    def _display_tick(self):
        """Display wrench and constant entrance-tip error at 4 Hz."""
        last_wrench = self._force_feedback.get_last_wrench()
        if last_wrench is not None:
            self.get_logger().info(
                "wrench | force: "
                f"x={last_wrench.wrench.force.x:.3f}, y={last_wrench.wrench.force.y:.3f}, z={last_wrench.wrench.force.z:.3f} "
                "| torque: "
                f"x={last_wrench.wrench.torque.x:.3f}, y={last_wrench.wrench.torque.y:.3f}, z={last_wrench.wrench.torque.z:.3f}"
            )
        elif self._force_feedback.get_num_wrench_msgs() == 0:
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

    def _force_stream_tick(self):
        """Stream force feedback while an insert operation is running."""
        feedback = self._force_feedback.stream_tick(time_window=0.1)
        if feedback is None:
            return

        delta_forces, forces_gradient, abs_forces = feedback
        self.get_logger().info(
            f"Abs force/torque: "
            f"force: x={abs_forces[0][0]:.3f}, y={abs_forces[0][1]:.3f}, z={abs_forces[0][2]:.3f} | "
            f"torque: x={abs_forces[1][0]:.3f}, y={abs_forces[1][1]:.3f}, z={abs_forces[1][2]:.3f}"
        )
        self.get_logger().info(
            f"Force delta over last 0.1s: "
            f"force: x={delta_forces[0][0]:.3f}, y={delta_forces[0][1]:.3f}, z={delta_forces[0][2]:.3f} | "
            f"torque: x={delta_forces[1][0]:.3f}, y={delta_forces[1][1]:.3f}, z={delta_forces[1][2]:.3f}"
        )
        self.get_logger().info(
            f"Force gradient over last 0.1s: "
            f"force: x={forces_gradient[0][0]:.3f}, y={forces_gradient[0][1]:.3f}, z={forces_gradient[0][2]:.3f} | "
            f"torque: x={forces_gradient[1][0]:.3f}, y={forces_gradient[1][1]:.3f}, z={forces_gradient[1][2]:.3f}"
        )

    def set_my_target_pose(self, move_robot: MoveRobotCallback,
                            pose: Pose,
                            offset_x= 0.0,
                            offset_y= 0.0,
                            offset_z= 0.0,
                            frame_id="base_link"):
        """Set the target pose for the robot, with an optional offset."""

        # Correct Target Pose with TCP Offset x=-0.0016, y=0.0010, z=-0.1073 (Not sure from where the offset is coming from)
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
        self.sleep_for(1.0)
        if abs(err_x) < 0.005 and abs(err_y) < 0.005 and abs(err_z) < 0.005:
            self.get_logger().info("Cable tip is well aligned with entrance (error < 5mm).")
            return True
        else:
            self.get_logger().warn(
                f"Cable tip is not well aligned with entrance: error_x={err_x:.4f}, error_y={err_y:.4f}, error_z={err_z:.4f}"
            )
            return False

    def plug_in(self, target_pos_in_base_link, target_rot_in_base_link, move_robot):
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
        self.sleep_for(3.0) 
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
        max_alignment_attempts = 20
        max_insertion_attempts = 30
        alignment_attempts = 0
        insertion_attempts = 0
        aligned_once = False

        try:
            # Prefer workspace-local logging directory used by this repository layout.
            data_dir = Path.cwd() / "aic_solution" / "aic_solution_policy" / "data"
            if not data_dir.parent.exists():
                data_dir = Path(__file__).resolve().parents[1] / "data"
            self._force_feedback.start_csv_logging(data_dir)
            self._force_feedback.set_stream_active(True)
            while insertion_attempts < max_insertion_attempts:
                # Zyklisch arbeiten, um TF und Sensorik ruhig nachzufuehren.
                self.sleep_for(0.25)

                '''
                Workflow:
                1) KeyPoint Prediction
                2) Triangulation -> Returns TF in base_link
                3) Call Allingnment
                4) Call plug_in(target_pose_in_base_link)
                '''

                # Entrance-Pose in base_link fuer Sollausrichtung laden.
                entrance_pos, entrance_rot = lookup_pose_in_base(
                    self._parent_node._tf_buffer,
                    "task_board/nic_card_mount_0/sfp_port_0_link_entrance",
                )

                # Calculatet 7-cm z to entrance 
                entrance_pos_offset = Point(
                    x=entrance_pos.x,
                    y=entrance_pos.y,
                    z=entrance_pos.z + 0.01
                )
                entrance_rot_offset = entrance_rot

                if not aligned_once:
                    alignment_attempts += 1
                    aligned_once = self.allign_connector(entrance_pos_offset, entrance_rot_offset, move_robot)
                    if aligned_once:
                        self.get_logger().info("Cable tip aligned with entrance, start insertion phase.")
                        send_feedback("Cable tip aligned, inserting cable...")
                    elif alignment_attempts >= max_alignment_attempts:
                        self.get_logger().warn("Alignment failed too often, aborting insert_cable().")
                        send_feedback("Alignment failed too often.")
                        return False
                    else:
                        self.get_logger().warn("Cable tip is not aligned with entrance, retrying alignment.")
                        send_feedback("Cable tip not aligned, retrying alignment...")
                    continue

                # Calculatet 7-cm z to entrance 
                target_pos_in_base_link = Point(
                    x=entrance_pos.x,
                    y=entrance_pos.y,
                    z=entrance_pos.z - 0.05
                )
                target_rot_in_base_link = entrance_rot

                insertion_attempts += 1
                inserted = self.plug_in(target_pos_in_base_link, target_rot_in_base_link, move_robot)
                if inserted:
                    self.get_logger().info("Cable successfully inserted.")
                    send_feedback("Cable successfully inserted.")
                    break
                else:
                    self.get_logger().warn(
                        f"Cable insertion failed (attempt {insertion_attempts}/{max_insertion_attempts})."
                    )
                    send_feedback("Cable insertion failed, retrying insertion...")
                    if insertion_attempts % 5 == 0:
                        aligned_once = False
                        self.get_logger().info("Re-running entrance alignment after repeated insertion failures.")

                observation = get_observation()
                if observation is None:
                    self.get_logger().info("No observation received.")

            if insertion_attempts >= max_insertion_attempts:
                self.get_logger().warn("Reached maximum insertion attempts without success.")
                send_feedback("Insertion failed after max attempts.")
                return False
        

        except Exception as exc:
            self.get_logger().error(f"Failed to get socket entrance transform: {exc}")
            send_feedback(f"Error: {exc}")
            return False
        finally:
            self._force_feedback.set_stream_active(False)
            self._force_feedback.stop_csv_logging()

        self.get_logger().info("Plug_in.insert_cable() exiting...")
        return True
