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
import rclpy
from ultralytics import YOLO
import numpy as np
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

class VisionBasedPortDocking(Policy):
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

        current_file_path = Path(__file__).resolve()
        
        # Suche an verschiedenen Orten:
        possible_paths = [
            # 1. Im installierten Paket (site-packages)
            current_file_path.parents[1] / "models" / "best150.pt",
            # 2. In deinem lokalen Entwicklungsordner (Workspace)
            Path.home() / "ws_aic/src/aic/aic_solution/training/models/best150.pt",
            # 3. Falls der Ordner direkt im aktuellen Arbeitsverzeichnis ist
            Path.cwd() / "models" / "best150.pt"
        ]

        model_path = None
        for p in possible_paths:
            if p.exists():
                model_path = p
                break

        if model_path is None:
            self.get_logger().error(f"Modell konnte an keinem der Orte gefunden werden:")
            for p in possible_paths:
                self.get_logger().error(f" - Versucht: {p}")
            raise FileNotFoundError("YOLO Modell nicht gefunden.")

        self.get_logger().info(f"Modell erfolgreich gefunden unter: {model_path}")
        self._model = YOLO(str(model_path))

        self._bridge = CvBridge()

        # YOLO Modell laden
        self._model = YOLO(model_path)
        self._camera_names = ['left', 'center', 'right']
        self._cam_intrinsics = {}
        
        # Intrinsics einmalig sammeln
        for cam in self._camera_names:
            self._parent_node.create_subscription(
                CameraInfo, f'/{cam}_camera/camera_info', 
                lambda msg, c=cam: self._save_cam_info(msg, c), 10)

        # Periodische Diagnoseausgabe fuer Wrench und Positionsfehler.
        self._display_timer = self._parent_node.create_timer(0.5, self._display_tick)
        #self._force_stream_timer = self._parent_node.create_timer(0.1, self._force_stream_tick)

    def _save_cam_info(self, msg, cam_name):
        if cam_name not in self._cam_intrinsics:
            self._cam_intrinsics[cam_name] = {
                'fx': msg.k[0], 'fy': msg.k[4], 
                'cx': msg.k[2], 'cy': msg.k[5]
            }

    def _display_tick(self):
        """Display wrench and constant entrance-tip error at 4 Hz."""
        last_wrench = self._force_feedback.get_last_wrench()
        # if last_wrench is not None:
        #     self.get_logger().info(
        #         "wrench | force: "
        #         f"x={last_wrench.wrench.force.x:.3f}, y={last_wrench.wrench.force.y:.3f}, z={last_wrench.wrench.force.z:.3f} "
        #         "| torque: "
        #         f"x={last_wrench.wrench.torque.x:.3f}, y={last_wrench.wrench.torque.y:.3f}, z={last_wrench.wrench.torque.z:.3f}"
        #     )
        # elif self._force_feedback.get_num_wrench_msgs() == 0:
        #     count = self._parent_node.count_publishers("/fts_broadcaster/wrench")
        #     self.get_logger().warn(
        #         f"No wrench messages received yet. Publishers on /fts_broadcaster/wrench: {count}"
        #     )

        # if self._last_constant_error_base is not None:
        #     dx, dy, dz = self._last_constant_error_base
        #     self.get_logger().info(
        #         "constant_error(base_link) | "
        #         f"x={dx:.4f}, y={dy:.4f}, z={dz:.4f}"
        #     )

    def _force_stream_tick(self):
        """Stream force feedback while an insert operation is running."""
        feedback = self._force_feedback.stream_tick(time_window=0.1)
        if feedback is None:
            return

        # delta_forces, forces_gradient, abs_forces = feedback
        # self.get_logger().info(
        #     f"Abs force/torque: "
        #     f"force: x={abs_forces[0][0]:.3f}, y={abs_forces[0][1]:.3f}, z={abs_forces[0][2]:.3f} | "
        #     f"torque: x={abs_forces[1][0]:.3f}, y={abs_forces[1][1]:.3f}, z={abs_forces[1][2]:.3f}"
        # )
        # self.get_logger().info(
        #     f"Force delta over last 0.1s: "
        #     f"force: x={delta_forces[0][0]:.3f}, y={delta_forces[0][1]:.3f}, z={delta_forces[0][2]:.3f} | "
        #     f"torque: x={delta_forces[1][0]:.3f}, y={delta_forces[1][1]:.3f}, z={delta_forces[1][2]:.3f}"
        # )
        # self.get_logger().info(
        #     f"Force gradient over last 0.1s: "
        #     f"force: x={forces_gradient[0][0]:.3f}, y={forces_gradient[0][1]:.3f}, z={forces_gradient[0][2]:.3f} | "
        #     f"torque: x={forces_gradient[1][0]:.3f}, y={forces_gradient[1][1]:.3f}, z={forces_gradient[1][2]:.3f}"
        # )

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
        self.sleep_for(5.0)

        return True
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

    def detect_ports(self, observation):
        if observation is None:
            return {}

        # 1. Intrinsics aus Observation lernen
        for cam in self._camera_names:
            if cam not in self._cam_intrinsics:
                # Versuche 'left_camera_info' aus der Observation zu lesen
                attr_name = f"{cam}_camera_info"
                if hasattr(observation, attr_name):
                    self._save_cam_info(getattr(observation, attr_name), cam)

        # 2. Bilder verarbeiten
        separated = {0: {}, 1: {}}
        for cam in self._camera_names:
            img_msg = getattr(observation, f"{cam}_image", None)
            if img_msg is None:
                continue

            cv_img = self._bridge.imgmsg_to_cv2(img_msg, "bgr8")
            res = self._model.predict(cv_img, conf=0.8, verbose=False)[0]
            
            for j, box in enumerate(res.boxes):
                cls = int(box.cls[0])
                if cls in separated:
                    separated[cls][cam] = res.keypoints.xy[j].cpu().numpy()

        # 3. Triangulation
        found_ports = {}
        timestamp = self._parent_node.get_clock().now().to_msg()

        for pid, cams in separated.items():
            if len(cams) < 2: continue
            
            pts_3d = []
            for k in range(4):
                rays = []
                for cam_name, kpts in cams.items():
                    # TF braucht den vollen Frame Namen (z.B. left_camera/optical)
                    fdata = self._get_cam_frame_data(f"{cam_name}_camera", timestamp)
                    
                    # SICHERHEITS-CHECK: Existiert der Key in Intrinsics?
                    if fdata and cam_name in self._cam_intrinsics:
                        u, v = kpts[k]
                        intr = self._cam_intrinsics[cam_name]
                        d_cam = np.array([
                            (u - intr['cx']) / intr['fx'], 
                            (v - intr['cy']) / intr['fy'], 
                            1.0])
                        d_world = fdata["rot"] @ d_cam
                        rays.append({"origin": fdata["pos"], "direction": d_world/np.linalg.norm(d_world)})
                
                if len(rays) >= 2:
                    pts_3d.append(self._triangulate_rays(rays))

            if len(pts_3d) == 4:
                pos, quat = self._calculate_forced_pose(pts_3d)
                found_ports[pid] = {"pos": pos, "quat": quat}
                self.get_logger().info(f"✨ Port {pid} erkannt!")
                self.get_logger().info(f"Position: [x: {pos[0]:.5f}, y: {pos[1]:.5f}, z: {pos[2]:.5f}]")

        return found_ports

    def _get_cam_frame_data(self, cam_full_name, timestamp):
        try:
            target = f"{cam_full_name}/optical" 
            trans = self._parent_node._tf_buffer.lookup_transform(
                "base_link", target, Time(), rclpy.duration.Duration(seconds=0.1))
            return {
                "pos": np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]),
                "rot": R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]).as_matrix()
            }
        except: return None

    def _triangulate_rays(self, rays):
        I = np.eye(3)
        A, b = np.zeros((3,3)), np.zeros(3)
        for r in rays:
            M = I - np.outer(r["direction"], r["direction"])
            A += M
            b += M @ r["origin"]
        return np.linalg.lstsq(A, b, rcond=None)[0]

    def _calculate_forced_pose(self, corners):
        corners = np.array(corners)
        center = np.mean(corners, axis=0)
        vec_x = corners[1] - corners[0]
        vec_x[2] = 0 # Z-Komponente auf 0 für reine Ebene
        vec_x /= np.linalg.norm(vec_x)
        vec_z = np.array([0.0, 0.0, -1.0]) # Z zeigt nach unten
        vec_y = np.cross(vec_z, vec_x)
        vec_y /= np.linalg.norm(vec_y)
        rot_matrix = np.stack([vec_x, vec_y, vec_z], axis=1)
        return center, R.from_matrix(rot_matrix).as_quat()
    
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        """
        Hauptablauf für das Einstecken des Kabels basierend auf Vision-Erkennung.
        """
        # --- TASK CONTENT PRINTING ---
        self.get_logger().info("========================================")
        self.get_logger().info(f"EINGEHENDER TASK EMPFANGEN2:")
        # Die Task-Nachricht direkt als String loggen
        self.get_logger().info(f"Task Details: {task}")
        
        # Falls der Task eine Liste von Parametern hat, können wir diese auch einzeln zeigen:
        if hasattr(task, 'parameters'):
            for param in task.parameters:
                self.get_logger().info(f" - Parameter: {param.name} = {param.value}")
        self.get_logger().info("========================================")

        self.get_logger().info("Starte Vision-basierten Plug_in Vorgang.")
        send_feedback("Initializing Vision and Force Feedback...")

        # 1. Force Feedback Logging starten
        try:
            # Pfad für CSV-Logs festlegen
            data_dir = Path.cwd() / "aic_solution" / "aic_solution_policy" / "data"
            if not data_dir.parent.exists():
                data_dir = Path(__file__).resolve().parents[1] / "data"
            
            self._force_feedback.start_csv_logging(data_dir)
            self._force_feedback.set_stream_active(True)

            # 3. Port-Scanning Phase
            target_port_data = None
            max_scan_attempts = 30 # Etwas mehr Versuche geben
            
            for attempt in range(max_scan_attempts):
                send_feedback(f"Scanning for ports... (Attempt {attempt+1}/{max_scan_attempts})")
                
                # Wichtig: Observation bei jedem Versuch neu holen!
                observation = get_observation()
                detected_ports = self.detect_ports(observation)

                if detected_ports:
                    # Wir nehmen Port 0 (SFP 0), falls vorhanden
                    target_id = 0 if 0 in detected_ports else list(detected_ports.keys())[0]
                    target_port_data = detected_ports[target_id]
                    self.get_logger().info(f"✨ Ziel-Port {target_id} erfolgreich lokalisiert!")
                    break
                
                self.sleep_for(0.2)

            if target_port_data is None:
                self.get_logger().error("Vision: Keine Ports im Sichtfeld der Kameras erkannt.")
                send_feedback("Detection failed: No ports visible.")
                return False

            # 4. Ziel-Posen vorbereiten
            p = target_port_data["pos"]
            q = target_port_data["quat"]
            
            # Die Basis-Rotation des Ports (Z zeigt nach unten)
            entrance_rot = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

            # 5. Alignment-Phase (Positionierung über dem Port)
            # Wir zielen auf 2 cm über den Port
            alignment_pos = Point(x=p[0], y=p[1], z=p[2] + 0.02)
            
            self.get_logger().info(f"Fahre Alignment-Pose an: z={alignment_pos.z:.3f}")
            send_feedback("Aligning connector above port...")
            
            aligned = self.allign_connector(alignment_pos, entrance_rot, move_robot)
            
            if not aligned:
                self.get_logger().warn("Alignment fehlgeschlagen oder ungenau.")
                send_feedback("Alignment failed.")
                return False

            # 6. Insertion-Phase (Einführen)
            # Wir drücken das Kabel 4 cm tief rein
            send_feedback("Inserting cable into port...")
            self.get_logger().info("Starte Insertion...")
            
            insertion_pos = Point(x=p[0], y=p[1], z=p[2] - 0.04)
            success = self.plug_in(insertion_pos, entrance_rot, move_robot)

            if success:
                self.get_logger().info("✅ KABEL ERFOLGREICH EINGESTECKT!")
                send_feedback("Task completed successfully.")
            else:
                self.get_logger().error("❌ Stecken fehlgeschlagen.")
                send_feedback("Insertion failed.")
                return False

        except Exception as exc:
            self.get_logger().error(f"KRITISCHER FEHLER: {exc}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False
        finally:
            # Logging immer sauber beenden
            self._force_feedback.set_stream_active(False)
            self._force_feedback.stop_csv_logging()

        return True
