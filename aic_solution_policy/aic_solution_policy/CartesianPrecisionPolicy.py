import numpy as np
import math
from rclpy.time import Time
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    Policy,
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

class CartesianPrecisionPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)

    def _get_diagonal_matrix(self, values):
        """Hilfsfunktion: Erzeugt eine flache 6x6 Matrix (36 Werte) aus 6 Diagonalelementen."""
        mat = [0.0] * 36
        for i in range(6):
            mat[i * 6 + i] = values[i]
        return mat

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info("Starting precision movement to target pose...")

        # 1. Erstelle die MotionUpdate Nachricht
        motion_update = MotionUpdate()
        motion_update.header.frame_id = "base_link"
        
        # Trajektorien-Modus auf POSITION (Mode 2 laut Doku-Beispiel)
        motion_update.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION

        # 2. Setze deine spezifischen Koordinaten
        motion_update.pose.position.x = -0.4321
        motion_update.pose.position.y = 0.2345
        motion_update.pose.position.z = 0.3211

        motion_update.pose.orientation.x = 1.0
        motion_update.pose.orientation.y = 0.0
        motion_update.pose.orientation.z = 0.0
        motion_update.pose.orientation.w = 0.0

        # 3. Parameter für Präzision (Stiffness & Damping)
        # Translation (X,Y,Z) hoch, Rotation (Rx,Ry,Rz) moderat
        stiff_diag = [1500.0, 1500.0, 1500.0, 150.0, 150.0, 150.0]
        damp_diag = [150.0, 150.0, 150.0, 15.0, 15.0, 15.0]
        
        motion_update.target_stiffness = self._get_diagonal_matrix(stiff_diag)
        motion_update.target_damping = self._get_diagonal_matrix(damp_diag)
        
        # 4. Der Bewegungs-Loop
        # Wir versuchen es für maximal 5 Sekunden (50 * 0.1s)
        for i in range(50):
            # Aktuellen Zeitstempel der Simulation setzen
            motion_update.header.stamp = self.get_clock().now().to_msg()
            
            # Befehl an den Controller senden
            move_robot(motion_update=motion_update)

            # Feedback über den Fortschritt holen
            obs = get_observation()
            curr = obs.controller_state.tcp_pose.position
            
            # Fehler berechnen (Euklidischer Abstand)
            error = math.sqrt(
                (curr.x - (-0.4321))**2 +
                (curr.y - 0.2345)**2 +
                (curr.z - 0.3211)**2
            )

            if i % 10 == 0:
                send_feedback(f"Abstand zum Ziel: {error*1000:.2f} mm")

            # Wenn wir näher als 0.5mm sind, brechen wir ab
            if error < 0.0005:
                send_feedback("Ziel mit hoher Präzision erreicht!")
                break

            self.sleep_for(0.1)

        # 5. Settling (Kurz warten, damit der Roboter ruhig steht)
        for _ in range(10):
            motion_update.header.stamp = self.get_clock().now().to_msg()
            move_robot(motion_update=motion_update)
            self.sleep_for(0.1)

        self.get_logger().info("Movement finished.")
        return True