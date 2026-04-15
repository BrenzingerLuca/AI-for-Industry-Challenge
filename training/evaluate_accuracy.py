import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
from rclpy.duration import Duration

class PoseEvaluator(Node):
    def __init__(self):
        super().__init__('pose_evaluator')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Wir vergleichen alle 0.5 Sekunden
        self.timer = self.create_timer(0.5, self.evaluate_callback)
        
        # Namen der Frames
        self.ground_truth_frame = "task_board/nic_card_mount_0/sfp_port_0_link_entrance"
        self.detected_frame = "detected_sfp_port_0"

        self.get_logger().info("Accuracy Evaluator gestartet...")

    def evaluate_callback(self):
        try:
            # Suche die Transformation zwischen KI-Erkennung und Ground Truth
            now = rclpy.time.Time(nanoseconds=0)
            trans = self.tf_buffer.lookup_transform(
                self.ground_truth_frame, 
                self.detected_frame, 
                now,
                timeout=Duration(seconds=0.1)
            )

            # Extrahiere den Fehler in Metern
            dx = trans.transform.translation.x
            dy = trans.transform.translation.y
            dz = trans.transform.translation.z
            
            # Euklidischer Abstand (3D Distanzfehler)
            error_m = np.sqrt(dx**2 + dy**2 + dz**2)
            error_mm = error_m * 1000.0

            self.get_logger().info(f"📍 Fehler: {error_mm:.2f} mm | dx:{dx*1000:.1f} dy:{dy*1000:.1f} dz:{dz*1000:.1f}")

        except Exception as e:
            self.get_logger().warn(f"Warte auf Frames... ({e})")

def main():
    rclpy.init()
    node = PoseEvaluator()
    
    # Die Zeile 'node.declare_parameter' wurde entfernt, 
    # da der Parameter bereits durch die Kommandozeile existiert.
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()