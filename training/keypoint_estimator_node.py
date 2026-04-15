import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

class SFPDetectorNode(Node):
    def __init__(self):
        super().__init__('sfp_detector_node')
        self.bridge = CvBridge()
        
        # 1. Load your trained model
        # Path to the best.pt you downloaded from Colab
        self.model = YOLO('models/best150.pt')
        
        # 2. Subscribers & Publishers
        self.subscription = self.create_subscription(
            Image, '/right_camera/image', self.image_callback, 10)
        
        self.publisher = self.create_publisher(
            Image, '/right_camera/detected_ports', 10)

        self.get_logger().info("🚀 SFP Detector Node started!")

    def image_callback(self, msg):
        # Convert ROS image to OpenCV
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # 3. Run YOLOv8-Pose Inference
        # conf=0.5 helps filter out weak detections
        results = self.model.predict(cv_img, conf=0.9, verbose=False)
        
        # 4. Process Results
        for r in results:
            # Get Keypoints (4 corners)
            if r.keypoints is not None:
                # Loop über JEDE einzelne Erkennung (jeden Port)
                for i in range(len(r.keypoints.data)):
                    # Hol dir die Keypoints für Port Nummer i
                    kpts = r.keypoints.xy[i].cpu().numpy()
                    
                    # Jetzt zeichne die Punkte für DIESEN Port
                    for kp in kpts:
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(cv_img, (x, y), 5, (0, 255, 0), -1)

            # Draw the standard Bounding Box
            if r.boxes is not None:
                for box in r.boxes:
                    b = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
                    cv2.rectangle(cv_img, (int(b[0]), int(b[1])), 
                                  (int(b[2]), int(b[3])), (255, 0, 0), 2)

        # 5. Publish the debug image
        self.publisher.publish(self.bridge.cv2_to_imgmsg(cv_img, "bgr8"))

def main():
    rclpy.init()
    node = SFPDetectorNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()