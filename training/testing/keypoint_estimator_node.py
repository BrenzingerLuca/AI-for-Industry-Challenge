"""
Script: SFP Detector Node
Description:
This script performs YOLOv8-Pose inference on live Gazebo simulation images.
Bounding boxes are color-coded by Port ID (e.g., ID 0 is Blue, ID 1 is Red) 
to allow for easy visual identification without text labels. Keypoints are 
also color-coded by their specific index to verify spatial consistency.

Usage:
1. Ensure Gazebo is running and publishing to the configured camera topic.
2. Run this node to visualize the detections on the output topic.
"""

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

# --- CONFIGURATION CONSTANTS ---
CAMERA_TOPIC = "/right_camera/image"
OUTPUT_TOPIC = "/right_camera/detected_ports"
MODEL_RELATIVE_PATH = "../models/best150.pt"
CONFIDENCE_THRESHOLD = 0.7
SHOW_KEYPOINTS = True
LINE_THICKNESS = 3

# Colors for Bounding Boxes (BGR format)
# Index 0 = Blue, Index 1 = Red, Index 2 = Green, etc.
BOX_COLOR_PALETTE = [
    (255, 0, 0),   # ID 0: Blue
    (0, 0, 255),   # ID 1: Red
    (0, 255, 0),   # ID 2: Green
    (0, 255, 255), # ID 3: Cyan
]

# Colors for Keypoints (BGR format)
KP_COLOR_PALETTE = [
    (0, 0, 255),   # KP 0: Red
    (0, 255, 0),   # KP 1: Green
    (255, 0, 0),   # KP 2: Blue
    (0, 255, 255), # KP 3: Cyan
    (255, 0, 255), # KP 4: Magenta
    (255, 255, 0), # KP 5: Yellow
]
# -------------------------------

class SFPDetectorNode(Node):
    def __init__(self):
        """
        Initialize the node, load the model, and log the color configuration.
        """
        super().__init__('sfp_detector_node')
        self.bridge = CvBridge()
        
        # Path Handling
        script_dir = os.path.dirname(os.path.realpath(__file__))
        model_full_path = os.path.abspath(os.path.join(script_dir, MODEL_RELATIVE_PATH))
        
        try:
            self.model = YOLO(model_full_path)
            self.get_logger().info(f"Model loaded: {model_full_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            return

        # Print Color Mapping for the user
        self.get_logger().info("--- VISUALIZATION COLOR MAP ---")
        self.get_logger().info("Port ID 0: BLUE Bbox")
        self.get_logger().info("Port ID 1: RED Bbox")
        self.get_logger().info("Port ID 2: GREEN Bbox")
        self.get_logger().info("Keypoints: Color-coded by index (Red=0, Green=1, Blue=2...)")
        self.get_logger().info("--------------------------------")

        self.subscription = self.create_subscription(
            Image, CAMERA_TOPIC, self.image_callback, 10)
        
        self.publisher = self.create_publisher(
            Image, OUTPUT_TOPIC, 10)

    def image_callback(self, msg):
        """
        Main callback for processing images and drawing color-coded detections.
        """
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Conversion error: {e}")
            return
        
        results = self.model.predict(
            cv_img, 
            conf=CONFIDENCE_THRESHOLD, 
            verbose=False
        )
        
        for r in results:
            if r.boxes is None:
                continue

            for i, box in enumerate(r.boxes):
                # Determine color based on detection index (ID)
                # This makes Port 0 Blue and Port 1 Red
                box_color = BOX_COLOR_PALETTE[i % len(BOX_COLOR_PALETTE)]
                
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)
                
                # --- 1. Draw Bounding Box (No Label Text) ---
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), box_color, LINE_THICKNESS)

                # --- 2. Draw Keypoints ---
                if SHOW_KEYPOINTS and r.keypoints is not None:
                    # Keypoints for this specific instance i
                    kpts = r.keypoints.xy[i].cpu().numpy()
                    
                    for kp_idx, kp in enumerate(kpts):
                        kx, ky = int(kp[0]), int(kp[1])
                        
                        if kx > 0 or ky > 0:
                            # Use keypoint specific palette
                            kp_color = KP_COLOR_PALETTE[kp_idx % len(KP_COLOR_PALETTE)]
                            
                            # Draw the keypoint circle
                            cv2.circle(cv_img, (kx, ky), 5, kp_color, -1)
                            
                            # Small ID next to keypoint for verification
                            cv2.putText(cv_img, str(kp_idx), (kx + 5, ky - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, kp_color, 1)

        # Publish the debug frame
        try:
            self.publisher.publish(self.bridge.cv2_to_imgmsg(cv_img, "bgr8"))
        except Exception as e:
            self.get_logger().error(f"Publishing failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SFPDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()