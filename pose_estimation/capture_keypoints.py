"""
One-shot debug helper: grabs the first confident detection from each camera,
draws the YOLO boxes + 4 corner keypoints on it (via ultralytics' own
Result.plot()), and saves each as a PNG. Not part of the policy pipeline --
just a tool for grabbing documentation screenshots.

Run from this directory (pose_estimation/) so the relative MODEL_PATH
resolves, with the sim already showing a port to at least one camera:

    pixi shell
    cd aic_solution/pose_estimation
    python3 capture_keypoints.py
"""

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

# Swap to '../training/models/single_sc_detection.pt' to capture SC keypoints instead.
MODEL_PATH = '../training/models/single_sc_detection.pt'
CAMERA_NAMES = ['left_camera', 'center_camera', 'right_camera']
OUTPUT_DIR = 'keypoint_captures'
CONFIDENCE_THRESHOLD = 0.7


class KeypointCaptureNode(Node):
    def __init__(self):
        super().__init__('keypoint_capture_node')
        self.bridge = CvBridge()
        self.model = YOLO(MODEL_PATH)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.captured = set()
        for cam in CAMERA_NAMES:
            self.create_subscription(Image, f'/{cam}/image', lambda msg, c=cam: self.on_image(msg, c), 10)
        self.get_logger().info(f"Waiting for a confident detection on {CAMERA_NAMES}...")

    def on_image(self, msg, cam_name):
        if cam_name in self.captured:
            return
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        result = self.model.predict(cv_img, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        if len(result.boxes) == 0:
            return

        annotated = result.plot()  # draws boxes + keypoints
        out_path = os.path.join(OUTPUT_DIR, f'{cam_name}_keypoints.png')
        cv2.imwrite(out_path, annotated)
        self.get_logger().info(f"Saved {out_path}")
        self.captured.add(cam_name)

        if len(self.captured) == len(CAMERA_NAMES):
            self.get_logger().info("All cameras captured, shutting down.")
            rclpy.shutdown()


def main():
    rclpy.init()
    node = KeypointCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
