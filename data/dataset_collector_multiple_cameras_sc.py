"""dataset_collector_multiple_cameras_sc.py - Multi-Camera Dataset Collection Node

This node subscribes to all three robot cameras simultaneously.

Two modes are supported:
1) Manual labeling mode (default):
   - Press ENTER to save the latest frame from *all three* cameras.
   - Only the 3 images are written (no label files), so you can label later by hand.

2) Auto labeling mode:
   - Projects SFP port corners from TF into each camera image.
   - Generates YOLO pose labels automatically and writes image + label files.

Output structure:
    <output_path>/images/<filename>.jpg
    <output_path>/labels/<filename>.txt   (only when writing labels)
"""
 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import tf2_ros
import tf2_geometry_msgs
from image_geometry import PinholeCameraModel
import numpy as np
import os
import threading
from rclpy.time import Time
import time
from functools import partial
 
 
# =============================================================================
# CONFIG — edit these values before running
# =============================================================================
 
# Output folder where images/ and labels/ subdirectories will be created
OUTPUT_PATH = "datasets/sfp_tip_dataset"
 
# Prefix for saved filenames, e.g. "run1" -> "run1_left_camera_1_....jpg"
OUTPUT_PREFIX = "img"
 
# Cameras to subscribe to
CAMERA_NAMES = ['left_camera', 'center_camera', 'right_camera']

# When True: save only images (no TF projection, no labels written).
# When False: generate labels from TF and (optionally) write label files.
MANUAL_LABELING_MODE = True

# Only relevant when MANUAL_LABELING_MODE is False.
# If True: writes YOLO label text files into OUTPUT_PATH/labels.
WRITE_LABEL_FILES = True
 
# TF frames to label, mapped to their YOLO class ID
# Add or remove frames here to change which ports are labeled
TARGET_FRAMES = {
    "cable_0/sfp_tip_link": 0
   # "task_board/nic_card_mount_0/sfp_port_1_link_entrance": 1,
}
 
# Physical dimensions of the SFP port opening in meters
PORT_WIDTH  = 0.0256
PORT_HEIGHT = 0.009
 
# =============================================================================
 
 
class MultiCameraDatasetCollector(Node):
    def __init__(self):
        super().__init__('multi_camera_collector')
        self.bridge = CvBridge()
 
        # TF buffer and listener to query frame transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
 
        self.save_count = 0
 
        # Create output directories
        os.makedirs(f"{OUTPUT_PATH}/images", exist_ok=True)
        if not MANUAL_LABELING_MODE and WRITE_LABEL_FILES:
            os.makedirs(f"{OUTPUT_PATH}/labels", exist_ok=True)
 
        # Initialize state for each camera: model, latest frame, label, and debug publisher
        self.camera_states = {}
        for cam in CAMERA_NAMES:
            self.camera_states[cam] = {
                'model': PinholeCameraModel(),
                'latest_image': None,
                'latest_label': "",
                'info_received': False,
                'pub': self.create_publisher(Image, f'/{cam}/debug_image', 10)
            }
            # Subscribe to camera info and image topics for each camera
            self.create_subscription(
                CameraInfo, f'/{cam}/camera_info',
                partial(self.info_cb, cam_name=cam), 10
            )
            self.create_subscription(
                Image, f'/{cam}/image',
                partial(self.image_cb, cam_name=cam), 10
            )
 
        # Run keyboard listener in a background thread so ROS keeps spinning
        threading.Thread(target=self.keyboard_listener, daemon=True).start()
 
        self.get_logger().info("\n--- MULTI-CAM DATASET COLLECTOR READY ---")
        self.get_logger().info(f"Saving to: {OUTPUT_PATH}/  (prefix: '{OUTPUT_PREFIX}')")
        if MANUAL_LABELING_MODE:
            self.get_logger().info("Mode: MANUAL (images only, label later by hand)")
            self.get_logger().info("Press ENTER to save the latest frames from all 3 cameras.")
        else:
            self.get_logger().info("Mode: AUTO (TF projection -> YOLO labels)")
            self.get_logger().info("Press ENTER to save the latest frame from all cameras.")
 
    def info_cb(self, msg, cam_name):
        """Store the camera intrinsics when received."""
        self.camera_states[cam_name]['model'].from_camera_info(msg)
        self.camera_states[cam_name]['info_received'] = True
 
    def image_cb(self, img_msg, cam_name):
        """
        On each incoming image:
        1. Look up the TF transform for each target port frame.
        2. Project the 4 physical port corners into pixel coordinates.
        3. Compute the YOLO bounding box and keypoint label string.
        4. Draw a debug visualization and publish it.
        5. Store the latest raw image and label string for saving on keypress.
        """
        state = self.camera_states[cam_name]

        if MANUAL_LABELING_MODE:
            img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            state['latest_image'] = img
            state['latest_label'] = ""
            state['pub'].publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
            return
 
        # Skip until camera intrinsics have been received
        if not state['info_received'] or state['model'].projection_matrix() is None:
            return
 
        img_h, img_w = float(img_msg.height), float(img_msg.width)
        all_labels = []
        debug_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8").copy()
 
        # Half-extents of the port opening for corner offsets (in meters)
        w2, h2 = PORT_WIDTH / 2.0, PORT_HEIGHT / 2.0
 
        # 4 corners in the port frame's local coordinate system (z=0 = port plane)
        # Order: TL, TR, BR, BL
        corners_3d = [
            (-w2,  h2, 0.0),
            ( w2,  h2, 0.0),
            ( w2, -h2, 0.0),
            (-w2, -h2, 0.0),
        ]
 
        for frame_id, class_id in TARGET_FRAMES.items():
            try:
                # Get the transform from the port frame to the camera frame
                transform = self.tf_buffer.lookup_transform(
                    img_msg.header.frame_id, frame_id, Time(nanoseconds=0)
                )
 
                # Transform each 3D corner into camera coordinates and project to 2D pixels
                pixels = []
                for pt in corners_3d:
                    p_msg = tf2_geometry_msgs.PointStamped()
                    p_msg.point.x, p_msg.point.y, p_msg.point.z = pt
                    p_cam = tf2_geometry_msgs.do_transform_point(p_msg, transform)
                    u, v = state['model'].project3dToPixel(
                        (p_cam.point.x, p_cam.point.y, p_cam.point.z)
                    )
                    pixels.append((u, v))
 
                u_coords = [p[0] for p in pixels]
                v_coords = [p[1] for p in pixels]
 
                # Skip this frame if the port is entirely outside the image
                if (max(u_coords) < 0 or min(u_coords) > img_w or
                        max(v_coords) < 0 or min(v_coords) > img_h):
                    continue
 
                # Clamp bounding box to image boundaries
                min_u = np.clip(min(u_coords), 0, img_w)
                max_u = np.clip(max(u_coords), 0, img_w)
                min_v = np.clip(min(v_coords), 0, img_h)
                max_v = np.clip(max(v_coords), 0, img_h)
 
                # Compute normalized YOLO bounding box (cx, cy, w, h)
                yolo_x = ((min_u + max_u) / 2) / img_w
                yolo_y = ((min_v + max_v) / 2) / img_h
                yolo_w = (max_u - min_u) / img_w
                yolo_h = (max_v - min_v) / img_h
 
                # Build YOLO pose label line with keypoints (visibility=2 means visible)
                line = f"{class_id} {yolo_x:.6f} {yolo_y:.6f} {yolo_w:.6f} {yolo_h:.6f}"
                for (u, v) in pixels:
                    line += f" {u/img_w:.6f} {v/img_h:.6f} 2"
                all_labels.append(line)
 
                # --- Debug visualization ---
                pts = np.array(pixels, np.int32).reshape((-1, 1, 2))
 
                # Green polygon: the actual port opening edges
                cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)
 
                # Magenta line: shows port orientation (corner 0 → corner 1 = top edge)
                cv2.line(debug_img,
                         (int(pixels[0][0]), int(pixels[0][1])),
                         (int(pixels[1][0]), int(pixels[1][1])),
                         (255, 0, 255), 3)
 
                # Cyan rectangle: the YOLO bounding box
                cv2.rectangle(debug_img,
                              (int(min_u), int(min_v)),
                              (int(max_u), int(max_v)),
                              (255, 255, 0), 1)
 
                cv2.putText(debug_img, f"ID:{class_id}",
                            (int(min_u), int(min_v) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
 
            except Exception:
                continue
 
        # Store the latest raw image and assembled label for this camera
        state['latest_image'] = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        state['latest_label'] = "\n".join(all_labels)
 
        # Publish the debug visualization
        state['pub'].publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
 
    def keyboard_listener(self):
        """Wait for ENTER keypress in the terminal and trigger a save."""
        while rclpy.ok():
            input()
            self.save_all_cameras()
 
    def save_all_cameras(self):
        """Save the latest frames from the three cameras.

        - In manual mode: always saves exactly 3 JPGs (left/center/right) and no labels.
        - In auto mode: saves JPGs and (optionally) labels; requires non-empty labels.
        """
        states = [self.camera_states[cam] for cam in CAMERA_NAMES]
        if any(state['latest_image'] is None for state in states):
            print("Not saved — waiting for images from all 3 cameras.")
            return

        self.save_count += 1
        timestamp = int(time.time() * 1000)

        saved = []
        for cam_name in CAMERA_NAMES:
            state = self.camera_states[cam_name]

            if not MANUAL_LABELING_MODE:
                if state['latest_label'] == "":
                    print(f"Not saved — no valid label for camera '{cam_name}' yet.")
                    return

            img_name = f"{OUTPUT_PREFIX}_{cam_name}_{self.save_count}_{timestamp}.jpg"
            img_path = os.path.join(OUTPUT_PATH, "images", img_name)
            cv2.imwrite(img_path, state['latest_image'])

            if not MANUAL_LABELING_MODE and WRITE_LABEL_FILES:
                label_path = os.path.join(
                    OUTPUT_PATH, "labels", img_name.replace('.jpg', '.txt')
                )
                with open(label_path, "w") as f:
                    f.write(state['latest_label'])

            saved.append(cam_name)

        print(f"[SAVE {self.save_count}] Saved images: {', '.join(saved)}")
 
 
def main():
    rclpy.init()
    node = MultiCameraDatasetCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()