"""
dataset_collector_single_cam.py - Single-Camera Dataset Collection Node

Earlier version of the dataset collector, using only the center camera.
Superseded by dataset_collector.py which supports all three cameras simultaneously.
Kept here as a simpler reference implementation.

Subscribes to the center camera, projects SFP port corners from TF into the image,
and generates YOLO pose labels automatically. Press ENTER in the terminal to save
the current frame as an image + label file pair.

YOLO Pose label format:
    <class> <cx> <cy> <bw> <bh> <kp1x> <kp1y> <v1> <kp2x> <kp2y> <v2> ...
    All values normalized to [0, 1]. Visibility flag: 2 = visible.

Usage:
    # Inside the pixi environment:
    python3 dataset_collector_single_cam.py

Configuration:
    Edit the CONFIG section below to set target frames, output path, and prefix.
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


# =============================================================================
# CONFIG — edit these values before running
# =============================================================================

# Output folder where images/ and labels/ subdirectories will be created
OUTPUT_PATH = "datasets/single_nic_card_dataset"

# Prefix for saved filenames, e.g. "run1" -> "run1_1_<timestamp>.jpg"
OUTPUT_PREFIX = "img"

# TF frames to label, mapped to their YOLO class ID
TARGET_FRAMES = {
    "task_board/nic_card_mount_0/sfp_port_0_link_entrance": 0,
    "task_board/nic_card_mount_0/sfp_port_1_link_entrance": 1,
}

# Physical dimensions of the SFP port opening in meters
PORT_WIDTH  = 0.014
PORT_HEIGHT = 0.009

# =============================================================================


class DatasetCollector(Node):
    def __init__(self):
        super().__init__('dataset_collector')
        self.bridge = CvBridge()

        # TF buffer and listener to query frame transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cam_model = PinholeCameraModel()

        self.latest_image = None
        self.latest_label = ""
        self.save_count = 0

        # Create output directories for images and labels
        os.makedirs(f"{OUTPUT_PATH}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_PATH}/labels", exist_ok=True)

        self.create_subscription(CameraInfo, '/center_camera/camera_info', self.info_cb, 10)
        self.create_subscription(Image, '/center_camera/image', self.image_cb, 10)
        self.publisher = self.create_publisher(Image, '/debug_image', 10)

        # Run keyboard listener in a background thread so ROS keeps spinning
        threading.Thread(target=self.keyboard_listener, daemon=True).start()

        self.get_logger().info("\n--- DATASET COLLECTOR READY ---")
        self.get_logger().info(f"Saving to: {OUTPUT_PATH}/  (prefix: '{OUTPUT_PREFIX}')")
        self.get_logger().info("Press ENTER to save the current frame.")

    def info_cb(self, msg):
        """Store the camera intrinsics when received."""
        self.cam_model.from_camera_info(msg)

    def image_cb(self, img_msg):
        """
        On each incoming image:
        1. Look up the TF transform for each target port frame.
        2. Project the 4 physical port corners into pixel coordinates.
        3. Compute the YOLO bounding box and keypoint label string.
        4. Draw a debug visualization and publish it.
        5. Store the latest raw image and label string for saving on keypress.
        """
        # Skip until camera intrinsics have been received
        if self.cam_model.projection_matrix() is None:
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
                    u, v = self.cam_model.project3dToPixel(
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

                # Debug visualization: color-coded per class ID
                color = (0, 255, 0) if class_id == 0 else (255, 255, 0)
                cv2.rectangle(debug_img,
                              (int(min_u), int(min_v)),
                              (int(max_u), int(max_v)),
                              color, 2)
                cv2.putText(debug_img, f"ID:{class_id}",
                            (int(min_u), int(min_v) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            except Exception:
                continue

        # Store the latest raw image and assembled label
        self.latest_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        self.latest_label = "\n".join(all_labels)

        # Publish the debug visualization
        self.publisher.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))

    def keyboard_listener(self):
        """Wait for ENTER keypress in the terminal and trigger a save."""
        while rclpy.ok():
            input()
            if self.latest_image is not None and self.latest_label != "":
                self.save_data()
            else:
                print("Nothing saved — no frame or no ports in view.")

    def save_data(self):
        """Save the latest image and label to disk."""
        self.save_count += 1
        timestamp = int(time.time() * 1000)
        img_name = f"{OUTPUT_PREFIX}_{self.save_count}_{timestamp}.jpg"
        txt_name = f"{OUTPUT_PREFIX}_{self.save_count}_{timestamp}.txt"

        cv2.imwrite(os.path.join(OUTPUT_PATH, "images", img_name), self.latest_image)
        with open(os.path.join(OUTPUT_PATH, "labels", txt_name), "w") as f:
            f.write(self.latest_label)

        print(f"[{self.save_count}] Saved: {img_name}\nLabels:\n{self.latest_label}")


def main():
    rclpy.init()
    node = DatasetCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()