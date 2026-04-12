"""
tf_keypoints_projection.py - Exploration Script
 
Extends tf_point_projection.py by projecting the 4 physical corners of the SFP port opening onto
the camera image, computing a YOLO-format bounding box + keypoint label string,
and publishing the annotated image.
 
This was used to verify that auto-generated YOLO pose labels from TF are
geometrically correct before building the full dataset collection pipeline.
 
YOLO Pose label format:
    <class> <cx> <cy> <bw> <bh> <kp1x> <kp1y> <v1> <kp2x> <kp2y> <v2> ...
    All values normalized to [0, 1]. Visibility flag: 2 = visible.
 
Usage: run as a ROS 2 node alongside the simulation.
"""
 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import tf2_ros
import tf2_geometry_msgs
from image_geometry import PinholeCameraModel
import message_filters
from rclpy.time import Time
import numpy as np
 
 
class TFToImageProjector(Node):
    def __init__(self):
        super().__init__('tf_projector')
        self.bridge = CvBridge()
 
        # TF buffer and listener to query frame transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
 
        self.cam_model = PinholeCameraModel()
 
        # TF frame of the SFP port entrance — origin is at the center of the opening
        self.target_frame = "task_board/nic_card_mount_0/sfp_port_1_link_entrance"
 
        # Synchronize image and camera info by timestamp to ensure consistent projection
        self.img_sub = message_filters.Subscriber(self, Image, '/center_camera/image')
        self.info_sub = message_filters.Subscriber(self, CameraInfo, '/center_camera/camera_info')
        self.ts = message_filters.TimeSynchronizer([self.img_sub, self.info_sub], 10)
        self.ts.registerCallback(self.callback)
 
        self.publisher = self.create_publisher(Image, '/center_camera/image_with_tf', 10)
 
    def callback(self, img_msg, info_msg):
        self.cam_model.fromCameraInfo(info_msg)
 
        try:
            # Physical dimensions of the SFP port opening (in meters), halved for corner offsets
            w2, h2 = 0.014 / 2.0, 0.009 / 2.0
 
            # 4 corners of the port in the target frame's local coordinate system (z=0 = port plane)
            # Order: TL, TR, BR, BL
            corners_3d = [
                (-w2,  h2, 0.0),
                ( w2,  h2, 0.0),
                ( w2, -h2, 0.0),
                (-w2, -h2, 0.0),
            ]
 
            # Get the transform from the port frame to the camera frame
            transform = self.tf_buffer.lookup_transform(
                img_msg.header.frame_id,
                self.target_frame,
                Time(nanoseconds=0)
            )
 
            # Transform each 3D corner into camera coordinates and project to 2D pixels
            pixels = []
            for pt in corners_3d:
                p_cam = self.transform_point(pt, transform)
                uv = self.cam_model.project3dToPixel(p_cam)
                pixels.append(uv)
 
            # Compute normalized YOLO bounding box from the projected pixel extents
            img_h, img_w = float(info_msg.height), float(info_msg.width)
            u_coords = [p[0] for p in pixels]
            v_coords = [p[1] for p in pixels]
            min_u, max_u = min(u_coords), max(u_coords)
            min_v, max_v = min(v_coords), max(v_coords)
 
            yolo_x = ((min_u + max_u) / 2) / img_w
            yolo_y = ((min_v + max_v) / 2) / img_h
            yolo_w = (max_u - min_u) / img_w
            yolo_h = (max_v - min_v) / img_h
 
            # Build YOLO pose label string (class 0, bbox, then keypoints with visibility=2)
            label_str = f"0 {yolo_x:.6f} {yolo_y:.6f} {yolo_w:.6f} {yolo_h:.6f}"
            for (u, v) in pixels:
                label_str += f" {u/img_w:.6f} {v/img_h:.6f} 2.000000"
 
            self.get_logger().info(f"\nLABEL: {label_str}")
 
            # Visualize the 4 projected corners on the image (color-coded per corner)
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            colors = [(0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 255)]  # R, G, Y, M
            for i, (u, v) in enumerate(pixels):
                cv2.circle(cv_img, (int(u), int(v)), 5, colors[i], -1)
 
            self.publisher.publish(self.bridge.cv2_to_imgmsg(cv_img, "bgr8"))
 
        except Exception as e:
            self.get_logger().warn(f"Could not project frame: {e}")
 
    def transform_point(self, point, transform):
        """Transform a 3D point from the target frame into the camera frame."""
        from geometry_msgs.msg import PointStamped
 
        p = PointStamped()
        p.point.x, p.point.y, p.point.z = point
        p_transformed = tf2_geometry_msgs.do_transform_point(p, transform)
        return (p_transformed.point.x, p_transformed.point.y, p_transformed.point.z)
 
 
def main():
    rclpy.init()
    node = TFToImageProjector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()