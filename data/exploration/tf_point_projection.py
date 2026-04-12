"""
tf_projector_v1.py - Exploration Script

Projects a single TF frame origin onto a camera image using ROS 2 and OpenCV.
This was an early prototype to verify that TF lookup + camera projection works
correctly before extending to bounding box / keypoint generation.

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


class TFToImageProjector(Node):
    def __init__(self):
        super().__init__('tf_projector')
        self.bridge = CvBridge()

        # TF buffer and listener to query frame transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cam_model = PinholeCameraModel()

        # The TF frame of the port entrance to project onto the image
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
            # Look up where the target frame is relative to the camera frame
            transform = self.tf_buffer.lookup_transform(
                img_msg.header.frame_id,
                self.target_frame,
                Time(nanoseconds=0)
            )

            # Extract the origin of the target frame in camera coordinates
            pt_cv = (
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            )

            # Project 3D point to 2D pixel coordinates using the camera model
            uv = self.cam_model.project3dToPixel(pt_cv)

            cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            u, v = int(uv[0]), int(uv[1])
            cv2.circle(cv_img, (u, v), 10, (0, 0, 255), -1)
            cv2.putText(cv_img, self.target_frame, (u + 15, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            self.publisher.publish(self.bridge.cv2_to_imgmsg(cv_img, "bgr8"))

        except Exception as e:
            self.get_logger().warn(f"Could not project frame: {e}")


def main():
    rclpy.init()
    node = TFToImageProjector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()