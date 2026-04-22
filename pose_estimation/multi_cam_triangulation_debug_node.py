"""
This script, SFP Multi-View Fusion Node, is designed to estimate the 3D position 
and orientation of SFP ports using a multi-camera setup.

What this script does:
1.) Synchronized Detection: It waits for synchronized image frames from three
    cameras (left, center, right).

2.) Keypoint Detection: It uses a YOLOv8 model to detect ports and their 4 corner
    keypoints in each 2D image.

3.) Ray Casting: For every detected corner, it calculates a 3D "ray" starting from 
    the camera's optical center passing through the 2D pixel into the world space.

4.) Triangulation: It uses a Least Squares solver to find the 3D point where rays 
    from different cameras intersect (the most likely 3D position of the corner).

5.) Pose Estimation: Based on the 4 triangulated corners, it constructs a coordinate
    system (TF frame) to determine the port's exact position and rotation.

6.) Visual Debugging: It publishes the calculated 3D rays as Markers to RViz, allowing
    you to see if the camera alignment and detections are accurate.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, Point
from cv_bridge import CvBridge
import tf2_ros
from ultralytics import YOLO
import numpy as np
from scipy.spatial.transform import Rotation as R
import message_filters
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

class SFPFusionNode(Node):
    """
    Node to fuse 2D keypoints from multiple cameras into 3D poses using triangulation.
    Includes debug visualization by publishing rays to RViz.
    """
    def __init__(self):
        super().__init__('sfp_fusion_node')
        self.bridge = CvBridge()
        
        # TF2 Setup for looking up camera extrinsics and broadcasting results
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Debug Publishers: One Topic for each of the 4 port corners
        self.ray_pubs = [
            self.create_publisher(MarkerArray, f'/debug/rays/point_{i}', 10) 
            for i in range(4)
        ]

        # --- Configuration ---
        self.model = YOLO('../training/models/single_sc_detection.pt')
        self.camera_names = ['left_camera', 'center_camera', 'right_camera']
        self.fixed_frame = 'base'  # The global reference frame (e.g., 'world' or 'robot_base')

        self.cam_intrinsics = {}
        self.sync_subs = []
        
        # Initialize Subscribers
        for cam in self.camera_names:
            # Subscribe to CameraInfo once to get focal lengths and principal points
            self.create_subscription(
                CameraInfo, f'/{cam}/camera_info', 
                lambda msg, c=cam: self.save_info(msg, c), 10
            )
            
            # Create synchronized image subscribers
            sub = message_filters.Subscriber(self, Image, f'/{cam}/image')
            self.sync_subs.append(sub)

        # Synchronize images: Ensures we process frames taken at the same time
        self.ts = message_filters.ApproximateTimeSynchronizer(
            self.sync_subs, queue_size=10, slop=0.05
        )
        self.ts.registerCallback(self.multi_view_callback)

        self.get_logger().info("SFP Multi-View Fusion Node started!")

    def save_info(self, msg, cam_name):
        """Stores camera intrinsic parameters (fx, fy, cx, cy)."""
        if cam_name not in self.cam_intrinsics:
            self.cam_intrinsics[cam_name] = {
                'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]
            }

    def get_cam_frame_data(self, cam_name, time):
        """Retrieves camera position and rotation matrix in the global frame."""
        try:
            # ROS Optical Frame: Z forward, X right, Y down
            target = f"{cam_name}/optical" 
            trans = self.tf_buffer.lookup_transform(
                self.fixed_frame, target, time, 
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            pos = np.array([
                trans.transform.translation.x, 
                trans.transform.translation.y, 
                trans.transform.translation.z
            ])
            
            # Convert Quaternion to 3x3 Rotation Matrix
            q = trans.transform.rotation
            rot_mat = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            
            return {"position": pos, "rotation_matrix": rot_mat}
        except Exception as e:
            self.get_logger().warn(f"TF Error for {cam_name}: {e}")
            return None

    def compute_ray(self, kp_pixel, cam_frame, intrinsics):
        """
        Projects a 2D pixel to a 3D unit vector (ray) in the world frame.
        """
        u, v = kp_pixel
        # 1. Unproject pixel to camera local coordinates (Z=1)
        x = (u - intrinsics['cx']) / intrinsics['fx']
        y = (v - intrinsics['cy']) / intrinsics['fy']
        
        ray_cam = np.array([x, y, 1.0])
        ray_cam /= np.linalg.norm(ray_cam) # Normalize to unit vector
        
        # 2. Rotate ray into the world (fixed_frame) coordinate system
        ray_world = cam_frame["rotation_matrix"] @ ray_cam
        
        return {"origin": cam_frame["position"], "direction": ray_world}

    def triangulate(self, rays_list, weights):
        """
        Performs Least Squares Triangulation to find the intersection of N rays.
        """
        I = np.eye(3)
        A_w = np.zeros((3, 3))
        b_w = np.zeros(3)

        for i, ray in enumerate(rays_list):
            o = ray["origin"]
            d = ray["direction"]
            w = weights[i]
            
            # Projection matrix onto the line
            M = I - np.outer(d, d)
            A_w += w * M
            b_w += w * (M @ o)

        # Solve A * x = b
        return np.linalg.lstsq(A_w, b_w, rcond=None)[0]

    def multi_view_callback(self, *img_msgs):
        """Main processing loop triggered when all camera images are synchronized."""
        if len(self.cam_intrinsics) < 3: 
            return
        
        # Group detections by Port Class (e.g., class 0 = SFP_A, class 1 = SFP_B)
        separated_data = {0: {}, 1: {}} 
        timestamp = img_msgs[0].header.stamp

        # --- STEP 1: Object Detection ---
        for i, msg in enumerate(img_msgs):
            cam_name = self.camera_names[i]
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Inference using YOLO
            results = self.model.predict(cv_img, conf=0.90, verbose=False)[0]
            
            for j, box in enumerate(results.boxes):
                cls = int(box.cls[0]) 
                if cls in separated_data:
                    # Save keypoints (4 corners) for this camera and class
                    separated_data[cls][cam_name] = results.keypoints.xy[j].cpu().numpy()

        # --- STEP 2: Triangulation per Class ---
        for port_id, cam_dict in separated_data.items():
            if len(cam_dict) < 2:
                continue # Need at least 2 cameras to triangulate

            final_3d_corners = []
            debug_rays_for_this_port = [[] for _ in range(4)]

            for k_idx in range(4): # Loop through the 4 corners of the port
                rays_for_this_corner = []
                for cam_name, kpts in cam_dict.items():
                    cam_frame = self.get_cam_frame_data(cam_name, timestamp)
                    if cam_frame:
                        ray = self.compute_ray(kpts[k_idx], cam_frame, self.cam_intrinsics[cam_name])
                        rays_for_this_corner.append(ray)
                
                debug_rays_for_this_port[k_idx] = rays_for_this_corner

                if len(rays_for_this_corner) >= 2:
                    corner_3d = self.triangulate(rays_for_this_corner, [1.0]*len(rays_for_this_corner))
                    final_3d_corners.append(corner_3d)

            # Publish Debug Rays to RViz
            self.publish_rays(debug_rays_for_this_port, timestamp)

            # --- STEP 3: Pose Construction ---
            if len(final_3d_corners) == 4:
                self.publish_final_pose(final_3d_corners, timestamp, child_frame=f"fused_port_{port_id}") 

    def publish_final_pose(self, corners, timestamp, child_frame):
        """
        Calculates center and orientation from 4 corners and broadcasts the TF.
        """
        corners = np.array(corners)
        center = np.mean(corners, axis=0)

        # Coordinate System Construction:
        # X-axis: Corner 0 -> Corner 1 (Width)
        # Y-axis: Corner 0 -> Corner 3 (Height)
        vec_x = corners[1] - corners[0]
        vec_y = corners[3] - corners[0]
        
        vec_x /= np.linalg.norm(vec_x)
        vec_y /= np.linalg.norm(vec_y)
        # Z-axis: Orthogonal to X and Y
        vec_z = np.cross(vec_x, vec_y)
        vec_z /= np.linalg.norm(vec_z)
        
        # Build Rotation Matrix and convert to Quaternion
        rot_matrix = np.stack([vec_x, vec_y, vec_z], axis=1)
        quat = R.from_matrix(rot_matrix).as_quat()

        # Broadcast TF
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = self.fixed_frame
        t.child_frame_id = child_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = center.tolist()
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = quat.tolist()
        
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info(f"📍 {child_frame} fused at: {center}")

    def publish_rays(self, all_rays_by_corner, timestamp):
        """Publishes visualization lines for debug purposes in RViz."""
        corner_colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9), # 0: Red
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.9), # 1: Green
            ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.9), # 2: Blue
            ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.9)  # 3: Yellow
        ]

        for k_idx, rays in enumerate(all_rays_by_corner):
            if not rays: continue

            marker_array = MarkerArray()
            line_marker = Marker()
            line_marker.header.frame_id = self.fixed_frame
            line_marker.header.stamp = timestamp
            line_marker.ns = "rays"
            line_marker.id = 0
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.002 
            line_marker.color = corner_colors[k_idx]

            for ray in rays:
                # Ray origin (Camera)
                p_start = Point(x=float(ray["origin"][0]), 
                                y=float(ray["origin"][1]), 
                                z=float(ray["origin"][2]))
                # Ray end (Projected 2 meters out)
                end_vec = ray["origin"] + ray["direction"] * 2.0
                p_end = Point(x=float(end_vec[0]), 
                            y=float(end_vec[1]), 
                            z=float(end_vec[2]))
                
                line_marker.points.append(p_start)
                line_marker.points.append(p_end)

            marker_array.markers.append(line_marker)
            self.ray_pubs[k_idx].publish(marker_array)

def main():
    rclpy.init()
    node = SFPFusionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()