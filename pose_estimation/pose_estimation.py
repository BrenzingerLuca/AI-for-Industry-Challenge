"""
This script, SFPPoseVisualizer, is a specialized version of the fusion node. 
Its primary purpose is to triangulate the 3D position of SFP ports and force 
a specific orientation (where the Z-axis always points straight down). 

Key Features:
Pose Estimation: 
 It calculates the X-axis from the corners but forces the Z-axis to point at [0, 0, -1] (downwards) 
 to ensure a stable coordinate system for a robot.

State Persistence: It stores the latest_results to allow periodic logging.

Periodic Status Logging: A timer-driven callback prints the current coordinates and quaternions to the console every few seconds for easier debugging.
"""

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import tf2_ros
from ultralytics import YOLO
import numpy as np
from scipy.spatial.transform import Rotation as R
import message_filters

# --- CONFIGURATION ---
MODEL_PATH = '../training/models/single_sc_detection.pt'
TARGET_FRAME = 'base'               
PRINT_INTERVAL = 3.0                
CONFIDENCE_THRESHOLD = 0.70         
# ---------------------

class SFPPoseVisualizer(Node):
    def __init__(self):
        super().__init__('sfp_pose_visualizer_node')
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.model = YOLO(MODEL_PATH)
        self.camera_names = ['left_camera', 'center_camera', 'right_camera']
        
        self.cam_intrinsics = {}
        self.sync_subs = []
        self.latest_results = {0: None, 1: None}

        for cam in self.camera_names:
            self.create_subscription(CameraInfo, f'/{cam}/camera_info', 
                                     lambda msg, c=cam: self.save_info(msg, c), 10)
            sub = message_filters.Subscriber(self, Image, f'/{cam}/image')
            self.sync_subs.append(sub)

        self.ts = message_filters.ApproximateTimeSynchronizer(self.sync_subs, 10, 0.05)
        self.ts.registerCallback(self.multi_view_callback)
        self.timer = self.create_timer(PRINT_INTERVAL, self.timer_print_callback)

        self.get_logger().info("Visualizer gestartet: Pose-Logik angepasst (Y & Z Vorzeichen geflippt).")

    def save_info(self, msg, cam_name):
        if cam_name not in self.cam_intrinsics:
            self.cam_intrinsics[cam_name] = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}

    def get_cam_frame_data(self, cam_name, timestamp):
        try:
            target = f"{cam_name}/optical" 
            trans = self.tf_buffer.lookup_transform(TARGET_FRAME, target, timestamp, rclpy.duration.Duration(seconds=0.1))
            pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            q = trans.transform.rotation
            return {"pos": pos, "rot": R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()}
        except: return None

    def triangulate(self, rays):
        I = np.eye(3)
        A, b = np.zeros((3,3)), np.zeros(3)
        for r in rays:
            M = I - np.outer(r["direction"], r["direction"])
            A += M
            b += M @ r["origin"]
        return np.linalg.lstsq(A, b, rcond=None)[0]

    def calculate_pose(self, corners):
        corners = np.array(corners)
        center = np.mean(corners, axis=0)

        # X-Achse aus Geometrie (2D, ignoriere Z)
        vec_x = corners[1] - corners[0]
        vec_x[2] = 0  # Z-Komponente auf 0 erzwingen (rein 2D)
        vec_x /= np.linalg.norm(vec_x)

        # Z-Achse: fix nach unten
        vec_z = np.array([0.0, 0.0, -1.0])

        # Y-Achse: aus Z × X (Rechtshändig, konsistent)
        vec_y = np.cross(vec_z, vec_x)
        vec_y /= np.linalg.norm(vec_y)

        rot_matrix = np.stack([vec_x, vec_y, vec_z], axis=1)
        quat = R.from_matrix(rot_matrix).as_quat()

        return center, quat

    def multi_view_callback(self, *img_msgs):
        if len(self.cam_intrinsics) < 3: return
        timestamp = img_msgs[0].header.stamp
        separated = {0: {}, 1: {}}

        for i, msg in enumerate(img_msgs):
            cam = self.camera_names[i]
            res = self.model.predict(self.bridge.imgmsg_to_cv2(msg, "bgr8"), conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
            for j, box in enumerate(res.boxes):
                cls = int(box.cls[0])
                if cls in separated: separated[cls][cam] = res.keypoints.xy[j].cpu().numpy()

        for pid, cams in separated.items():
            if len(cams) < 2: 
                self.latest_results[pid] = None
                continue
            
            pts_3d = []
            for k in range(4):
                rays = []
                for c, kpts in cams.items():
                    fdata = self.get_cam_frame_data(c, timestamp)
                    if fdata:
                        u, v = kpts[k]
                        d_cam = np.array([(u-self.cam_intrinsics[c]['cx'])/self.cam_intrinsics[c]['fx'], (v-self.cam_intrinsics[c]['cy'])/self.cam_intrinsics[c]['fy'], 1.0])
                        d_world = fdata["rot"] @ d_cam
                        rays.append({"origin": fdata["pos"], "direction": d_world/np.linalg.norm(d_world)})
                if len(rays) >= 2: pts_3d.append(self.triangulate(rays))

            if len(pts_3d) == 4:
                pos, quat = self.calculate_pose(pts_3d)
                self.latest_results[pid] = {"pos": pos, "quat": quat}
                
                t = TransformStamped()
                t.header.stamp = timestamp
                t.header.frame_id = TARGET_FRAME
                t.child_frame_id = f"port_{pid}"
                t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = pos.tolist()
                t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = quat.tolist()
                self.tf_broadcaster.sendTransform(t)
            else:
                self.latest_results[pid] = None

    def timer_print_callback(self):
        self.get_logger().info("--- Aktuelle Posen ---")
        for pid, data in self.latest_results.items():
            if data:
                p, q = data["pos"], data["quat"]
                self.get_logger().info(f"Port {pid}: Pos {[round(float(c),3) for c in p]}, Quat {[round(float(c),3) for c in q]}")

def main():
    rclpy.init(); node = SFPPoseVisualizer()
    try: rclpy.spin(node)
    except: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()