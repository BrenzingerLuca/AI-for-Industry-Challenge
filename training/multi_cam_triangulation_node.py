import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import tf2_ros
from ultralytics import YOLO
import numpy as np
from scipy.spatial.transform import Rotation as R
import message_filters # Für die Synchronisation der 3 Kameras

class SFPFusionNode(Node):
    def __init__(self):
        super().__init__('sfp_fusion_node')
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # 1. Konfiguration
        self.model = YOLO('models/best150.pt')
        self.camera_names = ['left_camera', 'center_camera', 'right_camera']
        self.fixed_frame = 'base' # Wir triangulieren alles relativ zur Welt

        # 2. Kamera-Daten Struktur
        self.cam_intrinsics = {}
        self.sync_subs = []
        
        for cam in self.camera_names:
            # Camera Info Subscriber (einmalig)
            self.create_subscription(CameraInfo, f'/{cam}/camera_info', 
                                     lambda msg, c=cam: self.save_info(msg, c), 10)
            
            # Image Subscriber für Synchronizer
            sub = message_filters.Subscriber(self, Image, f'/{cam}/image')
            self.sync_subs.append(sub)

        # Synchronizer: Wartet, bis alle 3 Kameras ein Bild zum fast gleichen Zeitpunkt haben
        self.ts = message_filters.ApproximateTimeSynchronizer(self.sync_subs, queue_size=10, slop=0.05)
        self.ts.registerCallback(self.multi_view_callback)

        self.get_logger().info("SFP Multi-View Fusion Node gestartet!")

    def save_info(self, msg, cam_name):
        if cam_name not in self.cam_intrinsics:
            self.cam_intrinsics[cam_name] = {
                'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]
            }

    # --- MATHEMATIK DEINES KUMPELS (Integriert & Angepasst) ---

    def get_cam_frame_data(self, cam_name, time):
        """ Holt Kamera-Position und Euler-Winkel aus TF """
        try:
            target = f"{cam_name}/optical"
            trans = self.tf_buffer.lookup_transform(self.fixed_frame, target, time)
            pos = np.array([trans.transform.translation.x, 
                           trans.transform.translation.y, 
                           trans.transform.translation.z])
            
            # Quaternion zu Euler (XYZ) für das Skript deines Kumpels
            q = trans.transform.rotation
            rot = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')
            
            return {"position": pos, "rotation": rot}
        except Exception as e:
            return None

    def euler_to_rotmat(self, rotation):
        rx, ry, rz = rotation
        Rx = np.array([[1,0,0], [0,np.cos(rx),-np.sin(rx)], [0,np.sin(rx),np.cos(rx)]])
        Ry = np.array([[np.cos(ry),0,np.sin(ry)], [0,1,0], [-np.sin(ry),0,np.cos(ry)]])
        Rz = np.array([[np.cos(rz),-np.sin(rz),0], [np.sin(rz),np.cos(rz),0], [0,0,1]])
        return Rz @ Ry @ Rx

    def compute_ray(self, kp_pixel, cam_frame, intrinsics):
        """ Berechnet einen 3D-Strahl für einen Keypoint """
        R_mat = self.euler_to_rotmat(cam_frame["rotation"])
        t = cam_frame["position"]

        u, v = kp_pixel
        x = (u - intrinsics['cx']) / intrinsics['fx']
        y = (v - intrinsics['cy']) / intrinsics['fy']

        ray_cam = np.array([x, y, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)
        
        # Strahl in das Welt-System rotieren
        ray_world = R_mat @ ray_cam
        return {"origin": t, "direction": ray_world}

    def triangulate(self, rays_list, weights):
        """ Least Squares Triangulation von N Strahlen """
        I = np.eye(3)
        A_w = np.zeros((3, 3))
        b_w = np.zeros(3)

        for i, ray in enumerate(rays_list):
            o = ray["origin"]
            d = ray["direction"]
            w = weights[i]
            
            M = I - np.outer(d, d)
            A_w += w * M
            b_w += w * (M @ o)

        return np.linalg.lstsq(A_w, b_w, rcond=None)[0]

    # --- HAUPT LOGIK ---

    def multi_view_callback(self, *img_msgs):
        """ Wird aufgerufen, wenn alle 3 Kamera-Bilder da sind """
        if len(self.cam_intrinsics) < 3: return

        # 1. Keypoints von allen Kameras sammeln via YOLO
        all_cams_kpts = {} # {cam_name: [4 keypoints]}
        timestamp = img_msgs[0].header.stamp

        for i, msg in enumerate(img_msgs):
            cam_name = self.camera_names[i]
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self.model.predict(cv_img, conf=0.6, verbose=False)
            
            if results and len(results[0].keypoints.data) > 0:
                # Wir nehmen nur den ersten erkannten Port (ID 0)
                # kpts shape: [4, 2]
                all_cams_kpts[cam_name] = results[0].keypoints.xy[0].cpu().numpy()

        if len(all_cams_kpts) < 2:
            return # Wir brauchen mindestens 2 Kameras für Triangulation

        # 2. Für jeden der 4 Keypoints (Ecken) eine eigene Triangulation durchführen
        final_3d_corners = []
        for k_idx in range(4): # 4 Ecken
            rays_for_this_corner = []
            weights = []

            for cam_name in all_cams_kpts.keys():
                cam_frame = self.get_cam_frame_data(cam_name, timestamp)
                if cam_frame:
                    kp_pixel = all_cams_kpts[cam_name][k_idx]
                    ray = self.compute_ray(kp_pixel, cam_frame, self.cam_intrinsics[cam_name])
                    rays_for_this_corner.append(ray)
                    weights.append(1.0) # Hier könnte man YOLO-Confidence nutzen

            if len(rays_for_this_corner) >= 2:
                corner_3d = self.triangulate(rays_for_this_corner, weights)
                final_3d_corners.append(corner_3d)

        if len(final_3d_corners) == 4:
            self.publish_final_pose(final_3d_corners, timestamp)

    def publish_final_pose(self, corners, timestamp):
        """ Berechnet Center und Orientierung aus den 4 triangulierten 3D-Punkten """
        corners = np.array(corners)
        center = np.mean(corners, axis=0)

        # Orientierung berechnen (X-Achse von Ecke 0 zu 1, Y-Achse von 0 zu 3)
        vec_x = corners[1] - corners[0]
        vec_y = corners[3] - corners[0]
        
        vec_x /= np.linalg.norm(vec_x)
        vec_y /= np.linalg.norm(vec_y)
        vec_z = np.cross(vec_x, vec_y)
        vec_z /= np.linalg.norm(vec_z)
        
        # Orthogonale Matrix bauen
        rot_matrix = np.stack([vec_x, vec_y, vec_z], axis=1)
        quat = R.from_matrix(rot_matrix).as_quat()

        # TF Senden
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = self.fixed_frame
        t.child_frame_id = "fused_sfp_port"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = center.tolist()
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = quat.tolist()
        
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info(f"📍 Fused Port detected at: {center}")

def main():
    rclpy.init()
    node = SFPFusionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()