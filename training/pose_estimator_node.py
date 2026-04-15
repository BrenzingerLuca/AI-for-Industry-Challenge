import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import tf2_ros
from ultralytics import YOLO
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

class SFPPoseEstimator(Node):
    def __init__(self):
        super().__init__('sfp_pose_estimator')
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # 1. Load your trained model
        self.model = YOLO('models/best150.pt')
        
        # 2. Define the 3D Model of the hole (14mm x 9mm)
        # Order MUST match your labeling: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        w2, h2 = 0.014 / 2.0, 0.009 / 2.0
        self.object_points = np.array([
            [-w2,  h2, 0.0], # P0: Top-Left
            [ w2,  h2, 0.0], # P1: Top-Right
            [ w2, -h2, 0.0], # P2: Bottom-Right
            [-w2, -h2, 0.0]  # P3: Bottom-Left
        ], dtype=np.float32)

        # 3. Camera Parameters (filled via CameraInfo)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        self.create_subscription(CameraInfo, '/center_camera/camera_info', self.info_cb, 10)
        self.create_subscription(Image, '/center_camera/image', self.image_cb, 10)
        self.debug_pub = self.create_publisher(Image, '/center_camera/debug_pose', 10)

        self.get_logger().info("SFP Pose Estimator Node started!")

    def info_cb(self, msg):
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

    def image_cb(self, msg):
        if self.camera_matrix is None: return
        
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model.predict(cv_img, conf=0.9, verbose=False)
        
        # Zeitstempel aus dem Bild-Header
        image_timestamp = msg.header.stamp 

        for r in results:
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                for i in range(len(r.keypoints.data)):
                    # 1. 2D-Punkte aus der KI (Pixel)
                    image_points = r.keypoints.xy[i].cpu().numpy().astype(np.float32)
                    
                    # 2. SolvePnP mit allen benötigten Parametern aufrufen
                    # Wir brauchen: 3D-Modell, 2D-Pixel, Kamera-Matrix, Verzeichnungskoeffizienten
                    success, rvec, tvec = cv2.solvePnP(
                        self.object_points, 
                        image_points, 
                        self.camera_matrix, 
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE # Bester Algorithmus für dein flaches Loch
                    )

                    if success:
                        class_id = int(r.boxes.cls[i])
                        # Zeitstempel an die TF-Funktion weitergeben
                        self.broadcast_tf(msg.header.frame_id, rvec, tvec, class_id, image_timestamp)
                        # Optional: Zeichne etwas zur Kontrolle ins Bild
                        self.draw_debug(cv_img, image_points, rvec, tvec, class_id)

        # Debug-Bild veröffentlichen
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_img, "bgr8"))

    def broadcast_tf(self, parent_frame, rvec, tvec, class_id, timestamp):
        t = TransformStamped()
        t.header.stamp = timestamp 
        t.header.frame_id = parent_frame
        t.child_frame_id = f"detected_sfp_port_{class_id}"

        # tvec ist oft [[x], [y], [z]], daher nutzen wir .flatten() 
        # um sicher [x, y, z] zu bekommen.
        t_flat = tvec.flatten()

        # Translation
        t.transform.translation.x = float(t_flat[0])
        t.transform.translation.y = float(t_flat[1])
        t.transform.translation.z = float(t_flat[2])

        # Rotation (Vorsicht: rvec ist auch [[r1], [r2], [r3]])
        rot_matrix, _ = cv2.Rodrigues(rvec)
        quat = R.from_matrix(rot_matrix).as_quat()
        
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

    def draw_debug(self, img, kpts, rvec, tvec, class_id):
        # 1. Draw Keypoints
        for i, kp in enumerate(kpts):
            cv2.circle(img, (int(kp[0]), int(kp[1])), 4, (0, 255, 0), -1)
        
        # 2. Draw 3D Axes at the center point
        axis_length = 0.02 # 2cm axis
        axis_points = np.float32([[axis_length,0,0], [0,axis_length,0], [0,0,axis_length], [0,0,0]]).reshape(-1, 3)
        imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        imgpts = imgpts.astype(int)
        
        origin = tuple(imgpts[3].ravel())
        cv2.line(img, origin, tuple(imgpts[0].ravel()), (0,0,255), 2) # X: Red
        cv2.line(img, origin, tuple(imgpts[1].ravel()), (0,255,0), 2) # Y: Green
        cv2.line(img, origin, tuple(imgpts[2].ravel()), (255,0,0), 2) # Z: Blue
        
        # 3. Print Distance
        dist = np.linalg.norm(tvec)
        cv2.putText(img, f"ID:{class_id} Dist:{dist:.3f}m", (origin[0], origin[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    rclpy.init()
    node = SFPPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()