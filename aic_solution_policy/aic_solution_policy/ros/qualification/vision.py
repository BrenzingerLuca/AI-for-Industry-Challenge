"""YOLO port-keypoint detection and multi-camera ray triangulation."""

import numpy as np
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R


class PortDetector:
    """Detects the two port openings per camera view (YOLO pose model, 4
    corner keypoints each) and triangulates their 3D pose in base_link from
    whichever cameras saw them.
    """

    def __init__(self, tf_buffer, bridge, models, camera_names, logger):
        self._tf_buffer = tf_buffer
        self._bridge = bridge
        self._models = models
        self._camera_names = camera_names
        self._logger = logger
        self._cam_intrinsics = {}
        # TF lookups can become temporarily unreliable after sim resets (time
        # jumps). Camera extrinsics are static, so cache successful lookups
        # and reuse them.
        self._tf_cam_frame_cache = {}

    def reset(self):
        """Drop cached TF data so stale transforms aren't reused after a sim reset."""
        self._tf_cam_frame_cache.clear()

    def _get_cam_frame_data(self, cam_full_name):
        """Camera pose in base_link frame, or the last cached value if TF is unavailable."""
        cached = self._tf_cam_frame_cache.get(cam_full_name)
        try:
            target = f"{cam_full_name}/optical"
            trans = self._tf_buffer.lookup_transform("base_link", target, Time())
            data = {
                "pos": np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]),
                "rot": R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]).as_matrix()
            }
            self._tf_cam_frame_cache[cam_full_name] = data
            return data
        except Exception:
            return cached

    def detect(self, observation, cable_type):
        """Runs YOLO on each camera view and triangulates the 3D port pose(s).

        Returns (found_ports, total_detections, num_unique_port_classes).
        found_ports maps port class id (0 or 1) -> {"pos": xyz, "quat": xyzw}.
        """
        if observation is None:
            return {}, 0, 0

        for cam in self._camera_names:
            attr = f"{cam}_camera_info"
            if hasattr(observation, attr):
                msg = getattr(observation, attr)
                self._cam_intrinsics[cam] = {
                    'fx': msg.k[0],
                    'fy': msg.k[4],
                    'cx': msg.k[2],
                    'cy': msg.k[5]
                }

        model = self._models.get(cable_type)
        if model is None:
            self._logger.error(f"No YOLO model available for cable_type='{cable_type}'")
            return {}, 0, 0

        unique_classes = set()
        total_detections = 0
        separated = {0: {}, 1: {}}

        for cam in self._camera_names:
            img_msg = getattr(observation, f"{cam}_image", None)
            if img_msg is None:
                continue

            cv_img = self._bridge.imgmsg_to_cv2(img_msg, "bgr8")
            res = model.predict(cv_img, conf=0.8, verbose=False)[0]

            for j, box in enumerate(res.boxes):
                cls = int(box.cls[0])
                total_detections += 1
                unique_classes.add(cls)
                if cls in separated:
                    separated[cls][cam] = res.keypoints.xy[j].cpu().numpy()

        # Triangulate each port's 4 corner keypoints from every camera that saw it.
        found_ports = {}
        for pid, cams in separated.items():
            if len(cams) < 2:
                continue

            pts_3d = []
            for k in range(4):
                rays = []
                for cam_name, kpts in cams.items():
                    fdata = self._get_cam_frame_data(f"{cam_name}_camera")
                    if fdata and cam_name in self._cam_intrinsics:
                        u, v = kpts[k]
                        intr = self._cam_intrinsics[cam_name]
                        d_cam = np.array([
                            (u - intr['cx']) / intr['fx'],
                            (v - intr['cy']) / intr['fy'],
                            1.0
                        ])
                        d_world = fdata["rot"] @ d_cam
                        d_world /= np.linalg.norm(d_world)
                        rays.append({"origin": fdata["pos"], "direction": d_world})

                if len(rays) >= 2:
                    # Least-squares closest point to all rays.
                    I = np.eye(3)
                    A = np.zeros((3, 3))
                    b = np.zeros(3)
                    for r in rays:
                        M = I - np.outer(r["direction"], r["direction"])
                        A += M
                        b += M @ r["origin"]
                    pts_3d.append(np.linalg.lstsq(A, b, rcond=None)[0])

            if len(pts_3d) == 4:
                corners = np.array(pts_3d)
                center = np.mean(corners, axis=0)

                vec_x = corners[1] - corners[0]
                vec_x[2] = 0
                vec_x /= np.linalg.norm(vec_x)

                vec_z = np.array([0.0, 0.0, -1.0])
                vec_y = np.cross(vec_z, vec_x)
                vec_y /= np.linalg.norm(vec_y)

                rot_matrix = np.stack([vec_x, vec_y, vec_z], axis=1)

                found_ports[pid] = {
                    "pos": center,
                    "quat": R.from_matrix(rot_matrix).as_quat()
                }

        return found_ports, total_detections, len(unique_classes)
