import numpy as np


"""
===========================================================
MULTI-VIEW TRIANGULATION (RAYS + LEAST SQUARES)
===========================================================

Dieses Skript berechnet 3D-Strahlen aus mehreren Kameras
und trianguliert einen Punkt im Raum relativ zu Kamera 1.

-----------------------
EINGABEN
-----------------------

1. keypoints (Liste, Länge N Kameras)

   Unterstützte Formate:
   - (u, v)
   - [(u, v), confidence]
   - {"u": u, "v": v}
   - {"fx": u, "fy": v}   (legacy support)

   Beispiel:
   keypoints = [
       [(320, 240), 0.98],
       [(300, 240), 0.95],
       [(320, 220), 0.88]
   ]

2. cam_frames (Liste von Dicts)

   [
       {"position": np.array([x,y,z]), "rotation": (rx, ry, rz)},
       ...
   ]

   - Rotation = Euler Winkel (Radiant)
   - Reihenfolge: XYZ

3. cam_intrinsics (Liste von Dicts)

   [
       {"fx": ..., "fy": ..., "cx": ..., "cy": ...},
       ...
   ]

-----------------------
AUSGABEN
-----------------------

1. Rays:
   dict:
   {
       "cam_0": {
           "origin": np.array(3),
           "direction": np.array(3),
           "confidence": float
       }
   }

2. Triangulierte Punkte:
   {
       "unweighted": np.array(3),
       "weighted": np.array(3)
   }

-----------------------
MATHEMATIK
-----------------------

Ray:
    X(t) = origin + t * direction

Triangulation:
    Minimiert Abstand zu allen Rays (Least Squares)

    A = Σ (I - d dᵀ)
    b = Σ (I - d dᵀ) o

    weighted:
    A = Σ w (I - d dᵀ)

-----------------------
GRENZEN / LIMITIERUNGEN
-----------------------

- Mindestens 2 Kameras erforderlich (besser ≥3)
- Rays sollten sich schneiden (Geometrie wichtig!)
- Schlechte Keypoints → großer Fehler
- Parallele Rays → numerisch instabil
- Keine Outlier-Filterung (kein RANSAC)
- Keine Lens Distortion berücksichtigt

-----------------------
EMPFEHLUNGEN
-----------------------

- Verwende Confidence als Gewicht (bereits integriert)
- Nutze ≥3 Kameras für Stabilität
- Normalisiere Eingaben sauber
- Für robuste Systeme:
    → RANSAC + Reprojection Error ergänzen

===========================================================
"""


def euler_to_rotmat(rotation):
    rx, ry, rz = rotation

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)]
    ])

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


def compute_camera_rays(keypoints, cam_frames, cam_intrinsics):
    """
    Berechnet Rays im Koordinatensystem von Kamera 1.
    """

    R_ref = euler_to_rotmat(cam_frames[0]["rotation"])
    t_ref = cam_frames[0]["position"]
    R_ref_inv = R_ref.T

    rays = {}

    for i in range(len(cam_frames)):
        cam = cam_frames[i]
        K = cam_intrinsics[i]

        kp = keypoints[i]

        # --- Keypoint Parsing ---
        confidence = 1.0

        if isinstance(kp, list) and len(kp) == 2 and isinstance(kp[1], (int, float)):
            (u, v), confidence = kp
        elif isinstance(kp, tuple):
            u, v = kp
        elif isinstance(kp, dict):
            u = kp.get("u", kp.get("fx"))
            v = kp.get("v", kp.get("fy"))
        else:
            raise ValueError(f"Unbekanntes Keypoint-Format: {kp}")

        u, v = float(u), float(v)
        confidence = float(confidence)

        # --- Extrinsics ---
        R = euler_to_rotmat(cam["rotation"])
        t = cam["position"]

        R_rel = R_ref_inv @ R
        t_rel = R_ref_inv @ (t - t_ref)

        # --- Intrinsics ---
        fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]

        x = (u - cx) / fx
        y = (v - cy) / fy

        ray_cam = np.array([x, y, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)

        ray_world = R_rel @ ray_cam

        rays[f"cam_{i}"] = {
            "origin": t_rel,
            "direction": ray_world,
            "confidence": confidence
        }

    return rays


def triangulate_rays(rays):
    """
    Berechnet den Punkt mit minimalem Abstand zu allen Rays.
    """

    I = np.eye(3)

    A = np.zeros((3, 3))
    b = np.zeros(3)

    A_w = np.zeros((3, 3))
    b_w = np.zeros(3)

    for ray in rays.values():
        o = ray["origin"]
        d = ray["direction"]
        w = ray.get("confidence", 1.0)

        d = d / np.linalg.norm(d)

        M = I - np.outer(d, d)

        A += M
        b += M @ o

        A_w += w * M
        b_w += w * (M @ o)

    # --- lösen ---
    try:
        P = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        P = np.linalg.lstsq(A, b, rcond=None)[0]

    try:
        P_w = np.linalg.solve(A_w, b_w)
    except np.linalg.LinAlgError:
        P_w = np.linalg.lstsq(A_w, b_w, rcond=None)[0]

    return {
        "unweighted": P,
        "weighted": P_w
    }