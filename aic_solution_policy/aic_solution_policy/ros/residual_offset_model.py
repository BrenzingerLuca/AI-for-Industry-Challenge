"""Vision-based residual offset-correction model, shared by both policies.

Trained in `residual_policy.ipynb` on data collected with `data_acquisition.py`:
given the three camera images, predicts the current 6-DoF offset between the
cable tip and the target port. Both the qualification and phase-1 policies use
this to nudge their approach pose before the spiral search, from the same
checkpoints (`dataset/checkpoints/regressor_best_{sfp,sc}.pt`).
"""

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from scipy.spatial.transform import Rotation as R

# Must exactly match residual_policy.ipynb's preprocessing/architecture -- we
# load that notebook's saved state_dict directly, so any mismatch here (image
# size, camera order, crop box, layer shapes) silently produces garbage
# predictions instead of an error.
RESIDUAL_IMAGE_SIZE = 128
RESIDUAL_CAMS = ['left', 'center', 'right']
# Must match CFG["crop"] in residual_policy.ipynb -- the model was trained on
# these fixed per-camera ROIs (crop, then resize), not the raw frame. Keyed by
# cable_type first since SC and SFP connectors sit in different parts of the frame.
RESIDUAL_CROP = {
    'sfp': {
        'left': (560, 600, 740, 760),
        'center': (480, 560, 660, 720),
        'right': (420, 600, 600, 760),
    },
    'sc': {
        'left': (570, 660, 710, 800),
        'center': (500, 660, 640, 800),
        'right': (430, 660, 570, 800),
    },
}
RESIDUAL_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RESIDUAL_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SharedViewEncoder(nn.Module):
    """Same architecture as residual_policy.ipynb's SharedViewEncoder."""

    def __init__(self, out_dim, pretrained=False):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x):
        return self.proj(self.backbone(x))


class MultiViewRegressor(nn.Module):
    """Same architecture as residual_policy.ipynb's MultiViewRegressor."""

    def __init__(self, num_cams, feat_dim, hidden, out_dim=6):
        super().__init__()
        self.num_cams = num_cams
        self.encoder = SharedViewEncoder(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * num_cams, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, images):
        b, n, c, h, w = images.shape
        feats = self.encoder(images.reshape(b * n, c, h, w)).reshape(b, n * self.encoder.proj.out_features)
        return self.mlp(feats)


def load_residual_model(checkpoint_path, device):
    # weights_only=False: trusted, self-produced checkpoint (contains numpy
    # target_mean/std alongside the state_dict, which torch's default
    # weights_only=True safe-unpickler may reject).
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MultiViewRegressor(num_cams=len(RESIDUAL_CAMS), feat_dim=256, hidden=256)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return {
        'model': model,
        'target_mean': np.asarray(checkpoint['target_mean'], dtype=np.float32),
        'target_std': np.asarray(checkpoint['target_std'], dtype=np.float32),
    }


def preprocess_image_for_residual_model(img_bgr, cam, cable_type):
    """Mirrors residual_policy.ipynb's OffsetDataset._load_image exactly (crop to
    the fixed per-camera ROI for this cable_type, then resize -- eval-mode
    preprocessing, no train-time augmentation)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    box = RESIDUAL_CROP.get(cable_type, {}).get(cam)
    if box is not None:
        x0, y0, x1, y1 = box
        img_rgb = img_rgb[y0:y1, x0:x1]
    img_resized = cv2.resize(img_rgb, (RESIDUAL_IMAGE_SIZE, RESIDUAL_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = (img_norm - RESIDUAL_IMAGENET_MEAN) / RESIDUAL_IMAGENET_STD
    return torch.from_numpy(img_norm.transpose(2, 0, 1))


class ResidualOffsetCorrector:
    """Loads the per-connector-type regressor checkpoints and runs them on live
    camera images. Runs on CPU: it's a single forward pass per insert_cable
    call, not worth risking GPU contention with the sim for.
    """

    def __init__(self, bridge, logger, checkpoint_paths, skip_description="correction"):
        self._bridge = bridge
        self._logger = logger
        self._device = torch.device('cpu')
        self._models = {}
        for cable_type, model_path in checkpoint_paths.items():
            if not model_path or not os.path.isfile(model_path):
                logger.warning(
                    f"No residual correction checkpoint for '{cable_type}' at {model_path!r}; "
                    f"{skip_description} will be skipped for this connector type."
                )
                continue
            try:
                self._models[cable_type] = load_residual_model(model_path, self._device)
                logger.info(f"Loaded residual correction model [{cable_type}]: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load residual correction model for '{cable_type}': {e}")

    def predict(self, observation, cable_type):
        """Runs the trained offset-correction model on the current camera images.

        Returns [dx,dy,dz] (meters) + [droll,dpitch,dyaw] (degrees): the
        model's estimate of the cable tip's current pose relative to the
        port, in the port's frame -- or None if no model/images are available.
        """
        bundle = self._models.get(cable_type)
        if bundle is None or observation is None:
            return None

        images = []
        for cam in RESIDUAL_CAMS:
            img_msg = getattr(observation, f"{cam}_image", None)
            if img_msg is None:
                self._logger.warning(f"Residual model: missing '{cam}' image, skipping correction")
                return None
            cv_img = self._bridge.imgmsg_to_cv2(img_msg, "bgr8")
            images.append(preprocess_image_for_residual_model(cv_img, cam, cable_type))
        images_tensor = torch.stack(images, dim=0).unsqueeze(0).to(self._device)

        with torch.no_grad():
            pred_norm = bundle['model'](images_tensor)[0].cpu().numpy()

        return pred_norm * bundle['target_std'] + bundle['target_mean']


def apply_predicted_correction(tcp_pos, tcp_quat, off_pos, off_quat, predicted_offset, correct_z=False):
    """Shifts a TCP pose so the cable tip cancels out the lateral part of
    `predicted_offset` (the tip's predicted pose relative to the port, in the
    port's frame -- same convention as compute_relative_offset() in
    residual_policy.ipynb: [dx,dy,dz] meters + [droll,dpitch,dyaw] degrees).

    Only lateral position (dx, dy) is corrected. Rotation is intentionally
    ignored (module height/orientation is fixed across connector configs, so
    there's nothing to correct there), and depth (dz) is dropped by default --
    Z is already governed by the approach/plug depths and the force-controlled
    descent, not by this model.

    The cable tip and gripper/tcp differ by a fixed, non-identity rotation
    (off_quat), so the correction -- expressed in the tip's own frame -- has to
    be conjugated by that offset before it can be applied to the TCP pose
    directly. When correct_z is False, the resulting world-frame Z is clamped
    back to the original tcp_pos[2] explicitly: a vector that's flat in the
    tilted tip frame is generally not flat once rotated into world frame, so
    zeroing dz alone isn't enough to keep this a purely lateral move.
    """
    dx, dy, dz, droll, dpitch, dyaw = predicted_offset
    if not correct_z:
        dz = 0.0

    droll = 0.0
    dpitch = 0.0
    dyaw = 0.0

    r_pred = R.from_euler('xyz', [droll, dpitch, dyaw], degrees=True)
    delta_pos_tip = -np.array([dx, dy, dz])
    delta_rot_tip = r_pred.inv()

    r_k = R.from_quat(off_quat)   # tip -> TCP fixed rotation
    t_k = np.array(off_pos)       # tip -> TCP fixed translation, in tip frame

    r_tcp_old = R.from_quat(tcp_quat)
    r_tip_old = r_tcp_old * r_k.inv()

    pos_correction = r_tip_old.apply(delta_pos_tip + delta_rot_tip.apply(t_k) - t_k)
    tcp_pos_new = np.array(tcp_pos) + pos_correction
    if not correct_z:
        tcp_pos_new[2] = tcp_pos[2]

    rot_correction_tcp_frame = r_k.inv() * delta_rot_tip * r_k
    tcp_quat_new = (r_tcp_old * rot_correction_tcp_frame).as_quat()

    return tcp_pos_new, tcp_quat_new
