"""Single-image vehicle keypoint inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from skimage import transform as skimage_transform

from trafficlab.keypoint.model import KeyPointModel

ORIENTATION_LABELS = ['front', 'rear', 'left', 'left-front', 'left-rear', 'right', 'right-front', 'right-rear']

KP_LABELS = [
    'left-front wheel',       # 0
    'left-back wheel',        # 1
    'right-front wheel',      # 2
    'right-back wheel',       # 3
    'right fog lamp',         # 4
    'left fog lamp',          # 5
    'right headlight',        # 6
    'left headlight',         # 7
    'front auto logo',        # 8
    'front license plate',    # 9
    'left rear-view mirror',  # 10
    'right rear-view mirror', # 11
    'right-front roof corner',# 12
    'left-front roof corner', # 13
    'left-back roof corner',  # 14
    'right-back roof corner', # 15
    'left rear lamp',         # 16
    'right rear lamp',        # 17
    'rear auto logo',         # 18
    'rear license plate',     # 19
]

# Index groups for heading calculation
WHEEL_INDICES = (0, 2, 1, 3)   # front-left, front-right, rear-left, rear-right
ROOF_INDICES  = (13, 12, 14, 15)  # front-left, front-right, rear-left, rear-right


_DEFAULT_MEAN_STD = Path(__file__).parent / 'veri_mean_std.pth.tar'


class PirazhPreprocessor:
    """Resize + normalize a crop for the Pirazh model."""

    def __init__(self, mean_std_path: str | Path = _DEFAULT_MEAN_STD):
        data = torch.load(str(mean_std_path), weights_only=False)
        self.mean = data['mean'].numpy()
        self.std = data['std'].numpy()

    def __call__(self, bgr_crop: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bgr_crop: H×W×3 uint8 BGR image (from cv2)
        Returns:
            (img224, img56): float tensors ready for the model, both (1,3,H,W)
        """
        rgb = bgr_crop[:, :, ::-1].astype(np.float64) / 255.0
        img224 = skimage_transform.resize(rgb, (224, 224), anti_aliasing=True)
        img56  = skimage_transform.resize(rgb, (56, 56),   anti_aliasing=True)
        for j in range(3):
            img224[:, :, j] = (img224[:, :, j] - self.mean[j]) / self.std[j]
            img56[:, :, j]  = (img56[:, :, j]  - self.mean[j]) / self.std[j]
        t224 = torch.from_numpy(img224.transpose(2, 0, 1)).float().unsqueeze(0)
        t56  = torch.from_numpy(img56.transpose(2, 0, 1)).float().unsqueeze(0)
        return t224, t56


class PirazhDetector:
    """Loads checkpoint and runs inference on vehicle crops."""

    def __init__(self, checkpoint_path: str | Path, device: str = 'cpu'):
        self.device = torch.device(device)
        self.preprocessor = PirazhPreprocessor()
        self.model = KeyPointModel().to(self.device)
        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        state = ckpt.get('net_state_dict', ckpt.get('state_dict', ckpt))
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, bgr_crop: np.ndarray) -> dict:
        """
        Run keypoint + orientation inference on a single BGR vehicle crop.

        Returns dict with:
            keypoints   : (20, 2) float array, coordinates in crop pixel space
            orientation : int 0-7
            orient_label: str
            orient_conf : float
            fine_heatmaps: (20, 56, 56) numpy array (raw, for debug)
        """
        t224, t56 = self.preprocessor(bgr_crop)
        t224, t56 = t224.to(self.device), t56.to(self.device)

        with torch.no_grad():
            _, fine_kp, orientation_logits = self.model(t224, t56)

        # keypoints: argmax of each 56×56 heatmap → (x, y) in 56×56 space
        # fine_kp has 21 channels (20 keypoints + background); take first 20
        kp_flat = fine_kp[:, :20, :, :].reshape(1, 20, -1)
        idx = kp_flat.argmax(dim=2).squeeze(0)  # (20,)
        kp56 = torch.stack([idx % 56, idx // 56], dim=1).float().cpu().numpy()  # (20, 2) x,y

        # scale to crop pixel coords
        h, w = bgr_crop.shape[:2]
        kp_crop = kp56 * np.array([w / 56.0, h / 56.0])

        # orientation
        probs = torch.softmax(orientation_logits, dim=1).squeeze(0).cpu().numpy()
        orient_idx = int(probs.argmax())

        return {
            'keypoints': kp_crop,
            'orientation': orient_idx,
            'orient_label': ORIENTATION_LABELS[orient_idx],
            'orient_conf': float(probs[orient_idx]),
            'fine_heatmaps': fine_kp.squeeze(0).cpu().numpy(),
        }


def heading_from_keypoints(kp_crop: np.ndarray) -> float | None:
    """
    Estimate heading (degrees) from keypoints in crop pixel space.

    Tries wheel keypoints first; falls back to roof corners.
    Returns None if both groups are at the origin (not detected).
    """
    def _axle_heading(fl, fr, rl, rr):
        front = (np.array(fl) + np.array(fr)) / 2.0
        rear  = (np.array(rl) + np.array(rr)) / 2.0
        dx, dy = front[0] - rear[0], front[1] - rear[1]
        return (np.degrees(np.arctan2(dy, dx)) + 360) % 360

    fl_w, fr_w, rl_w, rr_w = [kp_crop[i] for i in WHEEL_INDICES]
    if not (np.allclose(fl_w, 0) and np.allclose(fr_w, 0)):
        return _axle_heading(fl_w, fr_w, rl_w, rr_w)

    fl_r, fr_r, rl_r, rr_r = [kp_crop[i] for i in ROOF_INDICES]
    if not (np.allclose(fl_r, 0) and np.allclose(fr_r, 0)):
        return _axle_heading(fl_r, fr_r, rl_r, rr_r)

    return None
