"""
CarFusion 14-keypoint vehicle localization via fixed-scale 2D Procrustes (SVD).

Keypoint indices (CarFusion convention):
    0  wheel_fl   4  light_fl   8  roof_fl   12 exhaust
    1  wheel_fr   5  light_fr   9  roof_fr   13 center
    2  wheel_rl   6  light_rl  10  roof_rl
    3  wheel_rr   7  light_rr  11  roof_rr
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from trafficlab.motion.haware_localization import _FALLBACK_DIMS, _TRACK_RATIO, _WHEELBASE_RATIO

KP_NAMES = [
    'wheel_fl', 'wheel_fr', 'wheel_rl', 'wheel_rr',
    'light_fl', 'light_fr', 'light_rl', 'light_rr',
    'roof_fl',  'roof_fr',  'roof_rl',  'roof_rr',
    'exhaust',  'center',
]

_BASE_KP_HEIGHTS = [
    0.00, 0.00, 0.00, 0.00,
    0.65, 0.65, 0.65, 0.65,
    -1.0, -1.0, -1.0, -1.0,
    0.20, 0.70,
]


def build_carfusion_template(dims: dict) -> tuple:
    """Build (14,3) body-frame template and per-kp height list.

    Convention: x=lateral (left+), y=height, z=longitudinal (front-, rear+).
    Returns (template, kp_heights).
    """
    L  = dims['length']
    W  = dims['width']
    H  = dims['height']
    TW = dims.get('track_width', W * _TRACK_RATIO)
    WB = dims.get('wheelbase',   L * _WHEELBASE_RATIO)
    htw, hwb, hw, hl = TW/2, WB/2, W/2, L/2

    t = np.zeros((14, 3), dtype=np.float64)
    t[0]  = [ htw,       0.00, -hwb]
    t[1]  = [-htw,       0.00, -hwb]
    t[2]  = [ htw,       0.00, +hwb]
    t[3]  = [-htw,       0.00, +hwb]
    t[4]  = [ hw*0.85,   0.65, -hl ]
    t[5]  = [-hw*0.85,   0.65, -hl ]
    t[6]  = [ hw*0.85,   0.65, +hl ]
    t[7]  = [-hw*0.85,   0.65, +hl ]
    t[8]  = [ hw*0.70,   H,   -hl*0.50]
    t[9]  = [-hw*0.70,   H,   -hl*0.50]
    t[10] = [ hw*0.70,   H,   +hl*0.40]
    t[11] = [-hw*0.70,   H,   +hl*0.40]
    t[12] = [-hw*0.15,   0.20, +hl ]
    t[13] = [ 0.0,       0.70,  0.0]

    kp_heights = list(_BASE_KP_HEIGHTS)
    for i in (8, 9, 10, 11):
        kp_heights[i] = H
    return t, kp_heights


@dataclass
class CarFusionResult:
    sat_coords:  Optional[tuple]
    heading:     Optional[float]
    confidence:  float
    n_keypoints: int
    status:      str
    kp_sat:      dict = field(default_factory=dict)


class CarFusionLocalizer:
    def __init__(self, g_engine, template: np.ndarray, kp_heights: list, kp_conf: float = 0.2):
        self.g_engine   = g_engine
        self.template   = template
        self.kp_heights = kp_heights
        self.kp_conf    = kp_conf
        self._s         = g_engine.px_per_m

    def localize(self, kp_14: np.ndarray) -> CarFusionResult:
        """kp_14: (14,3) array [x_img, y_img, conf]."""
        kp_sat: dict[int, tuple] = {}
        for i in range(14):
            x, y, c = float(kp_14[i, 0]), float(kp_14[i, 1]), float(kp_14[i, 2])
            if c < self.kp_conf or (x == 0.0 and y == 0.0):
                continue
            kp_sat[i] = self.g_engine.cctv_to_sat(x, y, h=self.kp_heights[i])

        n = len(kp_sat)
        if n < 2:
            return CarFusionResult(None, None, 0.0, n, 'failed_insufficient_kp', kp_sat)

        idx = list(kp_sat.keys())
        Q   = self.template[idx][:, [0, 2]] * self._s
        P   = np.array([kp_sat[i] for i in idx])
        qb, pb = Q.mean(0), P.mean(0)
        U, _, Vt = np.linalg.svd((Q - qb).T @ (P - pb))
        det_sign = float(np.sign(np.linalg.det(Vt.T @ U.T)))
        R = Vt.T @ np.diag([1.0, det_sign]) @ U.T
        T_sat = pb - R @ qb

        heading = math.degrees(math.atan2(R[1, 1], -R[0, 1])) % 360.0
        z_vals  = self.template[idx, 2]
        if np.all(z_vals >= 0) or np.all(z_vals <= 0):
            return CarFusionResult(tuple(T_sat), None, 0.0, n, 'ambiguous_heading', kp_sat)

        P_pred = (Q - qb) @ R.T + pb
        rms    = float(np.sqrt(np.mean(np.sum((P - P_pred)**2, axis=1))))
        conf   = min(1.0, n/8.0) * max(0.0, 1.0 - rms / (5.0 * self._s))

        return CarFusionResult(tuple(T_sat), heading, conf, n, 'ok', kp_sat)
