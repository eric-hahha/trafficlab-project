"""
h-aware vehicle localization using 3D keypoint template matching.

Uses all 24 Apollo-24 keypoints with per-keypoint height priors to project
each detected point to satellite coordinates, then fits vehicle center and
heading via fixed-scale 2D Procrustes (SVD).

Reference: docs/3d-keypoint-template-localization.md § 3B
"""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Fallback dimensions (prior_dimensions.json "measurements_visdrone" car entry)
_FALLBACK_DIMS = {
    'length':      3.8,
    'width':       1.8,
    'height':      1.55,
    'track_width': 1.53,
    'wheelbase':   2.55,
}

# Same ratios as wheel_localization.py for when track/wheelbase are absent
_TRACK_RATIO     = 0.85   # track_width / width
_WHEELBASE_RATIO = 0.67   # wheelbase / length


def compute_car_dims_from_spec_csv(csv_path: str, body_type: str = 'Sedan') -> dict:
    """Parse ilyasozkurt/automobile-models-and-specs engines.csv and return median
    sedan dimensions in metres.

    The CSV stores specs as nested JSON with a "Dimensions" section whose values
    follow the pattern "X.X In (YYYY Mm)" (parenthesised mm value).
    Filters to plausible sedan ranges before computing medians.
    Falls back to _FALLBACK_DIMS if the file is missing or yields no rows.
    """
    def _extract_mm(val: str) -> Optional[float]:
        # Handles both "4509 Mm" and "1,590/1,570 Mm" (front/rear track average)
        m = re.findall(r'\(([0-9,./]+)\s*[Mm]m\)', val)
        if not m:
            return None
        s = m[0].replace(',', '')
        if '/' in s:
            parts = [float(p) for p in s.split('/')]
            return sum(parts) / len(parts)
        return float(s)

    buckets: dict[str, list[float]] = {k: [] for k in ('length', 'width', 'height', 'track', 'wheelbase')}
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    specs = json.loads(row.get('specs', '{}'))
                    dims = specs.get('Dimensions', {})
                    if not dims:
                        continue
                    L  = _extract_mm(dims.get('Length:', ''))
                    W  = _extract_mm(dims.get('Width:', ''))
                    H  = _extract_mm(dims.get('Height:', ''))
                    T  = _extract_mm(dims.get('Front/Rear Track:', ''))
                    WB = _extract_mm(dims.get('Wheelbase:', ''))
                    if None in (L, W, H, T, WB):
                        continue
                    # Plausible sedan range in mm
                    if not (3800 < L < 5200): continue
                    if not (1600 < W < 2000): continue
                    if not (1300 < H < 1650): continue
                    if not (1300 < T < 1700): continue
                    if not (2400 < WB < 3000): continue
                    buckets['length'].append(L)
                    buckets['width'].append(W)
                    buckets['height'].append(H)
                    buckets['track'].append(T)
                    buckets['wheelbase'].append(WB)
                except Exception:
                    continue
    except FileNotFoundError:
        return dict(_FALLBACK_DIMS)

    n = len(buckets['length'])
    if n == 0:
        return dict(_FALLBACK_DIMS)

    def _median(vals):
        s = sorted(vals)
        return s[len(s) // 2] / 1000.0  # mm → m

    result = {
        'length':      _median(buckets['length']),
        'width':       _median(buckets['width']),
        'height':      _median(buckets['height']),
        'track_width': _median(buckets['track']),
        'wheelbase':   _median(buckets['wheelbase']),
    }
    print(f"[haware] Spec CSV: {n} sedans → "
          f"L={result['length']:.3f} W={result['width']:.3f} H={result['height']:.3f} "
          f"TW={result['track_width']:.3f} WB={result['wheelbase']:.3f} m")
    return result


def build_car_template(dims: dict) -> np.ndarray:
    """Build a (24, 3) Apollo-24 keypoint template in metres.

    Coordinate system (Apollo-24 / CAR_POSE_24 convention):
      x: lateral — positive = vehicle left, negative = right
      y: height  — 0 = ground, positive = upward
      z: longitudinal — negative = front, positive = rear

    Keypoint height confidence:
      HIGH (measured): wheels (h=0), roof (h=dims.height)
      ESTIMATED: bumpers, lights, mirrors, plate
    """
    L  = dims['length']
    W  = dims['width']
    H  = dims['height']
    TW = dims.get('track_width', W * _TRACK_RATIO)
    WB = dims.get('wheelbase',   L * _WHEELBASE_RATIO)

    hl  = L  / 2   # half length
    hw  = W  / 2   # half width
    htw = TW / 2   # half track width  (real measurement)
    hwb = WB / 2   # half wheelbase    (real measurement)

    # Estimated heights for intermediate keypoints (not in spec sheets)
    H_BUMPER = 0.20   # front / rear bumper bottom
    H_CORNER = 0.50   # rear corners, rear plate
    H_LAMP   = 0.65   # head / tail lights
    H_MIRROR = 1.05   # side mirrors (sticks above door line)

    t = np.zeros((24, 3), dtype=np.float64)

    # ---- front upper area (roof front edge) ----
    t[0]  = [-hw * 0.70, H,        -hl * 0.55]   # front_up_right
    t[1]  = [ hw * 0.70, H,        -hl * 0.55]   # front_up_left

    # ---- headlights (middle height, front face) ----
    t[2]  = [-hw * 0.85, H_LAMP,   -hl]           # front_light_right
    t[3]  = [ hw * 0.85, H_LAMP,   -hl]           # front_light_left

    # ---- front bumper bottom ----
    t[4]  = [-hw,        H_BUMPER, -hl]            # front_low_right
    t[5]  = [ hw,        H_BUMPER, -hl]            # front_low_left

    # ---- roof centre ----
    t[6]  = [ hw * 0.85, H,         0.0]           # central_up_left

    # ---- wheels (real measured positions, h = 0) ----
    t[7]  = [ htw,       0.0,      -hwb]           # front_wheel_left
    t[8]  = [ htw,       0.0,       hwb]           # rear_wheel_left

    # ---- rear corners / rear area ----
    t[9]  = [ hw,        H_CORNER,  hl * 0.65]    # rear_corner_left
    t[10] = [ hw * 0.70, H,         hl * 0.40]    # rear_up_left
    t[11] = [-hw * 0.70, H,         hl * 0.40]    # rear_up_right
    t[12] = [ hw * 0.85, H_LAMP,    hl]            # rear_light_left
    t[13] = [-hw * 0.85, H_LAMP,    hl]            # rear_light_right
    t[14] = [ hw,        H_BUMPER,  hl]            # rear_low_left
    t[15] = [-hw,        H_BUMPER,  hl]            # rear_low_right

    # ---- roof centre right ----
    t[16] = [-hw * 0.85, H,         0.0]           # central_up_right
    t[17] = [-hw,        H_CORNER,  hl * 0.65]    # rear_corner_right

    # ---- wheels (real) ----
    t[18] = [-htw,       0.0,       hwb]           # rear_wheel_right
    t[19] = [-htw,       0.0,      -hwb]           # front_wheel_right

    # ---- rear licence plate ----
    t[20] = [ hw * 0.15, H_CORNER,  hl]            # rear_plate_left
    t[21] = [-hw * 0.15, H_CORNER,  hl]            # rear_plate_right

    # ---- side mirrors (stick out beyond body width) ----
    t[22] = [ hw * 1.05, H_MIRROR, -hl * 0.30]    # mirror_edge_left
    t[23] = [-hw * 1.05, H_MIRROR, -hl * 0.30]    # mirror_edge_right

    return t


@dataclass
class HawareResult:
    sat_coords:  Optional[tuple]         # (x, y) sat-image pixels; None on failure
    heading:     Optional[float]         # degrees, 0=East 90=North; None if ambiguous/failed
    confidence:  float                   # 0–1
    n_keypoints: int                     # number of keypoints used in fit
    status:      str                     # 'ok' | 'ambiguous_heading' | 'failed_insufficient_kp'
    p_sat:       dict = field(default_factory=dict)  # {kp_idx: (sat_x, sat_y)}


class HawareLocalizer:
    """Localize a single vehicle from its 24 Apollo-24 keypoints.

    Algorithm (doc §3B):
      1. For each confident keypoint, lift to sat coords using its template height h_i.
      2. If n < 2 detections, return failure.
      3. Fixed-scale 2D Procrustes (SVD) on template (x,z) vs observed sat (x,y).
      4. Output vehicle centre T and heading θ.
    """

    def __init__(self, g_engine, template_3d: np.ndarray, kp_conf: float = 0.2):
        self.g_engine = g_engine
        self.template = template_3d          # (24, 3) metres
        self.kp_conf  = kp_conf
        self._s       = g_engine.px_per_m   # satellite pixels per metre

    def localize(self, kp_24: np.ndarray) -> HawareResult:
        """Localize from a (24, 3) array of [x_img, y_img, conf] keypoints."""
        # Step 1 — project each confident keypoint to sat coords
        p_sat: dict[int, tuple] = {}
        for i in range(24):
            x_img, y_img, conf = float(kp_24[i, 0]), float(kp_24[i, 1]), float(kp_24[i, 2])
            if conf < self.kp_conf or (x_img == 0.0 and y_img == 0.0):
                continue
            h_i = float(self.template[i, 1])   # template height for this keypoint
            sat_xy = self.g_engine.cctv_to_sat(x_img, y_img, h=h_i)
            p_sat[i] = sat_xy

        n = len(p_sat)
        if n < 2:
            return HawareResult(
                sat_coords=None, heading=None, confidence=0.0,
                n_keypoints=n, status='failed_insufficient_kp', p_sat=p_sat,
            )

        # Step 2 — fixed-scale 2D Procrustes
        idx = list(p_sat.keys())
        s   = self._s

        # Template (x, z) columns scaled to sat pixels
        Q = self.template[idx][:, [0, 2]] * s      # (n, 2)
        P = np.array([p_sat[i] for i in idx])       # (n, 2)

        qb = Q.mean(0)
        pb = P.mean(0)
        Hc = (Q - qb).T @ (P - pb)                 # 2×2 cross-covariance
        U, _, Vt = np.linalg.svd(Hc)
        det_sign = float(np.sign(np.linalg.det(Vt.T @ U.T)))
        R = Vt.T @ np.diag([1.0, det_sign]) @ U.T  # 2×2 rotation (no reflection)
        T_sat = pb - R @ qb                         # vehicle centre in sat pixels

        # Step 3 — heading
        # Vehicle forward = template −z = (0,−1) in (x,z) space
        # After rotation: forward_sat = R @ [0,−1]^T = (−R[0,1], −R[1,1])
        # Project convention (_sat_heading in wheel_localization.py): atan2(−dy, dx)
        heading = math.degrees(math.atan2(R[1, 1], -R[0, 1])) % 360.0

        # Step 4 — ambiguity: all detected kps on the same longitudinal side?
        z_vals = self.template[idx, 2]
        symmetric = bool(np.all(z_vals >= 0) or np.all(z_vals <= 0))
        status = 'ambiguous_heading' if symmetric else 'ok'
        if symmetric:
            heading = None

        # Step 5 — confidence heuristic
        P_pred = (Q - qb) @ R.T + pb
        rms    = float(np.sqrt(np.mean(np.sum((P - P_pred) ** 2, axis=1))))
        conf   = min(1.0, n / 8.0) * max(0.0, 1.0 - rms / (5.0 * s))

        return HawareResult(
            sat_coords=tuple(T_sat),
            heading=heading,
            confidence=conf,
            n_keypoints=n,
            status=status,
            p_sat=p_sat,
        )


# ---------------------------------------------------------------------------
# Bbox IoU matching utilities (Method B: assign YOLO track IDs to h-aware detections)
# ---------------------------------------------------------------------------

def kp_bbox_xyxy(kp_data: np.ndarray, conf_thresh: float = 0.2) -> Optional[tuple]:
    """Compute tight xyxy bbox from a (24, 3) keypoint array [x, y, conf].

    Returns (x1, y1, x2, y2) in image pixels, or None if fewer than 2 visible keypoints.
    """
    vis = kp_data[kp_data[:, 2] > conf_thresh]
    if len(vis) < 2:
        return None
    return (float(vis[:, 0].min()), float(vis[:, 1].min()),
            float(vis[:, 0].max()), float(vis[:, 1].max()))


def match_by_bbox_iou(
    pifpaf_boxes: list,
    yolo_boxes: list,
    yolo_tids: list,
    iou_threshold: float = 0.3,
) -> list:
    """Match PifPaf detections to YOLO tracks by bbox IoU.

    Args:
        pifpaf_boxes: list of (x1,y1,x2,y2) or None, one per PifPaf detection.
        yolo_boxes:   list of (x1,y1,x2,y2), one per YOLO detection.
        yolo_tids:    list of track IDs (int or None), same length as yolo_boxes.
        iou_threshold: minimum IoU to accept a match.

    Returns:
        list of int|None, same length as pifpaf_boxes — the matched YOLO track ID,
        or None if no YOLO box overlaps above the threshold.
    """
    result = []
    for pb in pifpaf_boxes:
        if pb is None or not yolo_boxes:
            result.append(None)
            continue

        px1, py1, px2, py2 = pb
        pa = max(0.0, px2 - px1) * max(0.0, py2 - py1)

        best_iou = 0.0
        best_tid = None
        for (yx1, yy1, yx2, yy2), tid in zip(yolo_boxes, yolo_tids):
            iw = max(0.0, min(px2, yx2) - max(px1, yx1))
            ih = max(0.0, min(py2, yy2) - max(py1, yy1))
            inter = iw * ih
            ya = max(0.0, yx2 - yx1) * max(0.0, yy2 - yy1)
            union = pa + ya - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_tid = tid

        result.append(best_tid if best_iou >= iou_threshold else None)
    return result
