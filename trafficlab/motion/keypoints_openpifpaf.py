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
    H_ROOF   = 1.65   # roof-line keypoints (front/central/rear up) — fixed,
                       # not tied to dims.height (roof peak sits higher than
                       # the vehicle's overall body-height spec for most sedans)

    t = np.zeros((24, 3), dtype=np.float64)

    # ---- front upper area (roof front edge) ----
    t[0]  = [-hw * 0.70, H_ROOF,   -hl * 0.55]   # front_up_right
    t[1]  = [ hw * 0.70, H_ROOF,   -hl * 0.55]   # front_up_left

    # ---- headlights (middle height, front face) ----
    t[2]  = [-hw * 0.85, H_LAMP,   -hl]           # front_light_right
    t[3]  = [ hw * 0.85, H_LAMP,   -hl]           # front_light_left

    # ---- front bumper bottom ----
    t[4]  = [-hw,        H_BUMPER, -hl]            # front_low_right
    t[5]  = [ hw,        H_BUMPER, -hl]            # front_low_left

    # ---- roof centre ----
    t[6]  = [ hw * 0.85, H_ROOF,    0.0]           # central_up_left

    # ---- wheels (real measured positions, h = 0) ----
    t[7]  = [ htw,       0.0,      -hwb]           # front_wheel_left
    t[8]  = [ htw,       0.0,       hwb]           # rear_wheel_left

    # ---- rear corners / rear area ----
    t[9]  = [ hw,        H_CORNER,  hl * 0.65]    # rear_corner_left
    t[10] = [ hw * 0.70, H_ROOF,    hl * 0.40]    # rear_up_left
    t[11] = [-hw * 0.70, H_ROOF,    hl * 0.40]    # rear_up_right
    t[12] = [ hw * 0.85, H_LAMP,    hl]            # rear_light_left
    t[13] = [-hw * 0.85, H_LAMP,    hl]            # rear_light_right
    t[14] = [ hw,        H_BUMPER,  hl]            # rear_low_left
    t[15] = [-hw,        H_BUMPER,  hl]            # rear_low_right

    # ---- roof centre right ----
    t[16] = [-hw * 0.85, H_ROOF,    0.0]           # central_up_right
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


# Apollo-24 keypoint names, indexed to match build_car_template's rows.
KP_NAMES = [
    'front_up_right',    'front_up_left',
    'front_light_right', 'front_light_left',
    'front_low_right',   'front_low_left',
    'central_up_left',
    'front_wheel_left',  'rear_wheel_left',
    'rear_corner_left',  'rear_up_left',      'rear_up_right',
    'rear_light_left',   'rear_light_right',
    'rear_low_left',     'rear_low_right',
    'central_up_right',  'rear_corner_right',
    'rear_wheel_right',  'front_wheel_right',
    'rear_plate_left',   'rear_plate_right',
    'mirror_edge_left',  'mirror_edge_right',
]


# ---------------------------------------------------------------------------
# Keypoint pair tables for the geometric centerline-intersection localizer
# (localize_reprojection). Indices per build_car_template's Apollo-24 layout.
# ---------------------------------------------------------------------------

# Left-right symmetric pairs: same template z (and y), mirrored x. Every one
# of the 24 keypoints belongs to exactly one such pair.
_LR_PAIRS = [
    (0, 1),    # front_up_right     / front_up_left
    (2, 3),    # front_light_right  / front_light_left
    (4, 5),    # front_low_right    / front_low_left
    (6, 16),   # central_up_left    / central_up_right
    (7, 19),   # front_wheel_left   / front_wheel_right
    (8, 18),   # rear_wheel_left    / rear_wheel_right
    (9, 17),   # rear_corner_left   / rear_corner_right
    (10, 11),  # rear_up_left       / rear_up_right
    (12, 13),  # rear_light_left    / rear_light_right
    (14, 15),  # rear_low_left      / rear_low_right
    (20, 21),  # rear_plate_left    / rear_plate_right
    (22, 23),  # mirror_edge_left   / mirror_edge_right
]

# Same-side front/rear wheel pairs: only these two are symmetric about the
# vehicle's mid-wheelbase (z sums to 0), so their connecting-line midpoint
# falls exactly on the true lateral centerline. (front_idx, rear_idx).
_FR_WHEEL_PAIRS = [
    (7, 8),    # front_wheel_left  / rear_wheel_left
    (19, 18),  # front_wheel_right / rear_wheel_right
]

# Excluded from Method 2's "assumed centerline" position averaging (not from
# the heading consistency vote below) — lights, mirrors and rear corners sit
# at extremities and are judged less reliable position-wise for this average.
_EXCLUDE_FROM_MIDPOINT = frozenset([2, 3, 12, 13, 22, 23, 9, 17])


@dataclass
class HawareResult:
    sat_coords:  Optional[tuple]         # (x, y) sat-image pixels; None on failure
    heading:     Optional[float]         # degrees, 0=East 90=North; None if ambiguous/failed
    confidence:  float                   # 0–1
    n_keypoints: int                     # number of keypoints used in fit
    status:      str                     # 'ok' | 'failed_insufficient_kp'
    p_sat:       dict = field(default_factory=dict)  # {kp_idx: (sat_x, sat_y)}
    method:      Optional[int] = None    # localize_reprojection only: 1/2/3 (see its docstring); None for localize()


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
        # Convention matches trafficlab/motion/kinematics.py (the production
        # pipeline's heading source) and trafficlab/visualization/sat_renderer.py's
        # arrow drawing: plain atan2(dy, dx), no negation — NOT the
        # atan2(−dy, dx) convention documented in wheel_localization.py, which
        # doesn't actually match how the rest of the system draws/computes it.
        heading = math.degrees(math.atan2(-R[1, 1], -R[0, 1])) % 360.0

        # Step 4 — confidence heuristic
        P_pred = (Q - qb) @ R.T + pb
        rms    = float(np.sqrt(np.mean(np.sum((P - P_pred) ** 2, axis=1))))
        conf   = min(1.0, n / 8.0) * max(0.0, 1.0 - rms / (5.0 * s))

        return HawareResult(
            sat_coords=tuple(T_sat),
            heading=heading,
            confidence=conf,
            n_keypoints=n,
            status='ok',
            p_sat=p_sat,
        )

    def localize_reprojection(self, kp_24: np.ndarray) -> HawareResult:
        """Geometric centerline-intersection localizer.

        Trusts PifPaf's keypoint pixel positions and labels directly (each
        confident keypoint is still lifted to sat coords via cctv_to_sat +
        its own template height, same as localize(), but the pose is then
        built from pairwise keypoint geometry instead of an SVD fit over all
        points at once). Branches on which keypoint pairs are visible:

          Method 1 — a left-right symmetric pair AND a same-side front/rear
            wheel pair (_LR_PAIRS / _FR_WHEEL_PAIRS) are both visible. Each
            pair's perpendicular bisector is a line through the true vehicle
            centre (one runs along the heading axis, the other across it);
            their intersection is the centre. Heading is the front->rear
            wheel vector (unambiguous — front/rear are named keypoints).

          Method 2 — only one of the two pair types is visible. A single
            *named* pair still fully determines a proper rotation R (two
            labeled points pin down a 2D rotation exactly — matching the
            template's inter-point vector to the observed one leaves no
            reflection freedom), which gives one centerline plus a fully
            resolved heading. Every other confident keypoint (including the
            cue pair's own two points) is shifted along that centerline's
            own axis by its known template offset on the *other* body axis,
            to land on the perpendicular centerline; those shifted points
            are averaged into a second, perpendicular centerline. The two
            centerlines' intersection is the vehicle centre.

          Method 3 — neither pair type is visible: not enough independent
            geometric constraints to fix a pose -> 'failed_insufficient_kp'.

        Tie-breaking when multiple L-R or wheel pairs are simultaneously
        visible is not yet decided — this uses the first match in
        _LR_PAIRS / _FR_WHEEL_PAIRS order (see docs/localization-methods.md).
        Confidence/status semantics beyond ok/failed are also not yet
        decided; confidence is a flat placeholder for now.
        """
        p_sat: dict[int, tuple] = {}
        for i in range(24):
            x_img, y_img, conf = float(kp_24[i, 0]), float(kp_24[i, 1]), float(kp_24[i, 2])
            if conf < self.kp_conf or (x_img == 0.0 and y_img == 0.0):
                continue
            h_i = float(self.template[i, 1])
            p_sat[i] = self.g_engine.cctv_to_sat(x_img, y_img, h=h_i)

        n = len(p_sat)
        if n < 2:
            return HawareResult(
                sat_coords=None, heading=None, confidence=0.0,
                n_keypoints=n, status='failed_insufficient_kp', p_sat=p_sat, method=3,
            )

        s = self._s
        P = {i: np.array(p_sat[i], dtype=np.float64) for i in p_sat}

        def _perp(v):
            return np.array([-v[1], v[0]])

        def _rotation_from_vectors(v_body, v_world):
            """Proper 2D rotation R with R @ v_body pointing along v_world
            (angle-matching only — magnitude/scale mismatch is ignored, so
            this is robust to per-point pixel noise on the vector lengths).
            """
            ang = math.atan2(v_world[1], v_world[0]) - math.atan2(v_body[1], v_body[0])
            c, sn = math.cos(ang), math.sin(ang)
            return np.array([[c, -sn], [sn, c]])

        def _intersect(p1, d1, p2, d2):
            A = np.column_stack([d1, -d2])
            det = np.linalg.det(A)
            if abs(det) < 1e-9:
                return None
            t = np.linalg.solve(A, p2 - p1)
            return p1 + t[0] * d1

        def _heading_from_forward(fwd):
            # atan2(dy, dx), no negation — see the matching comment in
            # localize() for why (matches kinematics.py / sat_renderer.py).
            return math.degrees(math.atan2(fwd[1], fwd[0])) % 360.0

        def _avg_forward(fwd1, fwd2):
            """Circular mean of two forward directions: normalize each to a
            unit vector and sum, so 350° vs 10° averages to ~0°/360° instead
            of the wrong 180° an arithmetic mean of the angle numbers gives."""
            u1 = fwd1 / np.linalg.norm(fwd1)
            u2 = fwd2 / np.linalg.norm(fwd2)
            return u1 + u2

        def _resolve_lr_forward(lr_a, lr_b, R):
            """An L-R pair's own template z only tells us that pair's own
            longitudinal slot, not which way the *whole car* faces — that
            requires trusting this one pair's assumed front/rear identity,
            which fails whenever the detector's front/rear labeling for the
            whole instance is swapped (confirmed on real data: id=248,
            frame76 of test21-6 — every visible keypoint's template z sign
            came out inverted relative to its actual position once resolved
            through this pair alone).

            Cross-check instead: the template-assumed "+z/rear" world
            direction is rear_dir = R @ (0,1). For every other confident
            keypoint with a nonzero template z, check whether it actually
            sits on the side rear_dir points to (z>0) or the opposite side
            (z<0), and tally agreement vs disagreement. Flip rear_dir if the
            majority disagrees. This only needs the *sign* of a dot product,
            so it stays a simple linear check even with many candidate
            points (per user request, not weighted/more elaborate than that).

            Deliberately NOT excluding light/mirror/rear_corner keypoints
            here — that exclusion (_EXCLUDE_FROM_MIDPOINT) is scoped to the
            Method 2 position average only, not this vote.
            """
            rear_dir = R @ np.array([0.0, 1.0])
            agree = disagree = 0
            for i, p in P.items():
                if i in (lr_a, lr_b):
                    continue
                z_i = self.template[i, 2]
                if z_i == 0.0:
                    continue
                proj = np.dot(p - mid_world, rear_dir)
                if (proj > 0) == (z_i > 0):
                    agree += 1
                else:
                    disagree += 1
            if disagree > agree:
                rear_dir = -rear_dir
            return -rear_dir  # forward = opposite of the resolved "rear" direction

        lr_pair = next(((a, b) for a, b in _LR_PAIRS if a in p_sat and b in p_sat), None)
        fr_pair = next(((f, r) for f, r in _FR_WHEEL_PAIRS if f in p_sat and r in p_sat), None)

        if lr_pair is not None and fr_pair is not None:
            # ---- Method 1: both pair types visible ----
            a, b = lr_pair
            mid_lr, lr_dir = (P[a] + P[b]) / 2.0, _perp(P[b] - P[a])
            f, r = fr_pair
            mid_fr, fr_dir = (P[f] + P[r]) / 2.0, _perp(P[r] - P[f])

            center = _intersect(mid_lr, lr_dir, mid_fr, fr_dir)
            if center is None:
                return HawareResult(None, None, 0.0, n, 'failed_insufficient_kp', p_sat, method=1)

            mid_world = mid_lr
            lr_body = self.template[[a, b]][:, [0, 2]]
            R_lr = _rotation_from_vectors(lr_body[1] - lr_body[0], P[b] - P[a])
            bisector_forward = _resolve_lr_forward(a, b, R_lr)
            wheel_forward = P[f] - P[r]
            heading = _heading_from_forward(_avg_forward(bisector_forward, wheel_forward))
            conf = 0.6
            method = 1

        elif lr_pair is not None or fr_pair is not None:
            # ---- Method 2: only one pair type visible ----
            if lr_pair is not None:
                a, b = lr_pair
                offset_axis = 2   # shift others along template z (longitudinal)
            else:
                a, b = fr_pair
                offset_axis = 0   # shift others along template x (lateral)

            cue_body = self.template[[a, b]][:, [0, 2]]   # (2,2) metres, columns (x,z)
            v_body  = cue_body[1] - cue_body[0]
            v_world = P[b] - P[a]
            R = _rotation_from_vectors(v_body, v_world)

            mid_world = (P[a] + P[b]) / 2.0
            line1_dir = _perp(v_world)                     # this pair's own centerline
            shift_body = np.array([0.0, 1.0]) if offset_axis == 2 else np.array([1.0, 0.0])
            shift_world = R @ shift_body                    # correctly-signed, from R

            cross_points = [
                P[i] - float(self.template[i, offset_axis]) * s * shift_world
                for i in p_sat if i not in _EXCLUDE_FROM_MIDPOINT
                # includes the cue's own two points if not excluded — see write-up
            ]
            if not cross_points:
                return HawareResult(None, None, 0.0, n, 'failed_insufficient_kp', p_sat, method=2)
            line2_point = np.mean(cross_points, axis=0)
            line2_dir = v_world                             # perpendicular to line1_dir by construction

            center = _intersect(mid_world, line1_dir, line2_point, line2_dir)
            if center is None:
                return HawareResult(None, None, 0.0, n, 'failed_insufficient_kp', p_sat, method=2)

            if lr_pair is not None:
                forward = _resolve_lr_forward(a, b, R)
            else:
                front_idx, rear_idx = fr_pair
                forward = P[front_idx] - P[rear_idx]
            heading = _heading_from_forward(forward)
            conf = 0.4
            method = 2

        else:
            # ---- Method 3: neither pair type visible ----
            return HawareResult(None, None, 0.0, n, 'failed_insufficient_kp', p_sat, method=3)

        return HawareResult(
            sat_coords=tuple(center),
            heading=heading,
            confidence=conf,
            n_keypoints=n,
            status='ok',
            p_sat=p_sat,
            method=method,
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
