"""Method 7 core math: N-point least-squares camera parallax calibration.

See docs/height-correction-algorithm-survey.md section 7. Generalizes the
existing two-object head/foot manual calibration (pars_stage.py, which
solves a closed-form line intersection for exactly 2 subjects) to N >= 2
reference objects of known height, solved jointly via nonlinear least
squares. Reuses the existing ground homography via
GProjection.cctv_to_sat(..., h=0.0) -- no GUI dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from trafficlab.projection.g_projection import GProjection


@dataclass
class ReferencePoint:
    id: int
    frame_index: int
    head_px: tuple[float, float]
    foot_px: tuple[float, float] | None
    height_m: float
    # Pre-projected ground position (sat-plane px), used instead of foot_px
    # when the true ground position is already known exactly (e.g. from a
    # rigid template fit) rather than clicked on the CCTV frame -- avoids a
    # sat->cctv->sat round trip, which is not numerically exact (see
    # docs/car-template-calibration-tool-guide.md).
    foot_sat: tuple[float, float] | None = None


def _closed_form_two_point(F1, G1, h1, F2, G2, h2):
    """Line-intersection + similar-triangles solve for exactly two reference
    points (same math as pars_stage.py's _find_intersection/_on_compute),
    used only to seed the least-squares initial guess. Returns
    (cam_sat_x, cam_sat_y, z_cam) or None if degenerate."""

    def find_intersection(p1, p2, p3, p4):
        p1, p2, p3, p4 = (np.array(p, dtype=np.float64) for p in (p1, p2, p3, p4))
        d1 = p2 - p1
        d2 = p4 - p3
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-9:
            return None
        t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
        return p1 + t * d1

    cam_sat = find_intersection(G1, F1, G2, F2)
    if cam_sat is None:
        return None

    def dist(a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    d_app_1, d_true_1 = dist(cam_sat, G1), dist(cam_sat, F1)
    d_app_2, d_true_2 = dist(cam_sat, G2), dist(cam_sat, F2)
    if d_app_1 < 1e-6 or d_app_2 < 1e-6:
        return None
    ratio_1 = d_true_1 / d_app_1
    ratio_2 = d_true_2 / d_app_2
    if abs(1 - ratio_1) < 1e-6 or abs(1 - ratio_2) < 1e-6:
        return None
    z1 = h1 / (1 - ratio_1)
    z2 = h2 / (1 - ratio_2)
    z_cam = (z1 + z2) / 2.0
    if not np.isfinite(z_cam) or z_cam <= 0:
        return None
    return float(cam_sat[0]), float(cam_sat[1]), float(z_cam)


def _empty_result(message: str, n_references: int) -> dict:
    return {
        "success": False,
        "message": message,
        "cam_sat_xy": None,
        "z_cam_meters": None,
        "rmse_m": None,
        "n_references": n_references,
        "per_reference": [],
    }


def calibrate(references: list[ReferencePoint], g_projection: GProjection) -> dict:
    """Solve for the camera's sat-plane position (cam_sat_xy) and height
    (z_cam_meters) from N >= 2 reference objects of known height.

    For each reference, the head/foot image pixels are pushed through the
    existing (undistort + ground homography) pipeline at h=0 to get their
    apparent ground positions G_i (head) and F_i (foot, == ground truth
    since its real height is 0). The unknowns O=(Ox, Oy)=cam_sat_xy and
    z_cam are then fit jointly across all references against:

        F_i - O = (G_i - O) * (z_cam - h_i) / z_cam

    which is the same homothety model as GProjection.parallax_correct_ground_to_real,
    solved in reverse.
    """
    if len(references) < 2:
        return _empty_result(f"至少需要 2 個參考點，目前只有 {len(references)} 個。", len(references))

    G = np.array(
        [g_projection.cctv_to_sat(ref.head_px[0], ref.head_px[1], h=0.0) for ref in references],
        dtype=np.float64,
    )
    F = np.array(
        [
            ref.foot_sat if ref.foot_sat is not None
            else g_projection.cctv_to_sat(ref.foot_px[0], ref.foot_px[1], h=0.0)
            for ref in references
        ],
        dtype=np.float64,
    )
    H = np.array([ref.height_m for ref in references], dtype=np.float64)
    max_h = float(np.max(H))

    # Seed the least-squares solve with a closed-form two-point solve on the
    # pair of references with the largest foot-point separation (most
    # laterally spread out -- mirrors pars_stage.py's guidance to pick
    # subjects "further apart" for numerical stability).
    best_pair, best_dist = None, -1.0
    for i in range(len(references)):
        for j in range(i + 1, len(references)):
            d = float(np.linalg.norm(F[i] - F[j]))
            if d > best_dist:
                best_dist, best_pair = d, (i, j)
    i, j = best_pair
    init = _closed_form_two_point(F[i], G[i], H[i], F[j], G[j], H[j])

    if init is None:
        seed_z = getattr(g_projection, "z_cam", None) or 10.0
        if seed_z <= max_h:
            seed_z = max_h + 1.0
        mean_f = F.mean(axis=0)
        init = (float(mean_f[0]), float(mean_f[1]), float(seed_z))

    x0 = np.array(init, dtype=np.float64)

    def residuals(params):
        ox, oy, z = params
        origin = np.array([ox, oy])
        factor = (z - H) / z
        pred_f = origin + (G - origin) * factor[:, None]
        return (F - pred_f).ravel()

    lower = [-np.inf, -np.inf, max_h + 1e-3]
    upper = [np.inf, np.inf, np.inf]
    try:
        result = least_squares(residuals, x0, bounds=(lower, upper))
    except Exception as e:
        return _empty_result(f"最小二乘求解失敗：{type(e).__name__}: {e}", len(references))

    ox, oy, z_cam = result.x
    per_ref_resid_m = np.linalg.norm(residuals(result.x).reshape(-1, 2), axis=1)
    rmse = float(np.sqrt(np.mean(per_ref_resid_m ** 2)))

    per_reference = [
        {
            "id": ref.id,
            "frame_index": ref.frame_index,
            "height_m": ref.height_m,
            "residual_m": float(per_ref_resid_m[k]),
        }
        for k, ref in enumerate(references)
    ]

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "cam_sat_xy": (float(ox), float(oy)),
        "z_cam_meters": float(z_cam),
        "rmse_m": rmse,
        "n_references": len(references),
        "per_reference": per_reference,
    }
