"""Rigid placement of a car keypoint template onto the sat plane from
exactly 2 wheel ground-contact correspondences.

See docs/car-template-calibration-tool-guide.md and
docs/height-correction-algorithm-survey.md section 4. Given 2 tire
ground-contact points clicked on a CCTV frame (converted to sat-plane
coordinates via the existing ground homography, h=0 -- exact, no parallax
ambiguity), this fits a 2D similarity transform (rotation + scale +
translation) that maps a CAD keypoint template's body-frame ground
footprint onto those 2 points exactly, then applies that transform to
every keypoint in the template. Each transformed keypoint's true ground
position, combined with its real height (scaled by the same solved ratio)
and its raw OpenPifPaf pixel detection (from a replay JSON's kp_cctv),
becomes one trafficlab.projection.reference_point_calibration.ReferencePoint
for the existing least-squares solve.

Deliberately has no dependency on trafficlab.motion.* -- trafficlab.projection
does not currently depend on trafficlab.motion, and this module keeps that
direction (kp_names comes from the template JSON itself, not from the
production Apollo-24 KP_NAMES constant, so alternative templates are
naturally supported).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from trafficlab.projection.reference_point_calibration import ReferencePoint

# Sentinels observed in real replay JSON kp_cctv arrays for "not detected in
# this instance" -- PifPaf's own convention, not something this repo defines.
_UNDETECTED_XY = {(0.0, 0.0), (0.0, -4.0)}


@dataclass(frozen=True)
class CarTemplate:
    kp_names: list[str]
    xhz: np.ndarray  # (N, 3) [x, h, z] meters, x=lateral(+left), h=height(0=ground), z=longitudinal(-front/+rear)
    source_path: str | None = None

    @property
    def ground_xz(self) -> np.ndarray:
        """(N, 2) body-frame ground-plane footprint (x, z), meters."""
        return self.xhz[:, [0, 2]]

    @property
    def heights_m(self) -> np.ndarray:
        """(N,) real height above ground, meters."""
        return self.xhz[:, 1]

    def index_of(self, name: str) -> int:
        return self.kp_names.index(name)


def load_car_template(path: str) -> CarTemplate:
    with open(path) as f:
        data = json.load(f)
    kp_names = list(data["kp_names"])
    xhz = np.asarray(data["template"], dtype=np.float64)
    if xhz.shape != (len(kp_names), 3):
        raise ValueError(
            f"template JSON '{path}' malformed: kp_names has {len(kp_names)} entries "
            f"but template array has shape {xhz.shape} (expected ({len(kp_names)}, 3))"
        )
    return CarTemplate(kp_names=kp_names, xhz=xhz, source_path=path)


def _rotation_matrix(theta_rad: float) -> np.ndarray:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, -s], [s, c]])


def _body_xz_to_sat_frame(points_xz: np.ndarray) -> np.ndarray:
    """Mirror body-frame (x, z) into a frame with the same handedness as the
    sat image, so a *proper* rotation can then place it.

    The body frame (x=+left, z=+rear, read as a top-down 2D frame after
    dropping h) and the sat image frame (x=+right, y=+down) have OPPOSITE
    handedness. A proper rotation (det=+1) cannot map one onto the other, so
    fitting body (x, z) directly against sat pixels places the whole vehicle
    mirrored -- the template's +x ("vehicle left") side lands on the
    vehicle's physical right. The same mismatch is documented in
    trafficlab/motion/keypoints_openpifpaf.py's localize_wheel_pair (see its
    "NOTE the '+' here, not '-'" comment), which compensates with a sign flip
    on its lateral shift; here it is handled once, explicitly, on the way
    into the similarity transform.

    Which axis gets mirrored (x or z) does not change the fitted placement:
    the two choices differ by a 180 deg rotation, which the 2-anchor fit
    absorbs into theta. What matters is that a reflection is applied at all.
    """
    return np.asarray(points_xz, dtype=np.float64) * np.array([-1.0, 1.0])


@dataclass(frozen=True)
class TemplatePose:
    """A placed template's pose in sat-plane space. Always rebuilt from
    scratch on every nudge (never accumulated as a matrix product), so
    repeated rotate/translate nudges cannot drift or induce an unintended
    translation."""
    theta_rad: float
    t_sat: tuple[float, float]  # sat position of the template's body origin (0,0)
    scale: float  # sat px per template meter, solved per-placement (NOT the global px_per_m)

    def rotated(self, d_theta_rad: float) -> "TemplatePose":
        """Rotate in place around the body origin (t_sat) -- the vehicle
        spins around its own center, not around wherever the 2 anchors
        happened to be clicked."""
        return TemplatePose(self.theta_rad + d_theta_rad, self.t_sat, self.scale)

    def translated(self, dx_sat: float, dy_sat: float) -> "TemplatePose":
        return TemplatePose(
            self.theta_rad, (self.t_sat[0] + dx_sat, self.t_sat[1] + dy_sat), self.scale
        )

    def flipped_180(self) -> "TemplatePose":
        return TemplatePose(self.theta_rad + math.pi, self.t_sat, self.scale)

    def matrix(self) -> np.ndarray:
        return _rotation_matrix(self.theta_rad)


def fit_two_point_pose(
    template: CarTemplate, anchors: list[tuple[str, tuple[float, float]]]
) -> TemplatePose:
    """Exact closed-form 2D similarity transform (rotation + scale +
    translation) from exactly 2 named-keypoint correspondences. anchors is
    [(kp_name, sat_xy), (kp_name, sat_xy)] -- 2 distinct template keypoint
    names (typically 2 of the 4 wheel ground-contact points) paired with
    their sat-plane click positions. With exactly 2 points this system is
    fully (not over-) determined: the fitted template lands exactly on both
    anchors, no least-squares needed.
    """
    if len(anchors) != 2:
        raise ValueError(f"fit_two_point_pose requires exactly 2 anchors, got {len(anchors)}")
    (name0, p0), (name1, p1) = anchors
    if name0 == name1:
        raise ValueError(f"the 2 anchors must reference distinct keypoints, both got '{name0}'")

    Q = _body_xz_to_sat_frame(
        template.ground_xz[[template.index_of(name0), template.index_of(name1)]]
    )
    P = np.array([p0, p1], dtype=np.float64)

    v_body = Q[1] - Q[0]
    v_world = P[1] - P[0]
    norm_body = float(np.linalg.norm(v_body))
    norm_world = float(np.linalg.norm(v_world))
    if norm_body < 1e-6:
        raise ValueError(
            f"template keypoints '{name0}'/'{name1}' are coincident in the body frame "
            "-- pick 2 distinct wheel keypoints"
        )
    if norm_world < 1e-6:
        raise ValueError(
            "the 2 clicked points are (almost) coincident on the sat plane -- "
            "click 2 clearly separate tire ground-contact points"
        )

    theta = math.atan2(v_world[1], v_world[0]) - math.atan2(v_body[1], v_body[0])
    scale = norm_world / norm_body
    R = _rotation_matrix(theta)
    t_sat = P[0] - scale * (R @ Q[0])
    return TemplatePose(theta_rad=theta, t_sat=(float(t_sat[0]), float(t_sat[1])), scale=scale)


def apply_pose_to_points(pose: TemplatePose, points_xz: np.ndarray) -> np.ndarray:
    """(N, 2) sat-plane position for arbitrary body-frame (x, z) points
    (meters) -- e.g. a body-outline bounding box, not just the template's
    own keypoints. apply_pose_to_template is the keypoints-only case of
    this."""
    R = pose.matrix()
    t_sat = np.array(pose.t_sat, dtype=np.float64)
    return (_body_xz_to_sat_frame(points_xz) * pose.scale) @ R.T + t_sat


def apply_pose_to_template(template: CarTemplate, pose: TemplatePose) -> np.ndarray:
    """(N, 2) sat-plane ground-footprint position for every template keypoint."""
    return apply_pose_to_points(pose, template.ground_xz)


def scaled_heights_m(template: CarTemplate, pose: TemplatePose, nominal_px_per_m: float) -> np.ndarray:
    """(N,) real height per keypoint, scaled by how much bigger/smaller this
    placed instance is than the template's own nominal size (isotropic
    scaling assumption -- the same ratio used for the ground footprint is
    applied to height)."""
    return template.heights_m * (pose.scale / nominal_px_per_m)


def anchor_fit_report(
    pose: TemplatePose,
    nominal_px_per_m: float,
    *,
    warn_band: tuple[float, float] = (0.85, 1.15),
    hard_band: tuple[float, float] = (0.5, 2.0),
) -> dict:
    """Diagnostic on the solved per-placement scale vs. the location's
    nominal px_per_m -- a large deviation usually means click imprecision,
    a wrong wheel-label pairing, or a template that doesn't match the real
    vehicle's size."""
    scale_ratio = pose.scale / nominal_px_per_m
    ok = hard_band[0] <= scale_ratio <= hard_band[1]
    warning = None
    if not ok:
        warning = (
            f"解出的縮放比例 {scale_ratio:.2f} 遠離 1.0（超出 {hard_band} 範圍）——"
            "很可能是兩個輪子標籤選錯、點擊位置錯誤，或幾乎點在同一點上。"
        )
    elif not (warn_band[0] <= scale_ratio <= warn_band[1]):
        warning = (
            f"解出的縮放比例 {scale_ratio:.2f} 偏離 1.0（超出 {warn_band} 範圍）——"
            "請確認點擊位置精準，或這台車的實際大小跟所選模板差異較大。"
        )
    return {
        "scale_px_per_m": pose.scale,
        "nominal_px_per_m": nominal_px_per_m,
        "scale_ratio": scale_ratio,
        "ok": ok,
        "warning": warning,
    }


def build_reference_points(
    template: CarTemplate,
    pose: TemplatePose,
    placed_sat_xy: np.ndarray,
    kp_cctv: Sequence[Sequence[float]],
    frame_index: int,
    nominal_px_per_m: float,
    *,
    kp_conf: float = 0.2,
    min_height_m: float = 0.0,
    start_id: int = 0,
    exclude_indices: frozenset[int] = frozenset(),
) -> list[ReferencePoint]:
    """Pair the placed template's true per-keypoint ground position/height
    with that same car's actual OpenPifPaf pixel detections (kp_cctv, in
    template.kp_names order) to produce a batch of reference-point
    observations. A keypoint only becomes a reference if it was actually
    detected in the replay JSON (confidence >= kp_conf and not one of the
    "not detected" sentinels PifPaf stores)."""
    heights = scaled_heights_m(template, pose, nominal_px_per_m)
    refs: list[ReferencePoint] = []
    next_id = start_id
    for i, kp in enumerate(kp_cctv):
        if i in exclude_indices:
            continue
        x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
        if conf < kp_conf or (x, y) in _UNDETECTED_XY:
            continue
        h_i = float(heights[i])
        if h_i < min_height_m:
            continue
        refs.append(ReferencePoint(
            id=next_id,
            frame_index=frame_index,
            head_px=(x, y),
            foot_px=None,
            height_m=h_i,
            foot_sat=(float(placed_sat_xy[i][0]), float(placed_sat_xy[i][1])),
        ))
        next_id += 1
    return refs


def self_test() -> None:
    """Synthetic round-trip test -- no replay JSON/G_projection needed.
    Fabricates a known pose, projects 2 wheel keypoints through it to get
    synthetic "clicks", and asserts fit_two_point_pose recovers the exact
    same pose and apply_pose_to_template reproduces every keypoint's true
    position."""
    kp_names = ["front_wheel_center_left", "rear_wheel_center_left",
                "front_wheel_center_right", "rear_wheel_center_right", "roof_center"]
    xhz = np.array([
        [0.85, 0.34, -1.27],
        [0.85, 0.34, 1.27],
        [-0.85, 0.34, -1.27],
        [-0.85, 0.34, 1.27],
        [0.0, 1.45, 0.0],
    ])
    template = CarTemplate(kp_names=kp_names, xhz=xhz)

    known_pose = TemplatePose(theta_rad=0.4, t_sat=(1000.0, -250.0), scale=41.3)
    known_sat_xy = apply_pose_to_template(template, known_pose)

    anchors = [
        ("front_wheel_center_left", tuple(known_sat_xy[0])),
        ("rear_wheel_center_right", tuple(known_sat_xy[3])),
    ]
    fitted_pose = fit_two_point_pose(template, anchors)
    assert abs(fitted_pose.theta_rad - known_pose.theta_rad) < 1e-9, fitted_pose
    assert abs(fitted_pose.scale - known_pose.scale) < 1e-9, fitted_pose
    assert max(abs(fitted_pose.t_sat[0] - known_pose.t_sat[0]),
               abs(fitted_pose.t_sat[1] - known_pose.t_sat[1])) < 1e-6, fitted_pose

    fitted_sat_xy = apply_pose_to_template(template, fitted_pose)
    max_err = float(np.abs(fitted_sat_xy - known_sat_xy).max())
    assert max_err < 1e-6, f"fitted placement diverges from known by {max_err}"

    # Handedness -- the property that actually matters to a user marking
    # wheels, and the one that silently breaks if the body->sat reflection
    # is dropped. Anchor the two LEFT wheels so the car faces "up" the sat
    # image (rear wheel further down/south than the front wheel): looking
    # straight down at a north-facing car, its own left side is on the
    # image's left, so every right-side keypoint must land at a LARGER x
    # than its left-side counterpart. Without the reflection the whole
    # template is placed mirrored and this flips.
    wheelbase_px = 2.54 * known_pose.scale
    north_pose = fit_two_point_pose(template, [
        ("front_wheel_center_left", (1000.0, 500.0)),
        ("rear_wheel_center_left", (1000.0, 500.0 + wheelbase_px)),
    ])
    north_sat = apply_pose_to_template(template, north_pose)
    fl = north_sat[kp_names.index("front_wheel_center_left")]
    fr = north_sat[kp_names.index("front_wheel_center_right")]
    assert abs(fl[0] - 1000.0) < 1e-6 and abs(fl[1] - 500.0) < 1e-6, (
        f"anchored left wheel must land exactly on its click, got {fl}"
    )
    assert fr[0] > fl[0], (
        f"vehicle-right wheel landed at x={fr[0]:.1f} but vehicle-left at x={fl[0]:.1f} "
        "-- template placed mirrored (body->sat reflection missing)"
    )

    nominal_px_per_m = 40.0
    report = anchor_fit_report(fitted_pose, nominal_px_per_m)
    expected_ratio = known_pose.scale / nominal_px_per_m
    assert abs(report["scale_ratio"] - expected_ratio) < 1e-9, report

    heights = scaled_heights_m(template, fitted_pose, nominal_px_per_m)
    assert abs(heights[4] - 1.45 * expected_ratio) < 1e-9, heights

    kp_cctv = [
        [100.0, 100.0, 0.9],   # front_wheel_center_left -- detected
        [0.0, -4.0, 0.0],      # rear_wheel_center_left -- undetected sentinel
        [0.0, 0.0, 0.0],       # front_wheel_center_right -- undetected sentinel
        [120.0, 300.0, 0.05],  # rear_wheel_center_right -- below conf threshold
        [150.0, 90.0, 0.8],    # roof_center -- detected
    ]
    refs = build_reference_points(
        template, fitted_pose, fitted_sat_xy, kp_cctv, frame_index=7,
        nominal_px_per_m=nominal_px_per_m, kp_conf=0.2, start_id=1,
    )
    assert len(refs) == 2, refs
    assert {r.id for r in refs} == {1, 2}, refs
    assert refs[0].head_px == (100.0, 100.0) and refs[0].foot_px is None
    assert refs[1].head_px == (150.0, 90.0)

    print("SELF-TEST PASSED")
    print(f"  handedness: left wheel x={fl[0]:.1f}, right wheel x={fr[0]:.1f} (right must be larger)")
    print(f"  fitted pose: {fitted_pose}")
    print(f"  anchor_fit_report: {report}")
    print(f"  n reference points from synthetic kp_cctv: {len(refs)}")


if __name__ == "__main__":
    self_test()
