"""Rigid-alignment + stationary-freeze correction of a YOLO-box replay's
sat_coords against a wheel_pair replay of the same video.

Two deliberate constraints, backed by
docs/wheelbase-track-consistency-homography-check.md's evidence that this
pipeline's positional bias varies with screen position rather than being a
constant offset:

- The rigid (rotation + translation, no scale) fit for a track only uses
  matched frames where the wheel_pair vehicle actually moved (see
  `compute_stationary_anchors`). Including near-stationary frames would let
  sub-pixel keypoint jitter dominate the fit instead of the real distortion
  the transform is meant to correct.
- Near-stationary intervals are frozen onto the position of the frame right
  before the interval started (anchor-style), not copied frame-to-frame,
  so a long stationary run cannot drift.

`sat_coords`, `sat_center` (if present), `sat_floor_box` (rotated + translated
by the same transform as sat_coords, so the SAT-panel box stays attached to
the corrected point), and `heading` (rotated by the same angle) are all
updated. `bbox_3d` — a CCTV-image-space reprojection of the floor box — is
not recomputed (it needs the full GProjection inverse-projection, not just
the rigid transform) and may be stale relative to the corrected position;
out of scope for this pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from trafficlab.motion.keypoints_openpifpaf import KP_NAMES, WHEEL_KP

# Wheels plus front-door-top corners: the keypoints most reliably visible
# (and least prone to perspective/parallax wobble) even when a wheel_pair
# match itself failed for the frame, so the stationary check has something
# to compare when e.g. only one side's wheels are confidently detected.
_STATIONARY_CHECK_INDICES = (
    WHEEL_KP["front_wheel_left"],
    WHEEL_KP["front_wheel_right"],
    WHEEL_KP["rear_wheel_left"],
    WHEEL_KP["rear_wheel_right"],
    KP_NAMES.index("front_door_top_left"),
    KP_NAMES.index("front_door_top_right"),
)


def _valid_point(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    )


def _stationary_check_displacement(kp_a, kp_b, kp_conf: float) -> Optional[float]:
    """Max CCTV-pixel displacement across the stationary-check keypoints
    (wheels + front-door-top corners) confident in both frames, or None if
    none of them clears kp_conf in both."""
    if not kp_a or not kp_b:
        return None
    best = None
    for idx in _STATIONARY_CHECK_INDICES:
        if idx >= len(kp_a) or idx >= len(kp_b):
            continue
        xa, ya, ca = kp_a[idx]
        xb, yb, cb = kp_b[idx]
        if ca < kp_conf or cb < kp_conf:
            continue
        d = math.hypot(xb - xa, yb - ya)
        if best is None or d > best:
            best = d
    return best


def compute_stationary_anchors(
    wp_frames: list[dict],
    *,
    vehicle_class: str = "car",
    kp_conf: float = 0.2,
    stationary_px_threshold: float = 2.0,
) -> dict[tuple[int, int], int]:
    """Map each (wheel_pair tracked_id, frame_index) to the anchor
    frame_index it should report a position for.

    A frame anchors itself unless its stationary-check keypoints (wheels +
    front-door-top corners — see `_STATIONARY_CHECK_INDICES`) show no real
    motion (max displacement < stationary_px_threshold) relative to the
    previous frame of the same track, in which case it inherits that frame's
    anchor — so a whole stationary run collapses onto the frame right before
    it started, instead of drifting through chained frame-to-frame copies.
    Frame pairs with no comparable stationary-check keypoint are treated as
    moved (conservative default: don't freeze on an unknown).
    """
    by_track: dict[int, list[dict]] = {}
    for frame in wp_frames:
        frame_index = frame.get("frame_index")
        if frame_index is None:
            continue
        for obj in frame.get("objects") or []:
            if obj.get("class") != vehicle_class:
                continue
            tracked_id = obj.get("tracked_id")
            if tracked_id is None:
                continue
            by_track.setdefault(int(tracked_id), []).append(
                {"frame_index": int(frame_index), "kp_cctv": obj.get("kp_cctv")}
            )

    anchors: dict[tuple[int, int], int] = {}
    for tracked_id, records in by_track.items():
        records.sort(key=lambda r: r["frame_index"])
        prev = None
        prev_anchor = None
        for rec in records:
            fi = rec["frame_index"]
            if prev is None:
                anchor = fi
            else:
                d = _stationary_check_displacement(prev["kp_cctv"], rec["kp_cctv"], kp_conf)
                anchor = prev_anchor if (d is not None and d < stationary_px_threshold) else fi
            anchors[(tracked_id, fi)] = anchor
            prev, prev_anchor = rec, anchor
    return anchors


def match_frame_objects(
    target_objs: list[dict],
    wp_objs: list[dict],
    *,
    vehicle_class: str = "car",
    max_match_distance_px: float,
) -> list[tuple[dict, dict, float]]:
    """Greedy nearest-neighbor match of same-frame target/wheel_pair objects
    by sat_coords distance (track IDs come from two independent pipeline
    runs and are not comparable). Returns (target_obj, wp_obj, distance_px)
    triples; each object is used in at most one match."""
    candidates = []
    for t in target_objs:
        if t.get("class") != vehicle_class or not _valid_point(t.get("sat_coords")):
            continue
        for w in wp_objs:
            if w.get("class") != vehicle_class or not _valid_point(w.get("sat_coords")):
                continue
            tx, ty = t["sat_coords"][0], t["sat_coords"][1]
            wx, wy = w["sat_coords"][0], w["sat_coords"][1]
            d = math.hypot(tx - wx, ty - wy)
            if d <= max_match_distance_px:
                candidates.append((d, id(t), id(w), t, w))

    candidates.sort(key=lambda c: c[0])
    used_t: set = set()
    used_w: set = set()
    matches = []
    for d, tid_, wid_, t, w in candidates:
        if tid_ in used_t or wid_ in used_w:
            continue
        used_t.add(tid_)
        used_w.add(wid_)
        matches.append((t, w, d))
    return matches


def _fit_rigid(source_pts: np.ndarray, dest_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-scale 2D Procrustes (Kabsch): rotation R, translation T such
    that dest ~= source @ R + T (R applied as R @ point). Same derivation as
    OpenPifPafKeypointsLocalizer.localize() in
    trafficlab/motion/keypoints_openpifpaf.py."""
    qb = source_pts.mean(axis=0)
    pb = dest_pts.mean(axis=0)
    Hc = (source_pts - qb).T @ (dest_pts - pb)
    U, _, Vt = np.linalg.svd(Hc)
    det_sign = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, det_sign]) @ U.T
    T = pb - R @ qb
    return R, T


def _max_pairwise_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diffs = points[:, None, :] - points[None, :, :]
    return float(np.hypot(diffs[..., 0], diffs[..., 1]).max())


def _apply_rigid_to_obj(obj: dict, R: np.ndarray, T: np.ndarray, rotation_deg: float) -> None:
    """Apply the same rigid transform to every position/orientation field on
    obj so sat_coords, sat_center, sat_floor_box (the SAT-panel box corners)
    and heading (the arrow direction) stay mutually consistent."""
    pt = obj.get("sat_coords")
    if not _valid_point(pt):
        return
    new_pt = R @ np.array([pt[0], pt[1]], dtype=float) + T
    obj["sat_coords"] = [float(new_pt[0]), float(new_pt[1])]
    if obj.get("sat_center") is not None:
        obj["sat_center"] = list(obj["sat_coords"])

    floor_box = obj.get("sat_floor_box")
    if isinstance(floor_box, list) and floor_box and all(_valid_point(c) for c in floor_box):
        corners = np.array([[c[0], c[1]] for c in floor_box], dtype=float)
        new_corners = corners @ R.T + T
        obj["sat_floor_box"] = new_corners.tolist()

    if obj.get("heading") is not None:
        obj["heading"] = float((obj["heading"] + rotation_deg) % 360.0)


def _copy_position_fields(dst_obj: dict, src_obj: dict) -> None:
    """Freeze dst_obj's position/orientation to src_obj's (the anchor
    frame) — used to collapse a stationary interval onto one position."""
    dst_obj["sat_coords"] = list(src_obj["sat_coords"])
    if dst_obj.get("sat_center") is not None:
        dst_obj["sat_center"] = list(dst_obj["sat_coords"])
    src_floor_box = src_obj.get("sat_floor_box")
    if dst_obj.get("sat_floor_box") is not None and src_floor_box is not None:
        dst_obj["sat_floor_box"] = [list(c) for c in src_floor_box]
    if dst_obj.get("heading") is not None and src_obj.get("heading") is not None:
        dst_obj["heading"] = src_obj["heading"]


@dataclass
class TrackCorrectionResult:
    tracked_id: int
    status: str  # "corrected" | "skipped_insufficient_points" | "skipped_insufficient_span"
    n_fit_points: int
    n_frozen_frames: int = 0
    n_missing_anchor_frames: int = 0
    translation_px: Optional[float] = None
    rotation_deg: Optional[float] = None


@dataclass
class CorrectionReport:
    total_target_tracks: int
    results: list[TrackCorrectionResult] = field(default_factory=list)

    @property
    def corrected_tracks(self) -> list[TrackCorrectionResult]:
        return [r for r in self.results if r.status == "corrected"]

    @property
    def skipped_tracks(self) -> list[TrackCorrectionResult]:
        return [r for r in self.results if r.status != "corrected"]


def correct_replay(
    target_data: dict,
    wp_data: dict,
    *,
    vehicle_class: str = "car",
    kp_conf: float = 0.2,
    max_match_distance_px: float,
    stationary_px_threshold: float = 2.0,
    min_fit_points: int = 3,
    min_fit_span_px: float,
    track_id_map: Optional[dict[int, int]] = None,
) -> CorrectionReport:
    """Correct `target_data`'s sat_coords in place using `wp_data` as the
    positional reference. See module docstring for the algorithm.

    track_id_map: optional {target tracked_id: wheel_pair tracked_id}. When
    given, only these target tracks are touched — every other target track
    is left completely alone (not even auto-matched by sat_coords proximity).
    For a mapped track, each frame is looked up directly by the given
    wheel_pair tracked_id instead of guessed by nearest-neighbor distance,
    so a frame where the wheel_pair vehicle has no sat_coords (localization
    failed) can still be used to ask "did it move" via kp_cctv — it just
    can't be used to fit the rigid transform (no real position to fit
    against).
    """
    target_frames = target_data.get("frames") or []
    wp_frames = wp_data.get("frames") or []

    anchors = compute_stationary_anchors(
        wp_frames,
        vehicle_class=vehicle_class,
        kp_conf=kp_conf,
        stationary_px_threshold=stationary_px_threshold,
    )

    wp_by_frame: dict[int, list[dict]] = {}
    for frame in wp_frames:
        fi = frame.get("frame_index")
        if fi is None:
            continue
        wp_by_frame.setdefault(int(fi), []).extend(frame.get("objects") or [])

    target_by_frame: dict[int, list[dict]] = {}
    target_track_objs: dict[int, dict[int, dict]] = {}
    for frame in target_frames:
        fi = frame.get("frame_index")
        if fi is None:
            continue
        fi = int(fi)
        objs = frame.get("objects") or []
        target_by_frame.setdefault(fi, []).extend(objs)
        for obj in objs:
            if obj.get("class") != vehicle_class:
                continue
            tid = obj.get("tracked_id")
            if tid is None:
                continue
            target_track_objs.setdefault(int(tid), {})[fi] = obj

    if track_id_map is not None:
        target_track_objs = {
            tid: frame_map for tid, frame_map in target_track_objs.items()
            if tid in track_id_map
        }

    # Match per frame, aggregate correspondences by target tracked_id.
    matches_by_tid: dict[int, list[dict]] = {}
    if track_id_map is not None:
        # Correspondence is given, not guessed: look up each frame directly
        # by the mapped wheel_pair tracked_id, no sat_coords proximity
        # needed. This is also what lets a wheel_pair-localization-failure
        # frame (no sat_coords, but kp_cctv still present) participate —
        # it's only usable for the stationary check, never for the fit
        # (see has_position below / the fit_matches filter).
        for tid, wtid in track_id_map.items():
            frame_map = target_track_objs.get(tid)
            if frame_map is None:
                continue
            for fi in frame_map:
                w_obj = next(
                    (w for w in wp_by_frame.get(fi, [])
                     if w.get("class") == vehicle_class and w.get("tracked_id") == wtid),
                    None,
                )
                if w_obj is None:
                    continue
                anchor_fi = anchors.get((wtid, fi), fi)
                has_position = _valid_point(w_obj.get("sat_coords"))
                matches_by_tid.setdefault(tid, []).append({
                    "frame_index": fi,
                    "wp_sat": [w_obj["sat_coords"][0], w_obj["sat_coords"][1]] if has_position else None,
                    "is_stationary": anchor_fi != fi,
                    "anchor_frame_index": anchor_fi,
                    "has_position": has_position,
                })
    else:
        for fi, target_objs in target_by_frame.items():
            wp_objs = wp_by_frame.get(fi, [])
            if not wp_objs:
                continue
            pairs = match_frame_objects(
                target_objs, wp_objs,
                vehicle_class=vehicle_class,
                max_match_distance_px=max_match_distance_px,
            )
            for t, w, _d in pairs:
                tid = t.get("tracked_id")
                wtid = w.get("tracked_id")
                if tid is None or wtid is None:
                    continue
                anchor_fi = anchors.get((int(wtid), fi), fi)
                matches_by_tid.setdefault(int(tid), []).append({
                    "frame_index": fi,
                    "wp_sat": [w["sat_coords"][0], w["sat_coords"][1]],
                    "is_stationary": anchor_fi != fi,
                    "anchor_frame_index": anchor_fi,
                    "has_position": True,
                })

    report = CorrectionReport(total_target_tracks=len(target_track_objs))

    for tid, frame_map in target_track_objs.items():
        track_matches = matches_by_tid.get(tid, [])
        fit_matches = [
            m for m in track_matches
            if not m["is_stationary"] and m["has_position"]
            and _valid_point(frame_map.get(m["frame_index"], {}).get("sat_coords"))
        ]

        if len(fit_matches) < min_fit_points:
            report.results.append(TrackCorrectionResult(
                tracked_id=tid, status="skipped_insufficient_points",
                n_fit_points=len(fit_matches),
            ))
            continue

        src = np.array(
            [frame_map[m["frame_index"]]["sat_coords"][:2] for m in fit_matches], dtype=float
        )
        dst = np.array([m["wp_sat"] for m in fit_matches], dtype=float)

        if _max_pairwise_distance(src) < min_fit_span_px:
            report.results.append(TrackCorrectionResult(
                tracked_id=tid, status="skipped_insufficient_span",
                n_fit_points=len(fit_matches),
            ))
            continue

        R, T = _fit_rigid(src, dst)
        rotation_deg = float(math.degrees(math.atan2(R[1, 0], R[0, 0])))

        for obj in frame_map.values():
            _apply_rigid_to_obj(obj, R, T, rotation_deg)

        n_frozen = 0
        n_missing_anchor = 0
        for m in track_matches:
            if not m["is_stationary"]:
                continue
            cur_obj = frame_map.get(m["frame_index"])
            if cur_obj is None:
                continue
            anchor_obj = frame_map.get(m["anchor_frame_index"])
            if anchor_obj is None or not _valid_point(anchor_obj.get("sat_coords")):
                n_missing_anchor += 1
                continue
            _copy_position_fields(cur_obj, anchor_obj)
            n_frozen += 1

        report.results.append(TrackCorrectionResult(
            tracked_id=tid, status="corrected",
            n_fit_points=len(fit_matches),
            n_frozen_frames=n_frozen,
            n_missing_anchor_frames=n_missing_anchor,
            translation_px=float(np.hypot(T[0], T[1])),
            rotation_deg=rotation_deg,
        ))

    return report
