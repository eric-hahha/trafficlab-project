#!/usr/bin/env python3
"""Post-process TrafficLab replay JSON trajectories.

Initial scope:
  - Read a TrafficLab .json or .json.gz replay output.
  - Group objects by tracked_id.
  - Detect short, sharp direction reversals in motorcycle trajectories.
  - Replace those points with interpolation between nearby stable points.
  - Add optional B-spline visual coordinates without overwriting sat_coords.
  - Normalize output for trajectory smoothing/plotting tools that expect
    {"frames": [{"frame_id": ..., "objects": [{"sat_coords": ..., "sat_center": ...}]}]}.

This script intentionally operates after inference so the correction can see the
whole track instead of making frame-local decisions.

Usage:
  python postprocess.py output.json.gz
  python postprocess.py output.json.gz -o output.postprocessed.json.gz
  python postprocess.py output.json.gz --dry-run --verbose
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CONFIG_PATH = Path(__file__).with_name("postprocess_config.yaml")

TARGET_CLASSES = {
    "motor",
    "motorcycle",
    "two_wheeler",
    "two-wheeler",
    "twowheeler",
    "Two_Wheeler".lower(),
}


def _config_get(config: dict[str, Any], path: list[str], default: Any) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return default if value is None else value


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read postprocess config files") from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return data


def _preparse_config_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args, _ = parser.parse_known_args()
    return args.config


@dataclass
class TrackPoint:
    frame_index: int
    object_ref: dict[str, Any]
    point: list[float]


@dataclass
class Segment:
    start: int
    end: int
    max_angle_deg: float


def _read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _default_output_path(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith(".json.gz"):
        return input_path.with_name(name[:-8] + ".postprocessed.json.gz")
    if name.endswith(".json"):
        return input_path.with_name(name[:-5] + ".postprocessed.json")
    return input_path.with_name(name + ".postprocessed.json.gz")


def _frame_id(frame: dict[str, Any], fallback: int) -> int:
    value = frame.get("frame_id", frame.get("frame_index", fallback))
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _fps_from_meta(data: dict[str, Any]) -> float | None:
    meta = data.get("meta")
    if not isinstance(meta, dict):
        return None
    value = meta.get("fps")
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def _class_key(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _is_target_class(value: Any, target_classes: set[str]) -> bool:
    cls = _class_key(value)
    compact = cls.replace("_", "").replace("-", "")
    return cls in target_classes or compact in target_classes


def _normalize_point(value: Any) -> list[float] | None:
    if not _valid_point(value):
        return None
    return [float(value[0]), float(value[1])]


def _sub(a: list[float], b: list[float]) -> tuple[float, float]:
    return a[0] - b[0], a[1] - b[1]


def _add(a: list[float], b: tuple[float, float]) -> list[float]:
    return [a[0] + b[0], a[1] + b[1]]


def _mul(v: tuple[float, float], scale: float) -> tuple[float, float]:
    return v[0] * scale, v[1] * scale


def _norm(v: tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _angle_deg(v1: tuple[float, float], v2: tuple[float, float]) -> float | None:
    n1 = _norm(v1)
    n2 = _norm(v2)
    if n1 <= 1e-9 or n2 <= 1e-9:
        return None
    cos_value = max(-1.0, min(1.0, _dot(v1, v2) / (n1 * n2)))
    return math.degrees(math.acos(cos_value))


def _point_line_distance(point: list[float], a: list[float], b: list[float]) -> float:
    ab = _sub(b, a)
    ap = _sub(point, a)
    denom = _norm(ab)
    if denom <= 1e-9:
        return _norm(ap)
    return abs(ab[0] * ap[1] - ab[1] * ap[0]) / denom


def _interpolate(a: list[float], b: list[float], ratio: float) -> list[float]:
    return [
        a[0] + (b[0] - a[0]) * ratio,
        a[1] + (b[1] - a[1]) * ratio,
    ]


def _collect_tracks(data: dict[str, Any], target_classes: set[str]) -> dict[int, list[TrackPoint]]:
    tracks: dict[int, list[TrackPoint]] = {}
    for fallback_index, frame in enumerate(data.get("frames", [])):
        frame_index = _frame_id(frame, fallback_index)
        for obj in frame.get("objects", []):
            if not _is_target_class(obj.get("class"), target_classes):
                continue
            tracked_id = obj.get("tracked_id")
            sat_coords = _normalize_point(obj.get("sat_coords"))
            if tracked_id is None or sat_coords is None:
                continue
            tracks.setdefault(int(tracked_id), []).append(
                TrackPoint(
                    frame_index=frame_index,
                    object_ref=obj,
                    point=sat_coords,
                )
            )
    for points in tracks.values():
        points.sort(key=lambda item: item.frame_index)
    return tracks


def _valid_point(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    )


def _detect_sharp_indices(
    points: list[list[float]],
    *,
    sharp_turn_angle_deg: float,
    min_step_px: float,
    min_lateral_deviation_px: float,
) -> dict[int, float]:
    sharp: dict[int, float] = {}
    for i in range(1, len(points) - 1):
        prev_pt = points[i - 1]
        curr_pt = points[i]
        next_pt = points[i + 1]

        v1 = _sub(curr_pt, prev_pt)
        v2 = _sub(next_pt, curr_pt)
        if _norm(v1) < min_step_px or _norm(v2) < min_step_px:
            continue

        angle = _angle_deg(v1, v2)
        if angle is None or angle < sharp_turn_angle_deg:
            continue

        # A large angle is not enough: if the point is almost on the shortcut
        # line, it is usually just a slow dense cluster. Require lateral error.
        lateral_error = _point_line_distance(curr_pt, prev_pt, next_pt)
        if lateral_error < min_lateral_deviation_px:
            continue

        sharp[i] = angle
    return sharp


def _indices_to_segments(
    sharp_indices: dict[int, float],
    *,
    bridge_gap: int,
    max_bad_segment_len: int,
) -> list[Segment]:
    if not sharp_indices:
        return []

    ordered = sorted(sharp_indices)
    segments: list[Segment] = []
    start = prev = ordered[0]
    max_angle = sharp_indices[prev]

    for idx in ordered[1:]:
        if idx - prev <= bridge_gap + 1:
            prev = idx
            max_angle = max(max_angle, sharp_indices[idx])
            continue
        if prev - start + 1 <= max_bad_segment_len:
            segments.append(Segment(start, prev, max_angle))
        start = prev = idx
        max_angle = sharp_indices[idx]

    if prev - start + 1 <= max_bad_segment_len:
        segments.append(Segment(start, prev, max_angle))
    return segments


def _apply_segment_interpolation(
    original_points: list[list[float]],
    segments: Iterable[Segment],
) -> tuple[list[list[float]], dict[int, Segment]]:
    corrected = [p[:] for p in original_points]
    changed: dict[int, Segment] = {}

    for segment in segments:
        before = segment.start - 1
        after = segment.end + 1
        if before < 0 or after >= len(original_points):
            continue

        start_point = corrected[before]
        end_point = corrected[after]
        span = after - before
        if span <= 0:
            continue

        for idx in range(segment.start, segment.end + 1):
            ratio = (idx - before) / span
            corrected[idx] = _interpolate(start_point, end_point, ratio)
            changed[idx] = segment

    return corrected, changed


def _fit_pca_axis(points: list[list[float]]) -> tuple[list[float], tuple[float, float]] | None:
    if len(points) < 2:
        return None

    try:
        import numpy as np
    except ImportError:
        return None

    arr = np.array(points, dtype=float)
    center = arr.mean(axis=0)
    centered = arr - center
    if float(np.linalg.norm(centered)) <= 1e-9:
        return None

    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    axis = vh[0]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        return None
    axis = axis / norm
    return [float(center[0]), float(center[1])], (float(axis[0]), float(axis[1]))


def _track_global_axis(points: list[list[float]]) -> tuple[float, float] | None:
    fit = _fit_pca_axis(points)
    if fit is None:
        return None
    _, axis = fit
    overall = _sub(points[-1], points[0])
    if _norm(overall) > 1e-9 and _dot(axis, overall) < 0:
        axis = (-axis[0], -axis[1])
    return axis


def _local_axis_reference(
    points: list[list[float]],
    index: int,
    *,
    window_size: int,
    global_axis: tuple[float, float] | None,
) -> tuple[list[float], tuple[float, float]] | None:
    half = max(1, window_size // 2)
    start = max(0, index - half)
    end = min(len(points), index + half + 1)

    if end - start < 2:
        return None

    fit = _fit_pca_axis(points[start:end])
    if fit is None:
        return None

    reference, axis = fit
    if global_axis is not None and _dot(axis, global_axis) < 0:
        axis = (-axis[0], -axis[1])
    return reference, axis


def _angle_from_axis_deg(vector: tuple[float, float], axis: tuple[float, float]) -> float | None:
    n = _norm(vector)
    if n <= 1e-9:
        return None
    cos_value = max(-1.0, min(1.0, _dot(vector, axis) / n))
    return math.degrees(math.acos(cos_value))


def _merge_postprocess_entry(obj: dict[str, Any], *, original: list[float], corrected: list[float],
                             reason: str, diagnostics: dict[str, Any]) -> None:
    post = obj.setdefault("postprocess", {})
    if "original_sat_coords" not in post:
        post["original_sat_coords"] = original
    post["corrected"] = True
    post["reason"] = reason if not post.get("reason") else f"{post['reason']}+{reason}"
    post["corrected_sat_coords"] = corrected
    post.setdefault("diagnostics", {}).update(diagnostics)


def _apply_direction_correction(
    track: list[TrackPoint],
    *,
    enabled: bool,
    centerline_mode: str,
    window_size: int,
    min_points: int,
    max_angle_from_axis_deg: float,
    max_lateral_offset_px: float,
    lateral_retention: float,
    preserve_longitudinal_progress: bool,
    max_correction_px: float,
    dry_run: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "points": len(track),
            "corrected_points": 0,
            "skipped": "disabled",
        }

    if len(track) < min_points:
        return {
            "enabled": True,
            "points": len(track),
            "corrected_points": 0,
            "skipped": "short_track",
        }

    original_points = [item.point[:] for item in track]
    global_axis = _track_global_axis(original_points)
    global_fit = _fit_pca_axis(original_points)
    if global_axis is None or global_fit is None:
        return {
            "enabled": True,
            "points": len(track),
            "corrected_points": 0,
            "skipped": "no_stable_axis",
        }

    retention = max(0.0, min(1.0, lateral_retention))
    corrected_points = [point[:] for point in original_points]
    changed: dict[int, dict[str, Any]] = {}
    global_reference = global_fit[0]
    previous_longitudinal = None

    for idx, point in enumerate(original_points):
        if centerline_mode == "global_pca":
            global_reference, global_fit_axis = global_fit
            if _dot(global_fit_axis, global_axis) < 0:
                global_fit_axis = (-global_fit_axis[0], -global_fit_axis[1])
            local = (global_reference, global_fit_axis)
        else:
            local = _local_axis_reference(
                original_points,
                idx,
                window_size=window_size,
                global_axis=global_axis,
            )
        if local is None:
            continue

        reference, axis = local
        perpendicular = (-axis[1], axis[0])
        delta = _sub(point, reference)
        longitudinal = _dot(delta, axis)
        lateral = _dot(delta, perpendicular)
        lateral_abs = abs(lateral)
        global_longitudinal = _dot(_sub(point, global_reference), global_axis)

        step_angle = None
        if idx > 0:
            step_angle = _angle_from_axis_deg(_sub(point, original_points[idx - 1]), axis)

        angle_bad = step_angle is not None and step_angle > max_angle_from_axis_deg
        lateral_bad = lateral_abs > max_lateral_offset_px
        progress_bad = (
            preserve_longitudinal_progress
            and previous_longitudinal is not None
            and global_longitudinal < previous_longitudinal
        )
        if not angle_bad and not lateral_bad and not progress_bad:
            previous_longitudinal = global_longitudinal if previous_longitudinal is None else max(previous_longitudinal, global_longitudinal)
            continue

        retained_lateral = lateral * retention
        if max_lateral_offset_px >= 0:
            limit = max_lateral_offset_px
            retained_lateral = max(-limit, min(limit, retained_lateral))

        candidate = [
            reference[0] + longitudinal * axis[0] + retained_lateral * perpendicular[0],
            reference[1] + longitudinal * axis[1] + retained_lateral * perpendicular[1],
        ]

        if progress_bad and previous_longitudinal is not None:
            candidate_longitudinal = _dot(_sub(candidate, global_reference), global_axis)
            if candidate_longitudinal < previous_longitudinal:
                candidate = _add(
                    candidate,
                    _mul(global_axis, previous_longitudinal - candidate_longitudinal),
                )

        correction = _sub(candidate, point)
        correction_norm = _norm(correction)
        if max_correction_px > 0 and correction_norm > max_correction_px:
            scale = max_correction_px / correction_norm
            candidate = _add(point, _mul(correction, scale))
            correction_norm = max_correction_px

        if correction_norm <= 1e-6:
            continue

        corrected_points[idx] = candidate
        changed[idx] = {
            "axis_angle_deg": round(step_angle, 3) if step_angle is not None else None,
            "lateral_offset_px": round(lateral, 3),
            "retained_lateral_px": round(retained_lateral, 3),
            "correction_px": round(correction_norm, 3),
            "axis": [round(axis[0], 6), round(axis[1], 6)],
            "reference": [round(reference[0], 3), round(reference[1], 3)],
            "progress_clamped": bool(progress_bad),
        }
        corrected_global_longitudinal = _dot(_sub(candidate, global_reference), global_axis)
        previous_longitudinal = (
            corrected_global_longitudinal
            if previous_longitudinal is None
            else max(previous_longitudinal, corrected_global_longitudinal)
        )

    if not dry_run:
        for idx, diagnostics in changed.items():
            item = track[idx]
            old = original_points[idx]
            new = corrected_points[idx]
            delta = (new[0] - old[0], new[1] - old[1])
            item.object_ref["sat_coords"] = [new[0], new[1]]
            item.object_ref["sat_center"] = [new[0], new[1]]
            item.point = [new[0], new[1]]
            _translate_sat_floor_box(item.object_ref, delta)
            _merge_postprocess_entry(
                item.object_ref,
                original=old,
                corrected=new,
                reason="direction_corridor_projection",
                diagnostics=diagnostics,
            )

    return {
        "enabled": True,
        "points": len(track),
        "corrected_points": len(changed),
        "global_axis": [round(global_axis[0], 6), round(global_axis[1], 6)],
    }


def _translate_sat_floor_box(obj: dict[str, Any], delta: tuple[float, float]) -> None:
    pts = obj.get("sat_floor_box")
    if not isinstance(pts, list):
        return
    for pt in pts:
        if _valid_point(pt):
            pt[0] = float(pt[0]) + delta[0]
            pt[1] = float(pt[1]) + delta[1]


def _normalize_output_format(data: dict[str, Any]) -> dict[str, int]:
    """Make output compatible with traffic-trajectory-smooth input requirements."""
    stats = {
        "frames": 0,
        "objects": 0,
        "frame_id_added": 0,
        "sat_center_added": 0,
        "sat_coords_normalized": 0,
        "objects_missing_tracked_id": 0,
        "objects_missing_sat_coords": 0,
    }
    fps = _fps_from_meta(data)

    for fallback_index, frame in enumerate(data.get("frames", [])):
        if not isinstance(frame, dict):
            continue
        stats["frames"] += 1

        if "frame_id" not in frame:
            frame["frame_id"] = _frame_id(frame, fallback_index)
            stats["frame_id_added"] += 1

        if fps and "timestamp" not in frame:
            frame["timestamp"] = frame["frame_id"] / fps

        objects = frame.get("objects")
        if not isinstance(objects, list):
            frame["objects"] = []
            continue

        for obj in objects:
            if not isinstance(obj, dict):
                continue
            stats["objects"] += 1

            if obj.get("tracked_id") is None:
                stats["objects_missing_tracked_id"] += 1

            point = _normalize_point(obj.get("sat_coords"))
            if point is None:
                stats["objects_missing_sat_coords"] += 1
                continue

            if obj.get("sat_coords") != point:
                obj["sat_coords"] = point
                stats["sat_coords_normalized"] += 1

            if obj.get("sat_center") != point:
                obj["sat_center"] = point[:]
                stats["sat_center_added"] += 1

            if "class" not in obj or obj.get("class") is None:
                obj["class"] = "unknown"

    return stats


def _fit_bspline_visual_points(
    points: list[list[float]],
    *,
    degree: int,
    smooth_px: float,
    preserve_endpoints: bool,
) -> tuple[list[list[float]] | None, str | None]:
    if len(points) < max(4, degree + 1):
        return None, "short_track"

    distances = [0.0]
    for prev_pt, curr_pt in zip(points, points[1:]):
        distances.append(distances[-1] + _norm(_sub(curr_pt, prev_pt)))

    total_distance = distances[-1]
    if total_distance <= 1e-6:
        return None, "stationary_track"

    fit_points: list[list[float]] = []
    fit_distances: list[float] = []
    last_distance: float | None = None
    for point, distance in zip(points, distances):
        if last_distance is None or abs(distance - last_distance) > 1e-6:
            fit_points.append(point)
            fit_distances.append(distance)
            last_distance = distance

    if len(fit_points) < max(4, degree + 1):
        return None, "not_enough_unique_points"

    spline_degree = min(max(1, degree), len(fit_points) - 1)
    u_fit = [distance / total_distance for distance in fit_distances]
    u_eval = [distance / total_distance for distance in distances]
    x_values = [point[0] for point in fit_points]
    y_values = [point[1] for point in fit_points]
    smoothing = max(0.0, smooth_px) ** 2 * len(fit_points)

    try:
        from scipy.interpolate import splev, splprep

        tck, _ = splprep(
            [x_values, y_values],
            u=u_fit,
            s=smoothing,
            k=spline_degree,
        )
        x_smooth, y_smooth = splev(u_eval, tck)
    except Exception as exc:
        return None, f"bspline_failed:{exc.__class__.__name__}"

    visual_points = [[float(x), float(y)] for x, y in zip(x_smooth, y_smooth)]
    if preserve_endpoints and visual_points:
        visual_points[0] = points[0][:]
        visual_points[-1] = points[-1][:]

    return visual_points, None


def _add_visual_bspline_tracks(
    tracks: dict[int, list[TrackPoint]],
    *,
    min_track_points: int,
    degree: int,
    smooth_px: float,
    preserve_endpoints: bool,
) -> dict[str, Any]:
    summary = {
        "enabled": True,
        "tracks_processed": 0,
        "visual_points": 0,
        "skipped": {},
    }

    for track_id, track in tracks.items():
        if len(track) < min_track_points:
            summary["skipped"][str(track_id)] = "short_track"
            continue

        current_points: list[list[float]] = []
        for item in track:
            point = _normalize_point(item.object_ref.get("sat_coords"))
            if point is None:
                break
            current_points.append(point)
        else:
            visual_points, skip_reason = _fit_bspline_visual_points(
                current_points,
                degree=degree,
                smooth_px=smooth_px,
                preserve_endpoints=preserve_endpoints,
            )
            if visual_points is None:
                summary["skipped"][str(track_id)] = skip_reason or "unknown"
                continue

            for item, visual_point in zip(track, visual_points):
                item.object_ref["visual_sat_coords"] = visual_point
                item.object_ref["visual_sat_center"] = visual_point[:]
                post = item.object_ref.setdefault("postprocess", {})
                post["visual_bspline"] = True

            summary["tracks_processed"] += 1
            summary["visual_points"] += len(visual_points)
            continue

        summary["skipped"][str(track_id)] = "invalid_sat_coords"

    return summary


def _process_track(
    track: list[TrackPoint],
    *,
    min_track_points: int,
    sharp_turn_angle_deg: float,
    min_step_px: float,
    min_lateral_deviation_px: float,
    bridge_gap: int,
    max_bad_segment_len: int,
    dry_run: bool,
) -> dict[str, Any]:
    if len(track) < min_track_points:
        return {
            "points": len(track),
            "segments": 0,
            "corrected_points": 0,
            "skipped": "short_track",
        }

    original_points = [item.point for item in track]
    sharp = _detect_sharp_indices(
        original_points,
        sharp_turn_angle_deg=sharp_turn_angle_deg,
        min_step_px=min_step_px,
        min_lateral_deviation_px=min_lateral_deviation_px,
    )
    segments = _indices_to_segments(
        sharp,
        bridge_gap=bridge_gap,
        max_bad_segment_len=max_bad_segment_len,
    )
    corrected_points, changed = _apply_segment_interpolation(original_points, segments)

    if not dry_run:
        for idx, item in enumerate(track):
            if idx not in changed:
                continue

            old = original_points[idx]
            new = corrected_points[idx]
            delta = (new[0] - old[0], new[1] - old[1])
            item.object_ref["sat_coords"] = [new[0], new[1]]
            item.object_ref["sat_center"] = [new[0], new[1]]
            item.point = [new[0], new[1]]
            _translate_sat_floor_box(item.object_ref, delta)
            _merge_postprocess_entry(
                item.object_ref,
                original=old,
                corrected=new,
                reason="sharp_turn",
                diagnostics={"turn_angle_deg": round(changed[idx].max_angle_deg, 3)},
            )

    return {
        "points": len(track),
        "segments": len(segments),
        "corrected_points": len(changed),
        "sharp_candidates": len(sharp),
    }


def postprocess(data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    target_classes = {_class_key(v) for v in args.target_class}
    target_classes |= {v.replace("_", "").replace("-", "") for v in target_classes}
    tracks = _collect_tracks(data, target_classes)

    summary = {
        "tracks_seen": len(tracks),
        "tracks_processed": 0,
        "direction_corrected_points": 0,
        "segments": 0,
        "corrected_points": 0,
        "per_track": {},
    }

    for track_id, track in tracks.items():
        direction_result = _apply_direction_correction(
            track,
            enabled=args.direction_correction,
            centerline_mode=args.direction_centerline_mode,
            window_size=args.direction_window_size,
            min_points=args.direction_min_points,
            max_angle_from_axis_deg=args.direction_max_angle_deg,
            max_lateral_offset_px=args.direction_max_lateral_offset_px,
            lateral_retention=args.direction_lateral_retention,
            preserve_longitudinal_progress=args.direction_preserve_longitudinal_progress,
            max_correction_px=args.direction_max_correction_px,
            dry_run=False,
        )
        result = _process_track(
            track,
            min_track_points=args.min_track_points,
            sharp_turn_angle_deg=args.sharp_turn_angle_deg,
            min_step_px=args.min_step_px,
            min_lateral_deviation_px=args.min_lateral_deviation_px,
            bridge_gap=args.bridge_gap,
            max_bad_segment_len=args.max_bad_segment_len,
            dry_run=False,
        )
        result["direction_correction"] = direction_result
        summary["per_track"][str(track_id)] = result
        if result.get("skipped"):
            continue
        summary["tracks_processed"] += 1
        summary["direction_corrected_points"] += int(direction_result.get("corrected_points", 0))
        summary["segments"] += int(result["segments"])
        summary["corrected_points"] += int(result["corrected_points"])

    if not args.dry_run:
        format_stats = _normalize_output_format(data)
        visual_summary = {"enabled": False}
        if args.visual_bspline:
            visual_tracks = _collect_tracks(data, target_classes)
            visual_summary = _add_visual_bspline_tracks(
                visual_tracks,
                min_track_points=args.visual_bspline_min_points,
                degree=args.visual_bspline_degree,
                smooth_px=args.visual_bspline_smooth_px,
                preserve_endpoints=args.visual_bspline_preserve_endpoints,
            )
        data.setdefault("postprocess_meta", {})
        data["postprocess_meta"]["sharp_turn_filter"] = {
            "enabled": True,
            "target_classes": args.target_class,
            "sharp_turn_angle_deg": args.sharp_turn_angle_deg,
            "min_step_px": args.min_step_px,
            "min_lateral_deviation_px": args.min_lateral_deviation_px,
            "bridge_gap": args.bridge_gap,
            "max_bad_segment_len": args.max_bad_segment_len,
            "summary": {
                "tracks_seen": summary["tracks_seen"],
                "tracks_processed": summary["tracks_processed"],
                "direction_corrected_points": summary["direction_corrected_points"],
                "segments": summary["segments"],
                "corrected_points": summary["corrected_points"],
            },
        }
        data["postprocess_meta"]["direction_correction"] = {
            "enabled": bool(args.direction_correction),
            "centerline_mode": args.direction_centerline_mode,
            "window_size": args.direction_window_size,
            "min_points": args.direction_min_points,
            "max_angle_from_axis_deg": args.direction_max_angle_deg,
            "max_lateral_offset_px": args.direction_max_lateral_offset_px,
            "lateral_retention": args.direction_lateral_retention,
            "preserve_longitudinal_progress": args.direction_preserve_longitudinal_progress,
            "max_correction_px": args.direction_max_correction_px,
            "summary": {
                "tracks_seen": summary["tracks_seen"],
                "tracks_processed": summary["tracks_processed"],
                "corrected_points": summary["direction_corrected_points"],
            },
        }
        data["postprocess_meta"]["input_compatibility"] = {
            "format": "frames_objects_sat_coords",
            "normalized_for": "traffic-trajectory-smooth",
            "requirements": {
                "top_level_frames": True,
                "frame_id": True,
                "object_tracked_id": True,
                "object_sat_coords": True,
                "object_sat_center": True,
            },
            "stats": format_stats,
        }
        data["postprocess_meta"]["visual_bspline"] = {
            "enabled": bool(args.visual_bspline),
            "field": "visual_sat_coords",
            "center_field": "visual_sat_center",
            "degree": args.visual_bspline_degree,
            "smooth_px": args.visual_bspline_smooth_px,
            "preserve_endpoints": args.visual_bspline_preserve_endpoints,
            "summary": visual_summary,
        }
    else:
        if args.visual_bspline:
            summary["visual_bspline"] = {
                "enabled": True,
                "note": "dry-run does not write visual_sat_coords",
                "eligible_tracks": sum(
                    1 for track in tracks.values()
                    if len(track) >= args.visual_bspline_min_points
                ),
            }

    return summary


def build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove short sharp turns from TrafficLab output JSON trajectories.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_config_get(config, ["config_path"], DEFAULT_CONFIG_PATH),
        help="YAML config path. Defaults to postprocess_config.yaml next to this script.",
    )
    parser.add_argument("input", type=Path, help="Input .json or .json.gz replay file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path. Defaults to <input>.postprocessed.json.gz",
    )
    parser.add_argument(
        "--target-class",
        action="append",
        default=None,
        help="Vehicle class to post-process. Can be repeated. Overrides target_classes in YAML.",
    )
    parser.add_argument(
        "--min-track-points",
        type=int,
        default=_config_get(config, ["repair", "min_track_points"], 8),
    )
    parser.add_argument(
        "--sharp-turn-angle-deg",
        type=float,
        default=_config_get(config, ["repair", "sharp_turn_angle_deg"], 130.0),
    )
    parser.add_argument(
        "--min-step-px",
        type=float,
        default=_config_get(config, ["repair", "min_step_px"], 0.5),
    )
    parser.add_argument(
        "--min-lateral-deviation-px",
        type=float,
        default=_config_get(config, ["repair", "min_lateral_deviation_px"], 0.0),
        help=(
            "Require this much distance from the shortcut line before correcting. "
            "Default 0 also removes same-line backtracking, which is common in sharp S artifacts."
        ),
    )
    parser.add_argument(
        "--bridge-gap",
        type=int,
        default=_config_get(config, ["repair", "bridge_gap"], 1),
        help="Merge sharp indices separated by this many clean points.",
    )
    parser.add_argument(
        "--max-bad-segment-len",
        type=int,
        default=_config_get(config, ["repair", "max_bad_segment_len"], 8),
    )
    parser.add_argument(
        "--direction-correction",
        action=argparse.BooleanOptionalAction,
        default=_config_get(config, ["direction_correction", "enabled"], True),
        help="Enable or disable long-term direction corridor correction.",
    )
    parser.add_argument(
        "--direction-centerline-mode",
        default=_config_get(config, ["direction_correction", "centerline_mode"], "global_pca"),
        choices=["piecewise_pca", "global_pca"],
        help="Direction reference mode. Initial implementation uses piecewise PCA.",
    )
    parser.add_argument(
        "--direction-window-size",
        type=int,
        default=_config_get(config, ["direction_correction", "window_size"], 15),
    )
    parser.add_argument(
        "--direction-min-points",
        type=int,
        default=_config_get(config, ["direction_correction", "min_points"], 8),
    )
    parser.add_argument(
        "--direction-max-angle-deg",
        type=float,
        default=_config_get(config, ["direction_correction", "max_angle_from_axis_deg"], 25.0),
    )
    parser.add_argument(
        "--direction-max-lateral-offset-px",
        type=float,
        default=_config_get(config, ["direction_correction", "max_lateral_offset_px"], 3.0),
    )
    parser.add_argument(
        "--direction-lateral-retention",
        type=float,
        default=_config_get(config, ["direction_correction", "lateral_retention"], 0.15),
    )
    parser.add_argument(
        "--direction-preserve-longitudinal-progress",
        action=argparse.BooleanOptionalAction,
        default=_config_get(config, ["direction_correction", "preserve_longitudinal_progress"], True),
        help="Prevent short-term backward progress along the long-term axis.",
    )
    parser.add_argument(
        "--direction-max-correction-px",
        type=float,
        default=_config_get(config, ["direction_correction", "max_correction_px"], 12.0),
    )
    parser.add_argument(
        "--visual-bspline",
        action=argparse.BooleanOptionalAction,
        dest="visual_bspline",
        default=_config_get(config, ["visual_bspline", "enabled"], True),
        help="Enable or disable B-spline visual output fields.",
    )
    parser.add_argument(
        "--visual-bspline-min-points",
        type=int,
        default=_config_get(config, ["visual_bspline", "min_points"], 8),
        help="Minimum track length before adding visual_sat_coords.",
    )
    parser.add_argument(
        "--visual-bspline-degree",
        type=int,
        default=_config_get(config, ["visual_bspline", "degree"], 3),
        help="B-spline degree for visual_sat_coords.",
    )
    parser.add_argument(
        "--visual-bspline-smooth-px",
        type=float,
        default=_config_get(config, ["visual_bspline", "smooth_px"], 2.0),
        help="Approximate smoothing strength in satellite pixels.",
    )
    parser.add_argument(
        "--visual-bspline-preserve-endpoints",
        action=argparse.BooleanOptionalAction,
        default=_config_get(config, ["visual_bspline", "preserve_endpoints"], True),
        help="Keep first and last visual point equal to sat_coords.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and print summary without writing output.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    config_path = _preparse_config_path()
    config = _load_config(config_path)
    config["config_path"] = config_path

    parser = build_parser(config)
    args = parser.parse_args()
    if args.target_class is None:
        args.target_class = _config_get(config, ["target_classes"], sorted(TARGET_CLASSES))

    if not args.input.exists():
        parser.error(f"input does not exist: {args.input}")

    data = _read_json(args.input)
    if not isinstance(data, dict) or "frames" not in data:
        parser.error("input must be a TrafficLab replay object with a 'frames' key")

    output_path = args.output or _default_output_path(args.input)
    working_data = data if not args.dry_run else copy.deepcopy(data)
    summary = postprocess(working_data, args)

    print("Postprocess summary")
    print(f"  input: {args.input}")
    if not args.dry_run:
        print(f"  output: {output_path}")
    print(f"  tracks seen: {summary['tracks_seen']}")
    print(f"  tracks processed: {summary['tracks_processed']}")
    print(f"  direction-corrected points: {summary['direction_corrected_points']}")
    print(f"  sharp segments: {summary['segments']}")
    print(f"  corrected points: {summary['corrected_points']}")
    visual_meta = working_data.get("postprocess_meta", {}).get("visual_bspline", {})
    if visual_meta.get("enabled"):
        visual_summary = visual_meta.get("summary", {})
        print(f"  visual B-spline tracks: {visual_summary.get('tracks_processed', 0)}")
        print(f"  visual B-spline points: {visual_summary.get('visual_points', 0)}")
    elif args.visual_bspline and args.dry_run:
        dry_visual = summary.get("visual_bspline", {})
        print(f"  visual B-spline eligible tracks: {dry_visual.get('eligible_tracks', 0)}")

    if args.verbose:
        for track_id, result in sorted(summary["per_track"].items(), key=lambda item: int(item[0])):
            print(f"  track {track_id}: {result}")

    if not args.dry_run:
        _write_json(output_path, working_data)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
