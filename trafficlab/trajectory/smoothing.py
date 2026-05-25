"""Savitzky-Golay smoothing for TrafficLab satellite trajectories."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.signal import savgol_filter

from trafficlab.trajectory.io import (
    default_smooth_output_path,
    frames_from_data,
    load_json,
    write_json,
)


@dataclass(frozen=True)
class SmoothStats:
    """Summary of a trajectory smoothing run."""

    total_tracks: int
    selected_tracks: int
    smoothed_tracks: int
    skipped_short_tracks: int
    skipped_invalid_tracks: int
    updated_points: int


def _normalize_ids(selected_ids: Iterable[int] | None) -> set[int] | None:
    if selected_ids is None:
        return None
    return {int(track_id) for track_id in selected_ids}


def _valid_point(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    )


def _collect_tracks(
    frames: list[dict[str, Any]],
    selected_ids: set[int] | None,
) -> tuple[dict[int, list[dict[str, Any]]], int]:
    tracks: dict[int, list[dict[str, Any]]] = {}
    invalid_count = 0

    for frame in frames:
        for obj in frame.get("objects", []):
            tracked_id = obj.get("tracked_id")
            if tracked_id is None:
                continue

            track_id = int(tracked_id)
            if selected_ids is not None and track_id not in selected_ids:
                continue

            sat_coords = obj.get("sat_coords")
            if not _valid_point(sat_coords):
                invalid_count += 1
                continue

            tracks.setdefault(track_id, []).append(obj)

    return tracks, invalid_count


def _validate_filter_args(window_length: int, polyorder: int) -> None:
    if window_length <= 0:
        raise ValueError("window_length must be positive.")
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd.")
    if polyorder < 0:
        raise ValueError("polyorder must be non-negative.")
    if polyorder >= window_length:
        raise ValueError("polyorder must be smaller than window_length.")


def smooth_trajectories(
    data: Any,
    *,
    selected_ids: Iterable[int] | None = None,
    window_length: int = 45,
    polyorder: int = 3,
    update_sat_center: bool = True,
    in_place: bool = False,
) -> tuple[Any, SmoothStats]:
    """Smooth TrafficLab ``sat_coords`` values grouped by ``tracked_id``.

    The function preserves the original replay shape and only mutates
    ``sat_coords`` plus ``sat_center`` when present and requested.
    """
    _validate_filter_args(window_length, polyorder)

    output = data if in_place else copy.deepcopy(data)
    frames = frames_from_data(output)
    selected_id_set = _normalize_ids(selected_ids)
    tracks, invalid_count = _collect_tracks(frames, selected_id_set)

    smoothed_tracks = 0
    skipped_short_tracks = 0
    updated_points = 0

    for objects in tracks.values():
        if len(objects) < window_length:
            skipped_short_tracks += 1
            continue

        coords = np.array([obj["sat_coords"][:2] for obj in objects], dtype=float)
        smoothed_x = savgol_filter(coords[:, 0], window_length, polyorder)
        smoothed_y = savgol_filter(coords[:, 1], window_length, polyorder)

        for obj, x_value, y_value in zip(objects, smoothed_x, smoothed_y):
            point = [float(x_value), float(y_value)]
            obj["sat_coords"] = point
            if update_sat_center and "sat_center" in obj:
                obj["sat_center"] = point.copy()
            updated_points += 1

        smoothed_tracks += 1

    stats = SmoothStats(
        total_tracks=len(tracks),
        selected_tracks=len(tracks),
        smoothed_tracks=smoothed_tracks,
        skipped_short_tracks=skipped_short_tracks,
        skipped_invalid_tracks=invalid_count,
        updated_points=updated_points,
    )
    return output, stats


def smooth_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    selected_ids: Iterable[int] | None = None,
    window_length: int = 45,
    polyorder: int = 3,
    update_sat_center: bool = True,
) -> tuple[Path, SmoothStats]:
    """Smooth a TrafficLab replay JSON file and write the result."""
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else default_smooth_output_path(input_path)
    data = load_json(input_path)
    smoothed, stats = smooth_trajectories(
        data,
        selected_ids=selected_ids,
        window_length=window_length,
        polyorder=polyorder,
        update_sat_center=update_sat_center,
    )
    write_json(output_path, smoothed)
    return output_path, stats
