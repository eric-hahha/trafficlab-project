"""Satellite-image trajectory plotting for TrafficLab replay outputs."""

from __future__ import annotations

import os
import tempfile
import colorsys
from pathlib import Path
from typing import Any, Iterable

_CACHE_ROOT = Path(tempfile.gettempdir()) / "trafficlab-matplotlib-cache"
_MPL_CACHE = _CACHE_ROOT / "matplotlib"
_XDG_CACHE = _CACHE_ROOT / "xdg"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
_XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from PIL import Image

from trafficlab.motion.keypoints_openpifpaf import KP_NAMES
from trafficlab.trajectory.io import (
    frames_from_data,
    infer_location_code,
    load_json,
    resolve_satellite_image_path,
)

# Keypoint-name substring -> car part, checked in this order (first match
# wins). Mirrors trafficlab/visualization/sat_renderer.py's kp_color_mode=
# "part" palette (same hues, converted to 0-1 floats for matplotlib) so a
# keypoint's color means the same thing whether it's rendered by the Qt GUI
# renderer or this matplotlib plotter. Not shared code because the two
# renderers intentionally don't depend on each other's backend (Qt vs
# matplotlib). Every name in KP_NAMES matches exactly one of these; order
# only matters for 'front_low_fog_light_*', which carries both 'light' and
# 'bumper' — 'light' wins as the more specific/accurate part.
_KP_PART_COLORS = [
    ('wheel',  (34 / 255, 197 / 255, 94 / 255)),
    ('light',  (249 / 255, 115 / 255, 22 / 255)),
    ('plate',  (234 / 255, 179 / 255, 8 / 255)),
    ('door',   (34 / 255, 211 / 255, 238 / 255)),   # front-door top corners (was 'mirror')
    ('corner', (236 / 255, 72 / 255, 153 / 255)),
    ('bumper', (168 / 255, 85 / 255, 247 / 255)),   # was 'low'
    ('glass',  (59 / 255, 130 / 255, 246 / 255)),   # roof/glass-top corners (was 'up')
]


def _kp_part_color(kp_name: str) -> tuple[float, float, float]:
    for substr, color in _KP_PART_COLORS:
        if substr in kp_name:
            return color
    return (0.78, 0.78, 0.78)  # unmatched name, shouldn't happen


# Distinct per-keypoint (not per-part) palette for --keypoint-trajectories,
# where the whole point is telling individual named keypoints apart even
# when they share a _KP_PART_COLORS bucket (e.g. front_door_top_left and
# front_door_base_left are both 'door').
_KP_TRAJECTORY_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
]


class TrajectoryPlotter:
    """Plot TrafficLab trajectory points over a location satellite image."""

    _GOLDEN_RATIO_CONJUGATE = 0.618033988749895
    _LABEL_FONT_SIZE = 10
    _COLOR_VARIANTS = (
        (0.98, 0.95),
        (0.92, 0.72),
        (0.78, 0.98),
        (1.00, 0.82),
    )

    def __init__(
        self,
        data: Any,
        *,
        input_path: str | Path | None = None,
        location_code: str | None = None,
        satellite_image_path: str | Path | None = None,
        project_root: str | Path = ".",
    ) -> None:
        self.data = data
        self.frames = frames_from_data(data)
        self.location_code = location_code or infer_location_code(data, input_path)
        self.satellite_image_path = resolve_satellite_image_path(
            self.location_code,
            explicit_path=satellite_image_path,
            project_root=project_root,
        )
        if self.satellite_image_path is None:
            raise FileNotFoundError(
                "Could not resolve satellite image. Pass --sat-image or use "
                "location/<code>/sat_<code>.png."
            )
        self.satellite_image = Image.open(self.satellite_image_path)

    @classmethod
    def from_file(
        cls,
        input_path: str | Path,
        *,
        location_code: str | None = None,
        satellite_image_path: str | Path | None = None,
        project_root: str | Path = ".",
    ) -> "TrajectoryPlotter":
        data = load_json(input_path)
        return cls(
            data,
            input_path=input_path,
            location_code=location_code,
            satellite_image_path=satellite_image_path,
            project_root=project_root,
        )

    def extract_trajectories(
        self,
        *,
        min_points: int = 5,
    ) -> dict[int, list[tuple[float, float]]]:
        trajectories: dict[int, list[tuple[float, float]]] = {}

        for frame in self.frames:
            for obj in frame.get("objects", []):
                tracked_id = obj.get("tracked_id")
                sat_coords = obj.get("sat_coords") or obj.get("sat_coord")
                if tracked_id is None or not self._valid_point(sat_coords):
                    continue
                trajectories.setdefault(int(tracked_id), []).append(
                    (float(sat_coords[0]), float(sat_coords[1]))
                )

        return {
            track_id: points
            for track_id, points in trajectories.items()
            if len(points) >= min_points
        }

    def extract_classes(self) -> dict[int, str]:
        classes: dict[int, str] = {}
        for frame in self.frames:
            for obj in frame.get("objects", []):
                tracked_id = obj.get("tracked_id")
                if tracked_id is None:
                    continue
                classes[int(tracked_id)] = str(obj.get("class", "unknown"))
        return classes

    def extract_headings(self) -> dict[int, list[tuple[float, float, float | None]]]:
        headings: dict[int, list[tuple[float, float, float | None]]] = {}

        for frame in self.frames:
            for obj in frame.get("objects", []):
                tracked_id = obj.get("tracked_id")
                sat_coords = obj.get("sat_coords") or obj.get("sat_coord")
                if tracked_id is None or not self._valid_point(sat_coords):
                    continue

                heading = obj.get("heading", obj.get("heading_deg", obj.get("yaw")))
                heading_value = float(heading) if isinstance(heading, (int, float)) else None
                headings.setdefault(int(tracked_id), []).append(
                    (float(sat_coords[0]), float(sat_coords[1]), heading_value)
                )

        return headings

    def extract_keypoints(self) -> dict[int, list[tuple[float, float, int]]]:
        """Per-track (x, y, kp_idx) for every valid kp_sat entry across all frames."""
        keypoints: dict[int, list[tuple[float, float, int]]] = {}

        for frame in self.frames:
            for obj in frame.get("objects", []):
                tracked_id = obj.get("tracked_id")
                kp_sat = obj.get("kp_sat")
                if tracked_id is None or not kp_sat:
                    continue
                for kp_idx, kp in enumerate(kp_sat):
                    if not self._valid_point(kp):
                        continue
                    keypoints.setdefault(int(tracked_id), []).append(
                        (float(kp[0]), float(kp[1]), kp_idx)
                    )

        return keypoints

    def extract_keypoint_trajectory(
        self,
        kp_name: str,
        *,
        track_ids: Iterable[int] | None = None,
    ) -> dict[int, list[tuple[float, float]]]:
        """Per-track (x, y) path for a single named kp_sat keypoint, in frame order."""
        normalized = kp_name.replace("-", "_")
        if normalized not in KP_NAMES:
            raise ValueError(
                f"Unknown keypoint name {kp_name!r}. Valid names: {', '.join(KP_NAMES)}"
            )
        kp_idx = KP_NAMES.index(normalized)
        id_filter = {int(t) for t in track_ids} if track_ids is not None else None

        trajectory: dict[int, list[tuple[float, float]]] = {}
        for frame in self.frames:
            for obj in frame.get("objects", []):
                tracked_id = obj.get("tracked_id")
                if tracked_id is None:
                    continue
                tracked_id = int(tracked_id)
                if id_filter is not None and tracked_id not in id_filter:
                    continue
                kp_sat = obj.get("kp_sat")
                if not kp_sat or kp_idx >= len(kp_sat):
                    continue
                kp = kp_sat[kp_idx]
                if not self._valid_point(kp):
                    continue
                trajectory.setdefault(tracked_id, []).append((float(kp[0]), float(kp[1])))

        return trajectory

    def plot_scatter(
        self,
        output_path: str | Path,
        *,
        title: str | None = None,
        dpi: int = 200,
        color: str = "red",
        point_size: float = 6,
        alpha: float = 0.5,
    ) -> Path:
        """Plot all sat_coords as a scatter — for files without tracked_id."""
        xs, ys = [], []
        for frame in self.frames:
            for obj in frame.get("objects", []):
                sat_coords = obj.get("sat_coords") or obj.get("sat_coord")
                if self._valid_point(sat_coords):
                    xs.append(float(sat_coords[0]))
                    ys.append(float(sat_coords[1]))

        if not xs:
            raise ValueError("No valid sat_coords found in file.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(
            self.satellite_image,
            extent=[0, self.satellite_image.width, self.satellite_image.height, 0],
        )
        ax.scatter(xs, ys, s=point_size, c=color, alpha=alpha,
                   edgecolors="white", linewidths=0.4)
        ax.set_xlim(0, self.satellite_image.width)
        ax.set_ylim(self.satellite_image.height, 0)
        ax.set_aspect("equal")
        ax.set_title(title or f"TrafficLab Detection Scatter - {self.location_code}", fontsize=16)
        ax.set_xlabel("X Coordinate (pixels)", fontsize=12)
        ax.set_ylabel("Y Coordinate (pixels)", fontsize=12)
        ax.text(
            0.02,
            0.98,
            f"Total points: {len(xs)}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def compute_zoom_transform(
        self,
        selected_ids: Iterable[int] | None = None,
        *,
        margin_px: int = 200,
        min_points: int = 1,
        skip_out_of_bounds: bool = False,
    ) -> dict[str, float]:
        """Compute a single fixed zoom/crop window from a track's full trajectory
        (across every frame it appears in), for use as a shared viewport when
        rendering one plot per frame with plot_frame() — every frame gets the
        same view_min/max_x/y instead of independently fitting to just that
        frame's points, so the zoom and framing don't jump around frame to frame.
        """
        trajectories = self.extract_trajectories(min_points=min_points)
        if selected_ids is not None:
            selected_id_set = {int(track_id) for track_id in selected_ids}
            trajectories = {
                track_id: points
                for track_id, points in trajectories.items()
                if track_id in selected_id_set
            }
        if skip_out_of_bounds:
            trajectories = self._filter_visible_trajectories(trajectories)
        if not trajectories:
            raise ValueError("No trajectories matched the requested selection.")
        return self._zoom_transform(trajectories, margin_px=margin_px)

    def plot_frame(
        self,
        output_path: str | Path,
        frame_index: int,
        *,
        transform: dict[str, float],
        selected_ids: Iterable[int] | None = None,
        show_heading_arrows: bool = True,
        show_keypoints: bool = True,
        title: str | None = None,
        dpi: int = 200,
    ) -> Path:
        """Plot a single frame's object positions/headings/keypoints within a
        fixed, externally-supplied zoom window (see compute_zoom_transform).
        """
        frame = next((f for f in self.frames if f.get("frame_index") == frame_index), None)
        if frame is None:
            raise ValueError(f"frame_index {frame_index} not found in replay JSON.")

        objects = frame.get("objects", [])
        if selected_ids is not None:
            selected_id_set = {int(track_id) for track_id in selected_ids}
            objects = [o for o in objects if o.get("tracked_id") in selected_id_set]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        zoom_scale = transform["scale"]
        size_mult = max(1.0, zoom_scale) ** 0.5
        marker_size = min(16.0, 3.0 * size_mult)
        marker_edge_width = min(2.0, 0.6 * size_mult)
        arrow_lw = min(3.0, 1.0 * size_mult)
        arrow_mutation_scale = min(24.0, 8.0 * size_mult)
        keypoint_marker_size = min(80.0, 10.0 * size_mult)
        keypoint_marker_radius_pt = min(9.0, 3.0 * size_mult)
        kp_font_size = min(10.0, 6.0 * (size_mult ** 0.5))
        arrow_length = self._heading_arrow_length(transform)

        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(
            self.satellite_image,
            extent=[0, self.satellite_image.width, self.satellite_image.height, 0],
        )

        track_ids = sorted(
            {o.get("tracked_id") for o in objects if o.get("tracked_id") is not None}
        )
        color_map = self._color_map_for_ids(track_ids)

        id_label_requests = []   # (point, track_id, color)
        kp_label_requests = []   # (point, label)

        for obj in objects:
            tracked_id = obj.get("tracked_id")
            sat_coords = obj.get("sat_coords") or obj.get("sat_coord")
            color = color_map.get(tracked_id, "red")

            if show_keypoints:
                kp_sat = obj.get("kp_sat") or []
                kp_points = [
                    (float(kp[0]), float(kp[1]), kp_idx)
                    for kp_idx, kp in enumerate(kp_sat)
                    if self._valid_point(kp)
                ]
                if kp_points:
                    kp_x = [p[0] for p in kp_points]
                    kp_y = [p[1] for p in kp_points]
                    kp_colors = [_kp_part_color(KP_NAMES[p[2]]) for p in kp_points]
                    ax.scatter(
                        kp_x, kp_y,
                        s=keypoint_marker_size,
                        c=kp_colors,
                        edgecolors="white",
                        linewidths=0.4,
                        alpha=0.85,
                        zorder=2,
                    )
                    id_str = str(tracked_id) if tracked_id is not None else "?"
                    for x, y, kp_idx in kp_points:
                        kp_label_requests.append(((x, y), f"{id_str}-{KP_NAMES[kp_idx]}"))

            if not self._valid_point(sat_coords):
                continue

            ax.plot(
                sat_coords[0], sat_coords[1],
                marker="o",
                markersize=marker_size,
                color=color,
                markeredgecolor="white",
                markeredgewidth=marker_edge_width,
                linestyle="None",
                zorder=3,
            )

            if tracked_id is not None:
                id_label_requests.append(((sat_coords[0], sat_coords[1]), tracked_id, color))

            if show_heading_arrows:
                heading = obj.get("heading", obj.get("heading_deg", obj.get("yaw")))
                heading_rad = self._heading_to_radians(heading) if isinstance(heading, (int, float)) else None
                if heading_rad is not None:
                    dx = arrow_length * np.cos(heading_rad)
                    dy = arrow_length * np.sin(heading_rad)
                    ax.annotate(
                        "",
                        xy=(sat_coords[0] + dx, sat_coords[1] + dy),
                        xytext=(sat_coords[0], sat_coords[1]),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=color,
                            lw=arrow_lw,
                            mutation_scale=arrow_mutation_scale,
                        ),
                        zorder=4,
                    )

        ax.set_xlim(transform["view_min_x"], transform["view_max_x"])
        ax.set_ylim(transform["view_max_y"], transform["view_min_y"])
        ax.set_aspect("equal")
        ax.set_title(title or f"TrafficLab Frame {frame_index} - {self.location_code}", fontsize=16)
        ax.set_xlabel("X Coordinate (pixels)", fontsize=12)
        ax.set_ylabel("Y Coordinate (pixels)", fontsize=12)

        # Labels are placed after xlim/ylim are set (ax.transData needs current
        # data limits to be accurate) and after every marker is drawn, so their
        # occupied-box search can avoid covering markers as well as each other.
        # Vehicle ID labels are reserved/drawn first — one per vehicle, higher
        # priority to keep clear — before the (much more numerous) keypoint
        # labels compete for space around them.
        occupied_boxes = [self._marker_box(ax, point, marker_size / 2.0 + 2.0)
                           for point, _tid, _color in id_label_requests]
        occupied_boxes += [self._marker_box(ax, point, keypoint_marker_radius_pt)
                            for point, _label in kp_label_requests]

        for point, tracked_id, color in id_label_requests:
            self._draw_id_label(ax, tracked_id, point, color, occupied_boxes)

        for point, label in kp_label_requests:
            self._draw_kp_label(ax, point, label, occupied_boxes, font_size=kp_font_size)

        legend_lines = [f"Frame: {frame_index}", f"Zoom Scale: {zoom_scale:.2f}x", "", "Legend:", "  dot: position"]
        if show_heading_arrows:
            legend_lines.append("  arrow: heading")
        if show_keypoints:
            legend_lines.append("  small dot: keypoint, labeled id-part (color = car part)")
        ax.text(
            0.02, 0.98,
            "\n".join(legend_lines),
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot(
        self,
        output_path: str | Path,
        *,
        selected_ids: Iterable[int] | None = None,
        zoom_to_fit: bool = False,
        show_heading_arrows: bool = False,
        show_id_labels: bool = False,
        show_keypoints: bool = False,
        keypoint_trajectory_names: list[str] | None = None,
        skip_out_of_bounds: bool = True,
        title: str | None = None,
        dpi: int = 300,
        min_points: int = 5,
        zoom_margin_px: int = 200,
    ) -> Path:
        trajectories = self.extract_trajectories(min_points=min_points)
        if selected_ids is not None:
            selected_id_set = {int(track_id) for track_id in selected_ids}
            trajectories = {
                track_id: points
                for track_id, points in trajectories.items()
                if track_id in selected_id_set
            }

        if skip_out_of_bounds:
            trajectories = self._filter_visible_trajectories(trajectories)

        if not trajectories:
            raise ValueError("No trajectories matched the requested selection.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        object_classes = self.extract_classes()
        headings = self.extract_headings() if show_heading_arrows else {}
        keypoints_by_id = self.extract_keypoints() if show_keypoints else {}
        transform = self._zoom_transform(trajectories, margin_px=zoom_margin_px) if zoom_to_fit else None
        arrow_length = self._heading_arrow_length(transform) if show_heading_arrows else 0.0

        # Marker/line sizes below are fixed in points, so they don't grow on
        # their own as the view zooms in (unlike the satellite image, which
        # visibly pixelates) — without this they end up looking tiny and thin
        # against an increasingly blown-up background. Scale them up with the
        # zoom factor (dampened by sqrt so they don't balloon at extreme
        # zoom); no effect on the unzoomed default (scale == 1.0).
        zoom_scale = transform["scale"] if transform else 1.0
        size_mult = max(1.0, zoom_scale) ** 0.5
        marker_size = min(14.0, 2.0 * size_mult)
        marker_edge_width = min(2.0, 0.4 * size_mult)
        arrow_lw = min(3.0, 0.8 * size_mult)
        arrow_mutation_scale = min(20.0, 6.0 * size_mult)
        # scatter's `s` is a marker *area* (points^2), unlike ax.plot's
        # markersize (points) above — base value chosen for a small dot at
        # zoom_scale == 1, not derived from marker_size.
        keypoint_marker_size = min(60.0, 8.0 * size_mult)

        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(
            self.satellite_image,
            extent=[0, self.satellite_image.width, self.satellite_image.height, 0],
        )

        legend_items = []
        label_requests = []
        color_map = self._color_map_for_ids(sorted(trajectories))
        for track_id in sorted(trajectories):
            points = trajectories[track_id]
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]
            color = color_map[track_id]
            line = ax.plot(
                x_coords,
                y_coords,
                linestyle="None",
                marker="o",
                markersize=marker_size,
                color=color,
                markeredgecolor="white",
                markeredgewidth=marker_edge_width,
                alpha=0.92,
            )[0]

            if len(legend_items) < 15:
                obj_class = object_classes.get(track_id, "unknown")
                legend_items.append((line, f"ID {track_id} ({obj_class})"))

            if show_keypoints:
                kp_points = keypoints_by_id.get(track_id, [])
                if kp_points:
                    kp_x = [p[0] for p in kp_points]
                    kp_y = [p[1] for p in kp_points]
                    kp_colors = [_kp_part_color(KP_NAMES[p[2]]) for p in kp_points]
                    ax.scatter(
                        kp_x, kp_y,
                        s=keypoint_marker_size,
                        c=kp_colors,
                        edgecolors="white",
                        linewidths=0.3,
                        alpha=0.8,
                        zorder=1.5,
                    )

            if show_heading_arrows:
                for point_x, point_y, heading in headings.get(track_id, []):
                    heading_rad = self._heading_to_radians(heading)
                    if heading_rad is None:
                        continue
                    dx = arrow_length * np.cos(heading_rad)
                    dy = arrow_length * np.sin(heading_rad)
                    ax.annotate(
                        "",
                        xy=(point_x + dx, point_y + dy),
                        xytext=(point_x, point_y),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=color,
                            lw=arrow_lw,
                            mutation_scale=arrow_mutation_scale,
                        ),
                        zorder=4,
                    )

            if show_id_labels:
                label_point = self._label_point(points)
                if label_point is not None:
                    label_requests.append((track_id, label_point, color))

        if keypoint_trajectory_names:
            kp_line_width = min(3.0, 1.2 * size_mult)
            for kp_index, kp_name in enumerate(keypoint_trajectory_names):
                kp_color = _KP_TRAJECTORY_COLORS[kp_index % len(_KP_TRAJECTORY_COLORS)]
                kp_trajectory = self.extract_keypoint_trajectory(kp_name, track_ids=trajectories.keys())
                legend_line = None
                for track_id in sorted(kp_trajectory):
                    points = kp_trajectory[track_id]
                    kp_x = [point[0] for point in points]
                    kp_y = [point[1] for point in points]
                    legend_line = ax.plot(
                        kp_x,
                        kp_y,
                        linestyle="-",
                        linewidth=kp_line_width,
                        color=kp_color,
                        alpha=0.85,
                        zorder=1.9,
                    )[0]
                if legend_line is not None:
                    legend_items.append((legend_line, kp_name.replace("-", "_")))

        if transform:
            ax.set_xlim(transform["view_min_x"], transform["view_max_x"])
            ax.set_ylim(transform["view_max_y"], transform["view_min_y"])
        else:
            ax.set_xlim(0, self.satellite_image.width)
            ax.set_ylim(self.satellite_image.height, 0)

        if show_id_labels:
            # Seed with every plotted point's marker footprint (not just the
            # label anchor points) so placement search steers labels off the
            # dense trail of dots, not just away from other labels.
            occupied_label_boxes = [
                self._marker_box(ax, point, marker_size / 2.0 + 2.0)
                for points in trajectories.values()
                for point in points
            ]
            for track_id, label_point, color in label_requests:
                self._draw_id_label(ax, track_id, label_point, color, occupied_label_boxes)

        ax.set_aspect("equal")
        ax.set_title(title or f"TrafficLab Trajectories - {self.location_code}", fontsize=16)
        ax.set_xlabel("X Coordinate (pixels)", fontsize=12)
        ax.set_ylabel("Y Coordinate (pixels)", fontsize=12)

        if legend_items:
            lines, labels = zip(*legend_items)
            legend = ax.legend(lines, labels, loc="upper right", framealpha=0.9)
            legend.get_frame().set_facecolor("white")

        ax.text(
            0.02,
            0.98,
            self._stats_text(trajectories, object_classes, transform, show_heading_arrows, show_keypoints),
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    @staticmethod
    def _valid_point(value: Any) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        )

    @classmethod
    def _color_map_for_ids(cls, track_ids: list[int]) -> dict[int, str]:
        """Create a high-contrast color map for the currently visible tracks."""
        color_map = {}
        for index, track_id in enumerate(track_ids):
            hue = (index * cls._GOLDEN_RATIO_CONJUGATE) % 1.0
            saturation, value = cls._COLOR_VARIANTS[index % len(cls._COLOR_VARIANTS)]
            red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
            color_map[track_id] = mcolors.to_hex((red, green, blue))
        return color_map

    def _filter_visible_trajectories(
        self,
        trajectories: dict[int, list[tuple[float, float]]],
    ) -> dict[int, list[tuple[float, float]]]:
        return {
            track_id: points
            for track_id, points in trajectories.items()
            if any(self._point_in_image(point) for point in points)
        }

    def _point_in_image(self, point: tuple[float, float]) -> bool:
        return (
            0 <= point[0] <= self.satellite_image.width
            and 0 <= point[1] <= self.satellite_image.height
        )

    def _label_point(self, points: list[tuple[float, float]]) -> tuple[float, float] | None:
        visible_points = [point for point in points if self._point_in_image(point)]
        if not visible_points:
            return None
        return visible_points[len(visible_points) // 2]

    @classmethod
    def _draw_id_label(
        cls,
        ax,
        track_id: int,
        point: tuple[float, float],
        color: str,
        occupied_boxes: list[tuple[float, float, float, float]],
    ) -> None:
        label = str(track_id)
        offset, alignment, box = cls._label_placement(ax, point, label, occupied_boxes)
        occupied_boxes.append(box)
        ax.annotate(
            label,
            xy=point,
            xytext=offset,
            textcoords="offset points",
            color=color,
            fontsize=cls._LABEL_FONT_SIZE,
            fontweight="bold",
            horizontalalignment=alignment[0],
            verticalalignment=alignment[1],
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.86,
            },
            zorder=6,
        )

    @classmethod
    def _draw_kp_label(
        cls,
        ax,
        point: tuple[float, float],
        label: str,
        occupied_boxes: list[tuple[float, float, float, float]],
        *,
        font_size: float = 6.0,
    ) -> None:
        """Small leader-line label for a keypoint, distinct from _draw_id_label's
        bold colored-box style (there can be up to 24 of these per vehicle, so
        each one needs to stay visually light). Shares the same overlap-avoiding
        placement search as _draw_id_label via _label_placement/occupied_boxes."""
        offset, alignment, box = cls._label_placement(ax, point, label, occupied_boxes, font_size=font_size)
        occupied_boxes.append(box)
        ax.annotate(
            label,
            xy=point,
            xytext=offset,
            textcoords="offset points",
            color="black",
            fontsize=font_size,
            horizontalalignment=alignment[0],
            verticalalignment=alignment[1],
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
            zorder=5,
        )

    @staticmethod
    def _marker_box(ax, point: tuple[float, float], radius_pt: float) -> tuple[float, float, float, float]:
        """Occupied-box footprint for a plotted marker (not a label), so label
        placement can avoid covering markers as well as other labels."""
        px, py = ax.transData.transform(point)
        r = radius_pt * ax.figure.dpi / 72.0
        return (px - r, py - r, px + r, py + r)

    @classmethod
    def _label_placement(
        cls,
        ax,
        point: tuple[float, float],
        label: str,
        occupied_boxes: list[tuple[float, float, float, float]],
        *,
        font_size: float | None = None,
    ) -> tuple[tuple[int, int], tuple[str, str], tuple[float, float, float, float]]:
        for offset in cls._label_offsets():
            alignment = cls._label_alignment(offset)
            box = cls._estimate_label_box(ax, point, offset, alignment, label, font_size=font_size)
            if not any(cls._boxes_overlap(box, occupied) for occupied in occupied_boxes):
                return offset, alignment, box

        offset = (72, 72)
        alignment = ("left", "bottom")
        box = cls._estimate_label_box(ax, point, offset, alignment, label, font_size=font_size)
        return offset, alignment, box

    @staticmethod
    def _label_offsets() -> list[tuple[int, int]]:
        offsets = []
        directions = [
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ]
        for radius in (8, 18, 30, 44, 60, 78):
            offsets.extend((dx * radius, dy * radius) for dx, dy in directions)
        return offsets

    @staticmethod
    def _label_alignment(offset: tuple[int, int]) -> tuple[str, str]:
        horizontal = "left" if offset[0] >= 0 else "right"
        vertical = "bottom" if offset[1] >= 0 else "top"
        return horizontal, vertical

    @classmethod
    def _estimate_label_box(
        cls,
        ax,
        point: tuple[float, float],
        offset: tuple[int, int],
        alignment: tuple[str, str],
        label: str,
        *,
        font_size: float | None = None,
    ) -> tuple[float, float, float, float]:
        fs = font_size if font_size is not None else cls._LABEL_FONT_SIZE
        point_x, point_y = ax.transData.transform(point)
        offset_x = offset[0] * ax.figure.dpi / 72.0
        offset_y = offset[1] * ax.figure.dpi / 72.0
        anchor_x = point_x + offset_x
        anchor_y = point_y + offset_y
        width = max(24.0, len(label) * fs * 0.72 + 12.0)
        height = fs * 1.65

        if alignment[0] == "left":
            min_x, max_x = anchor_x, anchor_x + width
        else:
            min_x, max_x = anchor_x - width, anchor_x

        if alignment[1] == "bottom":
            min_y, max_y = anchor_y, anchor_y + height
        else:
            min_y, max_y = anchor_y - height, anchor_y

        padding = 4.0
        return min_x - padding, min_y - padding, max_x + padding, max_y + padding

    @staticmethod
    def _boxes_overlap(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> bool:
        return not (
            first[2] <= second[0]
            or second[2] <= first[0]
            or first[3] <= second[1]
            or second[3] <= first[1]
        )

    @staticmethod
    def _darker_color(color: str, factor: float = 0.65) -> tuple[float, float, float]:
        rgb = np.array(mcolors.to_rgb(color))
        return tuple(np.clip(rgb * factor, 0, 1))

    def _zoom_transform(
        self,
        trajectories: dict[int, list[tuple[float, float]]],
        margin_px: int = 200,
    ) -> dict[str, float]:
        points = [point for trajectory in trajectories.values() for point in trajectory]
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        span_x = max_x - min_x
        span_y = max_y - min_y

        image_width = self.satellite_image.width
        image_height = self.satellite_image.height

        # Crop window = trajectory bounding box padded by margin_px on every
        # side, in the same satellite-pixel units as span_x/span_y — so
        # margin_px means exactly "this many satellite pixels of padding"
        # regardless of trajectory size or image resolution. (Previously this
        # derived a "scale" from image_width/image_height minus margin, which
        # doesn't correspond to any real padding amount and made margin_px
        # nearly meaningless.)
        raw_view_width = span_x + 2 * margin_px
        raw_view_height = span_y + 2 * margin_px

        # Grow (never shrink) to match the image's aspect ratio, so imshow
        # isn't stretched and the requested margin is never reduced below
        # margin_px on the constraining axis.
        image_aspect = image_width / image_height
        if raw_view_width / raw_view_height > image_aspect:
            view_width = raw_view_width
            view_height = raw_view_width / image_aspect
        else:
            view_height = raw_view_height
            view_width = raw_view_height * image_aspect

        scale = image_width / view_width
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        view_min_x = max(center_x - view_width / 2.0, 0.0)
        view_max_x = min(center_x + view_width / 2.0, float(image_width))
        view_min_y = max(center_y - view_height / 2.0, 0.0)
        view_max_y = min(center_y + view_height / 2.0, float(image_height))

        return {
            "scale": float(scale),
            "span_x": float(span_x),
            "span_y": float(span_y),
            "view_min_x": float(view_min_x),
            "view_max_x": float(view_max_x),
            "view_min_y": float(view_min_y),
            "view_max_y": float(view_max_y),
            "view_width": float(view_max_x - view_min_x),
            "view_height": float(view_max_y - view_min_y),
            "margin_px": float(margin_px),
        }

    def _heading_arrow_length(self, transform: dict[str, float] | None = None) -> float:
        if transform:
            base_span = max(min(transform["view_width"], transform["view_height"]), 1.0)
        else:
            base_span = max(min(self.satellite_image.width, self.satellite_image.height), 1.0)
        return max(base_span * 0.035, 8.0)

    @staticmethod
    def _heading_to_radians(heading: float | None) -> float | None:
        if heading is None:
            return None
        if abs(heading) <= 2 * np.pi:
            return heading
        return float(np.deg2rad(heading))

    @staticmethod
    def _stats_text(
        trajectories: dict[int, list[tuple[float, float]]],
        object_classes: dict[int, str],
        transform: dict[str, float] | None,
        show_heading_arrows: bool,
        show_keypoints: bool = False,
    ) -> str:
        class_counts: dict[str, int] = {}
        for track_id in trajectories:
            obj_class = object_classes.get(track_id, "unknown")
            class_counts[obj_class] = class_counts.get(obj_class, 0) + 1

        lines = [f"Selected Trajectories: {len(trajectories)}", "Selected Classes:"]
        lines.extend(f"  {name}: {count}" for name, count in sorted(class_counts.items()))
        if transform:
            lines.extend(
                [
                    "",
                    f"Zoom Scale: {transform['scale']:.3f}x",
                    f"Trajectory Span: {transform['span_x']:.1f} x {transform['span_y']:.1f}",
                    f"View Window: {transform['view_width']:.1f} x {transform['view_height']:.1f}",
                ]
            )
        if show_heading_arrows or show_keypoints:
            lines.extend(["", "Legend:"])
            if show_heading_arrows:
                lines.append("  arrow: heading")
            if show_keypoints:
                lines.append("  dot: keypoint (color = car part)")
        return "\n".join(lines)
