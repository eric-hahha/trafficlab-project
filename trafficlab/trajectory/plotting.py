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

from trafficlab.trajectory.io import (
    frames_from_data,
    infer_location_code,
    load_json,
    resolve_satellite_image_path,
)


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

    def plot(
        self,
        output_path: str | Path,
        *,
        selected_ids: Iterable[int] | None = None,
        zoom_to_fit: bool = False,
        show_heading_arrows: bool = False,
        show_id_labels: bool = False,
        skip_out_of_bounds: bool = True,
        title: str | None = None,
        dpi: int = 300,
        min_points: int = 5,
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
        transform = self._zoom_transform(trajectories) if zoom_to_fit else None
        arrow_length = self._heading_arrow_length(transform) if show_heading_arrows else 0.0

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
                markersize=4,
                color=color,
                markeredgecolor=self._darker_color(color),
                markeredgewidth=1.0,
                alpha=0.92,
            )[0]

            if len(legend_items) < 15:
                obj_class = object_classes.get(track_id, "unknown")
                legend_items.append((line, f"ID {track_id} ({obj_class})"))

            ax.plot(
                x_coords[0],
                y_coords[0],
                "o",
                color="lime",
                markersize=3,
                markeredgecolor="darkgreen",
                markeredgewidth=2,
                alpha=0.9,
            )
            ax.plot(
                x_coords[-1],
                y_coords[-1],
                "s",
                color="red",
                markersize=3,
                markeredgecolor="darkred",
                markeredgewidth=2,
                alpha=0.9,
            )

            if show_heading_arrows:
                for point_x, point_y, heading in headings.get(track_id, []):
                    heading_rad = self._heading_to_radians(heading)
                    if heading_rad is None:
                        continue
                    ax.arrow(
                        point_x,
                        point_y,
                        arrow_length * np.cos(heading_rad),
                        arrow_length * np.sin(heading_rad),
                        color=color,
                        width=max(arrow_length * 0.03, 0.5),
                        head_width=max(arrow_length * 0.18, 3.0),
                        head_length=max(arrow_length * 0.22, 4.0),
                        length_includes_head=True,
                        alpha=0.75,
                        zorder=4,
                    )

            if show_id_labels:
                label_point = self._label_point(points)
                if label_point is not None:
                    label_requests.append((track_id, label_point, color))

        if transform:
            ax.set_xlim(transform["view_min_x"], transform["view_max_x"])
            ax.set_ylim(transform["view_max_y"], transform["view_min_y"])
        else:
            ax.set_xlim(0, self.satellite_image.width)
            ax.set_ylim(self.satellite_image.height, 0)

        if show_id_labels:
            occupied_label_boxes = []
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
            self._stats_text(trajectories, object_classes, transform, show_heading_arrows),
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
    def _label_placement(
        cls,
        ax,
        point: tuple[float, float],
        label: str,
        occupied_boxes: list[tuple[float, float, float, float]],
    ) -> tuple[tuple[int, int], tuple[str, str], tuple[float, float, float, float]]:
        for offset in cls._label_offsets():
            alignment = cls._label_alignment(offset)
            box = cls._estimate_label_box(ax, point, offset, alignment, label)
            if not any(cls._boxes_overlap(box, occupied) for occupied in occupied_boxes):
                return offset, alignment, box

        offset = (72, 72)
        alignment = ("left", "bottom")
        box = cls._estimate_label_box(ax, point, offset, alignment, label)
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
    ) -> tuple[float, float, float, float]:
        point_x, point_y = ax.transData.transform(point)
        offset_x = offset[0] * ax.figure.dpi / 72.0
        offset_y = offset[1] * ax.figure.dpi / 72.0
        anchor_x = point_x + offset_x
        anchor_y = point_y + offset_y
        width = max(24.0, len(label) * cls._LABEL_FONT_SIZE * 0.72 + 12.0)
        height = cls._LABEL_FONT_SIZE * 1.65

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
        margin_px: int = 10,
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
        drawable_width = max(image_width - 2 * margin_px, 1)
        drawable_height = max(image_height - 2 * margin_px, 1)

        if span_x == 0 and span_y == 0:
            scale = 1.0
        elif span_x == 0:
            scale = drawable_height / span_y
        elif span_y == 0:
            scale = drawable_width / span_x
        else:
            scale = min(drawable_width / span_x, drawable_height / span_y)

        view_width = image_width / scale
        view_height = image_height / scale
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
    ) -> str:
        class_counts: dict[str, int] = {}
        for track_id in trajectories:
            obj_class = object_classes.get(track_id, "unknown")
            class_counts[obj_class] = class_counts.get(obj_class, 0) + 1

        lines = [f"Selected Trajectories: {len(trajectories)}", "Selected Classes:"]
        lines.extend(f"  {name}: {count}" for name, count in sorted(class_counts.items()))
        lines.extend(["", "Legend:", "  green circle: start", "  red square: end"])
        if transform:
            lines.extend(
                [
                    "",
                    f"Zoom Scale: {transform['scale']:.3f}x",
                    f"Trajectory Span: {transform['span_x']:.1f} x {transform['span_y']:.1f}",
                    f"View Window: {transform['view_width']:.1f} x {transform['view_height']:.1f}",
                ]
            )
        if show_heading_arrows:
            lines.append("  arrow: heading")
        return "\n".join(lines)
