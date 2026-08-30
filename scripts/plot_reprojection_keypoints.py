#!/usr/bin/env python3
"""Plot per-vehicle keypoint reprojections (kp_sat) from an h-aware replay
JSON onto the satellite image for a single frame, with a black leader line
from each point to its "tracked_id-keypoint_name" label. Each vehicle's
localized position (sat_coords) is also plotted, marked with a black edge
to distinguish it from the white-edged keypoint dots.
"""

from __future__ import annotations

import argparse
import colorsys
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from trafficlab.trajectory.io import (
    infer_location_code,
    load_json,
    resolve_satellite_image_path,
)

# ApolloCar3D 24-keypoint order (matches KP_NAMES / the p_sat / kp_sat index
# used by trafficlab/motion/keypoints_openpifpaf.py).
KEYPOINT_NAMES = [
    'front_glass_top_right', 'front_glass_top_left', 'front_light_right', 'front_light_left',
    'front_low_fog_light_right', 'front_low_fog_light_left', 'front_door_top_left', 'front_wheel_center_left',
    'rear_wheel_center_left', 'rear_corner_left', 'rear_glass_up_left', 'rear_glass_up_right',
    'rear_light_left', 'rear_light_right', 'rear_bumper_left', 'rear_bumper_right',
    'front_door_top_right', 'rear_corner_right', 'rear_wheel_center_right', 'front_wheel_center_right',
    'rear_plate_left', 'rear_plate_right', 'front_door_base_left', 'front_door_base_right',
]

_GOLDEN_RATIO_CONJUGATE = 0.618033988749895
_LABEL_FONT_SIZE = 6
_LABEL_OFFSETS = [
    (dx * radius, dy * radius)
    for radius in (10, 18, 28, 40, 55, 72, 92, 115)
    for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1))
]
_POINT_MARKER_RADIUS_PT = 7.0
_POSITION_MARKER_RADIUS_PT = 9.0


def _parse_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _color_for_id(tracked_id: int) -> str:
    hue = (tracked_id * _GOLDEN_RATIO_CONJUGATE) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return mcolors.to_hex((r, g, b))


def _pick_richest_frame(frames: list[dict]) -> dict:
    def n_valid_kp(frame: dict) -> int:
        total = 0
        for obj in frame.get("objects", []):
            total += sum(1 for kp in (obj.get("kp_sat") or []) if kp is not None)
        return total

    return max(frames, key=n_valid_kp)


def _estimate_box(ax, point, offset, label):
    px, py = ax.transData.transform(point)
    ox = offset[0] * ax.figure.dpi / 72.0
    oy = offset[1] * ax.figure.dpi / 72.0
    ax_, ay_ = px + ox, py + oy
    width = max(20.0, len(label) * _LABEL_FONT_SIZE * 0.72 + 8.0)
    height = _LABEL_FONT_SIZE * 1.6
    min_x = ax_ - width / 2.0
    min_y = ay_ - height / 2.0
    return (min_x - 3.0, min_y - 3.0, min_x + width + 3.0, min_y + height + 3.0)


def _point_box(ax, point, radius_pt=_POINT_MARKER_RADIUS_PT):
    px, py = ax.transData.transform(point)
    r = radius_pt * ax.figure.dpi / 72.0
    return (px - r, py - r, px + r, py + r)


def _boxes_overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _within_bounds(box, bounds) -> bool:
    return (box[0] >= bounds[0] and box[1] >= bounds[1]
            and box[2] <= bounds[2] and box[3] <= bounds[3])


def _place_label(ax, point, label, occupied, bounds):
    """Pick the offset that overlaps nothing (a prior label box or a point
    marker box, both stored in `occupied`); if none is fully clear, fall back
    to the in-bounds offset with the fewest overlaps."""
    best_in_bounds = None  # (overlap_count, offset, box)
    best_any = None
    for offset in _LABEL_OFFSETS:
        box = _estimate_box(ax, point, offset, label)
        overlap_count = sum(1 for other in occupied if _boxes_overlap(box, other))
        in_bounds = _within_bounds(box, bounds)
        if overlap_count == 0 and in_bounds:
            occupied.append(box)
            return offset
        if in_bounds and (best_in_bounds is None or overlap_count < best_in_bounds[0]):
            best_in_bounds = (overlap_count, offset, box)
        if best_any is None or overlap_count < best_any[0]:
            best_any = (overlap_count, offset, box)

    _, offset, box = best_in_bounds if best_in_bounds is not None else best_any
    occupied.append(box)
    return offset


def plot_frame_keypoints(
    data: dict,
    frame_index: int | None,
    out_path: Path,
    *,
    sat_image_path: Path,
    ids: set[int] | None,
    dpi: int,
) -> tuple[Path, int]:
    frames = data.get("frames") or []
    if not frames:
        raise ValueError("No frames found in replay JSON.")

    if frame_index is None:
        frame = _pick_richest_frame(frames)
        frame_index = frame.get("frame_index")
        print(f"--frame-index not given; auto-selected frame {frame_index} "
              f"(most valid keypoints).")
    else:
        matches = [f for f in frames if f.get("frame_index") == frame_index]
        if not matches:
            raise ValueError(f"frame_index {frame_index} not found in replay JSON.")
        frame = matches[0]

    objects = frame.get("objects", [])
    if ids is not None:
        objects = [o for o in objects if o.get("tracked_id") in ids]

    points = []  # (x, y, tracked_id, kp_idx, color)
    pos_points = []  # (x, y, tracked_id, color) -- vehicle sat_coords
    for obj in objects:
        tracked_id = obj.get("tracked_id")
        color = _color_for_id(tracked_id if tracked_id is not None else 0)
        kp_sat = obj.get("kp_sat") or []
        for kp_idx, kp in enumerate(kp_sat):
            if kp is None:
                continue
            points.append((float(kp[0]), float(kp[1]), tracked_id, kp_idx, color))

        sat_coords = obj.get("sat_coords")
        if sat_coords is not None and sat_coords[0] is not None and sat_coords[1] is not None:
            pos_points.append((float(sat_coords[0]), float(sat_coords[1]), tracked_id, color))

    if not points and not pos_points:
        raise ValueError(f"No valid kp_sat or sat_coords points in frame {frame_index}.")

    from PIL import Image

    sat_image = Image.open(sat_image_path)

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(sat_image, extent=[0, sat_image.width, sat_image.height, 0])

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    colors = [p[4] for p in points]
    ax.scatter(xs, ys, s=14, c=colors, edgecolors="white", linewidths=0.5, zorder=3)

    if pos_points:
        pxs = [p[0] for p in pos_points]
        pys = [p[1] for p in pos_points]
        pcolors = [p[3] for p in pos_points]
        ax.scatter(pxs, pys, s=45, c=pcolors, edgecolors="black", linewidths=1.2, zorder=4)

    ax.set_xlim(0, sat_image.width)
    ax.set_ylim(sat_image.height, 0)
    ax.set_aspect("equal")
    location_code = data.get("location_code", "?")
    ax.set_title(
        f"Reprojected Keypoints - {location_code} - frame {frame_index}", fontsize=16
    )

    fig.tight_layout()
    fig.canvas.draw()
    axes_box = ax.get_window_extent()
    padding = 4.0
    bounds = (axes_box.x0 + padding, axes_box.y0 + padding,
              axes_box.x1 - padding, axes_box.y1 - padding)

    # Reserve every point marker's footprint up front (keypoints and vehicle
    # positions alike) so no label is placed on top of any dot.
    occupied_boxes = [_point_box(ax, (x, y)) for x, y, *_ in points]
    occupied_boxes += [_point_box(ax, (x, y), radius_pt=_POSITION_MARKER_RADIUS_PT)
                        for x, y, *_ in pos_points]

    def _draw_label(x, y, label):
        offset = _place_label(ax, (x, y), label, occupied_boxes, bounds)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=_LABEL_FONT_SIZE,
            color="black",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.75},
            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
            zorder=5,
        )

    for x, y, tracked_id, kp_idx, color in points:
        _draw_label(x, y, f"{tracked_id}-{KEYPOINT_NAMES[kp_idx]}")

    ax.text(
        0.02, 0.98,
        f"Frame: {frame_index}\n"
        f"Vehicles: {len({p[2] for p in points} | {p[2] for p in pos_points})}\n"
        f"Keypoints: {len(points)}\nPositions: {len(pos_points)}",
        transform=ax.transAxes, fontsize=11, verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path, len(points), len(pos_points)


def _default_output(input_path: Path, frame_index: int | None) -> Path:
    name = input_path.name
    stem = name[:-8] if name.endswith(".json.gz") else Path(name).stem
    suffix = f"frame{frame_index}" if frame_index is not None else "frameauto"
    return input_path.with_name(f"{stem}.keypoints_{suffix}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", help="Path to *_reprojection.json.gz replay file.")
    parser.add_argument("--frame-index", type=int, default=None,
                         help="Frame to plot. Default: auto-pick the frame with the most valid keypoints.")
    parser.add_argument("--ids", help="Comma-separated tracked_id values to include (default: all).")
    parser.add_argument("--location-code", help="Override inferred location code.")
    parser.add_argument("--sat-image", help="Override satellite image path.")
    parser.add_argument("-o", "--out", help="Output PNG path (default: next to input JSON).")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    input_path = Path(args.json_path)
    data = load_json(input_path)
    location_code = args.location_code or infer_location_code(data, input_path)
    sat_image_path = resolve_satellite_image_path(
        location_code, explicit_path=args.sat_image, project_root=REPO_ROOT
    )
    if sat_image_path is None:
        raise SystemExit(
            "Could not resolve satellite image. Pass --sat-image or --location-code."
        )

    out_path = Path(args.out) if args.out else _default_output(input_path, args.frame_index)
    ids = _parse_ids(args.ids)

    out_path, n_points, n_positions = plot_frame_keypoints(
        data, args.frame_index, out_path,
        sat_image_path=sat_image_path, ids=ids, dpi=args.dpi,
    )
    print(f"Saved {n_points} keypoints and {n_positions} vehicle positions to {out_path}")


if __name__ == "__main__":
    main()
