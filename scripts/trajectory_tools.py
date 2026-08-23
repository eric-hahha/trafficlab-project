#!/usr/bin/env python3
"""Trajectory smoothing and plotting CLI for TrafficLab replay outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_ids(value: str | None) -> list[int] | None:
    if not value:
        return None
    ids = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        ids.append(int(item))
    return ids or None


def _parse_kp_names(value: str | None) -> list[str] | None:
    if not value:
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    return names or None


def _default_plot_output(input_path: Path, suffix: str = "trajectories") -> Path:
    name = input_path.name
    if name.endswith(".json.gz"):
        stem = name[:-8]
    elif name.endswith(".json"):
        stem = name[:-5]
    else:
        stem = input_path.stem
    return input_path.with_name(f"{stem}.{suffix}.png")


def add_common_plot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ids", help="Comma-separated tracked_id values to include.")
    parser.add_argument("--location-code", help="Override inferred location code.")
    parser.add_argument("--sat-image", help="Explicit satellite image path.")
    parser.add_argument(
        "--min-points",
        type=int,
        default=5,
        help="Minimum points required for a track to be plotted. Defaults to 5.",
    )
    parser.add_argument("--zoom-to-fit", action="store_true", help="Zoom plot to selected tracks.")
    parser.add_argument(
        "--zoom-margin",
        type=int,
        default=200,
        help="Padding (satellite-image pixels) around the selected trajectory when "
        "--zoom-to-fit is set. Too small and the crop is just a blur of upsampled "
        "satellite pixels with no road/landmark context. Defaults to 200.",
    )
    parser.add_argument(
        "--include-out-of-bounds",
        action="store_true",
        help="Include tracks that are completely outside the satellite image bounds.",
    )
    parser.add_argument(
        "--show-id-labels",
        action="store_true",
        help="Draw each visible track's tracked_id next to the trajectory.",
    )
    parser.add_argument(
        "--show-keypoints",
        action="store_true",
        help="Draw each visible track's kp_sat keypoint projections, colored by car part.",
    )
    parser.add_argument(
        "--keypoint-trajectories",
        help="Comma-separated kp_sat keypoint names (see KP_NAMES in "
        "trafficlab/motion/keypoints_openpifpaf.py; hyphens or underscores both work) "
        "to draw as connected per-track paths, e.g. "
        "front-glass-top-left,front-door-top-left,front-door-base-left.",
    )
    parser.add_argument(
        "--show-heading-arrows",
        action="store_true",
        help="Draw heading/yaw arrows when heading fields exist.",
    )
    parser.add_argument("--title", help="Optional plot title.")


def run_scatter(args: argparse.Namespace) -> None:
    from trafficlab.trajectory import TrajectoryPlotter

    input_path = Path(args.input_path)
    output = Path(args.output) if args.output else _default_plot_output(input_path, "scatter")
    plotter = TrajectoryPlotter.from_file(
        input_path,
        location_code=args.location_code,
        satellite_image_path=args.sat_image,
    )
    output_path = plotter.plot_scatter(output, title=args.title)
    print(f"Scatter plot: {output_path}")


def run_smooth(args: argparse.Namespace) -> None:
    from trafficlab.trajectory import smooth_file

    output_path, stats = smooth_file(
        args.input_path,
        args.output,
        selected_ids=_parse_ids(args.ids),
        window_length=args.window_length,
        polyorder=args.polyorder,
        update_sat_center=not args.keep_sat_center,
    )
    print(f"Smoothed output: {output_path}")
    print(
        "Stats: "
        f"tracks={stats.total_tracks}, "
        f"smoothed={stats.smoothed_tracks}, "
        f"short_skipped={stats.skipped_short_tracks}, "
        f"invalid_points={stats.skipped_invalid_tracks}, "
        f"updated_points={stats.updated_points}"
    )


def run_plot(args: argparse.Namespace) -> None:
    from trafficlab.trajectory import TrajectoryPlotter

    input_path = Path(args.input_path)
    output = Path(args.output) if args.output else _default_plot_output(input_path)
    plotter = TrajectoryPlotter.from_file(
        input_path,
        location_code=args.location_code,
        satellite_image_path=args.sat_image,
    )
    output_path = plotter.plot(
        output,
        selected_ids=_parse_ids(args.ids),
        zoom_to_fit=args.zoom_to_fit,
        show_heading_arrows=args.show_heading_arrows,
        show_id_labels=args.show_id_labels,
        show_keypoints=args.show_keypoints,
        keypoint_trajectory_names=_parse_kp_names(args.keypoint_trajectories),
        skip_out_of_bounds=not args.include_out_of_bounds,
        title=args.title,
        min_points=args.min_points,
        zoom_margin_px=args.zoom_margin,
    )
    print(f"Trajectory plot: {output_path}")


def run_smooth_and_plot(args: argparse.Namespace) -> None:
    from trafficlab.trajectory import TrajectoryPlotter, smooth_file

    smoothed_path, stats = smooth_file(
        args.input_path,
        args.output,
        selected_ids=_parse_ids(args.ids),
        window_length=args.window_length,
        polyorder=args.polyorder,
        update_sat_center=not args.keep_sat_center,
    )
    plot_output = Path(args.plot_output) if args.plot_output else _default_plot_output(smoothed_path)
    plotter = TrajectoryPlotter.from_file(
        smoothed_path,
        location_code=args.location_code,
        satellite_image_path=args.sat_image,
    )
    plot_path = plotter.plot(
        plot_output,
        selected_ids=_parse_ids(args.ids),
        zoom_to_fit=args.zoom_to_fit,
        show_heading_arrows=args.show_heading_arrows,
        show_id_labels=args.show_id_labels,
        show_keypoints=args.show_keypoints,
        keypoint_trajectory_names=_parse_kp_names(args.keypoint_trajectories),
        skip_out_of_bounds=not args.include_out_of_bounds,
        title=args.title,
        min_points=args.min_points,
        zoom_margin_px=args.zoom_margin,
    )
    print(f"Smoothed output: {smoothed_path}")
    print(
        "Stats: "
        f"tracks={stats.total_tracks}, "
        f"smoothed={stats.smoothed_tracks}, "
        f"short_skipped={stats.skipped_short_tracks}, "
        f"invalid_points={stats.skipped_invalid_tracks}, "
        f"updated_points={stats.updated_points}"
    )
    print(f"Trajectory plot: {plot_path}")


def run_frames(args: argparse.Namespace) -> None:
    from trafficlab.trajectory import TrajectoryPlotter

    plotter = TrajectoryPlotter.from_file(
        args.input_path,
        location_code=args.location_code,
        satellite_image_path=args.sat_image,
    )
    ids = _parse_ids(args.ids)
    # Fixed once from the full trajectory (every frame this id appears in),
    # not per-frame, so the crop stays put across the whole sequence instead
    # of jumping to fit whatever's in just that one frame.
    transform = plotter.compute_zoom_transform(selected_ids=ids, margin_px=args.zoom_margin)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_set = set(ids) if ids is not None else None
    n_written = 0
    for frame in plotter.frames:
        objects = frame.get("objects", [])
        if id_set is not None:
            objects = [o for o in objects if o.get("tracked_id") in id_set]
        if not objects:
            continue
        frame_index = frame["frame_index"]
        plotter.plot_frame(
            out_dir / f"frame_{frame_index:04d}.png",
            frame_index,
            transform=transform,
            selected_ids=ids,
            show_heading_arrows=not args.hide_heading_arrows,
            show_keypoints=not args.hide_keypoints,
        )
        n_written += 1

    print(f"Wrote {n_written} frame(s) to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smooth and plot TrafficLab replay trajectories without mixing external scripts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scatter = subparsers.add_parser("scatter", help="Scatter-plot all sat_coords (for files without tracked_id).")
    scatter.add_argument("input_path", help="Input .json or .json.gz replay file.")
    scatter.add_argument("-o", "--output", help="Output PNG path.")
    scatter.add_argument("--location-code", help="Override inferred location code.")
    scatter.add_argument("--sat-image", help="Explicit satellite image path.")
    scatter.add_argument("--title", help="Optional plot title.")
    scatter.set_defaults(func=run_scatter)

    smooth = subparsers.add_parser("smooth", help="Smooth sat_coords in a replay JSON file.")
    smooth.add_argument("input_path", help="Input .json or .json.gz replay file.")
    smooth.add_argument("-o", "--output", help="Output .json or .json.gz path.")
    smooth.add_argument("--ids", help="Comma-separated tracked_id values to smooth.")
    smooth.add_argument("--window-length", type=int, default=45, help="Odd Savitzky-Golay window.")
    smooth.add_argument("--polyorder", type=int, default=3, help="Savitzky-Golay polynomial order.")
    smooth.add_argument(
        "--keep-sat-center",
        action="store_true",
        help="Do not mirror smoothed sat_coords into sat_center.",
    )
    smooth.set_defaults(func=run_smooth)

    plot = subparsers.add_parser("plot", help="Plot trajectories over a satellite image.")
    plot.add_argument("input_path", help="Input .json or .json.gz replay file.")
    plot.add_argument("-o", "--output", help="Output PNG path.")
    add_common_plot_args(plot)
    plot.set_defaults(func=run_plot)

    both = subparsers.add_parser("smooth-and-plot", help="Smooth trajectories, then plot the result.")
    both.add_argument("input_path", help="Input .json or .json.gz replay file.")
    both.add_argument("-o", "--output", help="Smoothed output .json or .json.gz path.")
    both.add_argument("--plot-output", help="Output PNG path.")
    both.add_argument("--window-length", type=int, default=45, help="Odd Savitzky-Golay window.")
    both.add_argument("--polyorder", type=int, default=3, help="Savitzky-Golay polynomial order.")
    both.add_argument(
        "--keep-sat-center",
        action="store_true",
        help="Do not mirror smoothed sat_coords into sat_center.",
    )
    add_common_plot_args(both)
    both.set_defaults(func=run_smooth_and_plot)

    frames = subparsers.add_parser(
        "frames",
        help="Plot one PNG per frame within a single fixed zoom window (e.g. to turn into a video with ffmpeg).",
    )
    frames.add_argument("input_path", help="Input .json or .json.gz replay file.")
    frames.add_argument("--out-dir", required=True, help="Directory to write per-frame PNGs.")
    frames.add_argument(
        "--ids",
        help="Comma-separated tracked_id values to include. Determines both which frames get "
        "written (only ones where a selected id appears) and the fixed zoom window (fit to "
        "these ids' combined trajectory). Omit to include every id and zoom to the whole image.",
    )
    frames.add_argument("--location-code", help="Override inferred location code.")
    frames.add_argument("--sat-image", help="Explicit satellite image path.")
    frames.add_argument(
        "--zoom-margin",
        type=int,
        default=200,
        help="Padding (satellite-image pixels) around the selected trajectory. Defaults to 200.",
    )
    frames.add_argument("--hide-heading-arrows", action="store_true", help="Don't draw heading arrows.")
    frames.add_argument("--hide-keypoints", action="store_true", help="Don't draw kp_sat keypoint projections.")
    frames.set_defaults(func=run_frames)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
