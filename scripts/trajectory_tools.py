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
        "--show-heading-arrows",
        action="store_true",
        help="Draw heading/yaw arrows when heading fields exist.",
    )
    parser.add_argument("--title", help="Optional plot title.")


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
        skip_out_of_bounds=not args.include_out_of_bounds,
        title=args.title,
        min_points=args.min_points,
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
        skip_out_of_bounds=not args.include_out_of_bounds,
        title=args.title,
        min_points=args.min_points,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smooth and plot TrafficLab replay trajectories without mixing external scripts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
