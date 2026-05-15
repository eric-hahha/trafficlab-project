import argparse
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

TMP_CACHE_DIR = Path("/private/tmp/trafficlab_plot_cache")
TMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_missing_heading(obj):
    return obj.get("heading") is None


def is_missing_speed(obj):
    return obj.get("speed_kmh") is None


def collect_missing_frames(data, selected_ids=None):
    selected_set = set(selected_ids) if selected_ids else None
    heading_missing = defaultdict(list)
    speed_missing = defaultdict(list)
    seen_tracks = set()

    for frame in data.get("frames", []):
        frame_index = frame.get("frame_index")
        for obj in frame.get("objects", []):
            track_id = obj.get("tracked_id")
            if track_id is None:
                continue
            if selected_set is not None and track_id not in selected_set:
                continue

            seen_tracks.add(track_id)

            if is_missing_heading(obj):
                heading_missing[track_id].append(frame_index)
            if is_missing_speed(obj):
                speed_missing[track_id].append(frame_index)

    return sorted(seen_tracks), heading_missing, speed_missing


def frames_to_ranges(frame_indices):
    if not frame_indices:
        return []

    ordered = sorted(frame_indices)
    ranges = []
    start = ordered[0]
    prev = ordered[0]

    for current in ordered[1:]:
        if current == prev + 1:
            prev = current
            continue
        ranges.append((start, prev))
        start = current
        prev = current

    ranges.append((start, prev))
    return ranges


def format_ranges(ranges, fps):
    formatted = []
    for start, end in ranges:
        start_s = start / fps if fps else start
        end_s = end / fps if fps else end
        duration_s = (end - start + 1) / fps if fps else (end - start + 1)
        formatted.append(
            {
                "start_frame": start,
                "end_frame": end,
                "start_sec": round(start_s, 3),
                "end_sec": round(end_s, 3),
                "duration_sec": round(duration_s, 3),
            }
        )
    return formatted


def print_summary(track_ids, heading_ranges, speed_ranges, fps):
    print(f"FPS: {fps}")
    print(f"Tracks analyzed: {track_ids}")
    for track_id in track_ids:
        heading_summary = format_ranges(heading_ranges.get(track_id, []), fps)
        speed_summary = format_ranges(speed_ranges.get(track_id, []), fps)
        print(f"tracked_id={track_id}")
        print(f"  missing_heading_ranges={heading_summary}")
        print(f"  missing_speed_ranges={speed_summary}")


def plot_ranges(track_ids, ranges_map, fps, axis, title, color):
    if not track_ids:
        axis.set_title(title)
        axis.text(0.5, 0.5, "No tracks found", ha="center", va="center", transform=axis.transAxes)
        axis.set_yticks([])
        return

    bar_height = 8
    row_gap = 4
    y_positions = []
    labels = []
    has_any_range = False

    for idx, track_id in enumerate(track_ids):
        y = idx * (bar_height + row_gap)
        y_positions.append(y + bar_height / 2)
        labels.append(str(track_id))

        spans = []
        for start, end in ranges_map.get(track_id, []):
            x_start = start / fps if fps else start
            width = (end - start + 1) / fps if fps else (end - start + 1)
            spans.append((x_start, max(width, 1e-6)))

        if spans:
            has_any_range = True
            axis.broken_barh(spans, (y, bar_height), facecolors=color)
            for start, end in ranges_map.get(track_id, []):
                x_start = start / fps if fps else start
                x_end = end / fps if fps else end
                x_mid = (x_start + x_end) / 2.0
                axis.text(
                    x_mid,
                    y + bar_height / 2,
                    f"{start}-{end}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.8,
                    },
                )
            for start, end in ranges_map.get(track_id, []):
                start_x = start / fps if fps else start
                end_x = end / fps if fps else end
                label = f"{start_x:.2f}s-{end_x:.2f}s" if fps else f"{start}-{end}"
                axis.text(
                    start_x,
                    y + bar_height + 0.8,
                    label,
                    fontsize=8,
                    color=color,
                    va="bottom",
                    ha="left",
                    clip_on=False,
                )

    axis.set_title(title)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels)
    axis.set_ylabel("tracked_id")
    axis.grid(axis="x", linestyle="--", alpha=0.35)

    if not has_any_range:
        axis.text(0.5, 0.5, "No missing ranges", ha="center", va="center", transform=axis.transAxes)


def save_plot(track_ids, heading_ranges, speed_ranges, fps, output_path):
    figure_height = max(4, 1.2 + len(track_ids) * 0.6)
    fig, axes = plt.subplots(2, 1, figsize=(14, figure_height + 2), sharex=True)

    plot_ranges(track_ids, heading_ranges, fps, axes[0], "Missing Heading Ranges", "#d64f4f")
    plot_ranges(track_ids, speed_ranges, fps, axes[1], "Missing Speed Ranges", "#f0a43a")

    axes[1].set_xlabel("Time (seconds)" if fps else "Frame Index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect TrafficLab replay JSON, find missing heading/speed ranges per tracked_id, "
            "print the ranges, and save a timeline chart."
        )
    )
    parser.add_argument("input_path", help="Path to output.json or output.json.gz")
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Path to output PNG file. Defaults next to the input file.",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        type=int,
        help="Optional tracked_id values to analyze. Defaults to all tracks in the file.",
    )
    return parser.parse_args()


def derive_output_path(input_path, explicit_output_path=None):
    if explicit_output_path:
        return Path(explicit_output_path)

    input_path = Path(input_path)
    if input_path.suffix == ".gz":
        base_name = input_path.stem
    else:
        base_name = input_path.stem
    scripts_dir = Path(__file__).resolve().parent
    return scripts_dir / f"{base_name}_missing_motion_ranges.png"


def main():
    args = parse_args()
    data = load_json(args.input_path)
    fps = data.get("meta", {}).get("fps")

    track_ids, heading_missing, speed_missing = collect_missing_frames(data, args.ids)
    heading_ranges = {track_id: frames_to_ranges(frames) for track_id, frames in heading_missing.items()}
    speed_ranges = {track_id: frames_to_ranges(frames) for track_id, frames in speed_missing.items()}

    print_summary(track_ids, heading_ranges, speed_ranges, fps)

    output_path = derive_output_path(args.input_path, args.output_path)
    save_plot(track_ids, heading_ranges, speed_ranges, fps, output_path)
    print(f"Saved chart to: {output_path}")


if __name__ == "__main__":
    main()
