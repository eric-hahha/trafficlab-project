#!/usr/bin/env python3
"""Correct a YOLO-box replay's sat_coords using a wheel_pair replay of the
same video as the positional reference.

Per-track rigid (rotation + translation, no scale) alignment fit only on
frames where the wheel_pair vehicle actually moved, plus freezing
near-stationary intervals onto the frame before they started. See
trafficlab/trajectory/wheel_pair_correction.py for the algorithm and
rationale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trafficlab.trajectory.io import (
    default_wheelpair_corrected_output_path,
    infer_location_code,
    load_json,
    write_json,
)
from trafficlab.trajectory.wheel_pair_correction import correct_replay


def _resolve_g_projection_path(location_code, *, explicit_path=None, project_root=REPO_ROOT):
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None
    if not location_code:
        return None
    candidates = [
        Path(project_root) / "location" / location_code / f"G_projection_{location_code}.json",
        Path(project_root) / "location" / location_code / f"G_projection_svg_{location_code}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_px_per_meter(target_data, wp_data, args) -> float:
    location_code = args.location_code or infer_location_code(target_data, args.target_json)
    g_proj_path = _resolve_g_projection_path(location_code, explicit_path=args.g_proj)
    if g_proj_path is None:
        print(
            f"[correct_trajectory_wheel_pair] Could not resolve a G_projection config "
            f"(--g-proj={args.g_proj!r} / --location-code={location_code!r}); "
            "px_per_meter is required to convert --max-match-distance/--min-fit-span-m to pixels.",
            file=sys.stderr,
        )
        sys.exit(1)
    g_proj_data = load_json(g_proj_path)
    px_per_m = g_proj_data.get("parallax", {}).get("px_per_meter")
    if not px_per_m or px_per_m <= 0:
        print(f"[correct_trajectory_wheel_pair] {g_proj_path} has no usable parallax.px_per_meter.",
              file=sys.stderr)
        sys.exit(1)
    return float(px_per_m)


def _parse_track_id_map(value):
    if not value:
        return None
    mapping = {}
    for pair in value.split(","):
        pair = pair.strip()
        if not pair:
            continue
        tid_str, wtid_str = pair.split(":")
        mapping[int(tid_str)] = int(wtid_str)
    return mapping


def _check_location_match(target_data, wp_data, args) -> None:
    if args.location_code:
        return
    target_loc = target_data.get("location_code")
    wp_loc = wp_data.get("location_code")
    if target_loc and wp_loc and target_loc != wp_loc:
        print(
            f"[correct_trajectory_wheel_pair] location_code mismatch: target={target_loc!r} "
            f"vs wheel_pair={wp_loc!r}. Pass --location-code to override if this is intentional.",
            file=sys.stderr,
        )
        sys.exit(1)


def _print_report(report) -> None:
    print(f"\n=== wheel_pair correction report (total target tracks={report.total_target_tracks}) ===")
    for r in sorted(report.results, key=lambda r: r.tracked_id):
        if r.status == "corrected":
            print(f"  tid={r.tracked_id:>6}  corrected  n_fit_points={r.n_fit_points:3d}  "
                  f"frozen_frames={r.n_frozen_frames:3d}  missing_anchor={r.n_missing_anchor_frames:3d}  "
                  f"translation={r.translation_px:7.2f}px  rotation={r.rotation_deg:+6.2f}deg")
        else:
            print(f"  tid={r.tracked_id:>6}  {r.status}  n_fit_points={r.n_fit_points}")
    n_corrected = len(report.corrected_tracks)
    n_skipped = len(report.skipped_tracks)
    print(f"\n  {n_corrected} corrected, {n_skipped} skipped (of {report.total_target_tracks} tracks).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_json", help="YOLO-box replay JSON(.gz) to correct.")
    parser.add_argument("--wheel-pair", required=True, help="wheel_pair replay JSON(.gz) used as reference.")
    parser.add_argument("-o", "--output", help="Output path (default: <stem>.wheelpair_corrected.json[.gz] next to input).")
    parser.add_argument("--g-proj", help="Path to a G_projection_<code>.json (provides px_per_meter).")
    parser.add_argument("--location-code", help="Override location code (also skips the location_code mismatch check).")
    parser.add_argument("--vehicle-class", default="car", help="Object class to correct (default: car).")
    parser.add_argument("--kp-conf", type=float, default=0.2, help="Wheel keypoint confidence threshold for stationary detection (default: 0.2).")
    parser.add_argument("--max-match-distance", type=float, default=7.0,
                         help="Max sat-space distance, in meters, to match a target object to a wheel_pair object in the same frame "
                              "(default: 7.0 — the YOLO bbox-bottom-center vs. wheel-grounded offset this tool corrects for can itself "
                              "be several meters at low camera angles per docs/localization-methods.md; on the Hsinchu verification data "
                              "the true match sat 4.6-5.4m apart while the nearest wrong candidate was 10m+, so 7.0 keeps clear margin "
                              "on both sides. Lower it in denser scenes where vehicles are closer than that.).")
    parser.add_argument("--stationary-px-threshold", type=float, default=2.0,
                         help="Max CCTV-pixel wheel-keypoint displacement between frames to call a frame stationary (default: 2.0).")
    parser.add_argument("--min-fit-points", type=int, default=3,
                         help="Minimum non-stationary matched frames required to fit a track's rigid transform (default: 3).")
    parser.add_argument("--min-fit-span-m", type=float, default=2.0,
                         help="Minimum spread (max pairwise distance, meters) among a track's fit points; tracks below this are skipped (default: 2.0).")
    parser.add_argument("--track-id-map",
                         help="Comma-separated target_id:wheel_pair_id pairs (e.g. '13:1,6:5') giving the "
                              "known correspondence between target tracked_ids and wheel_pair tracked_ids. "
                              "When given, ONLY these target tracks are corrected, matched frame-by-frame "
                              "directly by the given wheel_pair tracked_id instead of guessed by sat_coords "
                              "proximity — so a frame where the wheel_pair vehicle has no sat_coords (failed "
                              "localization) can still be checked for stationarity via kp_cctv and frozen, "
                              "it just can't be used to fit the rigid transform. Every target track not "
                              "listed here is left untouched.")
    args = parser.parse_args()

    target_path = Path(args.target_json)
    target_data = load_json(target_path)
    wp_data = load_json(args.wheel_pair)

    _check_location_match(target_data, wp_data, args)
    px_per_m = _resolve_px_per_meter(target_data, wp_data, args)

    report = correct_replay(
        target_data, wp_data,
        vehicle_class=args.vehicle_class,
        kp_conf=args.kp_conf,
        max_match_distance_px=args.max_match_distance * px_per_m,
        stationary_px_threshold=args.stationary_px_threshold,
        min_fit_points=args.min_fit_points,
        min_fit_span_px=args.min_fit_span_m * px_per_m,
        track_id_map=_parse_track_id_map(args.track_id_map),
    )

    _print_report(report)

    output_path = Path(args.output) if args.output else default_wheelpair_corrected_output_path(target_path)
    write_json(output_path, target_data)
    print(f"\nWrote corrected replay to {output_path}")


if __name__ == "__main__":
    main()
