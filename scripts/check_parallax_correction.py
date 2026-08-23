#!/usr/bin/env python3
"""Parallax-correction before/after plot for one frame.

Plots kp_sat (the height-corrected keypoint projections) alongside each
keypoint's uncorrected h=0 apparent point (recomputed from kp_cctv via
GProjection), joined by a line colored to match the keypoint's car part --
the direct visual of how much and in which direction each keypoint's own
height template moves it. See
trafficlab/diagnostics/parallax_correction_check.py for the underlying
computation.

By default kp_sat is read as already stored in the replay JSON (from
whichever G_projection config was active at inference time). Pass
--recompute to instead rebuild both points from kp_cctv using the
G_projection config resolved *now* -- e.g. after hand-correcting the
location's G_projection_<code>.json -- without re-running inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trafficlab.diagnostics.parallax_correction_check import (
    find_frame,
    compute_frame_records,
    pick_richest_frame,
    plot_frame_pre_post_keypoints,
)
from trafficlab.motion.keypoints_openpifpaf import (
    _FALLBACK_DIMS,
    build_car_template,
    compute_car_dims_from_spec_csv,
)
from trafficlab.projection.g_projection import GProjection
from trafficlab.trajectory.io import infer_location_code, load_json, resolve_satellite_image_path


def resolve_g_projection_path(location_code, *, explicit_path=None, project_root="."):
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None
    if not location_code:
        return None
    candidate = Path(project_root) / "location" / location_code / f"G_projection_{location_code}.json"
    return candidate if candidate.exists() else None


def resolve_dims(g_proj_dir: str, spec_csv: str | None, body_type: str, vehicle_class: str) -> dict:
    """Same fallback chain as run_keypoints_openpifpaf.py: --spec-csv, else
    walk up from the G_projection's directory looking for prior_dimensions.json,
    else the built-in defaults."""
    if spec_csv:
        return compute_car_dims_from_spec_csv(spec_csv, body_type=body_type)
    d = g_proj_dir
    dims_path = None
    for _ in range(5):
        candidate = os.path.join(d, "prior_dimensions.json")
        if os.path.exists(candidate):
            dims_path = candidate
            break
        d = os.path.dirname(d)
    if dims_path:
        with open(dims_path) as f:
            pj = json.load(f)
        print(f"[check_parallax_correction] Using {dims_path}")
        return pj.get("measurements_visdrone", {}).get(vehicle_class, dict(_FALLBACK_DIMS))
    print("[check_parallax_correction] No prior_dimensions.json found; using built-in fallback dims.")
    return dict(_FALLBACK_DIMS)


def _parse_ids(value: str | None):
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", help="Path to a replay JSON(.gz) with kp_cctv/kp_sat fields.")
    parser.add_argument("--frame-index", type=int, default=None,
                         help="Frame to plot. Default: auto-pick the frame with the most valid pairs.")
    parser.add_argument("--ids", help="Comma-separated tracked_id values to include (default: all).")
    parser.add_argument("--location-code", help="Override inferred location code.")
    parser.add_argument("--g-proj", help="Override G_projection config path.")
    parser.add_argument("--sat-image", help="Override satellite image path.")
    parser.add_argument("--kp-conf", type=float, default=0.2, help="Keypoint confidence threshold (default 0.2).")
    parser.add_argument("--recompute", action="store_true",
                         help="Recompute both pre and post points from kp_cctv using the G_projection "
                              "config resolved now, instead of reading kp_sat from the replay JSON. "
                              "Use after editing/correcting a location's G_projection_<code>.json, so "
                              "you don't have to re-run inference just to see the new projection.")
    parser.add_argument("--spec-csv", help="engines.csv for --recompute's car template (see run_keypoints_openpifpaf.py).")
    parser.add_argument("--body-type", default="Sedan", help="Used with --spec-csv (default Sedan).")
    parser.add_argument("--vehicle-class", default="car",
                         help="prior_dimensions.json key for --recompute's car template (default car).")
    parser.add_argument("-o", "--out", help="Output PNG path (default: next to input JSON).")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    input_path = Path(args.json_path)
    data = load_json(input_path)
    location_code = args.location_code or infer_location_code(data, input_path)

    g_proj_path = resolve_g_projection_path(location_code, explicit_path=args.g_proj, project_root=REPO_ROOT)
    if g_proj_path is None:
        print(f"[check_parallax_correction] Could not resolve G_projection config for "
              f"location_code={location_code!r} (pass --g-proj to override).", file=sys.stderr)
        sys.exit(1)
    with open(g_proj_path) as f:
        g_data = json.load(f)
    g_engine = GProjection(g_data, base_dir=str(g_proj_path.parent))

    template = None
    if args.recompute:
        dims = resolve_dims(str(g_proj_path.parent), args.spec_csv, args.body_type, args.vehicle_class)
        print(f"[check_parallax_correction] --recompute: using dims={dims}")
        template = build_car_template(dims)

    sat_image_path = resolve_satellite_image_path(
        location_code, explicit_path=args.sat_image, project_root=REPO_ROOT
    )
    if sat_image_path is None:
        raise SystemExit(
            "Could not resolve satellite image. Pass --sat-image or --location-code."
        )

    ids = _parse_ids(args.ids)

    if args.frame_index is None:
        frame_index, records = pick_richest_frame(data, g_engine, kp_conf=args.kp_conf, template=template)
        if frame_index is None:
            print("[check_parallax_correction] No frame has any valid kp_cctv/kp_sat pair.", file=sys.stderr)
            sys.exit(1)
        print(f"--frame-index not given; auto-selected frame {frame_index} (most valid pairs).")
    else:
        frame_index = args.frame_index
        frame = find_frame(data, frame_index)
        if frame is None:
            raise SystemExit(f"frame_index {frame_index} not found in replay JSON.")
        records = compute_frame_records(frame, g_engine, kp_conf=args.kp_conf, template=template)
        if not records:
            raise SystemExit(f"No valid kp_cctv/kp_sat pair in frame {frame_index}.")

    if args.out:
        out_path = Path(args.out)
    else:
        name = input_path.name
        stem = name[:-8] if name.endswith(".json.gz") else input_path.stem
        ids_suffix = f"_ids{args.ids.replace(',', '-')}" if args.ids else ""
        recompute_suffix = "_recomputed" if args.recompute else ""
        out_path = input_path.with_name(
            f"{stem}.parallax_frame{frame_index}{ids_suffix}{recompute_suffix}.png"
        )

    out_path, n_keypoints, n_objects = plot_frame_pre_post_keypoints(
        frame_index, records, sat_image_path, out_path, ids=ids, dpi=args.dpi,
        recomputed=args.recompute,
    )
    print(f"Saved {n_keypoints} pre/post keypoint pairs across {n_objects} object(s) to {out_path}")


if __name__ == "__main__":
    main()
