#!/usr/bin/env python3
"""Re-localize a replay's objects by ground-projecting their bbox_2d (tight
box) instead of using whatever keypoint-based sat_coords the replay already
carries. Lets a box-based localization (matching InferencePipeline's seg-mode
tight-box path) be compared against an existing keypoint-based result for the
same video, without re-running detection/segmentation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trafficlab.motion.kinematics import TrackSmoother
from trafficlab.projection.g_projection import GProjection
from trafficlab.trajectory.io import infer_location_code, load_json, write_json


def default_boxproj_output_path(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith(".json.gz"):
        return input_path.with_name(name[:-8] + ".boxproj.json.gz")
    if name.endswith(".json"):
        return input_path.with_name(name[:-5] + ".boxproj.json")
    return input_path.with_name(name + ".boxproj.json")


def resolve_g_proj_path(location_code: str | None) -> Path | None:
    if not location_code:
        return None
    loc_dir = REPO_ROOT / "location" / location_code
    for candidate in (
        loc_dir / f"G_projection_{location_code}.json",
        loc_dir / f"G_projection_svg_{location_code}.json",
    ):
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Ground-project each object's bbox_2d (tight box) to produce a box-based replay."
    )
    parser.add_argument("replay", help="Path to a replay .json/.json.gz containing bbox_2d per object")
    parser.add_argument("--g-proj", default=None, help="Override G_projection config path")
    parser.add_argument("--ref-method", default="center_bottom_side",
                         help="Ground-contact reference point on the box (default: center_bottom_side)")
    parser.add_argument("--proj-method", default="down_h",
                         help="Parallax correction method (default: down_h)")
    parser.add_argument("--prior-dimensions", default=None,
                         help="Key into prior_dimensions.json for per-class height/width/length "
                              "(default: the --config-name config's own 'prior_dimensions' value)")
    parser.add_argument("--config-path", default=str(REPO_ROOT / "inference_config.yaml"),
                         help="YAML config to source kinematics (heading/speed smoothing) settings from")
    parser.add_argument("--config-name", default="yolo_seg_tight_box",
                         help="Config key within --config-path (default: yolo_seg_tight_box, the seg tight-box config)")
    parser.add_argument("--out", default=None, help="Override output path")
    args = parser.parse_args()

    replay_path = Path(args.replay)
    data = load_json(replay_path)

    location_code = infer_location_code(data, replay_path)
    g_proj_path = Path(args.g_proj) if args.g_proj else resolve_g_proj_path(location_code)
    if g_proj_path is None or not g_proj_path.exists():
        parser.error(
            f"Could not resolve a G_projection config (location_code={location_code!r}); pass --g-proj explicitly."
        )

    with open(g_proj_path, "r") as f:
        g_data = json.load(f)
    g_engine = GProjection(g_data, base_dir=str(g_proj_path.parent))
    use_svg = g_data.get("use_svg", False)

    with open(args.config_path, "r") as f:
        full_config = yaml.safe_load(f)
    selected_cfg = full_config["configs"][args.config_name]
    kinematics_config = selected_cfg.get("kinematics", {})
    prior_dimensions_key = args.prior_dimensions or selected_cfg.get("prior_dimensions", "measurements_visdrone")

    real_fps = (data.get("meta") or {}).get("fps")
    if not real_fps:
        parser.error("Replay is missing meta.fps; cannot compute per-frame dt for kinematics smoothing.")

    with open(REPO_ROOT / "prior_dimensions.json", "r") as f:
        all_priors = json.load(f)
    prior_dims = all_priors.get(prior_dimensions_key, {})
    prior_dims_norm = {k.strip().lower(): v for k, v in prior_dims.items()}

    n_objects = 0
    n_projected = 0
    n_skipped_no_bbox = 0
    n_missing_dims = set()

    track_smoothers = {}
    last_seen_frame = {}

    frames = sorted(data.get("frames", []), key=lambda fr: fr.get("frame_index", 0))
    for frame in frames:
        i = frame.get("frame_index", 0)
        new_objects = []
        for obj in frame.get("objects", []):
            n_objects += 1
            bbox = obj.get("bbox_2d")
            if not bbox:
                n_skipped_no_bbox += 1
                continue

            cls_name = (obj.get("class") or "").strip().lower()
            dims = prior_dims_norm.get(cls_name)
            have_measurements = dims is not None
            if not have_measurements:
                n_missing_dims.add(cls_name)
            h_real = float(dims.get("height", 0.0)) if have_measurements else 0.0

            x1, y1, x2, y2 = bbox
            proj_res = g_engine.get_ground_contact_from_box(
                (x1, y1, x2 - x1, y2 - y1), h_real,
                ref_method=args.ref_method,
                proj_method=args.proj_method,
            )
            sat_coords = list(proj_res["sat_coords"])

            tid = obj.get("tracked_id")
            heading = None
            speed = 0.0
            is_def = False

            if tid is not None:
                if tid not in track_smoothers:
                    track_smoothers[tid] = TrackSmoother(kinematics_config)
                    last_seen_frame[tid] = i - 1

                svg_h = g_engine.get_svg_heading(sat_coords) if use_svg else None

                prev_f = last_seen_frame.get(tid, i - 1)
                dt = (i - prev_f) / real_fps
                if dt <= 0:
                    dt = 1.0 / real_fps

                k_res = track_smoothers[tid].update(sat_coords, dt, g_engine.px_per_m, svg_heading=svg_h)
                last_seen_frame[tid] = i

                speed = k_res["speed_kmh"]
                heading = k_res["heading"]
                is_def = k_res["default_heading"]

                corrected_sat_coords = k_res.get("corrected_position")
                if corrected_sat_coords is not None:
                    sat_coords = corrected_sat_coords

            have_heading = heading is not None
            if not have_heading:
                speed = 0.0

            sat_floor_box = None
            bbox_3d = None
            if have_heading and have_measurements:
                w_m, l_m = dims["width"], dims["length"]
                px_m = g_engine.px_per_m

                ang = np.radians(heading)
                c, s = np.cos(ang), np.sin(ang)
                dx, dy = (l_m * px_m) / 2, (w_m * px_m) / 2
                corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
                R = np.array([[c, -s], [s, c]])

                sat_floor_box = (corners @ R.T + sat_coords).tolist()
                bbox_3d = g_engine.sat_floor_to_cctv_3d(sat_floor_box, h_real)

            new_objects.append({
                "id": obj.get("id"),
                "tracked_id": tid,
                "class": obj.get("class"),
                "confidence": obj.get("confidence"),
                "bbox_2d": bbox,
                "reference_point": list(proj_res["cctv_ref_point"]),
                "sat_coords": sat_coords,
                "have_heading": have_heading,
                "have_measurements": have_measurements,
                "default_heading": is_def,
                "heading": heading,
                "speed_kmh": speed,
                "sat_floor_box": sat_floor_box,
                "bbox_3d": bbox_3d,
            })
            n_projected += 1
        frame["objects"] = new_objects

    data["frames"] = frames
    data["run_config"] = {
        "method": "box_ground_projection",
        "source_replay": str(replay_path),
        "ref_method": args.ref_method,
        "proj_method": args.proj_method,
        "prior_dimensions": prior_dimensions_key,
        "kinematics_config_name": args.config_name,
        "kinematics_config_path": args.config_path,
        "g_proj_path": str(g_proj_path),
    }

    out_path = Path(args.out) if args.out else default_boxproj_output_path(replay_path)
    write_json(out_path, data)

    print(f"Objects seen: {n_objects}")
    print(f"Projected: {n_projected}")
    if n_skipped_no_bbox:
        print(f"Skipped (no bbox_2d): {n_skipped_no_bbox}")
    if n_missing_dims:
        print(f"Classes missing prior dimensions (height=0.0 used): {sorted(n_missing_dims)}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
