#!/usr/bin/env python3
"""Convert Blender-exported raw CAD keypoints into a per-model Apollo-24
keypoint template JSON in this project's (x, h, z) convention.

Reads the intermediate JSON produced by scripts/blender_extract_cad_keypoints.py
(run separately, inside Blender), auto-detects the raw coordinate frame's
axis mapping against this project's convention, mirrors the 12 hand-labeled
*_left points into the full 24-point template, and writes the result plus a
dimension health-check report.

Does NOT wire the output into scripts/run_keypoints_openpifpaf.py /
build_car_template() — that pipeline integration is separate follow-up work.
See docs/keypoints-openpifpaf-intro.md and trafficlab/motion/keypoints_openpifpaf.py
for the surrounding template/localization context.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trafficlab.motion.cad_keypoint_template import (
    build_template,
    check_dimensions,
    compute_dimensions,
    detect_axes,
    self_test,
)
from trafficlab.motion.keypoints_openpifpaf import KP_NAMES

# Reference dimensions + tolerances for this specific CAD model, sourced from
# the verified glTF parse of cad_models/nissan_juke_nismo/nissan_juke_nismo.glb
# (chassis-only bounding box, excluding the non-body "glow" decal mesh). If
# more CAD models are added later these should become CLI flags.
_NISSAN_JUKE_NISMO_REFERENCE = {
    "wheelbase_m": 2.53, "track_width_m": 1.70, "length_m": 4.16, "height_m": 1.66,
}
_NISSAN_JUKE_NISMO_TOLERANCES = {
    "wheelbase_m": 0.15, "track_width_m": 0.20, "length_m": 0.20, "height_m": 0.15,
}


def _print_report(axes: dict, dims: dict, checks: list[dict]) -> bool:
    axis_label = {0: "Blender X", 1: "Blender Y", 2: "Blender Z"}
    print("\n=== Axis detection ===")
    print(f"  height axis       = {axis_label[axes['h_axis']]} (index {axes['h_axis']})")
    print(f"  longitudinal axis = {axis_label[axes['lon_axis']]} (index {axes['lon_axis']})  "
          f"sign_flip={'yes' if axes['lon_sign'] < 0 else 'no'}")
    print(f"  lateral axis       = {axis_label[axes['lat_axis']]} (index {axes['lat_axis']})  "
          f"sign_flip={'yes' if axes['lat_sign'] < 0 else 'no'}")
    for w in axes["warnings"]:
        print(f"  note: {w}")

    print("\n=== Dimension sanity ===")
    all_pass = True
    for c in checks:
        status = c["status"]
        all_pass = all_pass and status == "PASS"
        print(f"  {c['name']:<14}: computed={c['computed']:.3f}m  reference={c['reference']:.3f}m  "
              f"diff={c['pct_diff']:+.1f}%  (tol=±{c['tolerance_pct']:.0f}%)  {status}")

    print("\n=== Result ===")
    if all_pass:
        print("  ALL CHECKS PASSED")
    else:
        print("  ⚠ ONE OR MORE CHECKS NEED REVIEW (see above) — output still written, inspect before trusting it.")
    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", help="Path to the Blender-exported raw keypoints JSON.")
    parser.add_argument("--out", help="Path to write the final template JSON.")
    parser.add_argument("--self-test", action="store_true",
                         help="Run the synthetic round-trip self-test (no --input/--out needed) and exit.")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.input or not args.out:
        parser.error("--input and --out are required (unless --self-test)")

    input_path = Path(args.input)
    raw_data = json.loads(input_path.read_text())
    raw_points = raw_data["points"]

    axes = detect_axes(raw_points)
    template = build_template(raw_points, axes)
    dims = compute_dimensions(template)
    checks = check_dimensions(dims, _NISSAN_JUKE_NISMO_REFERENCE, _NISSAN_JUKE_NISMO_TOLERANCES)
    all_pass = _print_report(axes, dims, checks)

    out_data = {
        "schema_version": 1,
        "source": {
            "cad_model": "cad_models/nissan_juke_nismo/nissan_juke_nismo.glb",
            "intermediate_json": str(input_path),
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "generator_script": "scripts/build_cad_keypoint_template.py",
        },
        "coordinate_convention": "Apollo-24 (x=lateral,+left / h=height,0=ground,+up / z=longitudinal,-front/+rear), meters",
        "wheel_height_convention": "real_measured_hub_center",
        "axis_mapping": {
            "lateral_blender_axis": axes["lat_axis"],
            "longitudinal_blender_axis": axes["lon_axis"],
            "height_blender_axis": axes["h_axis"],
            "longitudinal_sign_flip": axes["lon_sign"] < 0,
            "lateral_sign_flip": axes["lat_sign"] < 0,
            "z_origin_blender": axes["z_origin"],
            "h_origin_blender": axes["h_origin"],
        },
        "ground_reference": {"method": "labeled_point", "point_name": "ground_ref"},
        "kp_names": list(KP_NAMES),
        "template": template.tolist(),
        "points_by_name": {name: template[i].tolist() for i, name in enumerate(KP_NAMES)},
        "validation": {**dims, "checks": checks},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nwrote {out_path}" + ("" if all_pass else "  (see WARN items above)"))


if __name__ == "__main__":
    main()
