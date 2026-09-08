#!/usr/bin/env python3
"""Convert a seg-mode inference replay into a record_car_masks.py-shaped file.

Lets run_keypoints_openpifpaf.py --method segmentation --seg-masks-json reuse
the mask polygons run_inference.py already computed, instead of running the
segmentation model a second time over the same video. See
trafficlab/trajectory/seg_mask_adapter.py for the field mapping and rationale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trafficlab.trajectory.io import infer_location_code, load_json, write_json
from trafficlab.trajectory.seg_mask_adapter import (
    DEFAULT_CAR_CLASSES,
    default_seg_masks_output_path,
    replay_to_seg_mask_records,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_json", help="Seg-mode replay JSON(.gz) carrying mask_contour.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output path (default: output/car_masks/<location>/seg-mask_<stem>.from-replay.json.gz).",
    )
    parser.add_argument(
        "--car-classes",
        default=",".join(DEFAULT_CAR_CLASSES),
        help=f"Comma-separated replay `class` values to emit as car instances (default: {','.join(DEFAULT_CAR_CLASSES)}).",
    )
    parser.add_argument("--location-code", help="Override the inferred location code (output path only).")
    args = parser.parse_args()

    replay_path = Path(args.replay_json)
    data = load_json(replay_path)

    car_classes = [c.strip() for c in args.car_classes.split(",") if c.strip()]
    try:
        out_data, stats = replay_to_seg_mask_records(data, car_classes=car_classes)
    except ValueError as exc:
        print(f"[replay_to_seg_masks] {exc}", file=sys.stderr)
        return 1

    if args.output:
        out_path = Path(args.output)
    else:
        location_code = args.location_code or infer_location_code(data, replay_path)
        out_path = default_seg_masks_output_path(replay_path, location_code)

    write_json(out_path, out_data)
    print(f"[replay_to_seg_masks] {stats.summary()}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
