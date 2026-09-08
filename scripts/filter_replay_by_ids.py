#!/usr/bin/env python3
"""Filter a replay JSON/JSON.GZ down to a subset of tracked_id values.

Every other field is left untouched -- unlike filter_and_enrich_output.py,
this does not recompute position_m/velocity_mps or require a G_projection /
prior_dimensions.json, since the kept objects already carry every field
downstream tools (GUI, trajectory_tools.py, export_cctv_sat_composite.py)
expect.
"""
import argparse
import gzip
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def filter_by_ids(data: dict, keep_ids: set) -> dict:
    for frame in data.get("frames", []):
        frame["objects"] = [
            obj for obj in frame.get("objects", []) if obj.get("tracked_id") in keep_ids
        ]
    return data


def default_output_path(input_path: Path, data: dict, keep_ids: list) -> Path:
    location_code = data.get("location_code", "unknown")
    ids_tag = "-".join(str(i) for i in keep_ids)
    stem = input_path.name.split(".")[0]
    suffix = "".join(input_path.suffixes)
    return Path("output") / "filtered_replay" / location_code / f"{stem}_ids-{ids_tag}{suffix}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter a replay JSON/JSON.GZ down to specific tracked_id values."
    )
    parser.add_argument("input_path", help="Path to input replay .json or .json.gz")
    parser.add_argument(
        "--ids", required=True, help="Comma-separated tracked_id values to keep"
    )
    parser.add_argument(
        "-o",
        "--out",
        help="Override output path (default: output/filtered_replay/<location_code>/<stem>_ids-<ids><suffix>)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    keep_ids = [int(x) for x in args.ids.split(",")]

    input_path = Path(args.input_path)
    data = load_json(input_path)
    filtered = filter_by_ids(data, set(keep_ids))

    out_path = Path(args.out) if args.out else default_output_path(input_path, data, keep_ids)
    write_json(out_path, filtered)
    print(f"Saved filtered replay to: {out_path}")


if __name__ == "__main__":
    main()
