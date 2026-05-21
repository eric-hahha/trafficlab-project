#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Iterable

import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TrafficLab inference without opening the GUI."
    )
    parser.add_argument(
        "--config-path",
        default="inference_config.yaml",
        help="Path to the inference YAML config.",
    )
    parser.add_argument(
        "--config-name",
        help="Config key inside a multi-config YAML. Defaults to the first config.",
    )
    parser.add_argument(
        "--location-root",
        default="location",
        help="Root folder containing TrafficLab location directories.",
    )
    parser.add_argument(
        "--output-root",
        default="output",
        help="Root folder for inference outputs.",
    )
    parser.add_argument(
        "--location",
        action="append",
        dest="locations",
        help="Only process this location code. Repeat to pass multiple.",
    )
    parser.add_argument(
        "--mp4",
        action="append",
        dest="mp4_paths",
        help="Process this specific mp4 path. Repeat to pass multiple files.",
    )
    parser.add_argument(
        "--all-pending",
        action="store_true",
        help="Scan all locations and process videos whose output does not exist yet.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run inference even if the output JSON already exists.",
    )
    return parser


def load_selected_config(config_path: Path, config_name: str | None) -> tuple[dict, str]:
    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    if isinstance(raw_cfg, dict) and "configs" in raw_cfg and isinstance(raw_cfg["configs"], dict):
        configs_map = raw_cfg["configs"]
        if not configs_map:
            raise ValueError(f"No configs found in {config_path}")
        if config_name:
            if config_name not in configs_map:
                choices = ", ".join(configs_map.keys())
                raise ValueError(f"Unknown config '{config_name}'. Available configs: {choices}")
            return configs_map[config_name], config_name

        first_key = next(iter(configs_map.keys()))
        return configs_map[first_key], first_key

    cfg = raw_cfg or {}
    return cfg, cfg.get("config_name", "default")


def output_base_dir(output_root: Path, selected_cfg: dict, config_name: str) -> Path:
    model_stem = Path(selected_cfg.get("model", {}).get("weights", "Unknown")).stem
    tracker_type = selected_cfg.get("tracking", {}).get("tracker_type", "Unknown")
    return output_root / f"model-{model_stem}_tracker-{tracker_type}" / config_name


def find_g_proj(location_root: Path, loc: str) -> Path | None:
    candidates = [
        location_root / loc / f"G_projection_{loc}.json",
        location_root / loc / f"G_projection_svg_{loc}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def expected_output_path(base_out: Path, loc: str, mp4_path: Path) -> Path:
    return base_out / loc / f"{mp4_path.stem}.json.gz"


def iter_location_mp4s(location_root: Path, locs: Iterable[str] | None) -> list[tuple[str, Path]]:
    if locs:
        target_locs = sorted(set(locs))
    else:
        target_locs = sorted(
            d.name for d in location_root.iterdir() if d.is_dir()
        )

    tasks: list[tuple[str, Path]] = []
    for loc in target_locs:
        footage_dir = location_root / loc / "footage"
        if not footage_dir.exists():
            continue
        for mp4_path in sorted(footage_dir.glob("*.mp4")):
            tasks.append((loc, mp4_path))
    return tasks


def infer_loc_from_mp4(location_root: Path, mp4_path: Path) -> str:
    try:
        rel = mp4_path.resolve().relative_to(location_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"MP4 path {mp4_path} is not inside location root {location_root}"
        ) from exc

    parts = rel.parts
    if len(parts) < 3 or parts[1] != "footage":
        raise ValueError(
            f"MP4 path {mp4_path} does not match expected layout location/<loc>/footage/*.mp4"
        )
    return parts[0]


def collect_tasks(
    args: argparse.Namespace,
    location_root: Path,
    base_out: Path,
) -> list[tuple[str, Path, Path]]:
    tasks: list[tuple[str, Path, Path]] = []

    if args.mp4_paths:
        for raw_mp4 in args.mp4_paths:
            mp4_path = Path(raw_mp4)
            if not mp4_path.exists():
                raise FileNotFoundError(f"MP4 not found: {mp4_path}")
            loc = infer_loc_from_mp4(location_root, mp4_path)
            g_proj_path = find_g_proj(location_root, loc)
            if not g_proj_path:
                print(f"[SKIP] {mp4_path.name}: no G-projection found for location '{loc}'")
                continue
            out_path = expected_output_path(base_out, loc, mp4_path)
            if out_path.exists() and not args.force:
                print(f"[SKIP] {mp4_path.name}: output already exists at {out_path}")
                continue
            tasks.append((loc, mp4_path, g_proj_path))
        return tasks

    scan_locs = args.locations if args.locations else None
    if not args.all_pending and not scan_locs:
        raise ValueError("Pass --all-pending, --location, or --mp4 to select work.")

    for loc, mp4_path in iter_location_mp4s(location_root, scan_locs):
        g_proj_path = find_g_proj(location_root, loc)
        if not g_proj_path:
            print(f"[SKIP] {loc}/{mp4_path.name}: no G-projection found")
            continue
        out_path = expected_output_path(base_out, loc, mp4_path)
        if out_path.exists() and not args.force:
            continue
        tasks.append((loc, mp4_path, g_proj_path))
    return tasks


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    config_path = Path(args.config_path)
    location_root = Path(args.location_root)
    output_root = Path(args.output_root)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not location_root.exists():
        raise FileNotFoundError(f"Location root not found: {location_root}")

    selected_cfg, config_name = load_selected_config(config_path, args.config_name)
    base_out = output_base_dir(output_root, selected_cfg, config_name)
    tasks = collect_tasks(args, location_root, base_out)

    if not tasks:
        print("No videos to process.")
        return 0

    print(f"Using config: {config_name}")
    print(f"Queue size: {len(tasks)}")

    from trafficlab.inference.pipeline import InferencePipeline

    for index, (loc, mp4_path, g_proj_path) in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] Starting {loc}/{mp4_path.name}")
        pipeline = InferencePipeline(
            location_code=loc,
            footage_path=str(mp4_path),
            config_path=str(config_path),
            output_root=str(output_root),
            g_proj_path=str(g_proj_path),
            config_name=config_name,
            log_fn=lambda msg, prefix=f"[{loc}/{mp4_path.name}] ": print(prefix + str(msg)),
            progress_fn=lambda pct, prefix=f"[{loc}/{mp4_path.name}] ": print(prefix + f"{pct}%"),
            stop_flag_fn=lambda: False,
        )
        pipeline.run()
        print(f"[{index}/{len(tasks)}] Finished {loc}/{mp4_path.name}")

    print("All inference tasks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
