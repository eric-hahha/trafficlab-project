"""Shared JSON I/O helpers for TrafficLab trajectory tools."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    """Load a .json or .json.gz file."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    """Write a .json or .json.gz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        return
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def frames_from_data(data: Any) -> list[dict[str, Any]]:
    """Return the frame list from supported TrafficLab replay shapes."""
    if isinstance(data, dict):
        frames = data.get("frames")
    elif isinstance(data, list):
        frames = data
    else:
        frames = None

    if not isinstance(frames, list):
        raise ValueError("Trajectory data must be a list or contain a 'frames' list.")
    return frames


def default_smooth_output_path(input_path: str | Path) -> Path:
    """Build a default path next to the input with .smoothed before the JSON suffix."""
    input_path = Path(input_path)
    name = input_path.name
    if name.endswith(".json.gz"):
        return input_path.with_name(name[:-8] + ".smoothed.json.gz")
    if name.endswith(".json"):
        return input_path.with_name(name[:-5] + ".smoothed.json")
    return input_path.with_name(name + ".smoothed.json")


def infer_location_code(data: Any, input_path: str | Path | None = None) -> str | None:
    """Infer a TrafficLab location code from metadata or a conventional path."""
    if isinstance(data, dict):
        location_code = data.get("location_code")
        if location_code:
            return str(location_code)

        meta = data.get("meta")
        if isinstance(meta, dict):
            location_code = meta.get("location_code")
            if location_code:
                return str(location_code)

    if input_path is None:
        return None

    path = Path(input_path)
    parts = path.parts
    if "location" in parts:
        index = parts.index("location")
        if index + 1 < len(parts):
            return parts[index + 1]

    if path.parent.name:
        return path.parent.name
    return None


def resolve_satellite_image_path(
    location_code: str | None,
    *,
    explicit_path: str | Path | None = None,
    project_root: str | Path = ".",
) -> Path | None:
    """Resolve the satellite image used as the plotting background."""
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None

    if not location_code:
        return None

    root = Path(project_root)
    candidates = [
        root / "location" / location_code / f"sat_{location_code}.png",
        root / "location" / location_code / f"{location_code}_sat.png",
        root / "location" / location_code / f"satellite_{location_code}.png",
        root / f"sat_{location_code}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
