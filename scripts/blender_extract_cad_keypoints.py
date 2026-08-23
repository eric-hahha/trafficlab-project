"""
BLENDER-ONLY SCRIPT.
Run this INSIDE Blender's Scripting workspace (Text > Open, then Run Script),
with the labeled .blend file already open as the active scene. Does NOT run
under `python` / the `trafficlab` conda env — `bpy` only exists inside
Blender's own bundled Python interpreter, a separate environment with no
access to this repo's packages.

Reads 13 named marker objects from the currently open scene — 12
hand-labeled Apollo-24 "*_left" keypoints plus a 13th "ground_ref" point
(snapped to a front tire's lowest/ground-contact vertex, used later to
establish the true h=0 ground plane) — and dumps their world-space
positions to a JSON file. Markers can be either Empty objects (e.g. Plain
Axes) or Mesh objects (e.g. a small UV Sphere) — either way, what's read is
the object's own origin (`matrix_world.translation`), so make sure the
marker was created (or its origin was set) exactly at the target vertex;
scaling/moving the object afterward is fine as long as you don't edit its
mesh data without re-centering the origin. Does no axis-convention or sign
interpretation itself; that happens in scripts/build_cad_keypoint_template.py
(run separately, in the trafficlab conda env, no Blender dependency).
"""
import json
import os

import bpy

# ---- CONFIG ----
# Defaults to a file sitting next to the currently open .blend. Set this to
# a string path to override.
OUTPUT_JSON_PATH = None

EXPECTED_NAMES = [
    "front_glass_top_left", "front_light_left", "front_low_fog_light_left",
    "front_door_top_left", "front_wheel_center_left", "rear_wheel_center_left",
    "rear_corner_left", "rear_glass_up_left", "rear_light_left",
    "rear_bumper_left", "rear_plate_left", "front_door_base_left",
    "ground_ref",
]
# NOTE: must stay in sync with trafficlab.motion.keypoints_openpifpaf.KP_NAMES
# / _LR_PAIRS's "*_left" half. This script can't import that module directly
# (bpy's Python has no access to the trafficlab package) — if
# keypoints_openpifpaf.py's keypoint names are renamed again, update this
# list by hand to match.


def _default_output_path() -> str:
    blend_path = bpy.data.filepath
    if not blend_path:
        raise RuntimeError(
            "The current Blender file hasn't been saved yet — save it first "
            "(so this script has a folder to write next to), or set "
            "OUTPUT_JSON_PATH above to an explicit path."
        )
    return os.path.join(os.path.dirname(blend_path), "blender_keypoints_raw.json")


def _collect_points() -> tuple[dict, list[str]]:
    problems = []
    points = {}
    for name in EXPECTED_NAMES:
        obj = bpy.data.objects.get(name)
        if obj is None:
            problems.append(f"missing marker object named '{name}'")
            continue
        if obj.type not in ('EMPTY', 'MESH'):
            problems.append(f"'{name}' exists but is type={obj.type!r}, expected EMPTY or MESH")
            continue
        points[name] = list(obj.matrix_world.translation)

    # Stray auto-suffixed duplicates (e.g. "front_light_left.001") usually
    # mean an accidental extra marker or a naming typo that created a second
    # object instead of overwriting/renaming the first.
    dupes = [
        o.name for o in bpy.data.objects
        if o.type in ('EMPTY', 'MESH') and any(o.name.startswith(n + ".") for n in EXPECTED_NAMES)
    ]

    if problems:
        raise RuntimeError(
            f"blender_extract_cad_keypoints: {len(problems)} problem(s) found:\n  "
            + "\n  ".join(problems)
        )
    return points, dupes


def main() -> None:
    points, dupes = _collect_points()

    scale_length = bpy.context.scene.unit_settings.scale_length
    if abs(scale_length - 1.0) > 1e-6:
        print(f"WARNING: scene.unit_settings.scale_length = {scale_length} (expected 1.0) "
              "— downstream math assumes 1 Blender unit = 1 meter.")

    if dupes:
        print(f"WARNING: possible stray duplicate markers (auto-suffixed): {dupes} "
              "— check these aren't accidental extra points or naming typos.")

    out_path = OUTPUT_JSON_PATH or _default_output_path()
    data = {
        "source_blend": bpy.data.filepath,
        "blender_unit_scale_length": scale_length,
        "coordinate_frame": "blender_world_meters (raw, unremapped)",
        "points": points,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {len(points)} points to {out_path}")


if __name__ == "__main__":
    main()
