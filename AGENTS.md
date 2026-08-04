# TrafficLab AI Instructions

This file is for AI coding agents working in this repository.

## Repository Purpose

TrafficLab 3D is a CCTV-to-satellite traffic analysis workflow. The main runtime
areas are:

- Calibration: create `G_projection_<location_code>.json` files under `location/<location_code>/`.
- Inference: run object detection, tracking, projection, kinematics, and write replay JSON files.
- Postprocess: correct, smooth, enrich, filter, or visualize replay JSON outputs after inference.
- Visualization: load replay JSON files and render CCTV/SAT synchronized views in the GUI.

Prefer keeping each concern in its own package or script. Do not turn one-off
experiments into root-level files unless the user explicitly asks for a scratch file.

## Environment

- Use the `trafficlab` conda environment for all Python commands.
- In this repository, the sandbox can activate the environment directly with:

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab
```

- Prefer direct activation over `conda run -n trafficlab ...` in this repo.
- Reason: `conda run` may fail in sandboxed sessions because Anaconda tries to create temporary files outside the writable area.
- When running TrafficLab inference or the GUI on Apple Silicon, set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- On `device: mps`, do not assume `half: true` is always safe.
- If inference reaches the model execution stage and then fails with dtype, backend, or FP16-related errors, inspect `half: true` in the selected config before debugging deeper.

## Repository Layout

- `main.py`: GUI entry point.
- `trafficlab/gui/`: PySide6 GUI implementation.
- `trafficlab/inference/`: inference pipeline shared by the GUI and CLI.
- `trafficlab/projection/`: G-projection and SVG projection helpers.
- `trafficlab/visualization/`: replay loading and rendering.
- `trafficlab/io/`: replay/config I/O helpers.
- `trafficlab/motion/`: kinematics utilities.
- `trafficlab/trajectory/`: post-inference trajectory smoothing and static plotting utilities.
- `scripts/`: CLI helpers and maintenance utilities.
- `location/<location_code>/`: calibration assets, satellite/CCTV images, projection files, and footage.
- `output/`: generated model/tracker/config/location replay outputs.
- `models/`: local detector/tracker checkpoints.
- `kiro/`: specs and historical task notes; do not treat these as runtime code unless asked.

Keep reusable code under `trafficlab/<domain>/`. Keep thin command-line wrappers
under `scripts/`. Avoid mixing external project files, sample outputs, pycache,
or notebooks into runtime packages.

## Command Rules

- Run Python commands from the repository root.
- From the repository root, `import trafficlab` resolves normally after activating the `trafficlab` environment.
- If a command must run from outside the repository root, set:

```bash
PYTHONPATH=/Users/eric/code/TrafficLab-3D-main
```

- Exception: `scripts/run_inference.py` does not add the repository root to `sys.path`, so running it by path (`python scripts/run_inference.py`) raises `ModuleNotFoundError: No module named 'trafficlab'` even with `cwd` at the repository root and the `trafficlab` environment active. Always set `PYTHONPATH` explicitly when invoking it — see the commands under "Run inference without the GUI" below.

- For regular Python scripts:

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python ...
```

- For inference or GUI commands that may use MPS fallback:

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && PYTORCH_ENABLE_MPS_FALLBACK=1 python ...
```

- Do not rely on the system `python` binary for project tasks.
- If a dependency appears missing, first verify that the `trafficlab` environment is active before concluding the package is unavailable.
- Prefer `python -m py_compile <files>` for quick syntax checks after edits.
- Do not use `conda run -n trafficlab ...` unless direct activation is impossible.
- If a command writes outputs for verification, prefer `/private/tmp` or another disposable writable path unless the output is intentionally part of the project.

## Common Commands

### Open the GUI

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py
```

### Run inference without the GUI

Process all pending videos with a chosen config:

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && PYTHONPATH=/Users/eric/code/TrafficLab-3D-main PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_inference.py --config-name slight_smoothing_best --all-pending
```

Process one location:

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && PYTHONPATH=/Users/eric/code/TrafficLab-3D-main PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_inference.py --config-name slight_smoothing_best --location test1
```

Process one mp4:

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && PYTHONPATH=/Users/eric/code/TrafficLab-3D-main PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_inference.py --config-name slight_smoothing_best --mp4 location/test1/footage/test1_8_100_s.mp4
```

Force re-run even if output already exists:

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && PYTHONPATH=/Users/eric/code/TrafficLab-3D-main PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_inference.py --config-name slight_smoothing_best --location test1 --force
```

### Postprocess

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python postprocess.py --help
```

### H-aware 3D keypoint localization

Run OpenPifPaf Apollo-24 detection on every frame and localize each vehicle via
height-aware keypoint template matching. Output is a standard TrafficLab replay
JSON loadable in the GUI (satellite position, heading, footprint, and CCTV
keypoint overlay). `--method` is required and selects one of two mutually
exclusive per-frame strategies — only the selected strategy's own flags take
effect.

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && \
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_keypoints_openpifpaf.py \
  --video location/test21/footage/test21-4.mp4 \
  --g-proj location/test21/G_projection_test21.json \
  --method geometric
```

Output: `output/haware/<location_code>/<video_stem>.json.gz`

`--method` choices:

| Method | What it does | Relevant flags |
|--------|---------------|-----------------|
| `geometric` | Bridge PifPaf detections to YOLO track IDs via bbox IoU. A PifPaf detection with only one confident keypoint has a zero-area bbox and never clears the IoU threshold on its own — a second pass recovers these: if the lone keypoint falls inside exactly one YOLO box, it's merged into the detection already IoU-matched to that track this frame (the fragment is then dropped so it doesn't also appear as its own object), or, if no such detection exists this frame, the fragment is tagged with the track id directly. | `--yolo` / `--yolo-conf` / `--yolo-classes` / `--iou-threshold` / `--yolo-boxes-json` / `--yolo-boxes-class` |
| `crop` | Crop each Pass-1 bbox and re-run PifPaf on the crop to recover more confident keypoints. | `--crop-redetect` / `--crop-padding` |

Shared options:

| Flag | Default | Notes |
|------|---------|-------|
| `--checkpoint` | `shufflenetv2k16-apollo-24` | PifPaf model |
| `--spec-csv` | *(none)* | `engines.csv` from automobile-models-and-specs; falls back to `prior_dimensions.json` then built-in defaults |
| `--body-type` | `Sedan` | Used with `--spec-csv` |
| `--kp-conf` | `0.2` | Keypoint confidence threshold |
| `--frames` | `-1` (all) | Limit frames for quick tests |
| `--start-frame` | `0` | First frame to process; frames before this are read and discarded, not seeked |
| `--localizer` | `procrustes` | `procrustes` (closed-form 2D Procrustes) or `reprojection` (nonlinear least-squares fit against PifPaf pixel positions) |
| `--out` | auto | Override output path |

`--method geometric` options:

| Flag | Default | Notes |
|------|---------|-------|
| `--yolo` | `models/best.pt` | YOLO model for track-ID matching (ByteTrack); pass `--yolo ""` to disable and leave `tracked_id=None` |
| `--yolo-conf` | `0.25` | YOLO detection confidence threshold |
| `--yolo-classes` | *(all)* | Comma-separated YOLO class indices to keep — model-specific, check the `--yolo` model's `.names` |
| `--iou-threshold` | `0.3` | Minimum bbox IoU to accept a PifPaf↔YOLO match |
| `--yolo-boxes-json` | *(none)* | Use a pre-computed replay JSON (e.g. `pipeline.py` output) as the YOLO box source instead of running a live model; takes priority over `--yolo` when set |
| `--yolo-boxes-class` | `car` | `class` value in `--yolo-boxes-json` to treat as a car |

`--method crop` options:

| Flag | Default | Notes |
|------|---------|-------|
| `--crop-redetect` | off | Crop each Pass-1 bbox with 50% padding and re-run PifPaf; without this flag `--method crop` is equivalent to plain Pass-1 PifPaf |
| `--crop-padding` | `0.5` | Fractional padding around the bbox for the crop |

Per-object fields added beyond the standard 14: `kp_cctv` (raw `[x, y, conf] × 24`
for GUI overlay), `n_keypoints`, `status` (`ok` / `ambiguous_heading` /
`failed_insufficient_kp`).

### Reprojection keypoint visualization

Plot per-vehicle `kp_sat` keypoint reprojections and `sat_coords` positions
from an h-aware (or other replay-shaped) JSON onto the satellite image for a
single frame. Each keypoint gets a black leader line to a
`tracked_id-keypoint_name` label; vehicle position points are drawn larger
with a black edge (keypoints use a white edge) and are not labeled.

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && \
python scripts/plot_reprojection_keypoints.py \
  output/haware/test21/test21-6_yolo_reprojection.json.gz \
  --frame-index 76
```

Options:

| Flag | Default | Notes |
|------|---------|-------|
| `--frame-index` | auto | Frame to plot; default auto-picks the frame with the most valid keypoints |
| `--ids` | *(all)* | Comma-separated `tracked_id` values to include |
| `--location-code` | inferred | Override inferred location code |
| `--sat-image` | inferred | Override satellite image path |
| `-o` / `--out` | next to input | Output PNG path |
| `--dpi` | `200` | Output resolution |

Output: PNG saved next to the input JSON by default
(`<stem>.keypoints_frame<N>.png`).

### CarFusion vehicle pose + tracking

Run CarFusion's YOLOv8-Pose model (one-stage bbox + 14 keypoints) on every
frame with ByteTrack cross-frame tracking enabled, project keypoints to
satellite coordinates, and fit vehicle center/heading via Procrustes SVD.
Output is a side-by-side CCTV+SAT composite per frame plus a standard
TrafficLab replay JSON loadable in the GUI.

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && \
python scripts/run_keypoints_carfusion.py \
  --video location/test21/footage/test21-4.mp4 \
  --weights models/carfusion_last.pt \
  --g-proj location/test21/G_projection_test21.json \
  --sat location/test21/sat_test21.png \
  --out /private/tmp/carfusion_sat/
```

Output: composite JPGs + `scatter.png` + `detections.json` under `--out`.

Options:

| Flag | Default | Notes |
|------|---------|-------|
| `--weights` | `models/carfusion_last.pt` | YOLOv8-Pose weights |
| `--start-frame` | `0` | First frame to process |
| `--frames` | `-1` (all) | Limit frames for quick tests |
| `--conf` | `0.25` | YOLO detection threshold — only decides what YOLO hands to the tracker, not what survives it (see below) |
| `--kp-conf` | `0.2` | Per-keypoint confidence threshold |
| `--tracker` | `trafficlab/inference/bytetrack.yaml` | Pinned ByteTrack config; not the bare ultralytics package default |

Tracking notes:
- `.track()` mode can drop a whole detection from the output entirely (not
  leave `tracked_id=None`) if ByteTrack never confirms it as a track. This is
  independent of `--conf` — see `docs/keypoints-carfusion-tracking.md` for the
  `is_activated` / `fuse_score` mechanics behind this.
- `trafficlab/inference/bytetrack.yaml` currently has `track_high_thresh` /
  `track_low_thresh` / `new_track_thresh` lowered to `0.01` and
  `fuse_score: False`, tuned for low-confidence / partially-cropped vehicles.
  Check this file's current values before assuming ultralytics defaults apply.

Per-object fields beyond the standard shape: `bbox_2d`, `bbox_cctv`, `kp_cctv`
(raw `[x, y, conf] × 14`), `n_keypoints`, `status` (`ok` / `ambiguous_heading` /
`failed_insufficient_kp`), `have_heading`, `have_measurements`,
`sat_floor_box`, `bbox_3d`. Top-level: `mp4_path`, `meta`, `location_code`,
`mp4_frame_count`, `animation_frame_count`.

To backfill these fields into a `detections.json` generated before this schema
existed, without re-running detection/tracking:

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && \
python scripts/patch_carfusion_replay_fields.py \
  --json /path/to/detections.json \
  --video location/test21/footage/test21-4.mp4 \
  --g-proj location/test21/G_projection_test21.json
```

### Trajectory smoothing and plotting

The integrated trajectory tools live in `trafficlab/trajectory/` and are exposed
through `scripts/trajectory_tools.py`.

Show help:

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python scripts/trajectory_tools.py --help
```

Smooth one replay JSON:

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python scripts/trajectory_tools.py smooth output/example.json.gz
```

Plot one replay JSON:

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python scripts/trajectory_tools.py plot output/example.smoothed.json.gz --location-code test1
```

Smooth and plot selected tracks:

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python scripts/trajectory_tools.py smooth-and-plot output/example.json.gz --ids 7,373 --zoom-to-fit
```

If the replay file does not contain `location_code` metadata, pass either
`--location-code <code>` or `--sat-image <path>`.

### Syntax check a script

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python -m py_compile scripts/run_inference.py
```

## Inference Notes

- The GUI inference tab and `scripts/run_inference.py` both use `trafficlab.inference.pipeline.InferencePipeline`.
- Location inputs are expected under `location/<location_code>/footage/*.mp4`.
- G-projection files are expected at one of:
  - `location/<location_code>/G_projection_<location_code>.json`
  - `location/<location_code>/G_projection_svg_<location_code>.json`
- Outputs are written under:

```text
output/model-<model_name>_tracker-<tracker_name>/<config_name>/<location_code>/*.json.gz
```

## Replay JSON Expectations

Most post-inference tools expect the standard replay shape:

```json
{
  "frames": [
    {
      "frame_index": 0,
      "objects": [
        {
          "tracked_id": 101,
          "class": "Car",
          "sat_coords": [1234.5, 678.9],
          "sat_center": [1234.5, 678.9]
        }
      ]
    }
  ]
}
```

- `tracked_id` links the same object across frames.
- `sat_coords` is the canonical satellite point used by smoothing and plotting.
- `sat_center` should be updated with `sat_coords` when a tool intentionally moves satellite points.
- `class` is optional for plotting but useful for summaries and filtering.
- `.json.gz` is the normal storage format for inference outputs; tools should preserve gzip support.

## Trajectory Tools Notes

- `trafficlab/trajectory/io.py` handles `.json` / `.json.gz` I/O and path resolution.
- `trafficlab/trajectory/smoothing.py` smooths `sat_coords` with Savitzky-Golay filtering grouped by `tracked_id`.
- `trafficlab/trajectory/plotting.py` renders trajectory points on a satellite image using matplotlib's non-interactive `Agg` backend.
- `scripts/trajectory_tools.py` is intentionally a thin CLI wrapper.
- Plotting skips tracks with fewer than 5 points by default. Use `--min-points` to override this.
- Plotting skips tracks that are completely outside the satellite image bounds by default. Use `--include-out-of-bounds` to override this.
- Use `--show-id-labels` to render same-color `tracked_id` labels next to visible trajectories.
- Tracks shorter than `--window-length` are left unchanged.
- The plotter looks for `location/<location_code>/sat_<location_code>.png` unless `--sat-image` is supplied.
- Do not reintroduce the old standalone `/Users/eric/code/traffic-trajectory-smooth` file layout into this repository. Integrate reusable logic into `trafficlab/trajectory/` and keep sample data outside the repo unless explicitly requested.

## External Integration Policy

When integrating another local project or script:

- Inspect the source project first and identify reusable logic, entry points, data files, generated outputs, and environment files.
- Copy or port only reusable source code and necessary documentation.
- Do not copy `.git/`, `.DS_Store`, `__pycache__/`, generated PNG/JSON outputs, sample datasets, or standalone environment files unless the user explicitly asks.
- Put reusable library code under a clear `trafficlab/<domain>/` package.
- Put runnable wrappers under `scripts/`.
- Add a short domain README when the integration creates a new subsystem.
- Prefer adapting code to existing TrafficLab I/O formats instead of creating parallel formats.
- Preserve existing user changes in the working tree. If unrelated files are already modified, do not revert or reformat them.

## Verification Checklist

For code changes, run the narrowest useful checks:

- Syntax check touched Python files:

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python -m py_compile path/to/file.py
```

- CLI help for changed scripts:

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python scripts/trajectory_tools.py --help
```

- For trajectory changes, omit `-o` / `--output` / `--plot-output` so outputs land next to the input JSON by default. Only redirect to `/private/tmp` for pure syntax/smoke tests that produce no meaningful artifact.
- For inference changes, use the same config name, mp4, environment, and working directory when comparing GUI and CLI behavior.
- For GUI changes, launch with MPS fallback when inference may be touched.

## Generated Files and Git Hygiene

- Do not commit or intentionally add `__pycache__/`, `.pyc`, `.DS_Store`, temporary plots, or throwaway JSON outputs.
- Prefer `/private/tmp` only for pure smoke-test artifacts (syntax checks, throwaway outputs). Trajectory plot and smooth outputs should use the default path next to the input JSON.
- Generated inference outputs belong under `output/` only when the user wants to keep them.
- Before reporting completion, check `git status --short` and distinguish your changes from pre-existing user changes.
- Never revert unrelated user changes.

## Agent Expectations

- If you need to run Python code, activate `trafficlab` and then use `python ...`.
- If you need to run inference, activate `trafficlab` and then use `PYTHONPATH=/Users/eric/code/TrafficLab-3D-main PYTORCH_ENABLE_MPS_FALLBACK=1 python ...`.
- If you compare GUI behavior and CLI behavior, keep the config name, mp4, environment, and working directory the same before drawing conclusions.
- Do not blame `half: true` for a failure unless the command has already reached actual inference/model execution and the error is consistent with FP16 or MPS backend issues.
- Keep README user-facing and concise. Put agent-only operational details here in `AGENTS.md`.
- Prefer small, focused changes over broad rewrites.
- When adding a new subsystem, document where it lives, how to run it, and what files should not be mixed into it.
