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
source /Users/eric/opt/anaconda3/bin/activate trafficlab && PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_inference.py --config-name car_heading_smooth --all-pending
```

Process one location:

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_inference.py --config-name car_heading_smooth --location test1
```

Process one mp4:

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_inference.py --config-name car_heading_smooth --mp4 location/test1/footage/test1_8_100_s.mp4
```

Force re-run even if output already exists:

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_inference.py --config-name car_heading_smooth --location test1 --force
```

### Postprocess

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python postprocess.py --help
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

- For trajectory changes, prefer a small disposable smoke test that writes to `/private/tmp`.
- For inference changes, use the same config name, mp4, environment, and working directory when comparing GUI and CLI behavior.
- For GUI changes, launch with MPS fallback when inference may be touched.

## Generated Files and Git Hygiene

- Do not commit or intentionally add `__pycache__/`, `.pyc`, `.DS_Store`, temporary plots, or throwaway JSON outputs.
- Prefer `/private/tmp` for smoke-test artifacts.
- Generated inference outputs belong under `output/` only when the user wants to keep them.
- Before reporting completion, check `git status --short` and distinguish your changes from pre-existing user changes.
- Never revert unrelated user changes.

## Agent Expectations

- If you need to run Python code, activate `trafficlab` and then use `python ...`.
- If you need to run inference, activate `trafficlab` and then use `PYTORCH_ENABLE_MPS_FALLBACK=1 python ...`.
- If you compare GUI behavior and CLI behavior, keep the config name, mp4, environment, and working directory the same before drawing conclusions.
- Do not blame `half: true` for a failure unless the command has already reached actual inference/model execution and the error is consistent with FP16 or MPS backend issues.
- Keep README user-facing and concise. Put agent-only operational details here in `AGENTS.md`.
- Prefer small, focused changes over broad rewrites.
- When adding a new subsystem, document where it lives, how to run it, and what files should not be mixed into it.
