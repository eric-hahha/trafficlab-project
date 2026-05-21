# TrafficLab AI Instructions

This file is for AI coding agents working in this repository.

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

## Agent Expectations

- If you need to run Python code, activate `trafficlab` and then use `python ...`.
- If you need to run inference, activate `trafficlab` and then use `PYTORCH_ENABLE_MPS_FALLBACK=1 python ...`.
- If you compare GUI behavior and CLI behavior, keep the config name, mp4, environment, and working directory the same before drawing conclusions.
- Do not blame `half: true` for a failure unless the command has already reached actual inference/model execution and the error is consistent with FP16 or MPS backend issues.
