# Trajectory Tools Integration

This folder contains the integrated version of `/Users/eric/code/traffic-trajectory-smooth`.
Only reusable TrafficLab code lives here; sample JSON files, PNG outputs, pycache files, and the
standalone conda project were intentionally not copied into this repository.

## Structure

- `io.py`: shared `.json` / `.json.gz` loading, writing, and path resolution helpers.
- `smoothing.py`: Savitzky-Golay smoothing for `sat_coords`, grouped by `tracked_id`.
- `plotting.py`: satellite-background trajectory plotting with optional zoom and heading arrows.
- `scripts/trajectory_tools.py`: command-line entry point for smoothing, plotting, or both.

## Usage

Run commands from the repository root with the `trafficlab` conda environment active:

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab
python scripts/trajectory_tools.py smooth output/example.json.gz
python scripts/trajectory_tools.py plot output/example.smoothed.json.gz --location-code test1
python scripts/trajectory_tools.py smooth-and-plot output/example.json.gz --ids 7,373 --zoom-to-fit
```

If the input file does not contain `location_code` metadata, pass `--location-code` or
`--sat-image`. By default the plotter looks for:

```text
location/<location_code>/sat_<location_code>.png
```

## Input Format

The tools accept the standard TrafficLab replay format:

```json
{
  "frames": [
    {
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

The plotter also accepts a top-level frame list, but smoothing expects objects to have
`tracked_id` and `sat_coords`. Plotting skips tracks with fewer than 5 points by default;
override this with `--min-points` if needed. It also skips tracks that are completely
outside the satellite image by default; use `--include-out-of-bounds` to include them.
Use `--show-id-labels` to draw same-color `tracked_id` labels next to visible tracks.
Tracks shorter than `--window-length` are left unchanged during smoothing.
