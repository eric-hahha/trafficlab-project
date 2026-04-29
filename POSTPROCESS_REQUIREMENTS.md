# Postprocess Trajectory Requirements

## Goal

Provide an offline trajectory postprocessor for TrafficLab inference output.

The postprocessor should prioritize trajectory geometry repair, especially removing sharp short-term turns and direction reversals, while preserving useful raw inference observations for debugging and future smoothing algorithms.

## Input

The tool accepts TrafficLab replay output in either format:

- `.json`
- `.json.gz`

The input must be a JSON object with a top-level `frames` array:

```json
{
  "frames": [
    {
      "frame_index": 0,
      "objects": [
        {
          "tracked_id": 373,
          "class": "Two_Wheeler",
          "sat_coords": [640.9, 587.5]
        }
      ]
    }
  ]
}
```

Required object fields:

- `tracked_id`: stable ID for grouping one trajectory.
- `sat_coords`: satellite-plane coordinate `[x, y]`.

Recommended object fields:

- `class`: vehicle/object class.
- `confidence`: detector confidence.
- `bbox_2d`: image-space detection box.
- `sat_center`: duplicate or center point for compatible smoothing tools.

## Output

The output must remain compatible with both TrafficLab visualization and the external trajectory smoothing/plotting project.

Required output structure:

```json
{
  "frames": [
    {
      "frame_id": 0,
      "timestamp": 0.0,
      "objects": [
        {
          "tracked_id": 373,
          "class": "Two_Wheeler",
          "sat_coords": [640.9, 587.5],
          "sat_center": [640.9, 587.5]
        }
      ]
    }
  ]
}
```

Output requirements:

- Top-level JSON must be an object.
- Top-level object must contain `frames`.
- Every frame should contain `frame_id`.
- Every object used for trajectory processing must contain `tracked_id`.
- Every object used for trajectory processing must contain numeric `sat_coords: [x, y]`.
- `sat_center` should be present and synchronized with `sat_coords`.
- Original TrafficLab fields should be preserved unless there is a clear reason to remove them.
- If `sat_coords` is corrected, `sat_center` must be corrected to the same value.
- If `sat_floor_box` exists, it should be translated by the same coordinate delta as `sat_coords`.
- `bbox_3d` should not be modified unless the projection can be recomputed correctly.

## Scope

The first production target is motorcycle/two-wheeler trajectory repair.

Default target classes:

- `motor`
- `motorcycle`
- `two_wheeler`
- `Two_Wheeler`

Other classes should be passed through unchanged unless explicitly configured.

## Core Problems To Solve

### 1. Long-term Direction Consistency

The postprocessor must use long-term trajectory evidence to estimate the object's dominant motion direction.

This is the primary repair goal.

Symptoms:

- The object should mostly move along one corridor, but short windows contain large side displacement.
- Consecutive displacements form a snake-like path around the long-term motion direction.
- Local smoothing makes the path visually smoother but still serpentine.

Expected behavior:

- Estimate a long-term centerline or dominant axis from the track.
- Decompose each displacement into longitudinal and lateral components relative to that long-term direction.
- Keep longitudinal progress when it is plausible.
- Suppress or project excessive lateral displacement back toward the centerline corridor.
- Preserve true long turns by switching to piecewise local centerlines instead of forcing one global straight line.

### 2. Sharp S-shaped Turns

The postprocessor must detect and correct short-term sharp S-shaped artifacts.

Symptoms:

- Direction changes abruptly within a few frames.
- A point or short segment creates a visually sharp corner.
- The trajectory briefly moves sideways or backward, then returns.

Expected behavior:

- Short abnormal segment is replaced by a smoother path between stable neighboring points.
- The large-scale trajectory direction is preserved.
- Natural long turns are not flattened.

Important clarification:

- Fixing a sharp S should not create a smoother snake.
- The corrected result should move closer to the long-term trajectory corridor.
- A local B-Spline that merely makes the snake prettier does not satisfy this requirement.

### 3. Short Direction Reversal

The postprocessor must detect short-term direction reversals.

Examples:

- Forward -> backward -> forward.
- Left lateral jump -> right lateral jump -> original direction.
- Near 180 degree turn over one or a few frames.

Expected behavior:

- If the reversal is short and isolated, correct it.
- If the reversal is part of a long valid turn, leave it unchanged.

### 4. Dense Stationary Clusters

The postprocessor should avoid turning stationary jitter clusters into long curves.

Symptoms:

- Multiple consecutive points remain in a small radius.
- Heading is unstable because movement is too small.

Expected behavior:

- Do not infer strong heading from tiny movement.
- Prefer leaving cluster points stable or replacing them with a robust local center.
- Do not stretch a dense cluster into a curved trajectory only for smoothness.

## Non-goals

The postprocessor is not expected to solve every tracking or projection issue in the first version.

The first version should not:

- Perform aggressive global smoothing.
- Force all tracks into straight lines.
- Overwrite raw observations without recording original values.
- Recalculate CCTV-space `bbox_3d` unless projection data is available.
- Modify cars/trucks by default.

## Algorithm Requirements

The postprocessor should be structured as a staged pipeline:

```text
raw sat_coords
  -> diagnostics
  -> long-term direction / centerline estimation
  -> outlier detection
  -> direction-corridor projection
  -> optional bad segment interpolation
  -> optional Savitzky-Golay smoothing
  -> optional visual B-Spline output
```

Each stage must be configurable. Debug metadata should make it clear which stage modified or annotated each point.

### Track Grouping

The tool must group objects by `tracked_id`.

For each track:

1. Sort points by frame order.
2. Ignore tracks shorter than `min_track_points`.
3. Process only configured target classes.

### Sharp Turn Detection

For consecutive points:

```text
P[i-1], P[i], P[i+1]
```

Compute:

```text
v1 = P[i] - P[i-1]
v2 = P[i+1] - P[i]
turn_angle = angle_between(v1, v2)
```

A point is a sharp-turn candidate when:

- `turn_angle >= sharp_turn_angle_deg`
- both step lengths are greater than `min_step_px`
- the candidate belongs to a short abnormal segment

Initial default:

```yaml
sharp_turn_angle_deg: 130
min_step_px: 0.5
```

This stage is secondary. It should catch extreme local errors, but it should not be the main mechanism for producing the final motorcycle trajectory.

### Long-term Direction / Centerline Estimation

The postprocessor should estimate the long-term motion path before deciding which points are abnormal.

Purpose:

- Avoid making decisions from only three points.
- Detect snake-like trajectories that are locally smooth but globally inconsistent.
- Repair points relative to the intended motion corridor.

Supported centerline modes:

```yaml
direction_correction:
  centerline_mode: "global_pca"      # global_pca | piecewise_pca | robust_regression
```

Initial implementation recommendation:

1. Split the track into windows or segments.
2. Estimate a dominant axis for each segment using PCA or robust linear regression.
3. Align axis signs so longitudinal progress is consistent over time.
4. Build a centerline using stable points, not every noisy point equally.
5. Use the centerline as the reference for lateral correction.

The centerline should be piecewise, not necessarily one global straight line.

Reason:

- Motorcycles can naturally turn.
- A single global straight line may incorrectly flatten true turns.
- A piecewise centerline can preserve large-scale curvature while suppressing short-term snake motion.

### Direction-corridor Projection

The main repair step should project excessive lateral motion back toward the long-term trajectory corridor.

For each point:

```text
reference = local centerline point
axis = local dominant direction
perpendicular = normal(axis)

delta = point - reference
longitudinal = dot(delta, axis)
lateral = dot(delta, perpendicular)
```

A point should be corrected when:

- the lateral offset exceeds a configured corridor width, or
- the movement vector angle differs too much from the local dominant direction, or
- the lateral sign alternates repeatedly in a short window, forming snake motion.

Initial correction method:

```text
corrected_point = reference
                + longitudinal * axis
                + clipped_lateral * perpendicular
```

Where:

```text
clipped_lateral = clamp(lateral, -max_lateral_offset_px, max_lateral_offset_px)
```

For stronger correction:

```text
clipped_lateral = lateral * lateral_retention
```

with `lateral_retention` between `0.0` and `0.3`.

Recommended default:

```yaml
direction_correction:
  enabled: true
  centerline_mode: "global_pca"
  window_size: 15
  min_points: 8
  max_angle_from_axis_deg: 18
  max_lateral_offset_px: 2.0
  lateral_retention: 0.05
  preserve_longitudinal_progress: true
  max_correction_px: 16.0
```

Behavior requirements:

- This stage may update `sat_coords`.
- If it updates `sat_coords`, it must update `sat_center`.
- It must preserve original coordinates in `postprocess`.
- It must not use B-Spline output as the source of truth.
- It should record the local axis, lateral offset, and correction amount when debug metadata is enabled.

Recommended metadata:

```json
{
  "postprocess": {
    "corrected": true,
    "reason": "direction_corridor_projection",
    "original_sat_coords": [730.0, 651.0],
    "corrected_sat_coords": [728.7, 648.2],
    "diagnostics": {
      "axis_angle_deg": 37.2,
      "lateral_offset_px": 7.4,
      "lateral_retention": 0.05
    }
  }
}
```

This stage should run before Savitzky-Golay smoothing and before visual B-Spline generation.

### Outlier Detection

The postprocessor should include a dedicated outlier detection stage before smoothing.

Purpose:

- Remove physically implausible points before they can pollute smoothing.
- Detect short spikes that are not necessarily caught by simple three-point angle checks.
- Mark invalid points or short invalid segments for interpolation.

Required diagnostics per track:

- step distance between consecutive points
- instantaneous speed
- acceleration or speed delta
- turn angle
- turn rate
- optional curvature radius
- optional lateral deviation from local trend

Outlier detection should support pixel-based thresholds first, because `px_per_m` may not always be available.

Initial pixel-based checks:

```yaml
outlier_detection:
  enabled: true
  max_step_px: null              # Optional hard cap for one-frame jumps
  max_accel_px_per_frame2: 8.0   # Reject sudden acceleration in satellite pixels
  max_turn_angle_deg: 150        # Reject near-reversal over one frame
  min_step_px_for_turn: 0.5      # Ignore tiny movements when computing turn
  max_invalid_segment_len: 8
```

Physical checks should be supported when `px_per_m` and `fps` are available:

```yaml
outlier_detection:
  max_speed_kmh: 130
  max_accel_kmh_per_s: 150
  speed_turn_rules:
    - min_speed_kmh: 50
      max_turn_angle_deg: 20
    - min_speed_kmh: 30
      max_turn_angle_deg: 35
```

Outlier examples:

- A point causes impossible acceleration.
- A point causes a short near-180-degree reversal.
- A point creates a sudden large lateral jump and immediately returns.
- A short segment strongly disagrees with local dominant direction.

Outlier repair requirement:

- Do not simply delete points from the JSON.
- Mark invalid points or invalid segments.
- Repair short invalid segments using neighboring stable anchors.
- Preserve original coordinates in `postprocess`.

Example metadata:

```json
{
  "postprocess": {
    "corrected": true,
    "reason": "outlier_acceleration",
    "original_sat_coords": [710.0, 650.0],
    "corrected_sat_coords": [708.4, 647.9],
    "diagnostics": {
      "accel_px_per_frame2": 12.4,
      "turn_angle_deg": 171.2
    }
  }
}
```

Outlier detection must run before Savitzky-Golay smoothing and before visual B-Spline generation.

### Savitzky-Golay Smoothing

The postprocessor should include an optional Savitzky-Golay smoothing stage after outlier repair.

Purpose:

- Remove high-frequency jitter in `sat_coords`.
- Preserve local velocity and acceleration trends better than simple EMA.
- Avoid using B-Spline as the only smoothing mechanism.

Recommended default:

```yaml
savgol:
  enabled: false                 # Start disabled until outlier repair is stable
  window_length: 9               # Must be odd
  polyorder: 2
  target_field: "sat_coords"
  update_sat_center: true
  preserve_endpoints: true
  min_track_points: 9
```

Behavior requirements:

- S-G smoothing runs after outlier interpolation.
- S-G may update `sat_coords` when enabled.
- If `sat_coords` is updated, `sat_center` must be updated to the same value.
- Original repaired coordinates before S-G should be preserved when debug mode is enabled.
- S-G should not run on tracks shorter than `window_length`.
- `window_length` must be adjusted or rejected if it is not odd or exceeds track length.
- Endpoints should be preserved by default or handled conservatively to avoid endpoint drift.

Recommended metadata:

```json
{
  "postprocess": {
    "savgol": true,
    "pre_savgol_sat_coords": [700.0, 650.0],
    "savgol_window_length": 9,
    "savgol_polyorder": 2
  }
}
```

Important limitation:

- S-G is not an outlier remover.
- S-G must not be applied before invalid points are repaired.
- If outliers remain, S-G may spread the error into neighboring frames.

### Direction Consistency Detection

The postprocessor should detect direction inconsistency over a long-term or piecewise-local window, not only three-point angles.

For each track:

1. Compute step vectors.
2. Estimate local dominant direction using a sliding or piecewise window.
3. Decompose steps into longitudinal and lateral components.
4. Mark steps whose lateral component or angle from axis is too large.
5. Detect repeated lateral sign alternation that forms snake motion.
6. Merge nearby bad steps into short or medium bad segments.

A segment should be considered abnormal when:

- Most vectors in the segment disagree with the local dominant direction.
- The segment's lateral offsets exceed the corridor.
- The segment alternates lateral direction repeatedly while maintaining forward progress.
- Stable points or a stable centerline exist before and after the segment.

### Bad Segment Repair

The tool should repair short bad segments using neighboring stable anchors.

For segment:

```text
bad_start ... bad_end
```

Find:

```text
anchor_before = point before bad_start
anchor_after = point after bad_end
```

Initial repair method:

```text
linear interpolation between anchor_before and anchor_after
```

Repair methods:

- Projection to local dominant axis or centerline corridor.
- Cubic interpolation.
- Robust local regression.

Interpolation should be used mainly for missing/invalid segments. For snake-like trajectories, projection to a long-term centerline is preferred because it preserves longitudinal progress better.

## Visual B-Spline Output

B-Spline smoothing may be generated as a second output field for visualization.

This field must not overwrite the analysis/repair coordinate:

```json
{
  "sat_coords": [700.0, 650.0],
  "sat_center": [700.0, 650.0],
  "visual_sat_coords": [699.2, 649.7],
  "visual_sat_center": [699.2, 649.7]
}
```

Requirements:

- `sat_coords` remains the primary repaired coordinate.
- `visual_sat_coords` is optional and intended for display only.
- `visual_sat_center` should be synchronized with `visual_sat_coords`.
- B-Spline smoothing should run after outlier repair.
- First and last visual points should preserve the repaired endpoints by default.
- If B-Spline fitting fails, the tool should leave `sat_coords` untouched and report the skipped track.

### Raw Data Preservation

If a point is modified, the original coordinate must be preserved:

```json
{
  "postprocess": {
    "corrected": true,
    "reason": "sharp_turn",
    "original_sat_coords": [700.0, 650.0],
    "corrected_sat_coords": [702.0, 648.0],
    "turn_angle_deg": 168.4
  }
}
```

The postprocessor should not destroy raw information needed for later debugging.

## Configuration Requirements

The tool should load default parameters from a YAML file:

```text
postprocess_config.yaml
```

The default config file should live next to `postprocess.py`.

The tool should also expose CLI parameters for temporary overrides:

- input path
- output path
- config path
- target classes
- minimum track length
- sharp turn angle threshold
- minimum step length
- maximum bad segment length
- bad segment merge gap
- dry-run mode
- verbose mode

Suggested defaults:

```yaml
target_classes:
  - motor
  - motorcycle
  - two_wheeler
  - Two_Wheeler
min_track_points: 8
sharp_turn_angle_deg: 130
min_step_px: 0.5
max_bad_segment_len: 8
bridge_gap: 1
```

Recommended YAML structure:

```yaml
target_classes:
  - motor
  - motorcycle
  - two_wheeler
  - Two_Wheeler

repair:
  min_track_points: 8
  sharp_turn_angle_deg: 130
  min_step_px: 0.5
  min_lateral_deviation_px: 0.0
  bridge_gap: 1
  max_bad_segment_len: 8

direction_correction:
  enabled: true
  centerline_mode: global_pca
  window_size: 15
  min_points: 8
  max_angle_from_axis_deg: 18
  max_lateral_offset_px: 2.0
  lateral_retention: 0.05
  preserve_longitudinal_progress: true
  max_correction_px: 16.0

outlier_detection:
  enabled: true
  max_step_px: null
  max_accel_px_per_frame2: 8.0
  max_turn_angle_deg: 150
  min_step_px_for_turn: 0.5
  max_invalid_segment_len: 8

savgol:
  enabled: false
  window_length: 9
  polyorder: 2
  target_field: sat_coords
  update_sat_center: true
  preserve_endpoints: true

visual_bspline:
  enabled: true
  min_points: 8
  degree: 3
  smooth_px: 2.0
  preserve_endpoints: true
```

## Validation Requirements

The tool should support dry-run analysis:

```bash
python postprocess.py output.json.gz --dry-run --verbose
```

Dry-run output should report:

- tracks seen
- tracks processed
- sharp candidates
- corrected segments
- corrected points
- per-track summary

For corrected output, the tool should report:

- output path
- total corrected points
- total corrected segments

## Acceptance Criteria

A result is acceptable when:

- The output JSON can be parsed.
- The output has top-level `frames`.
- Every processed object has numeric `sat_coords`.
- Every processed object has synchronized `sat_center`.
- Short, visually sharp S-shaped artifacts are reduced.
- Short near-180-degree local reversals are reduced.
- Dense stationary clusters are not stretched into long curves.
- Normal long turns are mostly preserved.
- Every changed point records original and corrected coordinates.

## Recommended Development Phases

### Phase 1: Format Compatibility

- Read `.json` and `.json.gz`.
- Write `.json` and `.json.gz`.
- Preserve original fields.
- Add `frame_id`.
- Add/sync `sat_center`.

### Phase 2: Sharp Turn Repair

- Detect local sharp turns.
- Merge nearby candidates.
- Interpolate short bad segments.
- Add debug metadata.

### Phase 3: Direction Consistency Repair

- Estimate local dominant direction.
- Detect short direction reversal segments.
- Repair those segments using stable anchors.

### Phase 4: Cluster Stabilization

- Detect low-motion dense clusters.
- Avoid generating artificial heading from stationary points.
- Optionally replace cluster jitter with robust local center.

### Phase 5: Visual QA Support

- Emit summary statistics.
- Optionally export before/after CSV for plotting.
- Optionally annotate corrected points for visualization.
