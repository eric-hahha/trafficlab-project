"""Wheel-only rectangle diagnostic for reprojection outputs.

Front/rear wheel keypoints (Apollo-24 indices 7/19/8/18) carry template
height h=0, so GProjection.cctv_to_sat (trafficlab/projection/g_projection.py)
applies no parallax correction to them — see its `if h != 0` branch. They
are therefore a control group: if their reprojected sat-pixel positions
already form a real-car-sized rectangle, the base homography/undistortion
is fine, and any distortion seen on the other 20 keypoints must come from
the per-keypoint height template used for parallax correction
(trafficlab/motion/keypoints_openpifpaf.py's H_BUMPER/H_CORNER/H_LAMP/
H_DOOR_BASE/H_ROOF constants).
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Iterator, Optional

# Apollo-24 keypoint indices; matches KEYPOINT_NAMES in
# scripts/plot_reprojection_keypoints.py and KP_NAMES in
# trafficlab/motion/keypoints_openpifpaf.py.
WHEEL_KP = {
    "front_wheel_left": 7,
    "rear_wheel_left": 8,
    "rear_wheel_right": 18,
    "front_wheel_right": 19,
}

# Edges of the wheel rectangle, named by the metric they represent.
EDGES = {
    "front_track": ("front_wheel_left", "front_wheel_right"),
    "rear_track": ("rear_wheel_left", "rear_wheel_right"),
    "left_wheelbase": ("front_wheel_left", "rear_wheel_left"),
    "right_wheelbase": ("front_wheel_right", "rear_wheel_right"),
    "diag_fl_rr": ("front_wheel_left", "rear_wheel_right"),
    "diag_fr_rl": ("front_wheel_right", "rear_wheel_left"),
}

# For each vertex, the two neighboring wheel names used to measure its
# interior angle's deviation from 90 degrees.
CORNERS = {
    "front_wheel_left": ("front_wheel_right", "rear_wheel_left"),
    "front_wheel_right": ("front_wheel_left", "rear_wheel_right"),
    "rear_wheel_left": ("front_wheel_left", "rear_wheel_right"),
    "rear_wheel_right": ("front_wheel_right", "rear_wheel_left"),
}

METRIC_NAMES = list(EDGES.keys()) + [f"angle_dev_{v}" for v in CORNERS]

Point = tuple[float, float]


def extract_wheel_points(obj: dict) -> dict[str, Optional[Point]]:
    """Pull the 4 wheel kp_sat points off one replay-JSON object record."""
    kp_sat = obj.get("kp_sat") or []
    points: dict[str, Optional[Point]] = {}
    for name, idx in WHEEL_KP.items():
        kp = kp_sat[idx] if idx < len(kp_sat) else None
        points[name] = (float(kp[0]), float(kp[1])) if kp is not None else None
    return points


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_deviation_deg(vertex: Point, p1: Point, p2: Point) -> Optional[float]:
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    cos_ang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return abs(math.degrees(math.acos(cos_ang)) - 90.0)


def compute_wheel_metrics(points: dict[str, Optional[Point]]) -> dict[str, Optional[float]]:
    """Compute whichever edge/diagonal/corner-angle metrics the available points allow.

    Never substitutes a default for a missing point — a metric that can't be
    computed from what's visible this frame is None, not zero.
    """
    metrics: dict[str, Optional[float]] = {}
    for metric_name, (a_name, b_name) in EDGES.items():
        a, b = points.get(a_name), points.get(b_name)
        metrics[metric_name] = _dist(a, b) if a is not None and b is not None else None

    for vertex_name, (n1_name, n2_name) in CORNERS.items():
        vertex, p1, p2 = points.get(vertex_name), points.get(n1_name), points.get(n2_name)
        key = f"angle_dev_{vertex_name}"
        metrics[key] = (
            _angle_deviation_deg(vertex, p1, p2)
            if vertex is not None and p1 is not None and p2 is not None
            else None
        )

    metrics["n_points"] = sum(1 for p in points.values() if p is not None)
    return metrics


def iter_instances(replay_data: dict) -> Iterator[dict]:
    """Yield one record per (frame, object) in the replay JSON."""
    for frame in replay_data.get("frames") or []:
        frame_index = frame.get("frame_index")
        for obj in frame.get("objects") or []:
            points = extract_wheel_points(obj)
            yield {
                "frame_index": frame_index,
                "tracked_id": obj.get("tracked_id"),
                "class": obj.get("class"),
                "points": points,
                "metrics": compute_wheel_metrics(points),
            }


def aggregate_metrics(instances: list[dict], min_points: int = 2) -> dict[str, dict]:
    """Summarize each metric across instances with at least `min_points` wheel keypoints."""
    values: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    n_considered = 0
    for inst in instances:
        if inst["metrics"]["n_points"] < min_points:
            continue
        n_considered += 1
        for name in METRIC_NAMES:
            v = inst["metrics"].get(name)
            if v is not None:
                values[name].append(v)

    summary: dict[str, dict] = {"n_instances_considered": n_considered}
    for name, vals in values.items():
        if not vals:
            summary[name] = {"count": 0}
            continue
        summary[name] = {
            "count": len(vals),
            "median": statistics.median(vals),
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }
    return summary


def px_to_meters(summary: dict[str, dict], px_per_m: float) -> dict[str, dict]:
    """Convert the edge-length metrics (not angle metrics) in `summary` to meters."""
    out: dict[str, dict] = {}
    for name in EDGES:
        stats = summary.get(name)
        if not stats or not stats.get("count"):
            continue
        out[name] = {
            "count": stats["count"],
            "median": stats["median"] / px_per_m,
            "mean": stats["mean"] / px_per_m,
            "std": stats["std"] / px_per_m,
            "min": stats["min"] / px_per_m,
            "max": stats["max"] / px_per_m,
        }
    return out


def compare_to_reference(summary_m: dict[str, dict], ref_dims: dict) -> dict[str, dict]:
    """Compare measured track/wheelbase medians (meters) to known reference dims."""
    comparisons: dict[str, dict] = {}
    pairs = {
        "front_track": "track_width",
        "rear_track": "track_width",
        "left_wheelbase": "wheelbase",
        "right_wheelbase": "wheelbase",
    }
    for metric_name, ref_key in pairs.items():
        stats = summary_m.get(metric_name)
        ref_val = ref_dims.get(ref_key)
        if not stats or not stats.get("count") or not ref_val:
            continue
        measured = stats["median"]
        comparisons[metric_name] = {
            "measured_m": measured,
            "reference_m": ref_val,
            "pct_error": (measured - ref_val) / ref_val * 100.0,
        }
    return comparisons


def pick_richest_instance(instances: list[dict]) -> Optional[dict]:
    """Pick the instance with the most visible wheel points (ties: first found)."""
    best = None
    for inst in instances:
        n = inst["metrics"]["n_points"]
        if n == 0:
            continue
        if best is None or n > best["metrics"]["n_points"]:
            best = inst
    return best


def plot_wheel_rectangle(instance: dict, sat_image_path, out_path, px_per_m: float | None = None) -> None:
    """Plot the available wheel points for one instance on the sat image,
    connecting every pair that's both present and annotating edge lengths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    points = instance["points"]
    colors = {
        "front_wheel_left": "#e41a1c",
        "front_wheel_right": "#377eb8",
        "rear_wheel_left": "#4daf4a",
        "rear_wheel_right": "#984ea3",
    }

    sat_image = Image.open(sat_image_path)
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(sat_image, extent=[0, sat_image.width, sat_image.height, 0])

    for edge_name, (a_name, b_name) in EDGES.items():
        a, b = points.get(a_name), points.get(b_name)
        if a is None or b is None:
            continue
        is_diag = edge_name.startswith("diag")
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black",
                linestyle="--" if is_diag else "-", linewidth=1.2, zorder=3)
        length_px = _dist(a, b)
        label = f"{length_px:.1f}px"
        if px_per_m:
            label += f" / {length_px / px_per_m:.2f}m"
        mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
        ax.annotate(label, (mx, my), fontsize=8, color="black", ha="center", zorder=5,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "yellow", "alpha": 0.85})

    for name, xy in points.items():
        if xy is None:
            continue
        ax.scatter(*xy, s=90, c=colors[name], edgecolors="white", linewidths=1.3, zorder=4)
        ax.annotate(name, xy, xytext=(6, 6), textcoords="offset points", fontsize=9,
                    color="black", zorder=5,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85})

    xs = [p[0] for p in points.values() if p is not None]
    ys = [p[1] for p in points.values() if p is not None]
    pad = 80.0
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(max(ys) + pad, min(ys) - pad)
    ax.set_aspect("equal")
    ax.set_title(
        f"Wheel rectangle check — frame {instance['frame_index']} "
        f"tracked_id {instance['tracked_id']} ({instance['metrics']['n_points']}/4 wheels)"
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
