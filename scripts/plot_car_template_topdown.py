#!/usr/bin/env python3
"""Top-down (bird's-eye) schematic of a car keypoint template's 24 points.

Plots a template JSON's own (x, h, z) body-frame coordinates directly (not
a replay/CCTV/SAT projection) — i.e. what the rigid template looks like
from directly above, with front at the top and vehicle-left on the left
(matches sitting in the driver's seat facing forward, looking straight
down). Works on either build_car_template()'s heuristic output (wrapped in
the same {"kp_names": [...], "template": [[x,h,z], ...]} shape) or a
scripts/build_cad_keypoint_template.py output JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from trafficlab.trajectory.plotting import _kp_part_color


def _abbrev(name: str) -> str:
    """Generic short code: initials of each underscore-separated token."""
    return "".join(tok[0].upper() for tok in name.split("_") if tok)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template_json", help="Path to a template JSON "
                         "(scripts/build_cad_keypoint_template.py output, or "
                         "any file with the same kp_names/template shape).")
    parser.add_argument("--out", default=None, help="Output PNG path "
                         "(default: <template_json_stem>.topdown.png next to the input).")
    parser.add_argument("--size", type=int, default=500, help="Output image is size x size pixels (default 500).")
    parser.add_argument("--font-scale", type=float, default=1.0,
                         help="Multiplier applied to every text element's font size (default 1.0).")
    args = parser.parse_args()
    fs = args.font_scale

    data = json.loads(Path(args.template_json).read_text())
    kp_names = data["kp_names"]
    template = data["template"]  # 24 x [x, h, z], meters

    xs = [row[0] for row in template]
    zs = [row[2] for row in template]

    # Top-down, front-at-top, vehicle-left-at-image-left: plot (-x, -z).
    plot_x = [-x for x in xs]
    plot_y = [-z for z in zs]

    dpi = 100
    fig, ax = plt.subplots(figsize=(args.size / dpi, args.size / dpi), dpi=dpi)

    # Body outline: simple rounded rectangle from the template's own bbox,
    # purely illustrative context, not a real silhouette.
    margin = 0.15
    x_min, x_max = min(plot_x) - margin, max(plot_x) + margin
    y_min, y_max = min(plot_y) - margin, max(plot_y) + margin
    ax.add_patch(FancyBboxPatch(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        boxstyle="round,pad=0,rounding_size=0.3",
        linewidth=1, edgecolor="#888888", facecolor="#f2f2f2", zorder=0,
    ))

    # Wheelbase/track construction lines between the 4 wheel keypoints, if present.
    idx = {n: i for i, n in enumerate(kp_names)}
    for a, b in (("front_wheel_center_left", "rear_wheel_center_left"),
                 ("front_wheel_center_right", "rear_wheel_center_right"),
                 ("front_wheel_center_left", "front_wheel_center_right"),
                 ("rear_wheel_center_left", "rear_wheel_center_right")):
        if a in idx and b in idx:
            ax.plot([plot_x[idx[a]], plot_x[idx[b]]], [plot_y[idx[a]], plot_y[idx[b]]],
                     color="#bbbbbb", linewidth=1, linestyle="--", zorder=1)

    for i, name in enumerate(kp_names):
        color = _kp_part_color(name)
        ax.scatter(plot_x[i], plot_y[i], s=40, color=color, edgecolors="black",
                   linewidths=0.5, zorder=3)
        ax.annotate(_abbrev(name), (plot_x[i], plot_y[i]), textcoords="offset points",
                    xytext=(4, 3), fontsize=4.5 * fs, zorder=4)

    # Orientation labels.
    ax.text(0, y_max + 0.08, "FRONT", ha="center", va="bottom", fontsize=8 * fs, weight="bold", color="#444444")
    ax.text(0, y_min - 0.08, "REAR", ha="center", va="top", fontsize=8 * fs, weight="bold", color="#444444")
    ax.text(x_min - 0.08, 0, "LEFT", ha="right", va="center", fontsize=7 * fs, rotation=90, color="#444444")
    ax.text(x_max + 0.08, 0, "RIGHT", ha="left", va="center", fontsize=7 * fs, rotation=-90, color="#444444")

    # 1-metre scale bar.
    bar_x0 = x_min + 0.1
    bar_y = y_min - 0.35
    ax.plot([bar_x0, bar_x0 + 1.0], [bar_y, bar_y], color="black", linewidth=2)
    ax.text(bar_x0 + 0.5, bar_y - 0.06, "1 m", ha="center", va="top", fontsize=7 * fs)

    # Legend: part -> color.
    legend_parts = ["wheel", "light", "plate", "door", "corner", "bumper", "glass"]
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=_kp_part_color(p), markersize=6)
               for p in legend_parts]
    ax.legend(handles, legend_parts, loc="upper left", bbox_to_anchor=(1.0, 1.0),
              fontsize=6 * fs, frameon=False, borderaxespad=0)

    cad_model = data.get("source", {}).get("cad_model", Path(args.template_json).stem)
    ax.set_title(f"{cad_model}\ntop-down keypoint template", fontsize=7 * fs)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(x_min - 0.9, x_max + 1.4)
    ax.set_ylim(y_min - 0.6, y_max + 0.3)

    out_path = Path(args.out) if args.out else Path(args.template_json).with_suffix("").with_suffix(".topdown.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out_path} ({args.size}x{args.size}px)")


if __name__ == "__main__":
    main()
