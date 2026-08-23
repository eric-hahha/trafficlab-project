"""Parallax-correction before/after diagnostic for reprojection outputs.

Every kp_sat[i] stored in a replay JSON is already parallax-corrected for
that keypoint's own template height (see GProjection.cctv_to_sat and the
per-keypoint height template in trafficlab/motion/keypoints_openpifpaf.py).
This recomputes the *uncorrected* apparent point for the same kp_cctv pixel
-- the h=0 ground-plane projection GProjection.cctv_to_sat gives without any
height correction -- so it can be plotted next to kp_sat and visually
compared: the line between them is exactly the correction displacement
applied by GProjection.parallax_correct_ground_to_real, proportional to
h / z_cam (keypoint height over camera height).

Both points can instead be recomputed fresh from kp_cctv with a given
car-template + GProjection (pass `template` to compute_pre_post_points /
compute_frame_records / etc.), ignoring the replay JSON's stored kp_sat
entirely. Use this to check a corrected/edited G_projection config against
the same raw detections without re-running inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from trafficlab.motion.keypoints_openpifpaf import KP_NAMES

Point = tuple[float, float]


def compute_pre_post_points(
    obj: dict, g_engine, kp_conf: float = 0.2, template=None,
) -> dict[int, dict]:
    """For every confident kp_cctv[i] this object has a usable pre/post pair
    for, return {kp_idx: {'pre': (x, y), 'post': (x, y)}}.

    'pre' is always the apparent h=0 (uncorrected) projection of the CCTV
    pixel, freshly computed via g_engine.

    'post' depends on `template` (a (24, 3) array from build_car_template):
    - template is None (default): 'post' is kp_sat[i] as already stored in
      the replay JSON -- not recomputed, so this mode doesn't need to know
      which per-keypoint height template the original run used.
    - template is given: 'post' is freshly recomputed too, via
      g_engine.cctv_to_sat(x, y, h=template[i, 1]) -- ignoring kp_sat
      entirely. Use this to check a corrected/edited G_projection against
      the same raw kp_cctv detections without re-running inference (the
      stored kp_sat would otherwise still reflect the old projection).
    """
    kp_cctv = obj.get("kp_cctv") or []
    kp_sat = obj.get("kp_sat") or []
    out: dict[int, dict] = {}
    for i, kp in enumerate(kp_cctv):
        x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
        if conf < kp_conf or (x == 0.0 and y == 0.0):
            continue
        if template is not None:
            h_i = float(template[i, 1])
            post_pt = g_engine.cctv_to_sat(x, y, h=h_i)
        else:
            if i >= len(kp_sat) or kp_sat[i] is None:
                continue
            post_pt = kp_sat[i]
        pre_pt = g_engine.cctv_to_sat(x, y, h=0.0)
        out[i] = {"pre": (float(pre_pt[0]), float(pre_pt[1])),
                  "post": (float(post_pt[0]), float(post_pt[1]))}
    return out


def compute_frame_records(frame: dict, g_engine, kp_conf: float = 0.2, template=None) -> list[dict]:
    """One record per object in `frame` that has >=1 valid pre/post pair."""
    records = []
    for obj in frame.get("objects") or []:
        pairs = compute_pre_post_points(obj, g_engine, kp_conf=kp_conf, template=template)
        if pairs:
            records.append({
                "tracked_id": obj.get("tracked_id"),
                "class": obj.get("class"),
                "pairs": pairs,
            })
    return records


def find_frame(replay_data: dict, frame_index: int) -> Optional[dict]:
    return next(
        (f for f in replay_data.get("frames") or [] if f.get("frame_index") == frame_index),
        None,
    )


def iter_frame_records(
    replay_data: dict, g_engine, kp_conf: float = 0.2, template=None,
) -> Iterator[tuple[int, list[dict]]]:
    """Yield (frame_index, records) for every frame that has >=1 valid pair."""
    for frame in replay_data.get("frames") or []:
        records = compute_frame_records(frame, g_engine, kp_conf=kp_conf, template=template)
        if records:
            yield frame.get("frame_index"), records


def compute_view_extent(
    records_list: list[list[dict]], pad: float = 60.0,
) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    """Union bounding box (with padding) of every pre/post point across one
    or more record sets, as (xlim, ylim) ready to pass straight to
    ax.set_xlim/ax.set_ylim (ylim is already inverted: (max+pad, min-pad),
    matching image y-down convention). Pass multiple record sets (e.g. a
    baseline and a recomputed pass over the same frame) to get one shared
    extent both plots can use, so they end up at the same zoom/position and
    can be overlaid. Returns None if there are no points at all.
    """
    xs = [pair["pre"][0] for records in records_list for record in records for pair in record["pairs"].values()]
    xs += [pair["post"][0] for records in records_list for record in records for pair in record["pairs"].values()]
    ys = [pair["pre"][1] for records in records_list for record in records for pair in record["pairs"].values()]
    ys += [pair["post"][1] for records in records_list for record in records for pair in record["pairs"].values()]
    if not xs:
        return None
    return (min(xs) - pad, max(xs) + pad), (max(ys) + pad, min(ys) - pad)


def pick_richest_frame(
    replay_data: dict, g_engine, kp_conf: float = 0.2, template=None,
) -> tuple[Optional[int], list[dict]]:
    """Auto-pick the frame with the most valid pairs, summed across its objects."""
    best_idx, best_records, best_n = None, [], -1
    for frame_index, records in iter_frame_records(replay_data, g_engine, kp_conf=kp_conf, template=template):
        n = sum(len(r["pairs"]) for r in records)
        if n > best_n:
            best_idx, best_records, best_n = frame_index, records, n
    return best_idx, best_records


def plot_frame_pre_post_keypoints(
    frame_index: int,
    records: list[dict],
    sat_image_path,
    out_path,
    *,
    ids: Optional[set] = None,
    title: Optional[str] = None,
    dpi: int = 200,
    recomputed: bool = False,
    view_extent: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
) -> tuple[Path, int, int]:
    """Plot every record's kp_sat (post-correction, labeled 'tid-part_name',
    colored by car part) next to its h=0 apparent point (pre-correction,
    small, unlabeled, gray dot), joined by a red line. Returns
    (out_path, n_keypoints_plotted, n_objects_plotted).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from trafficlab.trajectory.plotting import TrajectoryPlotter, _kp_part_color

    if ids is not None:
        records = [r for r in records if r["tracked_id"] in ids]
    if not records:
        raise ValueError(f"No object with a valid pre/post keypoint pair in frame {frame_index}.")

    sat_image = Image.open(sat_image_path)
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(sat_image, extent=[0, sat_image.width, sat_image.height, 0])

    # Frame the view on both pre and post points, so every correction line
    # is fully visible end to end -- not just the post (kp_sat) cluster. A
    # keypoint whose height is a large fraction of z_cam can have a pre
    # point thousands of pixels away (displacement scales with |apparent -
    # camera nadir|, which itself blows up for distant/oblique vehicles
    # under a flat-ground homography); when that happens the view zooms out
    # accordingly rather than cropping those points out of frame.
    #
    # If the caller passes view_extent (e.g. a shared extent computed across
    # a baseline + recomputed pair via compute_view_extent), use it as-is
    # instead of fitting to just this call's own records -- that's what lets
    # two separate plot_frame_pre_post_keypoints calls end up at the same
    # zoom/position and be overlaid pixel-for-pixel.
    if view_extent is not None:
        xlim, ylim = view_extent
    else:
        xlim, ylim = compute_view_extent([records], pad=60.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_title(title or f"Parallax correction before/after — frame {frame_index}", fontsize=16)
    ax.set_xlabel("X Coordinate (pixels)", fontsize=12)
    ax.set_ylabel("Y Coordinate (pixels)", fontsize=12)

    n_keypoints = 0
    occupied_boxes: list = []
    for record in records:
        for kp_idx, pair in record["pairs"].items():
            pre, post = pair["pre"], pair["post"]
            n_keypoints += 1
            line_color = _kp_part_color(KP_NAMES[kp_idx])
            ax.plot([pre[0], post[0]], [pre[1], post[1]], color=line_color, linewidth=1.2, zorder=2)
            ax.scatter(*pre, s=16, c="0.35", edgecolors="white", linewidths=0.4, zorder=2)
            occupied_boxes.append(TrajectoryPlotter._marker_box(ax, pre, 3.0))

        post_x = [p["post"][0] for p in record["pairs"].values()]
        post_y = [p["post"][1] for p in record["pairs"].values()]
        post_c = [_kp_part_color(KP_NAMES[kp_idx]) for kp_idx in record["pairs"]]
        ax.scatter(post_x, post_y, s=45, c=post_c, edgecolors="white", linewidths=0.5, zorder=3)
        occupied_boxes.extend(
            TrajectoryPlotter._marker_box(ax, p["post"], 4.0) for p in record["pairs"].values()
        )

    for record in records:
        id_str = str(record["tracked_id"]) if record["tracked_id"] is not None else "?"
        for kp_idx, pair in record["pairs"].items():
            label = f"{id_str}-{KP_NAMES[kp_idx]}"
            TrajectoryPlotter._draw_kp_label(ax, pair["post"], label, occupied_boxes, font_size=7.0)

    post_label = "recomputed from kp_cctv with the current G_projection" if recomputed else "as stored in the replay JSON"
    ax.text(
        0.02, 0.98,
        f"Frame: {frame_index}\nObjects: {len(records)}\nKeypoints: {n_keypoints}\n\n"
        "Legend:\n"
        f"  colored dot + label: kp_sat, height-corrected ({post_label})\n"
        "  gray dot (no label): apparent point at h=0 (before correction)\n"
        "  line: correction displacement (color = car part, matches the dot)",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if view_extent is not None:
        # tight_layout()/bbox_inches="tight" both crop to fit whatever text
        # this call happened to draw (title/legend length varies between a
        # baseline and a recomputed pass), which would leave two same-extent
        # plots at different pixel sizes -- fixed margins keep the axes rect
        # (and therefore the output canvas) identical across paired calls.
        fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.07)
        fig.savefig(out_path, dpi=dpi)
    else:
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path, n_keypoints, len(records)
