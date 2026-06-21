"""
Evaluate CarFusion YOLOv8-Pose keypoints → satellite coordinates.

Runs the CarFusion YOLO-Pose model on a video, projects 14 keypoints to sat
coordinates via per-keypoint height priors and G-projection, fits vehicle
centre and heading via Procrustes SVD, then saves side-by-side composite
images (CCTV left, satellite right).

Usage:
    source /Users/eric/opt/anaconda3/bin/activate trafficlab && \\
    python scripts/eval_carfusion_sat.py \\
        --video location/test21/footage/test21-4.mp4 \\
        --weights /private/tmp/carfusion_weights/weights/last.pt \\
        --g-proj  location/test21/G_projection_test21.json \\
        --sat     location/test21/sat_test21.png \\
        --out     /private/tmp/carfusion_sat/ \\
        --start-frame 120
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trafficlab.projection.g_projection import GProjection
from trafficlab.motion.haware_localization import _FALLBACK_DIMS
from trafficlab.motion.carfusion_localization import (
    CarFusionLocalizer,
    build_carfusion_template,
    KP_NAMES,
)

# BGR colour palette
_C_WHEEL   = (0,   255,   0)    # green   – wheel keypoints
_C_LIGHT   = (0,   165, 255)    # orange  – light keypoints
_C_ROOF    = (255,   0,   0)    # blue    – roof keypoints
_C_EXHAUST = (128,   0, 128)    # purple
_C_CENTER  = (0,     0, 255)    # red
_C_BOX     = (200, 200,   0)    # cyan    – YOLO bbox
_C_SAT_DOT = (0,   255, 255)    # yellow  – localized centre on sat
_C_ARROW   = (255, 200,   0)    # yellow-blue – heading arrow
_C_KP_SAT  = (180, 180, 180)    # grey    – individual kp projections on sat


def _kp_color(name: str) -> tuple:
    if name.startswith('wheel'):   return _C_WHEEL
    if name.startswith('light'):   return _C_LIGHT
    if name.startswith('roof'):    return _C_ROOF
    if name.startswith('exhaust'): return _C_EXHAUST
    return _C_CENTER


def _draw_arrow(img, cx, cy, heading_deg, length=30, color=_C_ARROW, thickness=2):
    rad = math.radians(heading_deg)
    ex  = int(cx + math.cos(rad) * length)
    ey  = int(cy - math.sin(rad) * length)   # sat: North = -y on screen
    cv2.arrowedLine(img, (int(cx), int(cy)), (ex, ey), color, thickness, tipLength=0.35)


def _load_dims(g_proj_dir: str) -> dict:
    d = g_proj_dir
    for _ in range(5):
        cand = os.path.join(d, 'prior_dimensions.json')
        if os.path.exists(cand):
            with open(cand) as f:
                pj = json.load(f)
            dims = pj.get('measurements_visdrone', {}).get('car', dict(_FALLBACK_DIMS))
            print(f'[carfusion] dims from prior_dimensions.json: {dims}')
            return dims
        d = os.path.dirname(d)
    print(f'[carfusion] using built-in fallback dims')
    return dict(_FALLBACK_DIMS)


def process_frame(frame, sat_img, g_engine, yolo_model, localizer,
                  conf_thresh: float, kp_conf: float):
    H_sat, W_sat = sat_img.shape[:2]
    cctv_vis = frame.copy()
    sat_vis  = sat_img.copy()

    results = yolo_model(frame, conf=conf_thresh, verbose=False)
    result  = results[0]

    n_vehicles = 0
    n_ok = n_amb = n_fail = 0
    sat_coords_out = []
    det_records    = []

    if result.keypoints is not None and result.boxes is not None and len(result.boxes) > 0:
        for i in range(len(result.boxes)):
            n_vehicles += 1
            box_conf = float(result.boxes.conf[i])
            x1, y1, x2, y2 = [int(v) for v in result.boxes.xyxy[i]]

            cv2.rectangle(cctv_vis, (x1, y1), (x2, y2), _C_BOX, 1)

            kp_xy   = result.keypoints.xy[i].cpu().numpy()    # (14, 2)
            kp_conf_arr = result.keypoints.conf[i].cpu().numpy()  # (14,)
            kp_14   = np.column_stack([kp_xy, kp_conf_arr])   # (14, 3)

            # Draw keypoints on CCTV
            for k in range(14):
                if kp_conf_arr[k] < kp_conf:
                    continue
                kx, ky = int(kp_xy[k, 0]), int(kp_xy[k, 1])
                if kx == 0 and ky == 0:
                    continue
                cv2.circle(cctv_vis, (kx, ky), 4, _kp_color(KP_NAMES[k]), -1)

            # Localize
            res = localizer.localize(kp_14)

            if res.status == 'ok':
                n_ok += 1
            elif res.status == 'ambiguous_heading':
                n_amb += 1
            else:
                n_fail += 1

            # Status label above bbox
            label = f"{res.status[:3]} kp={res.n_keypoints} c={res.confidence:.2f}"
            cv2.putText(cctv_vis, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, _C_BOX, 1)

            # SAT view: individual kp projections
            for kp_idx, (sx, sy) in res.kp_sat.items():
                wx = int(np.clip(sx, 0, W_sat - 1))
                wy = int(np.clip(sy, 0, H_sat - 1))
                cv2.circle(sat_vis, (wx, wy), 2, _kp_color(KP_NAMES[kp_idx]), -1)

            # SAT view: localized centre + heading
            if res.sat_coords is not None:
                sx = int(np.clip(res.sat_coords[0], 0, W_sat - 1))
                sy = int(np.clip(res.sat_coords[1], 0, H_sat - 1))
                cv2.circle(sat_vis, (sx, sy), 6, _C_SAT_DOT, -1)
                sat_coords_out.append((res.sat_coords, res.heading))
                if res.heading is not None:
                    _draw_arrow(sat_vis, sx, sy, res.heading)

            det_records.append({
                'class':       'Car',
                'sat_coords':  list(res.sat_coords) if res.sat_coords is not None else None,
                'sat_center':  list(res.sat_coords) if res.sat_coords is not None else None,
                'heading':     res.heading,
                'confidence':  round(res.confidence, 4),
                'n_keypoints': res.n_keypoints,
                'status':      res.status,
                'bbox_cctv':   [x1, y1, x2, y2],
                'bbox_conf':   round(box_conf, 4),
                'kp_cctv':     kp_14.tolist(),
            })

    return cctv_vis, sat_vis, {
        'n_vehicles': n_vehicles, 'n_ok': n_ok, 'n_amb': n_amb, 'n_fail': n_fail
    }, sat_coords_out, det_records


def _save_scatter(sat_img, all_detections: list, out_path: str):
    """all_detections: list of (sat_coords, heading_deg_or_None)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    H, W = sat_img.shape[:2]
    arrow_len = max(W, H) * 0.018

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB))
    for coords, heading in all_detections:
        x, y = coords
        ax.scatter(x, y, s=8, c='yellow', linewidths=0.4,
                   edgecolors='black', alpha=0.8, zorder=3)
        if heading is not None:
            dx =  math.cos(math.radians(heading)) * arrow_len
            dy = -math.sin(math.radians(heading)) * arrow_len
            ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color='red',
                                        lw=0.8, mutation_scale=6),
                        zorder=4)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f'Scatter plot: {out_path}  ({len(all_detections)} points)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video',       default='location/test21/footage/test21-4.mp4')
    ap.add_argument('--weights',     default='/private/tmp/carfusion_weights/weights/last.pt')
    ap.add_argument('--g-proj',      default='location/test21/G_projection_test21.json')
    ap.add_argument('--sat',         default='location/test21/sat_test21.png')
    ap.add_argument('--out',         default='/private/tmp/carfusion_sat/')
    ap.add_argument('--start-frame', type=int, default=0,
                    help='First frame index to process (default 0)')
    ap.add_argument('--frames',      type=int, default=-1,
                    help='Max frames to process after start-frame (-1 = all)')
    ap.add_argument('--conf',        type=float, default=0.25)
    ap.add_argument('--kp-conf',     type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.g_proj) as f:
        g_data = json.load(f)
    g_proj_dir = os.path.dirname(os.path.abspath(args.g_proj))
    g_engine   = GProjection(g_data, base_dir=g_proj_dir)

    dims = _load_dims(g_proj_dir)
    template, kp_heights = build_carfusion_template(dims)
    localizer = CarFusionLocalizer(g_engine, template, kp_heights, kp_conf=args.kp_conf)

    from ultralytics import YOLO
    print(f'Loading weights: {args.weights}')
    yolo_model = YOLO(args.weights)

    sat_img = cv2.imread(args.sat)
    if sat_img is None:
        print(f'ERROR: cannot read sat image: {args.sat}')
        sys.exit(1)

    cap   = cv2.VideoCapture(args.video)
    fps   = cap.get(cv2.CAP_PROP_FPS)

    # Count actual readable frames (CAP_PROP_FRAME_COUNT is unreliable for some codecs)
    actual_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Video: ~{actual_total} frames @ {fps:.1f} fps')
    print(f'Processing from frame {args.start_frame} to end')

    # Skip to start frame by sequential read (seek via cap.set is unreliable)
    for _ in range(args.start_frame):
        ret, _ = cap.read()
        if not ret:
            print(f'WARNING: video ended before reaching start-frame {args.start_frame}')
            cap.release()
            sys.exit(1)

    agg = {'n_vehicles': 0, 'n_ok': 0, 'n_amb': 0, 'n_fail': 0, 'frames_with_det': 0}
    all_detections = []
    replay_frames  = []
    det_id         = 1
    frame_idx = args.start_frame
    processed = 0
    limit = args.frames if args.frames > 0 else 99999

    while processed < limit:
        ret, frame = cap.read()
        if not ret:
            break

        cctv_vis, sat_vis, stats, sat_coords, det_records = process_frame(
            frame, sat_img.copy(), g_engine, yolo_model, localizer,
            conf_thresh=args.conf, kp_conf=args.kp_conf,
        )
        all_detections.extend(sat_coords)

        for rec in det_records:
            rec['tracked_id'] = det_id
            det_id += 1
        replay_frames.append({'frame_index': frame_idx, 'objects': det_records})

        # Overlay frame index
        cv2.putText(cctv_vis, f'frame {frame_idx}', (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Side-by-side composite
        h_cctv = cctv_vis.shape[0]
        scale  = h_cctv / sat_vis.shape[0]
        sat_resized = cv2.resize(sat_vis, (0, 0), fx=scale, fy=scale)
        composite = np.hstack([cctv_vis, sat_resized])

        out_path = os.path.join(args.out, f'frame_{frame_idx:05d}.jpg')
        cv2.imwrite(out_path, composite)

        for k in stats:
            agg[k] = agg.get(k, 0) + stats[k]
        if stats['n_vehicles'] > 0:
            agg['frames_with_det'] += 1

        if processed % 10 == 0 or stats['n_vehicles'] > 0:
            print(f'  frame {frame_idx:5d}: veh={stats["n_vehicles"]} '
                  f'ok={stats["n_ok"]} amb={stats["n_amb"]} fail={stats["n_fail"]}')

        frame_idx += 1
        processed += 1

    cap.release()

    print()
    print('=' * 50)
    print(f'Frames processed   : {processed}')
    print(f'Frames with detect : {agg["frames_with_det"]}')
    print(f'Total vehicles     : {agg["n_vehicles"]}')
    print(f'Localizations ok   : {agg["n_ok"]}')
    print(f'Ambiguous heading  : {agg["n_amb"]}')
    print(f'Failed             : {agg["n_fail"]}')
    print(f'Output             : {args.out}')
    print('=' * 50)

    scatter_path = os.path.join(args.out, 'scatter.png')
    _save_scatter(sat_img, all_detections, scatter_path)

    json_path = os.path.join(args.out, 'detections.json')
    with open(json_path, 'w') as f:
        json.dump({'frames': replay_frames}, f)
    print(f'Detections JSON: {json_path}')


if __name__ == '__main__':
    main()
