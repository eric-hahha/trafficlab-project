"""
Run h-aware 3D keypoint localization using pre-computed YOLO boxes as the
crop + track-ID source, and output a TrafficLab replay JSON.

This is a distinct cross-frame tracking method from geometric matching
(scripts/run_keypoints_openpifpaf.py --method geometric): instead of running YOLO
live and bridging it to PifPaf's own instances via bbox IoU, this script
takes a YOLO-produced replay JSON (car boxes + track_id, e.g. pipeline.py
output) as ground truth for *both* the crop region and the id — PifPaf's
only job is to re-detect keypoints inside each YOLO box and compute that
id's position/heading. No PifPaf<->YOLO matching happens; there is no
degenerate-box problem because there is nothing to match.

Frames where the YOLO file has no car that frame produce an empty object
list — there is nothing to crop, so nothing is emitted.

Usage:
    source /Users/eric/opt/anaconda3/bin/activate trafficlab && \\
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_keypoints_openpifpaf_yolo_crop.py \\
        --yolo-boxes-json /Users/eric/Desktop/test21-3sf.json.gz \\
        --g-proj location/test21/G_projection_test21.json
"""
import argparse
import gzip
import json
import math
import os
import re
import sys

import cv2
import numpy as np
import PIL.Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trafficlab.projection.g_projection import GProjection
from trafficlab.motion.keypoints_openpifpaf import (
    OpenPifPafKeypointsLocalizer,
    build_car_template,
    compute_car_dims_from_spec_csv,
    _FALLBACK_DIMS,
)
from trafficlab.io.replay_writer import ReplayWriter


def _infer_location_code(g_proj_path: str) -> str:
    """Extract location code from path like .../location/<code>/G_projection_*.json."""
    m = re.search(r'location[/\\]([^/\\]+)[/\\]', os.path.abspath(g_proj_path))
    if m:
        return m.group(1)
    return os.path.basename(os.path.dirname(os.path.abspath(g_proj_path)))


def _sat_floor_box(sat_coords, heading_deg: float, dims: dict, px_m: float):
    """Compute 4-corner vehicle footprint in sat pixels (same formula as pipeline.py)."""
    ang = math.radians(heading_deg)
    c, s = math.cos(ang), math.sin(ang)
    dx = (dims['length'] * px_m) / 2
    dy = (dims['width']  * px_m) / 2
    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    R = np.array([[c, -s], [s, c]])
    return (corners @ R.T + np.array(sat_coords)).tolist()


def _load_yolo_boxes(path: str, car_class: str):
    """Load a replay JSON and return (mp4_path, {frame_index: [objects]}), filtered to car_class."""
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        data = json.load(f)
    by_frame = {}
    for fr in data['frames']:
        cars = [o for o in fr['objects'] if o.get('class') == car_class]
        if cars:
            by_frame[fr['frame_index']] = cars
    return data.get('mp4_path'), by_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--yolo-boxes-json', required=True,
                        help='Pre-computed replay JSON (e.g. pipeline.py output) supplying '
                             'car bbox_2d + tracked_id per frame')
    parser.add_argument('--car-class', default='car',
                        help='class value in --yolo-boxes-json to treat as a car (default "car")')
    parser.add_argument('--video', default=None,
                        help='Input video path (default: mp4_path recorded in --yolo-boxes-json)')
    parser.add_argument('--g-proj', required=True, help='G_projection_*.json path')
    parser.add_argument('--out', default=None,
                        help='Output .json.gz path '
                             '(default: output/haware/<location>/<video_stem>_yolo_crop.json.gz)')
    parser.add_argument('--checkpoint', default='shufflenetv2k16-apollo-24')
    parser.add_argument('--spec-csv',   default=None)
    parser.add_argument('--body-type',  default='Sedan')
    parser.add_argument('--kp-conf',    type=float, default=0.2)
    parser.add_argument('--frames',     type=int,   default=-1, help='-1 = all')
    parser.add_argument('--pifpaf-threshold', type=float, default=0.01)
    parser.add_argument('--seed-threshold',   type=float, default=0.01)
    parser.add_argument('--crop-padding', type=float, default=0.5,
                        help='Fractional padding around each YOLO box before re-running PifPaf')
    args = parser.parse_args()

    # --- Pre-computed YOLO boxes ---
    file_mp4_path, boxes_by_frame = _load_yolo_boxes(args.yolo_boxes_json, args.car_class)
    video_path = args.video or file_mp4_path
    if args.video and file_mp4_path and os.path.normpath(args.video) != os.path.normpath(file_mp4_path):
        print(f'[haware-yolo-crop] WARNING: --video ({args.video}) does not match the '
              f'mp4_path recorded in --yolo-boxes-json ({file_mp4_path}). '
              f'Frame indices may not correspond to the right footage.')
    print(f'[haware-yolo-crop] Loaded {sum(len(v) for v in boxes_by_frame.values())} '
          f'"{args.car_class}" boxes across {len(boxes_by_frame)} frames from {args.yolo_boxes_json}')

    # --- G projection ---
    g_proj_dir = os.path.dirname(os.path.abspath(args.g_proj))
    with open(args.g_proj) as f:
        g_data = json.load(f)
    g_engine = GProjection(g_data, base_dir=g_proj_dir)
    location_code = _infer_location_code(args.g_proj)

    # --- Dimensions & template ---
    if args.spec_csv:
        dims = compute_car_dims_from_spec_csv(args.spec_csv, body_type=args.body_type)
    else:
        d = g_proj_dir
        dims_path = None
        for _ in range(5):
            candidate = os.path.join(d, 'prior_dimensions.json')
            if os.path.exists(candidate):
                dims_path = candidate
                break
            d = os.path.dirname(d)
        if dims_path:
            with open(dims_path) as f:
                pj = json.load(f)
            dims = pj.get('measurements_visdrone', {}).get('car', dict(_FALLBACK_DIMS))
            print(f'[haware-yolo-crop] Using prior_dimensions.json: {dims}')
        else:
            dims = dict(_FALLBACK_DIMS)
            print(f'[haware-yolo-crop] Using built-in fallback dims: {dims}')

    template = build_car_template(dims)
    localizer = OpenPifPafKeypointsLocalizer(g_engine, template, kp_conf=args.kp_conf)
    px_m = g_engine.px_per_m

    # --- OpenPifPaf ---
    import openpifpaf
    import openpifpaf.plugins.apollocar3d as _apc
    _apc.register()
    import argparse as _ap
    _dec_p = _ap.ArgumentParser()
    openpifpaf.decoder.cli(_dec_p)
    _dec_args = _dec_p.parse_args([])
    _dec_args.instance_threshold = args.pifpaf_threshold
    _dec_args.seed_threshold     = args.seed_threshold
    openpifpaf.decoder.configure(_dec_args)
    predictor = openpifpaf.Predictor(checkpoint=args.checkpoint)

    # --- Video metadata ---
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit = total if args.frames < 0 else min(total, args.frames)

    # --- Output path ---
    if args.out:
        out_path = args.out
    else:
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join('output', 'haware', location_code, f'{video_stem}_yolo_crop.json.gz')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    out_data = {
        'mp4_path':              video_path,
        'meta':                  {'resolution': [W, H], 'fps': fps},
        'location_code':         location_code,
        'mp4_frame_count':       total,
        'animation_frame_count': 0,
        'frames':                [],
    }

    n_boxes = n_ok = n_ambig = n_fail = n_empty_crop = 0
    print(f'Processing {limit}/{total} frames → {out_path}')

    for frame_idx in range(limit):
        ret, frame = cap.read()
        if not ret:
            break

        cars = boxes_by_frame.get(frame_idx)
        if not cars:
            out_data['frames'].append({'frame_index': frame_idx, 'objects': []})
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = PIL.Image.fromarray(rgb)

        frame_objects = []
        for j, car in enumerate(cars):
            n_boxes += 1
            bx1, by1, bx2, by2 = car['bbox_2d']
            bw, bh = bx2 - bx1, by2 - by1
            pad_x, pad_y = bw * args.crop_padding, bh * args.crop_padding
            x0 = max(0, int(bx1 - pad_x))
            y0 = max(0, int(by1 - pad_y))
            x1 = min(W, int(bx2 + pad_x))
            y1 = min(H, int(by2 + pad_y))

            kp_24 = np.zeros((24, 3), dtype=np.float32)
            if x1 > x0 and y1 > y0:
                try:
                    crop_preds, _, _ = predictor.pil_image(pil.crop((x0, y0, x1, y1)))
                except Exception:
                    crop_preds = []
                if crop_preds:
                    best = max(crop_preds,
                               key=lambda a: sum(1 for kp in a.data if kp[2] >= args.kp_conf))
                    kp_24 = best.data.copy()
                    kp_24[:, 0] += x0
                    kp_24[:, 1] += y0
                else:
                    n_empty_crop += 1
            else:
                n_empty_crop += 1

            result = localizer.localize(kp_24)

            sat_coords = list(result.sat_coords) if result.sat_coords is not None else None
            have_heading = result.heading is not None

            sfb = None
            if have_heading and sat_coords is not None:
                sfb = _sat_floor_box(sat_coords, result.heading, dims, px_m)

            if result.status == 'ok':
                n_ok += 1
            elif result.status == 'ambiguous_heading':
                n_ambig += 1
            else:
                n_fail += 1

            frame_objects.append({
                'id':               j,
                'tracked_id':       car.get('tracked_id'),
                'class':            'car',
                'confidence':       result.confidence,
                'bbox_2d':          list(car['bbox_2d']),
                'reference_point':  None,
                'sat_coords':       sat_coords,
                'have_heading':     have_heading,
                'have_measurements': True,
                'default_heading':  False,
                'heading':          result.heading,
                'speed_kmh':        0.0,
                'sat_floor_box':    sfb,
                'bbox_3d':          None,
                'n_keypoints':      result.n_keypoints,
                'status':           result.status,
                'kp_cctv':          kp_24.tolist(),
            })

        out_data['frames'].append({'frame_index': frame_idx, 'objects': frame_objects})

        if (frame_idx + 1) % 50 == 0:
            print(f'  frame {frame_idx + 1}/{limit} — '
                  f'ok={n_ok} ambig={n_ambig} fail={n_fail}')

    cap.release()
    n_frames_processed = len(out_data['frames'])
    out_data['animation_frame_count'] = (
        out_data['frames'][-1]['frame_index'] if out_data['frames'] else 0
    )

    ReplayWriter.write(out_path, out_data)

    print(f'\nDone: {n_frames_processed} frames, {n_boxes} YOLO car boxes')
    print(f'  ok={n_ok}  ambiguous={n_ambig}  failed={n_fail}')
    print(f'  crops with no PifPaf re-detection: {n_empty_crop}/{n_boxes}')
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
