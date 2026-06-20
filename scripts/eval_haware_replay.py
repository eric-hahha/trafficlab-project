"""
Run h-aware 3D keypoint localization and output a TrafficLab replay JSON.

Identical detection loop to eval_haware.py, but writes a .json.gz in the
standard TrafficLab replay format so results can be loaded in the GUI.

Usage:
    source /Users/eric/opt/anaconda3/bin/activate trafficlab && \\
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/eval_haware_replay.py \\
        --video location/test21/footage/test21-4.mp4 \\
        --g-proj location/test21/G_projection_test21.json \\
        --spec-csv /tmp/autospec/engines.csv
"""
import argparse
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
from trafficlab.motion.haware_localization import (
    HawareLocalizer,
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
    # fallback: parent directory name
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


def main():
    parser = argparse.ArgumentParser(
        description='h-aware localization → TrafficLab replay JSON')
    parser.add_argument('--video',      required=True,  help='Input video path')
    parser.add_argument('--g-proj',     required=True,  help='G_projection_*.json path')
    parser.add_argument('--out',        default=None,
                        help='Output .json.gz path '
                             '(default: output/haware/<location>/<video_stem>.json.gz)')
    parser.add_argument('--checkpoint', default='shufflenetv2k16-apollo-24')
    parser.add_argument('--spec-csv',   default=None,
                        help='engines.csv from ilyasozkurt/automobile-models-and-specs')
    parser.add_argument('--body-type',  default='Sedan')
    parser.add_argument('--kp-conf',    type=float, default=0.2)
    parser.add_argument('--frames',          type=int,   default=-1,  help='-1 = all')
    parser.add_argument('--pifpaf-threshold', type=float, default=0.2,
                        help='PifPaf instance score threshold (default 0.2)')
    parser.add_argument('--seed-threshold',   type=float, default=0.2,
                        help='PifPaf CIF seed threshold (default 0.2)')
    args = parser.parse_args()

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
            print(f'[haware] Using prior_dimensions.json: {dims}')
        else:
            dims = dict(_FALLBACK_DIMS)
            print(f'[haware] Using built-in fallback dims: {dims}')

    template = build_car_template(dims)
    localizer = HawareLocalizer(g_engine, template, kp_conf=args.kp_conf)
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
    cap   = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit = total if args.frames < 0 else min(total, args.frames)

    # --- Output path ---
    if args.out:
        out_path = args.out
    else:
        video_stem = os.path.splitext(os.path.basename(args.video))[0]
        out_path = os.path.join('output', 'haware', location_code, f'{video_stem}.json.gz')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # --- Replay JSON skeleton ---
    out_data = {
        'mp4_path':             args.video,
        'meta':                 {'resolution': [W, H], 'fps': fps},
        'location_code':        location_code,
        'mp4_frame_count':      total,
        'animation_frame_count': 0,
        'frames':               [],
    }

    # --- Frame loop ---
    n_det = n_ok = n_ambig = n_fail = 0
    print(f'Processing {limit}/{total} frames → {out_path}')

    for frame_idx in range(limit):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = PIL.Image.fromarray(rgb)
        try:
            predictions, _, _ = predictor.pil_image(pil)
        except Exception as e:
            print(f'  frame {frame_idx}: predictor error — {e}')
            out_data['frames'].append({'frame_index': frame_idx, 'objects': []})
            continue

        frame_objects = []
        for j, ann in enumerate(predictions):
            n_det += 1
            result = localizer.localize(ann.data)

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
                'tracked_id':       None,
                'class':            'car',
                'confidence':       result.confidence,
                'bbox_2d':          None,
                'reference_point':  None,
                'sat_coords':       sat_coords,
                'have_heading':     have_heading,
                'have_measurements': True,
                'default_heading':  False,
                'heading':          result.heading,
                'speed_kmh':        0.0,
                'sat_floor_box':    sfb,
                'bbox_3d':          None,
                # h-aware diagnostic fields
                'n_keypoints':      result.n_keypoints,
                'status':           result.status,
                # raw CCTV keypoints [x, y, conf] × 24 for overlay rendering
                'kp_cctv':          ann.data.tolist(),
            })

        out_data['frames'].append({'frame_index': frame_idx, 'objects': frame_objects})

        if (frame_idx + 1) % 50 == 0:
            print(f'  frame {frame_idx + 1}/{limit} — '
                  f'ok={n_ok} ambig={n_ambig} fail={n_fail}')

    cap.release()
    out_data['animation_frame_count'] = limit

    ReplayWriter.write(out_path, out_data)

    print(f'\nDone: {limit} frames, {n_det} detections')
    print(f'  ok={n_ok}  ambiguous={n_ambig}  failed={n_fail}')
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
