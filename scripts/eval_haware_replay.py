"""
Run h-aware 3D keypoint localization and output a TrafficLab replay JSON.

Identical detection loop to eval_haware.py, but writes a .json.gz in the
standard TrafficLab replay format so results can be loaded in the GUI.

--method selects one of two mutually exclusive per-frame strategies:
    geometric  Bridge PifPaf detections to YOLO track IDs via bbox IoU.
               Reads --yolo / --yolo-classes / --yolo-conf / --iou-threshold.
    crop       Crop each Pass-1 bbox and re-run PifPaf on the crop to recover
               more confident keypoints. Reads --crop-redetect / --crop-padding.
Both strategies' own flags keep their existing meaning and defaults; --method
only decides which one actually runs this invocation.

Usage:
    source /Users/eric/opt/anaconda3/bin/activate trafficlab && \\
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/eval_haware_replay.py \\
        --video location/test21/footage/test21-4.mp4 \\
        --g-proj location/test21/G_projection_test21.json \\
        --method geometric \\
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


def _kp_bbox_xyxy(kp_24: np.ndarray, kp_conf: float):
    """Return (x1,y1,x2,y2) tight box around confident keypoints, or None."""
    pts = kp_24[(kp_24[:, 2] >= kp_conf) & ~((kp_24[:, 0] == 0) & (kp_24[:, 1] == 0))]
    if len(pts) == 0:
        return None
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def _match_by_bbox_iou(pifpaf_boxes, yolo_boxes, yolo_tids, iou_threshold=0.3):
    """Match each PifPaf bbox to the best-IoU YOLO bbox.

    Returns (tracked_ids, matched_boxes): parallel lists, one entry per PifPaf
    box. matched_boxes holds the assigned YOLO bbox (xyxy) or None when
    nothing cleared iou_threshold.
    """
    def _iou(a, b):
        if a is None or b is None:
            return 0.0
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    tracked_ids, matched_boxes = [], []
    for pb in pifpaf_boxes:
        best_tid, best_box, best_iou = None, None, iou_threshold
        for yb, yt in zip(yolo_boxes, yolo_tids):
            v = _iou(pb, yb)
            if v > best_iou:
                best_iou, best_tid, best_box = v, yt, yb
        tracked_ids.append(best_tid)
        matched_boxes.append(best_box)
    return tracked_ids, matched_boxes


def _extract_yolo(results) -> tuple:
    """Extract (boxes_xyxy, track_ids) from a YOLO result list."""
    boxes, tids = [], []
    for r in results:
        if r.boxes is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        ids  = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None] * len(xyxy)
        for box, tid in zip(xyxy, ids):
            boxes.append(tuple(float(v) for v in box))
            tids.append(int(tid) if tid is not None else None)
    return boxes, tids


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
    parser.add_argument('--pifpaf-threshold', type=float, default=0.01,
                        help='PifPaf instance score threshold (default 0.01)')
    parser.add_argument('--seed-threshold',   type=float, default=0.01,
                        help='PifPaf CIF seed threshold (default 0.01)')
    parser.add_argument('--method', required=True, choices=['geometric', 'crop'],
                        help='geometric = bridge to YOLO track IDs via bbox IoU '
                             '(reads --yolo*/--iou-threshold); '
                             'crop = crop-and-redetect for better keypoints '
                             '(reads --crop-redetect/--crop-padding). Mutually exclusive: '
                             'only the selected strategy runs.')
    parser.add_argument('--crop-redetect', action='store_true',
                        help='Crop each Pass-1 bbox with 50%% padding and re-run PifPaf '
                             '(only takes effect when --method crop)')
    parser.add_argument('--crop-padding', type=float, default=0.5,
                        help='Fractional padding around bbox for crop-and-redetect crop (default 0.5)')
    # geometric matching: YOLO track-ID matching (only when --method geometric)
    parser.add_argument('--yolo',          default='models/best.pt',
                        help='YOLO model path/name for track-ID matching (default: models/best.pt, '
                             'ByteTrack tracker); pass --yolo "" to disable and leave tracked_id=None')
    parser.add_argument('--yolo-conf',     type=float, default=0.25,
                        help='YOLO detection confidence threshold (default 0.25)')
    parser.add_argument('--yolo-classes',  default=None,
                        help='Comma-separated YOLO class indices to keep, e.g. "3" for car '
                             'in models/best.pt — class indices are model-specific, check '
                             '--yolo model.names (default: all classes)')
    parser.add_argument('--iou-threshold', type=float, default=0.3,
                        help='Minimum bbox IoU to accept a PifPaf↔YOLO match (default 0.3)')
    args = parser.parse_args()

    if args.method == 'geometric' and not args.yolo:
        print('[haware] --method geometric but --yolo is empty: no track-ID matching '
              'will happen, tracked_id will be null for every detection.')
    if args.method == 'crop' and not args.crop_redetect:
        print('[haware] --method crop but --crop-redetect was not passed: no re-detection '
              'will happen, this run is equivalent to plain Pass-1 PifPaf.')

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

    # --- YOLO tracker (only loaded for --method geometric) ---
    yolo_model = None
    yolo_classes = None
    if args.method == 'geometric' and args.yolo:
        from ultralytics import YOLO as _YOLO
        yolo_model = _YOLO(args.yolo)
        if args.yolo_classes:
            yolo_classes = [int(c) for c in args.yolo_classes.split(',')]
        cls_str = str(yolo_classes) if yolo_classes else 'all'
        print(f'[haware] YOLO model loaded: {args.yolo} classes={cls_str} (IoU threshold={args.iou_threshold})')

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
    n_matched = 0
    n_yolo_det = 0
    n_yolo_frames = 0
    n_frames_pifpaf_fewer = 0        # frames where PifPaf count < YOLO count
    yolo_tid_frames: dict = {}       # tid → frames YOLO detected it
    matched_tid_frames: dict = {}    # tid → frames PifPaf matched it
    no_pifpaf_tid_frames: dict = {}  # tid → YOLO saw it but PifPaf detected nothing
    no_match_tid_frames: dict = {}   # tid → PifPaf detected but IoU match failed
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

        # --- geometric matching (--method geometric only): YOLO tracking + IoU matching ---
        tracked_ids = [None] * len(predictions)
        bbox_2d_list = [None] * len(predictions)
        if yolo_model is not None:
            yolo_res = yolo_model.track(frame, persist=True, conf=args.yolo_conf,
                                        classes=yolo_classes, tracker='bytetrack.yaml',
                                        verbose=False)
            yolo_boxes, yolo_tids = _extract_yolo(yolo_res)
            n_yolo_det += len(yolo_boxes)
            if yolo_boxes:
                n_yolo_frames += 1
            for yt in yolo_tids:
                if yt is not None:
                    yolo_tid_frames[yt] = yolo_tid_frames.get(yt, 0) + 1
            if len(predictions) < len(yolo_boxes):
                n_frames_pifpaf_fewer += 1
            if predictions:
                pifpaf_boxes = [_kp_bbox_xyxy(ann.data, args.kp_conf) for ann in predictions]
                tracked_ids, matched_boxes = _match_by_bbox_iou(
                    pifpaf_boxes, yolo_boxes, yolo_tids, iou_threshold=args.iou_threshold)
                n_matched += sum(1 for t in tracked_ids if t is not None)
                matched_this_frame = set(t for t in tracked_ids if t is not None)
                for t in matched_this_frame:
                    matched_tid_frames[t] = matched_tid_frames.get(t, 0) + 1
                for yt in set(yt for yt in yolo_tids if yt is not None):
                    if yt not in matched_this_frame:
                        no_match_tid_frames[yt] = no_match_tid_frames.get(yt, 0) + 1

                # bbox_2d: the matched YOLO box when there is one; otherwise
                # fall back to the PifPaf-keypoint box (no YOLO box to borrow).
                bbox_2d_list = [mb if mb is not None else pb
                                for mb, pb in zip(matched_boxes, pifpaf_boxes)]

                # Detections that didn't clear the IoU threshold keep their own
                # PifPaf per-frame instance index as an id (offset by 500 to
                # stay clear of real YOLO track IDs), so they stay
                # colour-distinguishable downstream. This is NOT a real track:
                # the offset index has no meaning from one frame to the next.
                for j in range(len(tracked_ids)):
                    if tracked_ids[j] is None:
                        tracked_ids[j] = j + 500
            else:
                # True no-PifPaf frames: YOLO ran but PifPaf found nothing
                for yt in set(yt for yt in yolo_tids if yt is not None):
                    no_pifpaf_tid_frames[yt] = no_pifpaf_tid_frames.get(yt, 0) + 1

        frame_objects = []
        for j, ann in enumerate(predictions):
            n_det += 1

            # crop-and-redetect (--method crop only): crop around Pass-1 bbox and re-detect
            kp_24 = ann.data
            if args.method == 'crop' and args.crop_redetect:
                bx, by, bw, bh = ann.bbox()
                pad_x, pad_y = bw * args.crop_padding, bh * args.crop_padding
                x0 = max(0, int(bx - pad_x))
                y0 = max(0, int(by - pad_y))
                x1 = min(W, int(bx + bw + pad_x))
                y1 = min(H, int(by + bh + pad_y))
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
                'tracked_id':       tracked_ids[j],
                'class':            'car',
                'confidence':       result.confidence,
                'bbox_2d':          list(bbox_2d_list[j]) if bbox_2d_list[j] is not None else None,
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
                'kp_cctv':          kp_24.tolist(),
            })

        out_data['frames'].append({'frame_index': frame_idx, 'objects': frame_objects})

        if (frame_idx + 1) % 50 == 0:
            print(f'  frame {frame_idx + 1}/{limit} — '
                  f'ok={n_ok} ambig={n_ambig} fail={n_fail}')

    cap.release()
    # Not `limit`: that's the pre-computed cap (possibly from an unreliable
    # CAP_PROP_FRAME_COUNT), not how far the loop actually got before a failed
    # cap.read() broke it early. Read back the last frame actually recorded instead,
    # same pattern as pipeline.py's animation_frame_count (= the loop's actual
    # last index, not the pre-computed frames_to_process cap).
    n_frames_processed = len(out_data['frames'])
    out_data['animation_frame_count'] = (
        out_data['frames'][-1]['frame_index'] if out_data['frames'] else 0
    )

    ReplayWriter.write(out_path, out_data)

    print(f'\nDone: {n_frames_processed} frames, {n_det} detections')
    print(f'  ok={n_ok}  ambiguous={n_ambig}  failed={n_fail}')
    if yolo_model is not None:
        pct = 100 * n_matched / n_det if n_det > 0 else 0.0
        print(f'  YOLO frames:      {n_yolo_frames}/{n_frames_processed}  detections: {n_yolo_det}')
        print(f'  PifPaf < YOLO 幀: {n_frames_pifpaf_fewer}/{n_frames_processed}')
        print(f'  track-ID matched: {n_matched}/{n_det} ({pct:.1f}%)')
        if yolo_tid_frames:
            print(f'\n  {"ID":>4}  {"YOLO幀":>6}  {"配對幀":>6}  {"無PifPaf":>8}  {"IoU失敗":>7}  {"配對率":>6}')
            print(f'  {"----":>4}  {"------":>6}  {"------":>6}  {"--------":>8}  {"-------":>7}  {"------":>6}')
            for tid in sorted(yolo_tid_frames):
                yf = yolo_tid_frames[tid]
                mf = matched_tid_frames.get(tid, 0)
                np_ = no_pifpaf_tid_frames.get(tid, 0)
                nm  = no_match_tid_frames.get(tid, 0)
                ratio = 100 * mf / yf if yf > 0 else 0.0
                print(f'  {tid:>4}  {yf:>6}  {mf:>6}  {np_:>8}  {nm:>7}  {ratio:>5.1f}%')
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
