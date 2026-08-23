"""
Run h-aware 3D keypoint localization and output a TrafficLab replay JSON.

Identical detection loop to eval_haware.py, but writes a .json.gz in the
standard TrafficLab replay format so results can be loaded in the GUI.

--method selects one of three mutually exclusive per-frame strategies:
    geometric     Bridge PifPaf detections to YOLO track IDs via bbox IoU.
                  Reads --yolo / --yolo-classes / --yolo-conf / --iou-threshold.
    crop          Crop each Pass-1 bbox and re-run PifPaf on the crop to recover
                  more confident keypoints. Reads --crop-redetect / --crop-padding.
    segmentation  Merge PifPaf fragments that belong to the same vehicle using
                  car-segmenter instance masks instead of a bbox-IoU heuristic —
                  fixes cases geometric's single-keypoint recovery can't (two
                  multi-keypoint fragments split from one vehicle). Uses
                  car-segmenter's own tracker for tracked_id, not YOLO.
                  Reads --seg-model / --seg-conf / --seg-device, or
                  --seg-masks-json to reuse a record_car_masks.py file instead
                  of running car-segmenter live (e.g. when the same video also
                  needs a --localizer reprojection pass and shouldn't pay for
                  YOLO11-seg inference twice).
Each strategy's own flags keep their existing meaning and defaults; --method
only decides which one actually runs this invocation.

Usage:
    source /opt/anaconda3/bin/activate trafficlab && \\
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_keypoints_openpifpaf.py \\
        --video location/test21/footage/test21-4.mp4 \\
        --g-proj location/test21/G_projection_test21.json \\
        --method geometric \\
        --spec-csv /tmp/autospec/engines.csv
"""
import argparse
import gzip
import json
import math
import os
import re
import sys
from collections import Counter

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
    KP_NAMES,
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


def _localize_and_assemble(kp_24, obj_id, tracked_id, bbox_2d, localizer, localizer_name, dims, px_m):
    """Run the localizer on kp_24 and assemble one frame_objects entry.

    Shared by every --method branch so the output schema stays identical
    regardless of how kp_24/tracked_id/bbox_2d were derived (raw PifPaf
    annotation, YOLO-bridged, or segmentation-merged).
    Returns (obj_dict, result.status).
    """
    if localizer_name == 'reprojection':
        result = localizer.localize_reprojection(kp_24)
    elif localizer_name == 'wheel_pair':
        result = localizer.localize_wheel_pair(kp_24)
    else:
        result = localizer.localize(kp_24)

    sat_coords = list(result.sat_coords) if result.sat_coords is not None else None
    have_heading = result.heading is not None

    sfb = None
    if have_heading and sat_coords is not None:
        sfb = _sat_floor_box(sat_coords, result.heading, dims, px_m)

    obj = {
        'id':               obj_id,
        'tracked_id':       tracked_id,
        'class':            'car',
        'confidence':       result.confidence,
        'bbox_2d':          list(bbox_2d) if bbox_2d is not None else None,
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
        'method':           result.method,
        # raw CCTV keypoints [x, y, conf] × 24 for overlay rendering
        'kp_cctv':          kp_24.tolist(),
        # per-keypoint sat-plane projection, same indexing as kp_cctv;
        # null where that keypoint wasn't confident enough to lift
        'kp_sat':           [list(result.p_sat[i]) if i in result.p_sat else None
                             for i in range(24)],
    }
    return obj, result.status


def _kp_bbox_xyxy(kp_24: np.ndarray, kp_conf: float):
    """Return (x1,y1,x2,y2) tight box around confident keypoints, or None."""
    pts = kp_24[(kp_24[:, 2] >= kp_conf) & ~((kp_24[:, 0] == 0) & (kp_24[:, 1] == 0))]
    if len(pts) == 0:
        return None
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def _single_confident_kp(kp_24: np.ndarray, kp_conf: float):
    """Return (kp_index, x, y) if exactly one keypoint clears kp_conf, else None.

    Counts directly rather than inferring from a degenerate (zero-area)
    _kp_bbox_xyxy box, since two different keypoint indices landing on the
    same pixel would also produce a degenerate box despite n=2.
    """
    mask = (kp_24[:, 2] >= kp_conf) & ~((kp_24[:, 0] == 0) & (kp_24[:, 1] == 0))
    idx = np.nonzero(mask)[0]
    if len(idx) != 1:
        return None
    i = int(idx[0])
    return i, float(kp_24[i, 0]), float(kp_24[i, 1])


def _point_in_box(x: float, y: float, box) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


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


def _load_yolo_boxes_json(path: str, car_class: str) -> tuple:
    """Load a pre-computed replay JSON (e.g. from a separate YOLO run) and return
    ({frame_index: (boxes_xyxy, track_ids)}, source_meta), filtered to car_class —
    an alternative YOLO box source for --method geometric IoU matching, in place
    of a live model. source_meta carries whatever provenance the replay file
    itself records (only meta.config_name today) so run_config can reflect the
    actual box source instead of the (unused) --yolo/--yolo-classes defaults."""
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        data = json.load(f)
    by_frame = {}
    for fr in data['frames']:
        cars = [o for o in fr['objects'] if o.get('class') == car_class and o.get('bbox_2d')]
        if cars:
            boxes = [tuple(o['bbox_2d']) for o in cars]
            tids  = [o.get('tracked_id') for o in cars]
            by_frame[fr['frame_index']] = (boxes, tids)
    source_meta = {'config_name': data.get('meta', {}).get('config_name')}
    return by_frame, source_meta


def _load_seg_masks_json(path: str) -> tuple:
    """Load a record_car_masks.py output file and return
    ({frame_index: records}, source_meta), each record {'tracker_id', 'bbox_xyxy',
    'confidence', 'polygon'} — an alternative car-segmenter source for --method
    segmentation, in place of running the model live. source_meta carries the
    seg_model/seg_conf/seg_device record_car_masks.py recorded at capture time,
    so run_config can reflect the model that actually produced these instances
    instead of the (unused) --seg-model/--seg-conf/--seg-device defaults."""
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        data = json.load(f)
    by_frame = {fr['frame_index']: fr['instances'] for fr in data['frames']}
    source_meta = {
        'seg_model':  data.get('seg_model'),
        'seg_conf':   data.get('seg_conf'),
        'seg_device': data.get('seg_device'),
    }
    return by_frame, source_meta


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
    parser.add_argument('--cad-template', default=None,
                        help='Path to a per-model keypoint template JSON from '
                             'scripts/build_cad_keypoint_template.py (e.g. '
                             'cad_models/nissan_juke_nismo/keypoint_template_nissan_juke_nismo.json). '
                             'When set, this exact 24-keypoint template is used directly instead of '
                             'build_car_template(dims) — --spec-csv/--body-type and '
                             'prior_dimensions.json are ignored.')
    parser.add_argument('--kp-conf',    type=float, default=0.2)
    parser.add_argument('--frames',          type=int,   default=-1,  help='-1 = all')
    parser.add_argument('--pifpaf-threshold', type=float, default=0.01,
                        help='PifPaf instance score threshold (default 0.01)')
    parser.add_argument('--seed-threshold',   type=float, default=0.01,
                        help='PifPaf CIF seed threshold (default 0.01)')
    parser.add_argument('--method', required=True, choices=['geometric', 'crop', 'segmentation'],
                        help='geometric = bridge to YOLO track IDs via bbox IoU '
                             '(reads --yolo*/--iou-threshold); '
                             'crop = crop-and-redetect for better keypoints '
                             '(reads --crop-redetect/--crop-padding); '
                             'segmentation = merge PifPaf fragments via car-segmenter '
                             'instance masks (reads --seg-*). Mutually exclusive: '
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
    parser.add_argument('--yolo-boxes-json', default=None,
                        help='Path to a pre-computed replay JSON (e.g. pipeline.py output) '
                             'supplying car bbox_2d + tracked_id per frame, used as the YOLO '
                             'box source for --method geometric IoU matching instead of '
                             'running a live YOLO model. When set, --yolo/--yolo-conf/'
                             '--yolo-classes are ignored.')
    parser.add_argument('--yolo-boxes-class', default='car',
                        help='class value in --yolo-boxes-json to treat as a car (default "car")')
    # segmentation matching: car-segmenter instance masks (only when --method segmentation)
    parser.add_argument('--seg-model', default='models/yolo11m-seg.pt',
                        help='Ultralytics *-seg checkpoint for car-segmenter (default models/yolo11m-seg.pt); '
                             'auto-downloaded into that path on first use if not present locally')
    parser.add_argument('--seg-conf', type=float, default=0.3,
                        help='car-segmenter detection confidence threshold (default 0.3)')
    parser.add_argument('--seg-device', default=None,
                        help='"cuda" / "mps" / "cpu", or leave unset to let ultralytics pick')
    parser.add_argument('--seg-masks-json', default=None,
                        help='Path to a record_car_masks.py output file supplying per-frame car '
                             'instances (tracker_id/bbox/mask polygon), used as the segmentation '
                             'source for --method segmentation instead of running car-segmenter '
                             'live. When set, --seg-model/--seg-conf/--seg-device are ignored and '
                             'car-segmenter is never loaded — use this so a video that needs both '
                             '--localizer procrustes and --localizer reprojection passes doesn\'t '
                             'pay for YOLO11-seg inference twice.')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='First frame index to process (default 0). Frames before this '
                             'are read and discarded, not seeked — CAP_PROP_POS_FRAMES '
                             'seeking has been unreliable on some test videos. --frames counts '
                             'from this point, not from frame 0.')
    parser.add_argument('--localizer', choices=['procrustes', 'reprojection', 'wheel_pair'], default='procrustes',
                        help='procrustes = closed-form 2D Procrustes on lifted sat coords '
                             '(default); reprojection = fit the 3D template directly against '
                             'PifPaf pixel positions via nonlinear least-squares '
                             '(OpenPifPafKeypointsLocalizer.localize_reprojection); '
                             'wheel_pair = position/heading from a single same-side front/rear '
                             'wheel pair only, ignoring every other keypoint '
                             '(OpenPifPafKeypointsLocalizer.localize_wheel_pair)')
    args = parser.parse_args()

    if args.method == 'geometric' and not args.yolo and not args.yolo_boxes_json:
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
    if args.cad_template:
        with open(args.cad_template) as f:
            cad_data = json.load(f)
        if cad_data.get('kp_names') != list(KP_NAMES):
            raise ValueError(
                f"--cad-template {args.cad_template!r} was built against a different "
                "KP_NAMES ordering than the current trafficlab.motion.keypoints_openpifpaf "
                "— regenerate it with scripts/build_cad_keypoint_template.py."
            )
        template = np.array(cad_data['template'], dtype=np.float64)
        if template.shape != (24, 3):
            raise ValueError(f"--cad-template {args.cad_template!r} has template shape "
                              f"{template.shape}, expected (24, 3)")
        cad_model = cad_data.get('source', {}).get('cad_model', '?')
        print(f'[haware] Using CAD-derived template: {args.cad_template} (cad_model={cad_model})')
        # _sat_floor_box (sat_floor_box output field) needs a length/width
        # footprint regardless of template source -- derive it from the CAD
        # template's own x/z extents (build_car_template's dims-driven
        # template uses the same half-length/half-width convention).
        dims = {
            'length': float(template[:, 2].max() - template[:, 2].min()),
            'width': float(template[:, 0].max() - template[:, 0].min()),
        }
    else:
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

    # --- YOLO box source (only loaded for --method geometric) ---
    # Either a live model (--yolo) or a pre-computed replay JSON (--yolo-boxes-json);
    # the latter takes priority when both are set.
    yolo_model = None
    yolo_classes = None
    yolo_boxes_by_frame = None
    yolo_source_meta = None
    if args.method == 'geometric' and args.yolo_boxes_json:
        yolo_boxes_by_frame, yolo_source_meta = _load_yolo_boxes_json(
            args.yolo_boxes_json, args.yolo_boxes_class)
        n_loaded = sum(len(v[0]) for v in yolo_boxes_by_frame.values())
        print(f'[haware] Loaded {n_loaded} "{args.yolo_boxes_class}" boxes across '
              f'{len(yolo_boxes_by_frame)} frames from {args.yolo_boxes_json} '
              f'(IoU threshold={args.iou_threshold})')
    elif args.method == 'geometric' and args.yolo:
        from ultralytics import YOLO as _YOLO
        yolo_model = _YOLO(args.yolo)
        if args.yolo_classes:
            yolo_classes = [int(c) for c in args.yolo_classes.split(',')]
        cls_str = str(yolo_classes) if yolo_classes else 'all'
        print(f'[haware] YOLO model loaded: {args.yolo} classes={cls_str} (IoU threshold={args.iou_threshold})')

    # --- car-segmenter mask source (only loaded for --method segmentation) ---
    # Either a live model (--seg-model, the default) or a pre-recorded
    # record_car_masks.py file (--seg-masks-json) — the latter skips loading
    # car-segmenter entirely, so a video that needs both --localizer
    # procrustes and --localizer reprojection passes doesn't run YOLO11-seg
    # twice. Both sources converge on the same per-frame record shape
    # ({'tracker_id','bbox_xyxy','confidence','polygon'}), but matching
    # keypoints against them uses two different implementations (dense mask
    # for the live source, polygon for the loaded one — see
    # assign_predictions_to_masks vs assign_predictions_to_polygons).
    mask_source = None
    seg_records_by_frame = None
    seg_source_meta = None
    if args.method == 'segmentation':
        from trafficlab.motion.segmentation_car import (
            assign_predictions_to_masks, assign_predictions_to_polygons, merge_keypoints_by_group,
        )
        if args.seg_masks_json:
            seg_records_by_frame, seg_source_meta = _load_seg_masks_json(args.seg_masks_json)
            n_loaded = sum(len(v) for v in seg_records_by_frame.values())
            print(f'[haware] Loaded {n_loaded} car-segmenter instances across '
                  f'{len(seg_records_by_frame)} frames from {args.seg_masks_json} '
                  f'(car-segmenter not loaded — --seg-model/--seg-conf/--seg-device ignored)')
        else:
            from trafficlab.motion.segmentation_car import CarMaskSource
            mask_source = CarMaskSource(model_path=args.seg_model, confidence=args.seg_conf,
                                         device=args.seg_device)
            print(f'[haware] car-segmenter loaded: {args.seg_model} conf={args.seg_conf} '
                  f'device={args.seg_device or "auto"}')

    # --- run_config.detector: which box/mask source this run actually used ---
    # Not simply echoing --yolo*/--seg-* CLI defaults: those weren't loaded at
    # all on the replay branches, so the real model identity there has to come
    # from the replay file's own recorded metadata instead.
    if args.method == 'geometric' and yolo_boxes_by_frame is not None:
        detector_info = {
            'source': 'replay', 'path': args.yolo_boxes_json,
            'iou_threshold': args.iou_threshold,
            'source_config_name': yolo_source_meta.get('config_name'),
        }
    elif args.method == 'geometric' and yolo_model is not None:
        detector_info = {
            'source': 'live', 'model': args.yolo, 'conf': args.yolo_conf,
            'iou_threshold': args.iou_threshold,
        }
    elif args.method == 'geometric':
        detector_info = {'source': None}
    elif args.method == 'crop':
        detector_info = {'crop_redetect': args.crop_redetect, 'crop_padding': args.crop_padding}
    elif seg_records_by_frame is not None:
        detector_info = {
            'source': 'replay', 'path': args.seg_masks_json,
            'recorded_model':  seg_source_meta.get('seg_model'),
            'recorded_conf':   seg_source_meta.get('seg_conf'),
            'recorded_device': seg_source_meta.get('seg_device'),
        }
    else:
        detector_info = {
            'source': 'live', 'model': args.seg_model, 'conf': args.seg_conf,
            'device': args.seg_device,
        }

    # --- Video metadata ---
    cap   = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start = max(0, args.start_frame)
    limit = total if args.frames < 0 else min(total, start + args.frames)

    # --- Output path ---
    if args.out:
        out_path = args.out
    else:
        video_stem = os.path.splitext(os.path.basename(args.video))[0]
        # wheel_pair writes to its own subfolder rather than output/haware/ so
        # it never collides with a procrustes/reprojection run on the same video.
        out_base = 'wheel_pair' if args.localizer == 'wheel_pair' else 'haware'
        out_path = os.path.join('output', out_base, location_code, f'{video_stem}.json.gz')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # --- Replay JSON skeleton ---
    out_data = {
        'mp4_path':             args.video,
        'meta':                 {'resolution': [W, H], 'fps': fps},
        'location_code':        location_code,
        'mp4_frame_count':      total,
        'animation_frame_count': 0,
        'run_config': {
            'checkpoint': args.checkpoint,
            'method':     args.method,
            'localizer':  args.localizer,
            'detector':   detector_info,
        },
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
    # --method segmentation stats
    n_seg_frames = 0        # frames where car-segmenter found >=1 instance
    n_seg_instances = 0     # total car-segmenter instances across all frames
    n_seg_no_pifpaf = 0     # segmented instances with no PifPaf keypoint inside them
    n_merge_events = 0      # groups where >=2 PifPaf fragments merged into one instance

    def _update_status_counters(status):
        nonlocal n_ok, n_ambig, n_fail
        if status == 'ok':
            n_ok += 1
        elif status == 'ambiguous_heading':
            n_ambig += 1
        else:
            n_fail += 1

    print(f'Processing frames {start}..{limit - 1} ({limit - start}/{total}) → {out_path}')

    for frame_idx in range(limit):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < start:
            continue

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
        kp_override = {}   # j (post-filter index) -> merged kp_24, set below for
                            # detections that absorbed a stray single-keypoint match
        if yolo_model is not None or yolo_boxes_by_frame is not None:
            if yolo_model is not None:
                yolo_res = yolo_model.track(frame, persist=True, conf=args.yolo_conf,
                                            classes=yolo_classes, tracker='bytetrack.yaml',
                                            verbose=False)
                yolo_boxes, yolo_tids = _extract_yolo(yolo_res)
            else:
                yolo_boxes, yolo_tids = yolo_boxes_by_frame.get(frame_idx, ([], []))
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

                # Single-keypoint recovery: a PifPaf detection with only one
                # confident keypoint has a zero-area bbox, so its IoU against
                # every YOLO box is 0 and it never clears iou_threshold above —
                # even when it's really just a fragment of a car YOLO already
                # tracks. If its one point falls inside exactly one (unpadded)
                # YOLO box, borrow that YOLO box's track: merge the point into
                # the detection already IoU-matched to that track this frame
                # (more keypoints -> better localizer fit) and drop this
                # fragment so it doesn't also appear as its own object; if no
                # such detection exists this frame, just tag the fragment with
                # the track id directly (it will still fail localization on
                # its own, but stays colour-consistent downstream).
                absorbed = set()
                for j, ann in enumerate(predictions):
                    if tracked_ids[j] is not None:
                        continue
                    single = _single_confident_kp(ann.data, args.kp_conf)
                    if single is None:
                        continue
                    kp_idx, x, y = single
                    containing = [k for k, yb in enumerate(yolo_boxes) if _point_in_box(x, y, yb)]
                    if len(containing) != 1:
                        continue
                    tid = yolo_tids[containing[0]]
                    if tid is None:
                        continue
                    main_j = next((jj for jj in range(len(predictions))
                                   if jj != j and tracked_ids[jj] == tid), None)
                    if main_j is None:
                        tracked_ids[j] = tid
                        continue
                    main_kp = kp_override.get(main_j, predictions[main_j].data.copy())
                    if main_kp[kp_idx, 2] < args.kp_conf or (
                            main_kp[kp_idx, 0] == 0 and main_kp[kp_idx, 1] == 0):
                        main_kp[kp_idx] = ann.data[kp_idx]
                        kp_override[main_j] = main_kp
                    absorbed.add(j)

                if absorbed:
                    keep = [j for j in range(len(predictions)) if j not in absorbed]
                    predictions  = [predictions[j] for j in keep]
                    tracked_ids  = [tracked_ids[j] for j in keep]
                    bbox_2d_list = [bbox_2d_list[j] for j in keep]
                    kp_override  = {new_j: kp_override[old_j]
                                    for new_j, old_j in enumerate(keep) if old_j in kp_override}

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
        if args.method == 'segmentation':
            # --- segmentation matching: merge PifPaf fragments via car-segmenter masks ---
            # Two sources, two matching implementations (see the loading block
            # above for why): --seg-masks-json loads per-frame records and
            # matches by polygon; otherwise the live model is queried and
            # matched by dense mask, exactly as before.
            if seg_records_by_frame is not None:
                records = seg_records_by_frame.get(frame_idx, [])
                polygons = [r['polygon'] for r in records]
                group_of = assign_predictions_to_polygons(predictions, polygons, args.kp_conf)
                n_instances_this_frame = len(records)
            else:
                seg_dets = mask_source.detect(frame)
                masks = seg_dets.mask if seg_dets.mask is not None else np.zeros((0, H, W), dtype=bool)
                group_of = assign_predictions_to_masks(predictions, masks, args.kp_conf)
                n_instances_this_frame = masks.shape[0]

            merged, leftover = merge_keypoints_by_group(predictions, group_of, args.kp_conf)

            n_seg_instances += n_instances_this_frame
            n_seg_no_pifpaf += n_instances_this_frame - len(merged)
            if n_instances_this_frame:
                n_seg_frames += 1
            group_counts = Counter(g for g in group_of if g is not None)
            n_merge_events += sum(1 for c in group_counts.values() if c >= 2)

            obj_id = 0
            for g, kp_24 in merged.items():
                n_det += 1
                if seg_records_by_frame is not None:
                    tid = records[g]['tracker_id']
                    bbox_2d = tuple(records[g]['bbox_xyxy'])
                else:
                    tid = None
                    if seg_dets.tracker_id is not None and seg_dets.tracker_id[g] >= 0:
                        tid = int(seg_dets.tracker_id[g])
                    bbox_2d = tuple(float(v) for v in seg_dets.xyxy[g])
                obj, status = _localize_and_assemble(
                    kp_24, obj_id, tid, bbox_2d, localizer, args.localizer, dims, px_m)
                _update_status_counters(status)
                frame_objects.append(obj)
                obj_id += 1

            # Fragments whose keypoints didn't fall inside any car-segmenter mask
            # (segmentation missed that vehicle, or the fragment is spurious) —
            # kept, not dropped, tagged the same way geometric tags an unmatched
            # PifPaf detection: its own per-frame index offset by 500.
            for j in leftover:
                n_det += 1
                kp_24 = predictions[j].data
                bbox_2d = _kp_bbox_xyxy(kp_24, args.kp_conf)
                obj, status = _localize_and_assemble(
                    kp_24, obj_id, j + 500, bbox_2d, localizer, args.localizer, dims, px_m)
                _update_status_counters(status)
                frame_objects.append(obj)
                obj_id += 1
        else:
            for j, ann in enumerate(predictions):
                n_det += 1

                # crop-and-redetect (--method crop only): crop around Pass-1 bbox and re-detect
                # kp_override (--method geometric only): use the merged keypoints when this
                # detection absorbed a stray single-keypoint match (see geometric matching above)
                kp_24 = kp_override.get(j, ann.data)
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

                bbox_2d = bbox_2d_list[j]
                obj, status = _localize_and_assemble(
                    kp_24, j, tracked_ids[j], bbox_2d, localizer, args.localizer, dims, px_m)
                _update_status_counters(status)
                frame_objects.append(obj)

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
    if yolo_model is not None or yolo_boxes_by_frame is not None:
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
    if args.method == 'segmentation':
        print(f'  segmentation frames: {n_seg_frames}/{n_frames_processed}  '
              f'instances: {n_seg_instances}')
        print(f'  merge events (>=2 PifPaf fragments → 1 instance): {n_merge_events}')
        print(f'  segmented-but-no-PifPaf-keypoints (not output): {n_seg_no_pifpaf}/{n_seg_instances}')
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
