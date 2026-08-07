"""
Run car-segmenter (YOLO11-seg) over a whole video and record per-frame car
instances (tracker_id, bbox, confidence, mask polygon) to a file — nothing
else. No PifPaf, no G-projection, no reprojection: pure image-space
recording.

Purpose: run_keypoints_openpifpaf.py --method segmentation --seg-masks-json
can load this file instead of running car-segmenter live. When the same
video needs both --localizer procrustes and --localizer reprojection passes,
record once here and point both runs at the same file instead of paying for
YOLO11-seg inference twice.

Usage:
    source /Users/eric/opt/anaconda3/bin/activate trafficlab && \\
    python scripts/record_car_masks.py \\
        --video location/test21/footage/test21-4.mp4
"""
import argparse
import os
import re
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trafficlab.motion.segmentation_car import CarMaskSource
from trafficlab.io.replay_writer import ReplayWriter


def _infer_location_code(video_path: str) -> str:
    """Extract location code from a path like .../location/<code>/footage/*.mp4.
    Same regex as run_keypoints_openpifpaf.py's _infer_location_code, applied
    to the video path directly since this script has no G_projection file."""
    m = re.search(r'location[/\\]([^/\\]+)[/\\]', os.path.abspath(video_path))
    if m:
        return m.group(1)
    return os.path.splitext(os.path.basename(video_path))[0]


def main():
    parser = argparse.ArgumentParser(
        description='Record car-segmenter instances (mask polygon + tracker_id) per frame')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--out', default=None,
                        help='Output .json.gz path '
                             '(default: output/car_masks/<location>/seg-mask_<video_stem>.json.gz)')
    parser.add_argument('--seg-model', default='models/yolo11n-seg.pt',
                        help='Ultralytics *-seg checkpoint (default models/yolo11n-seg.pt); '
                             'auto-downloaded into that path on first use if not present locally')
    parser.add_argument('--seg-conf', type=float, default=0.3,
                        help='car-segmenter detection confidence threshold (default 0.3)')
    parser.add_argument('--seg-device', default=None,
                        help='"cuda" / "mps" / "cpu", or leave unset to let ultralytics pick')
    parser.add_argument('--frames', type=int, default=-1, help='-1 = all')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='First frame index to process (default 0). Frames before this '
                             'are read and discarded, not seeked — same rationale as '
                             'run_keypoints_openpifpaf.py.')
    args = parser.parse_args()

    location_code = _infer_location_code(args.video)

    if args.out:
        out_path = args.out
    else:
        video_stem = os.path.splitext(os.path.basename(args.video))[0]
        out_path = os.path.join('output', 'car_masks', location_code, f'seg-mask_{video_stem}.json.gz')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    mask_source = CarMaskSource(model_path=args.seg_model, confidence=args.seg_conf,
                                 device=args.seg_device)
    print(f'[car-masks] car-segmenter loaded: {args.seg_model} conf={args.seg_conf} '
          f'device={args.seg_device or "auto"}')

    cap   = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start = max(0, args.start_frame)
    limit = total if args.frames < 0 else min(total, start + args.frames)

    out_data = {
        'video_path': args.video,
        'seg_model':  args.seg_model,
        'seg_conf':   args.seg_conf,
        'seg_device': args.seg_device,
        'meta':       {'resolution': [W, H], 'fps': fps},
        'frame_count': total,
        'frames':     [],
    }

    print(f'Processing frames {start}..{limit - 1} ({limit - start}/{total}) → {out_path}')
    n_instances = 0
    for frame_idx in range(limit):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < start:
            continue

        records = mask_source.detect_records(frame)
        n_instances += len(records)
        out_data['frames'].append({'frame_index': frame_idx, 'instances': records})

        if (frame_idx + 1) % 50 == 0:
            print(f'  frame {frame_idx + 1}/{limit} — instances so far: {n_instances}')

    cap.release()
    n_frames_processed = len(out_data['frames'])

    ReplayWriter.write(out_path, out_data)
    print(f'\nDone: {n_frames_processed} frames, {n_instances} instances')
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
