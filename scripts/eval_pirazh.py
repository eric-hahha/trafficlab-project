"""
Evaluate Pirazh keypoint detection on CCTV footage.

Usage:
    python scripts/eval_pirazh.py \\
        --checkpoint checkpoints/pirazh_stage2.pth.tar \\
        --video location/test1/footage/test1_8_100_s.mp4 \\
        --yolo models/yolov8n.pt \\
        --out /private/tmp/kp_eval \\
        [--frames 30] [--conf 0.3] [--device cpu]

Each processed frame is saved as a PNG showing:
- YOLO bounding boxes
- 20 keypoints overlaid on each crop (colour-coded)
- Orientation label + heading angle
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Resolve project root so imports work regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


KP_COLORS = [
    (0, 0, 255),    # 0  left-front wheel        — red
    (0, 0, 200),    # 1  left-back wheel
    (0, 255, 0),    # 2  right-front wheel        — green
    (0, 200, 0),    # 3  right-back wheel
    (255, 128, 0),  # 4  right fog lamp
    (255, 180, 0),  # 5  left fog lamp
    (255, 255, 0),  # 6  right headlight          — yellow
    (200, 255, 0),  # 7  left headlight
    (0, 255, 255),  # 8  front auto logo
    (0, 200, 255),  # 9  front license plate
    (255, 0, 255),  # 10 left rear-view mirror
    (200, 0, 255),  # 11 right rear-view mirror
    (255, 128, 128),# 12 right-front roof corner  — pink
    (255, 160, 160),# 13 left-front roof corner
    (128, 128, 255),# 14 left-back roof corner    — blue
    (160, 160, 255),# 15 right-back roof corner
    (0, 128, 255),  # 16 left rear lamp
    (0, 100, 255),  # 17 right rear lamp
    (128, 255, 128),# 18 rear auto logo
    (100, 255, 100),# 19 rear license plate
]


def draw_keypoints_on_crop(crop_bgr: np.ndarray, keypoints: np.ndarray, radius: int = 4) -> np.ndarray:
    vis = crop_bgr.copy()
    for i, (x, y) in enumerate(keypoints):
        if x < 1 and y < 1:
            continue
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(vis, (cx, cy), radius, KP_COLORS[i], -1)
        cv2.circle(vis, (cx, cy), radius + 1, (0, 0, 0), 1)
    return vis


def draw_heading_arrow(img: np.ndarray, cx: int, cy: int, heading_deg: float, length: int = 40) -> None:
    rad = np.radians(heading_deg)
    ex = int(cx + np.cos(rad) * length)
    ey = int(cy + np.sin(rad) * length)
    cv2.arrowedLine(img, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.3)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Pirazh keypoints on CCTV footage')
    parser.add_argument('--checkpoint', required=True, help='Path to Pirazh stage2 checkpoint (.pth.tar)')
    parser.add_argument('--video',      required=True, help='Path to CCTV mp4 footage')
    parser.add_argument('--yolo',       required=True, help='Path to YOLO weights (.pt)')
    parser.add_argument('--out',        default='/private/tmp/kp_eval', help='Output directory for result images')
    parser.add_argument('--frames',     type=int, default=20,  help='Number of frames to sample')
    parser.add_argument('--conf',       type=float, default=0.3, help='YOLO confidence threshold')
    parser.add_argument('--device',     default='cpu', help='Torch device (cpu / mps / cuda:0)')
    parser.add_argument('--classes',    default=None, help='Comma-separated class names to keep, e.g. car,van,truck')
    args = parser.parse_args()
    args.class_filter = set(args.classes.split(',')) if args.classes else None

    from ultralytics import YOLO
    from trafficlab.keypoint.inference import PirazhDetector, heading_from_keypoints, KP_LABELS

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading YOLO: {args.yolo}')
    yolo = YOLO(args.yolo)

    print(f'Loading Pirazh checkpoint: {args.checkpoint}')
    detector = PirazhDetector(args.checkpoint, device=args.device)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Video: {total} frames @ {fps:.1f} fps, {w}×{h}')

    # Sample evenly spaced frames
    sample_indices = [int(i * total / args.frames) for i in range(args.frames)]

    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        results = yolo(frame, conf=args.conf, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            continue

        vis_frame = frame.copy()
        boxes = results.boxes.xyxy.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy()

        for j, box in enumerate(boxes):
            cls_name = results.names[int(cls_ids[j])]
            if args.class_filter and cls_name not in args.class_filter:
                continue
            x1, y1, x2, y2 = map(int, box)
            # Clamp
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w - 1, x2); y2 = min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            pred = detector.predict(crop)
            kp   = pred['keypoints']
            heading = heading_from_keypoints(kp)

            # Draw bbox on full frame
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"{cls_name} | {pred['orient_label']} ({pred['orient_conf']:.0%})"
            if heading is not None:
                label += f' | hdg={heading:.0f}°'
            cv2.putText(vis_frame, label, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

            # Draw heading arrow from bbox center
            if heading is not None:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                draw_heading_arrow(vis_frame, cx, cy, heading)

            # Draw keypoints on crop, paste back
            vis_crop = draw_keypoints_on_crop(crop, kp, radius=max(3, min(crop.shape[:2]) // 20))
            vis_frame[y1:y2, x1:x2] = vis_crop

        out_path = out_dir / f'frame_{frame_idx:05d}.png'
        cv2.imwrite(str(out_path), vis_frame)
        print(f'  frame {frame_idx:5d}: {len(boxes)} detections → {out_path.name}')

    cap.release()
    print(f'\nDone. Results in: {out_dir}')


if __name__ == '__main__':
    main()
