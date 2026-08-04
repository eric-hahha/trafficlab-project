"""
Evaluate YOLO-bbox geometric heading estimator on CCTV footage.

No extra ML model needed — heading is estimated purely from bbox aspect ratio
and camera-to-vehicle geometry.

Usage:
    python scripts/archive/eval_bbox_heading.py \
        --video location/test21/footage/test21-3sf.mp4 \
        --config location/test21/G_projection_test21.json \
        --out /tmp/bbox_heading_eval \
        [--frames 20] [--conf 0.35] [--weights models/yolo11s-visdrone-v2-ft.pt]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trafficlab.projection.g_projection import GProjection
from bbox_heading import estimate_heading_from_bbox


VEHICLE_CLASSES = {'car', 'van', 'truck', 'bus', 'three_wheeler', 'motor', 'two_wheeler', 'bicycle',
                   'people', 'pedestrian'}  # excluded from heading arrows — kept for completeness
ARROW_LENGTH = 55
CONF_THRESHOLD_DISPLAY = 0.3  # only draw arrow if confidence >= this


def draw_arrow(img, cx, cy, heading_deg, length, color, thickness=2):
    rad = math.radians(heading_deg)
    ex = int(cx + math.cos(rad) * length)
    ey = int(cy - math.sin(rad) * length)  # y inverted in image space
    cv2.arrowedLine(img, (cx, cy), (ex, ey), color, thickness, tipLength=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',   required=True)
    parser.add_argument('--config',  required=True, help='G_projection_*.json path')
    parser.add_argument('--out',     default='/tmp/bbox_heading_eval')
    parser.add_argument('--frames',  type=int,   default=20)
    parser.add_argument('--conf',    type=float, default=0.35)
    parser.add_argument('--weights', default='models/yolo11s-visdrone-v2-ft.pt')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = json.load(f)

    base_dir = str(Path(args.config).parent)
    g_proj = GProjection(cfg, base_dir=base_dir)
    cam_pos = tuple(g_proj.cam_sat)

    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = model.names

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Video: {total} frames @ {fps:.1f} fps  {W}×{H_vid}')
    print(f'Camera sat pos: {cam_pos}')

    indices = [int(i * total / args.frames) for i in range(args.frames)]

    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        results = model(frame, conf=args.conf, iou=0.5, verbose=False)[0]
        vis = frame.copy()
        n_arrows = 0

        NON_VEHICLE = {'pedestrian', 'people'}
        for box in results.boxes:
            cls_name = names[int(box.cls[0])]
            if cls_name.lower() in NON_VEHICLE:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w_px = x2 - x1
            h_px = y2 - y1
            aspect = w_px / max(h_px, 1.0)

            # Project bbox to sat coords (use bottom-center, no parallax for simplicity)
            rect_xywh = (x1, y1, w_px, h_px)
            gc = g_proj.get_ground_contact_from_box(rect_xywh, h_meters=1.55)
            sat_coords = gc['sat_coords']

            heading_result = estimate_heading_from_bbox(
                [x1, y1, x2, y2], sat_coords, cam_pos, road_heading=None
            )

            # Draw bbox
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)

            # Label: class + aspect ratio
            label = f'{cls_name} ar={aspect:.1f}'
            cv2.putText(vis, label, (int(x1), max(int(y1) - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

            if heading_result is not None:
                heading_deg, confidence = heading_result
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Arrow colour: bright magenta if confident, grey if not
                if confidence >= CONF_THRESHOLD_DISPLAY:
                    color = (255, 0, 255)   # magenta
                    thickness = 2
                else:
                    color = (140, 140, 140)  # grey
                    thickness = 1

                draw_arrow(vis, cx, cy, heading_deg, ARROW_LENGTH, color, thickness)

                conf_label = f'{heading_deg:.0f}° c={confidence:.2f}'
                cv2.putText(vis, conf_label, (int(x1), min(int(y2) + 12, H_vid - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                n_arrows += 1

        out_path = out_dir / f'frame_{frame_idx:05d}.png'
        cv2.imwrite(str(out_path), vis)
        print(f'  frame {frame_idx:5d}: {len(results.boxes)} detections, '
              f'{n_arrows} arrows → {out_path.name}')

    cap.release()
    print(f'\nDone. Results in: {out_dir}')
    print('Arrow key: magenta = confident (>=0.30), grey = low confidence')


if __name__ == '__main__':
    main()
