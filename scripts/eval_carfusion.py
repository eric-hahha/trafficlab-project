"""
Evaluate CarFusion YOLOv8-Pose weights on TrafficLab footage.

Runs the Habib0905/Vehicle-Pose-Estimation model and produces:
  - Annotated output video (or sampled frames as PNG)
  - Per-frame stats: n_vehicles, avg_keypoints_per_vehicle, avg_confidence

Usage:
    python scripts/eval_carfusion.py \
        --video location/test21/footage/test21-3sf.mp4 \
        --weights models/carfusion_last.pt \
        --frames 40 \
        --out /private/tmp/carfusion_eval/
"""
from __future__ import annotations

import argparse
import os
import sys
import statistics
from pathlib import Path

import cv2
import numpy as np

# 14 keypoint names from CarFusion / Habib0905 repo
KP_NAMES = [
    'wheel_fl', 'wheel_fr', 'wheel_rl', 'wheel_rr',   # 0-3
    'light_fl', 'light_fr', 'light_rl', 'light_rr',   # 4-7
    'roof_fl',  'roof_fr',  'roof_rl',  'roof_rr',    # 8-11
    'exhaust',  'center',                               # 12-13
]

KP_COLORS = {
    'wheel': (0, 255, 0),    # green
    'light': (0, 165, 255),  # orange
    'roof':  (255, 0, 0),    # blue
    'exhaust': (128, 0, 128),
    'center': (0, 0, 255),
}


def _kp_color(name: str) -> tuple:
    for prefix, color in KP_COLORS.items():
        if name.startswith(prefix):
            return color
    return (200, 200, 200)


def run_eval(video_path: str, weights_path: str, max_frames: int, out_dir: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading weights: {weights_path}")
    model = YOLO(weights_path)
    print(f"Model loaded. Task: {model.task}, names: {model.names}")
    print()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = cap.get(cv2.CAP_PROP_FPS)
    w           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames > 0:
        step = max(1, total_frames // max_frames)
    else:
        step = 1

    frame_stats = []   # list of dicts per processed frame
    saved_frames = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames > 0 and frame_idx % step != 0:
            frame_idx += 1
            continue

        results = model(frame, verbose=False)
        result  = results[0]

        n_vehicles = 0
        kp_counts  = []
        confs      = []

        vis = frame.copy()

        if result.keypoints is not None and len(result.boxes) > 0:
            boxes = result.boxes
            kps   = result.keypoints  # shape: (N, 14, 3)

            for i in range(len(boxes)):
                n_vehicles += 1
                box_conf = float(boxes.conf[i])
                confs.append(box_conf)

                # Draw bounding box
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 0), 1)
                cv2.putText(vis, f"{box_conf:.2f}", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

                # Draw keypoints
                kp_xy   = kps.xy[i].cpu().numpy()    # (14, 2)
                kp_conf = kps.conf[i].cpu().numpy()  # (14,)

                visible = 0
                for k in range(min(14, len(KP_NAMES))):
                    kconf = float(kp_conf[k])
                    if kconf < 0.2:
                        continue
                    kx, ky = int(kp_xy[k, 0]), int(kp_xy[k, 1])
                    if kx == 0 and ky == 0:
                        continue
                    visible += 1
                    color = _kp_color(KP_NAMES[k])
                    cv2.circle(vis, (kx, ky), 4, color, -1)
                    # Label every other frame to avoid clutter
                    if saved_frames < 3:
                        cv2.putText(vis, KP_NAMES[k][:5], (kx + 4, ky),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

                kp_counts.append(visible)

        frame_stats.append({
            'frame': frame_idx,
            'n_vehicles': n_vehicles,
            'avg_kp': statistics.mean(kp_counts) if kp_counts else 0,
            'avg_conf': statistics.mean(confs) if confs else 0,
        })

        # Save annotated frame
        if saved_frames < 10:
            out_path = out_dir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), vis)
            saved_frames += 1

        frame_idx += 1

        processed = len(frame_stats)
        if processed % 10 == 0:
            print(f"  frame {frame_idx}/{total_frames}: "
                  f"{n_vehicles} vehicles, "
                  f"avg {statistics.mean(kp_counts):.1f} kp/veh" if kp_counts
                  else f"  frame {frame_idx}/{total_frames}: 0 vehicles")

        if max_frames > 0 and processed >= max_frames:
            break

    cap.release()

    # --- Summary ---
    print()
    print("=" * 50)
    print(f"Video:   {video_path}")
    print(f"Weights: {weights_path}")
    print(f"Frames processed: {len(frame_stats)}")
    print()

    total_veh = sum(s['n_vehicles'] for s in frame_stats)
    frames_with_det = sum(1 for s in frame_stats if s['n_vehicles'] > 0)
    all_kps = [s['avg_kp'] for s in frame_stats if s['avg_kp'] > 0]

    print(f"Frames with detections : {frames_with_det} / {len(frame_stats)} "
          f"({100*frames_with_det/len(frame_stats):.1f}%)")
    print(f"Total vehicle instances: {total_veh}")
    print(f"Avg vehicles per frame : {total_veh / len(frame_stats):.1f}")
    print(f"Avg keypoints/vehicle  : {statistics.mean(all_kps):.1f}" if all_kps else
          "Avg keypoints/vehicle  : N/A")
    print()
    print(f"Annotated frames saved to: {out_dir}")
    print("=" * 50)


def main():
    ap = argparse.ArgumentParser(description="Eval CarFusion YOLOv8-Pose on TrafficLab footage")
    ap.add_argument('--video',   required=True,  help="Path to .mp4 file")
    ap.add_argument('--weights', required=True,  help="Path to .pt weights file")
    ap.add_argument('--frames',  type=int, default=40,
                    help="Number of frames to sample (-1 = all)")
    ap.add_argument('--out',     default='/private/tmp/carfusion_eval/',
                    help="Output directory for annotated frames")
    args = ap.parse_args()

    run_eval(args.video, args.weights, args.frames, args.out)


if __name__ == '__main__':
    main()
