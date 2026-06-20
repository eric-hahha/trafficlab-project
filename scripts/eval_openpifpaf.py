"""
Quick visual eval of OpenPifPaf ApolloCar3D keypoints on test21 footage.
Samples N frames, draws 24-keypoint skeleton on each detected car, saves to /tmp/pifpaf_eval/.
"""
import argparse
import os
import cv2
import numpy as np
import torch
import openpifpaf
import openpifpaf.plugins.apollocar3d as apc

apc.register()

KEYPOINT_NAMES = [
    'front_up_right', 'front_up_left', 'front_light_right', 'front_light_left',
    'front_low_right', 'front_low_left', 'central_up_left', 'front_wheel_left',
    'rear_wheel_left', 'rear_corner_left', 'rear_up_left', 'rear_up_right',
    'rear_light_left', 'rear_light_right', 'rear_low_left', 'rear_low_right',
    'central_up_right', 'rear_corner_right', 'rear_wheel_right', 'front_wheel_right',
    'rear_plate_left', 'rear_plate_right', 'mirror_edge_left', 'mirror_edge_right',
]

# left-right horizontal pairs (same as Vehicle_Orientation_Detect utils.py)
HORIZONTAL_PAIRS = [
    [0, 1], [2, 3], [4, 5], [6, 16], [7, 19], [8, 18],
    [9, 17], [10, 11], [12, 13], [14, 15], [20, 21], [22, 23],
]

COLORS = {
    'wheel': (0, 255, 0),
    'light': (0, 180, 255),
    'mirror': (255, 0, 200),
    'other': (180, 180, 180),
}

WHEEL_IDX = {7, 8, 18, 19}
LIGHT_IDX = {2, 3, 12, 13}
MIRROR_IDX = {22, 23}


def kp_color(idx):
    if idx in WHEEL_IDX:
        return COLORS['wheel']
    if idx in LIGHT_IDX:
        return COLORS['light']
    if idx in MIRROR_IDX:
        return COLORS['mirror']
    return COLORS['other']


def draw_annotation(frame, ann, conf_thresh=0.2):
    kps = ann.data  # (24, 3) — x, y, confidence
    h, w = frame.shape[:2]

    # draw keypoints
    for idx, (x, y, c) in enumerate(kps):
        if c < conf_thresh:
            continue
        cx, cy = int(x), int(y)
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        color = kp_color(idx)
        cv2.circle(frame, (cx, cy), 4, color, -1)

    # draw horizontal pairs (orientation cues) in yellow
    for i, j in HORIZONTAL_PAIRS:
        xi, yi, ci = kps[i]
        xj, yj, cj = kps[j]
        if ci < conf_thresh or cj < conf_thresh:
            continue
        cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), (0, 255, 255), 2)

    # compute heading from best available horizontal pair
    best_pair, best_conf = None, 0.0
    for i, j in HORIZONTAL_PAIRS:
        xi, yi, ci = kps[i]
        xj, yj, cj = kps[j]
        conf = min(ci, cj)
        if conf > best_conf:
            best_conf = conf
            best_pair = (xi, yi, xj, yj)

    if best_pair and best_conf >= conf_thresh:
        xi, yi, xj, yj = best_pair
        mx, my = int((xi + xj) / 2), int((yi + yj) / 2)
        # perpendicular direction = vehicle forward axis
        dx, dy = xj - xi, yj - yi
        length = max(1, (dx**2 + dy**2)**0.5)
        nx, ny = -dy / length, dx / length  # rotate 90°
        arrow_len = 40
        ex, ey = int(mx + nx * arrow_len), int(my + ny * arrow_len)
        cv2.arrowedLine(frame, (mx, my), (ex, ey), (255, 80, 0), 2, tipLength=0.3)
        cv2.putText(frame, f"c={best_conf:.2f}", (mx + 5, my - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 80, 0), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='location/test21/footage/test21-3sf.mp4')
    parser.add_argument('--out', default='/tmp/pifpaf_eval')
    parser.add_argument('--frames', type=int, default=40, help='number of frames to sample')
    parser.add_argument('--checkpoint', default='shufflenetv2k16-apollo-24')
    parser.add_argument('--conf', type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    predictor = openpifpaf.Predictor(checkpoint=args.checkpoint)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total} frames @ {fps:.1f} fps")

    # sample evenly across video
    sample_frames = sorted(set(
        int(i * total / args.frames) for i in range(args.frames)
    ))

    for frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = openpifpaf.datasets.pil_image.fromarray(rgb) if hasattr(
            openpifpaf.datasets, 'pil_image') else __import__('PIL.Image', fromlist=['Image']).fromarray(rgb)

        predictions, gt_anns, image_meta = predictor.pil_image(pil_img)

        vis = frame.copy()
        n_drawn = 0
        for ann in predictions:
            if not hasattr(ann, 'data'):
                continue
            draw_annotation(vis, ann, conf_thresh=args.conf)
            n_drawn += 1

        # legend
        cv2.putText(vis, f"frame {frame_idx}  cars={n_drawn}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for label, color in [('wheel', COLORS['wheel']), ('light', COLORS['light']),
                              ('mirror', COLORS['mirror']), ('heading arrow', (255, 80, 0))]:
            pass  # skip verbose legend

        out_path = os.path.join(args.out, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(out_path, vis)
        print(f"  frame {frame_idx:5d}: {n_drawn} annotations → {out_path}")

    cap.release()
    print(f"\nDone. Images saved to {args.out}/")
    print("Legend: GREEN=wheels  ORANGE=lights  PURPLE=mirrors  YELLOW LINE=horizontal pair  BLUE ARROW=heading")


if __name__ == '__main__':
    main()
