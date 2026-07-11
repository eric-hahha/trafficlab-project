"""Composite CCTV + SAT overlays for each frame of an h-aware replay JSON.

Reuses the existing GUI renderers (CCTRenderer, SatRenderer) headlessly —
no new drawing logic, just a batch driver that writes one PNG per frame.
"""
import argparse
import gzip
import json
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter

from trafficlab.motion.haware_localization import kp_bbox_xyxy
from trafficlab.visualization.cctv_renderer import CCTRenderer
from trafficlab.visualization.sat_renderer import SatRenderer


def load_replay(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', required=True, help='Path to replay .json/.json.gz')
    p.add_argument('--out-dir', required=True, help='Directory to write composite PNGs')
    p.add_argument('--cctv-video', default=None, help='Override video path (default: mp4_path in replay)')
    p.add_argument('--sat-image', default=None, help='Override SAT background path '
                                                       '(default: location/<code>/sat_<code>.png)')
    p.add_argument('--kp-conf', type=float, default=0.2, help='Keypoint confidence threshold for the 2D box (default 0.2)')
    args = p.parse_args()

    data = load_replay(args.replay)
    video_path = args.cctv_video or data['mp4_path']
    loc = data['location_code']
    sat_path = args.sat_image or os.path.join('location', loc, f'sat_{loc}.png')

    os.makedirs(args.out_dir, exist_ok=True)

    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)

    sat_bg = QPixmap(sat_path)
    if sat_bg.isNull():
        raise SystemExit(f'could not load SAT image: {sat_path}')
    sw, sh = sat_bg.width(), sat_bg.height()

    cct_renderer = CCTRenderer()
    sat_renderer = SatRenderer()

    frames_by_index = {fr['frame_index']: fr['objects'] for fr in data['frames']}

    cap = cv2.VideoCapture(video_path)
    idx = 0
    n_written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        objects = frames_by_index.get(idx)
        if objects is not None:
            for obj in objects:
                kp = obj.get('kp_cctv')
                if kp and obj.get('bbox_2d') is None:
                    box = kp_bbox_xyxy(np.array(kp), conf_thresh=args.kp_conf)
                    if box is not None:
                        obj['bbox_2d'] = list(box)

            cctv_pix = cct_renderer.render(frame, objects, show_3d=False)

            sat_canvas = QPixmap(sat_bg)
            overlay = sat_renderer.render(objects, sw, sh,
                                          show_sat_box=True,
                                          show_sat_arrow=True,
                                          show_sat_coords_dot=True,
                                          sat_use_svg=False,
                                          show_3d=False,
                                          show_sat_label=True)
            painter = QPainter(sat_canvas)
            painter.drawPixmap(0, 0, overlay)
            painter.end()

            target_h = cctv_pix.height()
            sat_scaled = sat_canvas.scaledToHeight(target_h, Qt.SmoothTransformation)

            combo = QPixmap(cctv_pix.width() + sat_scaled.width(), target_h)
            combo.fill(Qt.black)
            cp = QPainter(combo)
            cp.drawPixmap(0, 0, cctv_pix)
            cp.drawPixmap(cctv_pix.width(), 0, sat_scaled)
            cp.end()

            out_path = os.path.join(args.out_dir, f'frame_{idx:04d}.png')
            combo.save(out_path)
            n_written += 1

        idx += 1

    cap.release()
    print(f'wrote {n_written} composite frame(s) to {args.out_dir}')


if __name__ == '__main__':
    main()
