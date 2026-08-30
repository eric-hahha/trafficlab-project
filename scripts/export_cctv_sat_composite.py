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
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPixmap, QPainter

from trafficlab.projection.parallax_reprojection import compute_frame_records, compute_view_extent
from trafficlab.motion.keypoints_openpifpaf import kp_bbox_xyxy
from trafficlab.projection.g_projection import GProjection
from trafficlab.visualization.cctv_renderer import CCTRenderer
from trafficlab.visualization.sat_renderer import SatRenderer


def load_replay(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        return json.load(f)


def _parse_ids(value):
    if not value:
        return None
    ids = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        ids.append(int(item))
    return set(ids) or None


def resolve_g_projection_path(location_code, *, explicit_path=None, project_root='.'):
    if explicit_path:
        return explicit_path if os.path.exists(explicit_path) else None
    if not location_code:
        return None
    candidate = os.path.join(project_root, 'location', location_code, f'G_projection_{location_code}.json')
    return candidate if os.path.exists(candidate) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', required=True, help='Path to replay .json/.json.gz')
    p.add_argument('--out-dir', required=True, help='Directory to write composite PNGs')
    p.add_argument('--ids', default=None,
                   help='Comma-separated tracked_id values to include (default: all objects). '
                        'Frames with no matching object are skipped entirely.')
    p.add_argument('--cctv-video', default=None, help='Override video path (default: mp4_path in replay)')
    p.add_argument('--sat-image', default=None, help='Override SAT background path '
                                                       '(default: location/<code>/sat_<code>.png)')
    p.add_argument('--kp-conf', type=float, default=0.2,
                   help='Keypoint confidence threshold. Used to derive the 2D box when bbox_2d is '
                        'missing; with --right-panel parallax, also the threshold for a valid '
                        'kp_cctv/kp_sat pair (default 0.2)')
    p.add_argument('--sat-only', action='store_true',
                   help='Write only the SAT panel (with keypoints), skip the CCTV panel and hstack')
    p.add_argument('--right-panel', choices=['sat', 'parallax'], default='sat',
                   help='sat (default): normal SAT overlay (box/arrow/coords dot/label/keypoints). '
                        'parallax: replace the panel with a parallax-correction before/after view — '
                        'each kp_sat (colored dot, labeled, height-corrected) joined by a line to its '
                        'uncorrected h=0 apparent point (gray dot), mirroring '
                        'trafficlab/projection/parallax_reprojection.py. Requires a resolvable '
                        'G_projection config (see --g-proj).')
    p.add_argument('--g-proj', default=None,
                   help='G_projection config path, only used with --right-panel parallax '
                        '(default: location/<code>/G_projection_<code>.json)')
    p.add_argument('--kp-color-mode', choices=['track', 'part'], default='track',
                   help='Only applies to --right-panel sat: color sat keypoints by vehicle track '
                        '(default) or by car part (wheel/light/plate/mirror/corner/low/up). '
                        '--right-panel parallax always colors by car part.')
    p.add_argument('--no-sat-keypoints', action='store_true',
                   help='Do not draw keypoints on the SAT panel (box/arrow/coords dot/label unaffected). '
                        'Only valid with --right-panel sat.')
    p.add_argument('--zoom-to-ids', action='store_true',
                   help='Crop the SAT panel to a single fixed window covering --ids\' sat_coords '
                        '(or, with --right-panel parallax, both pre- and post-correction keypoints) '
                        'across every frame it appears in (same window for every output frame, so '
                        'the crop doesn\'t jump around) instead of showing the full SAT image. '
                        'Requires --ids.')
    p.add_argument('--zoom-margin', type=int, default=200,
                   help='Padding in SAT-image pixels added around the --zoom-to-ids bounding box '
                        '(default 200, matches trajectory_tools.py\'s --zoom-margin default)')
    args = p.parse_args()

    if args.zoom_to_ids and args.ids is None:
        p.error('--zoom-to-ids requires --ids (the zoom window is computed from those tracks)')
    if args.right_panel == 'parallax' and args.no_sat_keypoints:
        p.error('--no-sat-keypoints has no effect with --right-panel parallax '
                 '(that panel is keypoints-only) — remove it')

    data = load_replay(args.replay)
    video_path = args.cctv_video or data['mp4_path']
    loc = data['location_code']
    sat_path = args.sat_image or os.path.join('location', loc, f'sat_{loc}.png')
    id_filter = _parse_ids(args.ids)

    os.makedirs(args.out_dir, exist_ok=True)

    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)

    sat_bg = QPixmap(sat_path)
    if sat_bg.isNull():
        raise SystemExit(f'could not load SAT image: {sat_path}')
    sw, sh = sat_bg.width(), sat_bg.height()

    sat_renderer = SatRenderer()

    g_engine = None
    if args.right_panel == 'parallax':
        g_proj_path = resolve_g_projection_path(loc, explicit_path=args.g_proj, project_root=REPO_ROOT)
        if g_proj_path is None:
            raise SystemExit(f'--right-panel parallax: could not resolve G_projection config for '
                              f'location_code={loc!r} (pass --g-proj to override)')
        with open(g_proj_path) as f:
            g_data = json.load(f)
        g_engine = GProjection(g_data, base_dir=os.path.dirname(g_proj_path))
        print(f'[composite] --right-panel parallax: using G_projection {g_proj_path}')

    zoom_rect = None
    if args.zoom_to_ids and args.right_panel == 'sat':
        xs, ys = [], []
        for fr in data['frames']:
            for o in fr['objects']:
                if o.get('tracked_id') not in id_filter:
                    continue
                pt = o.get('sat_coords') or o.get('sat_center')
                if pt:
                    xs.append(pt[0])
                    ys.append(pt[1])
        if not xs:
            raise SystemExit(f'--zoom-to-ids: no sat_coords found for --ids {sorted(id_filter)}')
        m = args.zoom_margin
        x0 = max(0, int(min(xs) - m))
        y0 = max(0, int(min(ys) - m))
        x1 = min(sw, int(max(xs) + m))
        y1 = min(sh, int(max(ys) + m))
        zoom_rect = QRect(x0, y0, x1 - x0, y1 - y0)
        print(f'[composite] zoom-to-ids window: x={x0}..{x1} y={y0}..{y1} '
              f'(from {len(xs)} points, margin={m}px)')
    elif args.zoom_to_ids and args.right_panel == 'parallax':
        records_list = [
            compute_frame_records(
                {'objects': [o for o in fr['objects'] if o.get('tracked_id') in id_filter]},
                g_engine, kp_conf=args.kp_conf,
            )
            for fr in data['frames']
        ]
        extent = compute_view_extent(records_list, pad=args.zoom_margin)
        if extent is None:
            raise SystemExit(f'--zoom-to-ids: no valid kp_cctv/kp_sat pair found for --ids {sorted(id_filter)}')
        xlim, ylim = extent
        x0 = max(0, int(xlim[0]))
        x1 = min(sw, int(xlim[1]))
        y0 = max(0, int(min(ylim)))
        y1 = min(sh, int(max(ylim)))
        zoom_rect = QRect(x0, y0, x1 - x0, y1 - y0)
        print(f'[composite] zoom-to-ids window (parallax, pre+post extent): x={x0}..{x1} y={y0}..{y1} '
              f'(margin={args.zoom_margin}px)')

    def render_sat(objects, frame_index=None):
        sat_canvas = QPixmap(sat_bg)
        if args.right_panel == 'parallax':
            records = compute_frame_records({'objects': objects}, g_engine, kp_conf=args.kp_conf)
            overlay = sat_renderer.render_parallax(records, sw, sh, frame_index=frame_index)
        else:
            overlay = sat_renderer.render(objects, sw, sh,
                                          show_sat_box=True,
                                          show_sat_arrow=True,
                                          show_sat_coords_dot=True,
                                          sat_use_svg=False,
                                          show_3d=False,
                                          show_sat_label=True,
                                          show_sat_keypoints=not args.no_sat_keypoints,
                                          kp_color_mode=args.kp_color_mode)
        painter = QPainter(sat_canvas)
        painter.drawPixmap(0, 0, overlay)
        painter.end()
        if zoom_rect is not None:
            sat_canvas = sat_canvas.copy(zoom_rect)
        return sat_canvas

    n_written = 0

    if args.sat_only:
        # No CCTV panel needed, so no need to open the video either.
        for fr in sorted(data['frames'], key=lambda fr: fr['frame_index']):
            idx = fr['frame_index']
            objects = fr['objects']
            if id_filter is not None:
                objects = [o for o in objects if o.get('tracked_id') in id_filter]
            if not objects:
                continue
            sat_canvas = render_sat(objects, idx)
            out_path = os.path.join(args.out_dir, f'frame_{idx:04d}.png')
            sat_canvas.save(out_path)
            n_written += 1
    else:
        cct_renderer = CCTRenderer()
        frames_by_index = {fr['frame_index']: fr['objects'] for fr in data['frames']}

        cap = cv2.VideoCapture(video_path)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_objects = frames_by_index.get(idx)
            # --ids only decides which frames get rendered and what the SAT
            # panel is filtered/zoomed to — the CCTV panel always shows every
            # object in the frame (full scene context), not just --ids.
            sat_objects = all_objects
            if all_objects is not None and id_filter is not None:
                sat_objects = [o for o in all_objects if o.get('tracked_id') in id_filter]
                if not sat_objects:
                    all_objects = None  # no matching object this frame — skip, same as "frame absent"
            if all_objects is not None:
                for obj in all_objects:
                    kp = obj.get('kp_cctv')
                    if kp and obj.get('bbox_2d') is None:
                        box = kp_bbox_xyxy(np.array(kp), conf_thresh=args.kp_conf)
                        if box is not None:
                            obj['bbox_2d'] = list(box)

                cctv_pix = cct_renderer.render(frame, all_objects, show_3d=False)
                sat_canvas = render_sat(sat_objects, idx)

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
