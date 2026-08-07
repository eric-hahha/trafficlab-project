import math
from typing import Optional

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (QColor, QPen, QBrush, QPolygonF,
                         QImage, QPixmap, QPainter, QFont, QFontMetricsF)

from trafficlab.visualization.cctv_renderer import get_color_from_string

# ApolloCar3D 24-keypoint order (matches scripts/eval_openpifpaf.py KEYPOINT_NAMES,
# scripts/plot_reprojection_keypoints.py, and the p_sat / kp_sat index used by
# trafficlab/motion/keypoints_openpifpaf.py).
_KEYPOINT_NAMES = [
    'front_up_right', 'front_up_left', 'front_light_right', 'front_light_left',
    'front_low_right', 'front_low_left', 'central_up_left', 'front_wheel_left',
    'rear_wheel_left', 'rear_corner_left', 'rear_up_left', 'rear_up_right',
    'rear_light_left', 'rear_light_right', 'rear_low_left', 'rear_low_right',
    'central_up_right', 'rear_corner_right', 'rear_wheel_right', 'front_wheel_right',
    'rear_plate_left', 'rear_plate_right', 'mirror_edge_left', 'mirror_edge_right',
]
_KP_LABEL_OFFSETS = [
    (dx * radius, dy * radius)
    for radius in (10, 18, 28, 40, 55, 72, 92, 115)
    for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1))
]
_KP_MARKER_RADIUS = 3.0
_KP_LABEL_FONT_SIZE = 6.0

# Keypoint-name substring -> car part, checked in this order (first match wins;
# 'up' must come after the more specific parts since e.g. 'front_up_right' has
# no other matching substring). Colors chosen for max hue separation.
_KP_PART_COLORS = [
    ('wheel',  QColor(34, 197, 94)),    # green
    ('light',  QColor(249, 115, 22)),   # orange
    ('plate',  QColor(234, 179, 8)),    # yellow
    ('mirror', QColor(34, 211, 238)),   # cyan
    ('corner', QColor(236, 72, 153)),   # pink
    ('low',    QColor(168, 85, 247)),   # purple
    ('up',     QColor(59, 130, 246)),   # blue
]


def _kp_part_color(kp_name: str) -> QColor:
    for substr, color in _KP_PART_COLORS:
        if substr in kp_name:
            return color
    return QColor(200, 200, 200)  # unmatched name, shouldn't happen


class SatRenderer:
    """Draws per-frame object overlays onto a single cached QPixmap (satellite view).

    Instead of creating individual QGraphicsItem objects (which each trigger Qt
    scene-graph bookkeeping), all per-frame drawing is done with a QPainter on a
    reusable QImage buffer.  Only one QGraphicsPixmapItem in the scene is updated
    per frame, reducing Qt overhead from O(N) to O(1) scene operations.
    """

    def __init__(self):
        # Cached image buffer — reused every frame when size is unchanged.
        self._img: Optional[QImage] = None
        self._img_size: tuple = (0, 0)

    def render(self, objects, scene_w: int, scene_h: int, *,
               show_tracking=True,
               sat_box_thick=2,
               show_sat_box=True,
               show_sat_arrow=False,
               show_sat_coords_dot=False,
               sat_use_svg=True,
               show_3d=True,
               show_sat_label=False,
               show_sat_keypoints=False,
               kp_color_mode="track",
               sat_label_size=12,
               text_color_mode="White",
               speed_display_cache=None,
               speed_update_delay_frames=30,
               current_frame_idx=0) -> QPixmap:
        """Render all objects for the current frame into a single transparent QPixmap.

        Returns a QPixmap of size (scene_w × scene_h) with all object overlays
        painted on it.  The caller should assign the result to a QGraphicsPixmapItem
        positioned at the origin (0, 0) of the satellite scene — matching the SAT
        background image.

        The internal QImage buffer is reused across frames when the scene size is
        unchanged, so the only per-frame cost is a memset clear + N draw calls.
        """
        if speed_display_cache is None:
            speed_display_cache = {}

        # Reuse the buffer when dimensions are unchanged; reallocate only on resize.
        if (scene_w, scene_h) != self._img_size or self._img is None:
            self._img = QImage(scene_w, scene_h, QImage.Format_ARGB32_Premultiplied)
            self._img_size = (scene_w, scene_h)
        self._img.fill(Qt.transparent)

        painter = QPainter(self._img)
        painter.setRenderHint(QPainter.Antialiasing)

        for obj in objects:
            cls = obj.get("class", "?")
            tid = obj.get("tracked_id")
            seed = f"{cls}_{tid}" if (show_tracking and tid is not None) else cls
            col = get_color_from_string(seed)
            pen = QPen(col, sat_box_thick)
            brush = QBrush(QColor(col.red(), col.green(), col.blue(), 100))

            have_heading = obj.get("have_heading", False)
            have_measurements = obj.get("have_measurements", False)
            coord = obj.get("sat_coords") or obj.get("sat_coord")
            pts = obj.get("sat_floor_box")

            # --- 1. Floor Box (heading + measurements required) ---
            if show_sat_box and have_heading and have_measurements and pts and len(pts) >= 3:
                painter.setPen(pen)
                painter.setBrush(brush)
                painter.drawPolygon(QPolygonF([QPointF(p[0], p[1]) for p in pts]))

            # --- 2. Heading Arrow ---
            default_heading = obj.get("default_heading", False)
            if (show_sat_arrow and have_heading and (not default_heading)
                    and coord and pts and len(pts) >= 3):
                heading = obj.get("heading")
                if heading is not None:
                    rad = math.radians(heading)
                    x1, y1 = coord[0], coord[1]
                    painter.setPen(QPen(Qt.yellow, 2))
                    painter.drawLine(
                        QPointF(x1, y1),
                        QPointF(x1 + 40 * math.cos(rad), y1 + 40 * math.sin(rad)),
                    )

            # --- 3a. Coordinate Dot (user-toggled) ---
            _has_floor = pts and len(pts) >= 3
            _no_svg_no_3d = (not sat_use_svg) and (not show_3d)
            if show_sat_coords_dot and coord and (_has_floor or _no_svg_no_3d):
                radius = 4.0
                if pts and len(pts) >= 3:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    avg_dim = ((max(xs) - min(xs)) + (max(ys) - min(ys))) / 2.0
                    radius = max(3.0, avg_dim * 0.15)
                painter.setPen(QPen(Qt.black, 1))
                painter.setBrush(QBrush(col))
                painter.drawEllipse(QPointF(coord[0], coord[1]), radius, radius)

            # --- 3b. Legacy Fallback Dot (no heading, has measurements) ---
            elif ((not have_heading) and have_measurements and (not show_3d)
                  and coord and (_has_floor or _no_svg_no_3d)):
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(col))
                painter.drawEllipse(QPointF(coord[0], coord[1]), 3.0, 3.0)

            # --- 4. Speed Label ---
            if show_sat_label and coord and (_has_floor or _no_svg_no_3d):
                raw_s = obj.get("speed_kmh", 0)
                disp_s = raw_s
                if tid is not None:
                    cache = speed_display_cache.get(tid, {"val": raw_s, "last_frame": -999})
                    if ((current_frame_idx - cache["last_frame"]) >= speed_update_delay_frames
                            or current_frame_idx < cache["last_frame"]):
                        cache["val"] = raw_s
                        cache["last_frame"] = current_frame_idx
                    speed_display_cache[tid] = cache
                    disp_s = cache["val"]

                id_prefix = f"#{tid} " if tid is not None else ""
                label_str = f"{id_prefix}{cls} {disp_s:.1f}km/h"
                font = QFont()
                font.setPointSize(sat_label_size)
                painter.setFont(font)

                if text_color_mode == "Black":
                    painter.setPen(QPen(Qt.black))
                elif text_color_mode == "Yellow":
                    painter.setPen(QPen(QColor(255, 255, 143)))
                else:
                    painter.setPen(QPen(Qt.white))

                painter.drawText(QPointF(coord[0], coord[1]), label_str)

        if show_sat_keypoints:
            self._draw_keypoints(painter, objects, show_tracking, scene_w, scene_h, kp_color_mode)

        painter.end()
        return QPixmap.fromImage(self._img)

    # --- Keypoint reprojection overlay ---
    # Mirrors scripts/plot_reprojection_keypoints.py: white-edged dots per
    # kp_sat entry, each with a "tracked_id-keypoint_name" label joined to its
    # point by a black leader line, placed to avoid covering any point marker
    # or other label.
    def _draw_keypoints(self, painter, objects, show_tracking, scene_w, scene_h, kp_color_mode="track"):
        kp_points = []  # (x, y, tracked_id, kp_idx, color)
        for obj in objects:
            kp_sat = obj.get("kp_sat")
            if not kp_sat:
                continue
            tid = obj.get("tracked_id")
            cls = obj.get("class", "?")
            seed = f"{cls}_{tid}" if (show_tracking and tid is not None) else cls
            track_col = get_color_from_string(seed)
            for kp_idx, kp in enumerate(kp_sat):
                if kp is None:
                    continue
                if kp_color_mode == "part":
                    col = _kp_part_color(_KEYPOINT_NAMES[kp_idx])
                else:
                    col = track_col
                kp_points.append((float(kp[0]), float(kp[1]), tid, kp_idx, col))

        if not kp_points:
            return

        painter.setPen(QPen(Qt.white, 0.8))
        for x, y, _tid, _kp_idx, col in kp_points:
            painter.setBrush(QBrush(col))
            painter.drawEllipse(QPointF(x, y), _KP_MARKER_RADIUS, _KP_MARKER_RADIUS)

        font = QFont()
        font.setPointSizeF(_KP_LABEL_FONT_SIZE)
        painter.setFont(font)
        metrics = QFontMetricsF(font)

        bounds = (0.0, 0.0, float(scene_w), float(scene_h))
        occupied = [
            (x - _KP_MARKER_RADIUS, y - _KP_MARKER_RADIUS,
             x + _KP_MARKER_RADIUS, y + _KP_MARKER_RADIUS)
            for x, y, *_ in kp_points
        ]

        for x, y, tid, kp_idx, _col in kp_points:
            label = f"{tid if tid is not None else '?'}-{_KEYPOINT_NAMES[kp_idx]}"
            rect = metrics.boundingRect(label)
            offset = self._place_kp_label(x, y, (rect.width(), rect.height()), occupied, bounds)
            lx, ly = x + offset[0], y + offset[1]

            painter.setPen(QPen(Qt.black, 0.6))
            painter.drawLine(QPointF(x, y), QPointF(lx, ly))

            pad = 2.0
            box = QRectF(lx - rect.width() / 2.0 - pad, ly - rect.height() / 2.0 - pad,
                         rect.width() + 2 * pad, rect.height() + 2 * pad)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 255, 255, 190)))
            painter.drawRoundedRect(box, 2, 2)

            painter.setPen(QPen(Qt.black))
            painter.drawText(box, Qt.AlignCenter, label)

    @staticmethod
    def _kp_label_box(x, y, offset, size):
        w, h = size
        cx, cy = x + offset[0], y + offset[1]
        pad = 2.0
        return (cx - w / 2.0 - pad, cy - h / 2.0 - pad, cx + w / 2.0 + pad, cy + h / 2.0 + pad)

    @staticmethod
    def _kp_boxes_overlap(a, b):
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

    @staticmethod
    def _kp_within_bounds(box, bounds):
        return (box[0] >= bounds[0] and box[1] >= bounds[1]
                and box[2] <= bounds[2] and box[3] <= bounds[3])

    def _place_kp_label(self, x, y, size, occupied, bounds):
        best_in_bounds = None  # (overlap_count, offset, box)
        best_any = None
        for offset in _KP_LABEL_OFFSETS:
            box = self._kp_label_box(x, y, offset, size)
            overlap_count = sum(1 for other in occupied if self._kp_boxes_overlap(box, other))
            in_bounds = self._kp_within_bounds(box, bounds)
            if overlap_count == 0 and in_bounds:
                occupied.append(box)
                return offset
            if in_bounds and (best_in_bounds is None or overlap_count < best_in_bounds[0]):
                best_in_bounds = (overlap_count, offset, box)
            if best_any is None or overlap_count < best_any[0]:
                best_any = (overlap_count, offset, box)

        _, offset, box = best_in_bounds if best_in_bounds is not None else best_any
        occupied.append(box)
        return offset
