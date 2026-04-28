import math
from typing import Optional

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import (QColor, QPen, QBrush, QPolygonF,
                         QImage, QPixmap, QPainter, QFont)

from trafficlab.visualization.cctv_renderer import get_color_from_string


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

                label_str = f"{cls} {disp_s:.1f}km/h"
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

        painter.end()
        return QPixmap.fromImage(self._img)
