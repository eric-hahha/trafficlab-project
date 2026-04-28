import hashlib
import functools

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (QImage, QPixmap, QColor, QPen, QBrush,
                         QPolygonF, QPainter)


@functools.lru_cache(maxsize=512)
def get_color_from_string(s):
    """Deterministic QColor derived from a string hash (stable across runs)."""
    hex_hash = hashlib.md5(s.encode()).hexdigest()
    r = int(hex_hash[0:2], 16)
    g = int(hex_hash[2:4], 16)
    b = int(hex_hash[4:6], 16)
    return QColor(r, g, b)


class CCTRenderer:
    """Draws detection/tracking overlays onto a CCTV video frame.

    All rendering is done with Qt (QPainter) on a QPixmap derived from
    the raw OpenCV BGR frame.  The caller is responsible for any
    pre-processing of *frame* (e.g. ROI blending) before passing it in.
    """

    def render(self, frame, objects,
               show_tracking=True,
               show_3d=True,
               box_thickness=2,
               face_opacity=50,
               show_label=True):
        """Render tracked-object overlays onto *frame* and return a QPixmap.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR frame as returned by cv2 (will not be modified).
        objects : list[dict]
            Per-frame object list from the replay JSON.
        show_tracking : bool
            Colour boxes by track-ID seed when True; by class name when False.
        show_3d : bool
            Draw 3D projected bounding boxes when True, 2D flat boxes when False.
        box_thickness : int
            Line thickness for box edges.
        face_opacity : int
            Alpha (0-255) for 3D face fill colour.
        show_label : bool
            Annotate each box with its class / track-ID label.

        Returns
        -------
        QPixmap
            The frame converted to a pixmap with overlays painted on top.
        """
        h, w, ch = frame.shape
        qt_img = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qt_img)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)

        for obj in objects:
            cls = obj.get("class", "?")
            tid = obj.get("tracked_id")
            seed = f"{cls}_{tid}" if (show_tracking and tid is not None) else cls
            col = get_color_from_string(seed)
            lbl = f"{tid} {cls}" if tid is not None else cls

            bbox_3d = obj.get("bbox_3d")
            have_heading = obj.get("have_heading", False)
            have_measurements = obj.get("have_measurements", False)

            # 3D MODE: heading + measurements + valid 8-point box required
            can_draw_3d = (
                show_3d and have_heading and have_measurements
                and bbox_3d and len(bbox_3d) == 8
            )

            if can_draw_3d:
                try:
                    pts = [QPointF(p[0], p[1]) for p in bbox_3d]
                    faces = [
                        [0, 1, 2, 3], [4, 5, 6, 7],  # Bot, Top
                        [0, 1, 5, 4], [1, 2, 6, 5],
                        [2, 3, 7, 6], [3, 0, 4, 7],
                    ]
                    pen = QPen(col, box_thickness)
                    painter.setPen(pen)
                    fill = QColor(col)
                    fill.setAlpha(face_opacity)
                    painter.setBrush(QBrush(fill))
                    for f_idx in faces:
                        poly = QPolygonF([pts[i] for i in f_idx])
                        painter.drawPolygon(poly)
                except Exception:
                    pass

            # 2D MODE
            elif not show_3d:
                bbox = obj.get("bbox_2d")
                if bbox:
                    x1, y1, x2, y2 = map(int, bbox)
                    rect = QRectF(x1, y1, x2 - x1, y2 - y1)
                    painter.setPen(QPen(col, box_thickness))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(rect)

                    # Reference point when heading unknown but measurements present
                    if (not have_heading) and have_measurements:
                        ref_pt = obj.get("reference_point")
                        if ref_pt:
                            rx, ry = ref_pt
                            painter.setBrush(QBrush(col))
                            painter.drawEllipse(QPointF(rx, ry), 4, 4)

                    if show_label:
                        painter.setPen(QPen(Qt.white))
                        fm = painter.fontMetrics()
                        tw, th = fm.width(lbl), fm.height()
                        painter.fillRect(QRectF(x1, y1 - th, tw + 4, th), col)
                        painter.drawText(QPointF(x1 + 2, y1 - 2), lbl)

        painter.end()
        return pix
