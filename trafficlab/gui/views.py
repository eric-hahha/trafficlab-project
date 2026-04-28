"""
Reusable QGraphicsView subclasses shared across GUI tabs.

  SatGraphicsView   — zoomable/pannable satellite map panel
  CCTVGraphicsView  — zoomable/pannable CCTV video panel
  MediaViewer       — general-purpose image viewer with placeholder text support
"""

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QFont, QPixmap, QPainter
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView


_ZOOM_IN  = 1.15
_ZOOM_OUT = 1.0 / _ZOOM_IN


class SatGraphicsView(QGraphicsView):
    """Zoomable/pannable view for the satellite map panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        # Set by the owning tab so the view can call back into the inspector panel.
        self.parent_inspector = None

    def wheelEvent(self, event):
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        factor = _ZOOM_IN if event.angleDelta().y() > 0 else _ZOOM_OUT
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)


class CCTVGraphicsView(QGraphicsView):
    """Zoomable/pannable view for the CCTV video panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event):
        factor = _ZOOM_IN if event.angleDelta().y() > 0 else _ZOOM_OUT
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)


class MediaViewer(QGraphicsView):
    """Zoomable image viewer with placeholder-text support.

    Used in the Location tab to preview CCTV / SAT images before a location
    is created.
    """

    PLACEHOLDER_POINT_SIZE = 18
    PLACEHOLDER_BOLD = True

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pixmap_item = None
        self._placeholder_item = None
        self._last_image_path = None
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def load_image(self, path: str):
        try:
            self.scene().clear()
        except Exception:
            pass
        self._pixmap_item = None
        self._placeholder_item = None
        try:
            self.resetTransform()
        except Exception:
            pass
        pix = QPixmap(path)
        if pix and not pix.isNull():
            self._pixmap_item = self.scene().addPixmap(pix)
            self._last_image_path = path
            self.scene().setSceneRect(self._pixmap_item.boundingRect())
            self.fit_view()

    def set_placeholder(self, text: str):
        try:
            self.scene().clear()
        except Exception:
            pass
        try:
            self.resetTransform()
        except Exception:
            pass
        self._pixmap_item = None
        self._placeholder_item = None
        try:
            font = QFont()
            font.setPointSize(self.PLACEHOLDER_POINT_SIZE)
            font.setBold(self.PLACEHOLDER_BOLD)
            self._placeholder_item = self.scene().addText(text, font)
            rect = self._placeholder_item.boundingRect()
            scene_rect = self.scene().sceneRect()
            if scene_rect.isNull():
                vw = max(400, self.viewport().width() or 400)
                vh = max(700, self.viewport().height() or 700)
                self.scene().setSceneRect(0, 0, vw, vh)
                scene_rect = self.scene().sceneRect()
            x = (scene_rect.width() - rect.width()) / 2
            y = (scene_rect.height() - rect.height()) / 2
            self._placeholder_item.setPos(QPointF(x, y))
        except Exception:
            self._placeholder_item = None

    def clear(self):
        try:
            self.scene().clear()
        except Exception:
            pass
        try:
            self.resetTransform()
        except Exception:
            pass
        self._pixmap_item = None
        self._placeholder_item = None
        self._last_image_path = None

    def fit_view(self):
        if self._pixmap_item is None:
            return
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if self._pixmap_item is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
