import os
import numpy as np
from typing import Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QCheckBox,
    QSplitter, QMessageBox, QGraphicsView, QGraphicsPixmapItem
)

from .undistort_stage import remap_with_supersample
from trafficlab.gui.tools.reference_point_calibration_tool import CrosshairMarker, DraggableMarkerViewer

MIN_ANCHORS = 4
MARKER_COLOR = QColor(0, 255, 0)

class RightClickImageViewer(DraggableMarkerViewer):
    """DraggableMarkerViewer already swaps ScrollHandDrag for NoDrag so
    CrosshairMarker's left-click-drag isn't eaten by view panning; this
    adds the right-click-to-place/remove behavior anchors need on top of
    that: right-clicking empty space emits right_clicked(x, y, None),
    right-clicking an existing CrosshairMarker emits right_clicked(x, y,
    marker), so the stage can decide add vs. remove without a separate
    list UI. A distinct signal (not the inherited 2-arg `clicked`) since
    it carries the hit-tested marker too."""

    right_clicked = pyqtSignal(float, float, object)

    def mousePressEvent(self, event):
        if self._pixmap_item is not None and event.button() == Qt.RightButton:
            pt = self.mapToScene(event.pos())
            hit = self._marker_at(pt)
            self.right_clicked.emit(float(pt.x()), float(pt.y()), hit)
            event.accept()
        else:
            QGraphicsView.mousePressEvent(self, event)

    def _marker_at(self, scene_pt):
        item = self.scene().itemAt(scene_pt, self.transform())
        while item is not None and not isinstance(item, CrosshairMarker):
            item = item.parentItem()
        return item


class HomAStage(QWidget):
    def __init__(self, project_root: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.project_root = project_root

        self.anchors = [
            {'id': i, 'cctv': None, 'sat': None, 'cctv_marker': None, 'sat_marker': None}
            for i in range(MIN_ANCHORS)
        ]
        self._current_cctv_cv = None
        self._current_sat_cv = None
        self._loaded_loc = None # Track what is currently loaded in UI

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet("background-color: #2b2b2b;")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(10, 10, 10, 10)
        side_layout.setSpacing(10)

        lbl_title = QLabel("Homography Anchors")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #fff;")
        side_layout.addWidget(lbl_title)

        inst = QLabel(
            "Controls:\n"
            "• Right Click (empty): Add next point\n"
            "• Right Click (on a point): Remove it\n"
            "• Left Drag on Point: Move Point\n"
            "• Wheel: Zoom\n"
            "• Scrollbars: Pan Image\n\n"
            f"Place the same points on both images, in the same order. At least "
            f"{MIN_ANCHORS} pairs are required; more pairs let RANSAC ignore "
            f"mis-clicked outliers."
        )
        inst.setWordWrap(True)
        inst.setStyleSheet("color: #aaa; font-size: 11px;")
        side_layout.addWidget(inst)

        side_layout.addStretch()

        self.chk_ransac = QCheckBox("Use RANSAC (ignore outlier points)")
        self.chk_ransac.setChecked(True)
        self.chk_ransac.setStyleSheet("color: #ccc;")
        side_layout.addWidget(self.chk_ransac)

        self.btn_compute = QPushButton("Compute Homography")
        self.btn_compute.setStyleSheet("background-color: #2a84ff; color: white; font-weight: bold; padding: 6px;")
        self.btn_compute.clicked.connect(self._on_compute)
        side_layout.addWidget(self.btn_compute)

        self.btn_proceed = QPushButton("Proceed")
        self.btn_proceed.clicked.connect(self._on_proceed)
        side_layout.addWidget(self.btn_proceed)

        main_layout.addWidget(sidebar)

        splitter = QSplitter(Qt.Horizontal)

        left_cont = QWidget()
        l_vbox = QVBoxLayout(left_cont)
        l_vbox.setContentsMargins(0,0,0,0)
        l_vbox.addWidget(QLabel("Undistorted CCTV (Right-click to place)"))
        self.view_cctv = RightClickImageViewer()
        self.view_cctv.right_clicked.connect(self._on_cctv_right_clicked)
        l_vbox.addWidget(self.view_cctv)

        right_cont = QWidget()
        r_vbox = QVBoxLayout(right_cont)
        r_vbox.setContentsMargins(0,0,0,0)
        r_vbox.addWidget(QLabel("Satellite / Layout (Right-click to place)"))
        self.view_sat = RightClickImageViewer()
        self.view_sat.right_clicked.connect(self._on_sat_right_clicked)
        r_vbox.addWidget(self.view_sat)

        splitter.addWidget(left_cont)
        splitter.addWidget(right_cont)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter, 1)

    def showEvent(self, event):
        super().showEvent(event)
        if not HAS_CV2: return

        host = getattr(self, 'host_tab', None) or self.parent()
        if not host: return
        obj = getattr(host, 'inspect_obj', None)
        if not obj: return

        loc_code = obj.get('meta', {}).get('location_code')

        # Sync any in-progress drag back into self.anchors and drop the
        # marker refs before _load_images() below replaces the pixmap --
        # ImageViewer.load_pixmap() calls QGraphicsScene.clear(), which
        # deletes the underlying C++ marker items (see its comment in
        # undistort_stage.py), so holding onto them past this point risks a
        # dangling-pointer segfault.
        self._sync_anchors_from_markers()
        for data in self.anchors:
            data['cctv_marker'] = None
            data['sat_marker'] = None

        # Always reload the CCTV/satellite images -- undistort K/D (or the
        # source PNGs) may have changed in an earlier stage since this one
        # was last shown, and a stale cached pixmap would silently hide that.
        self._load_images(obj)

        if self._loaded_loc != loc_code:
            self._loaded_loc = loc_code

            hom = obj.get('homography', {})
            saved_list = hom.get('anchors_list', [])

            n = max(MIN_ANCHORS, len(saved_list))
            self.anchors = [
                {'id': i, 'cctv': None, 'sat': None, 'cctv_marker': None, 'sat_marker': None}
                for i in range(n)
            ]
            for data, item in zip(self.anchors, saved_list):
                c_list = item.get('coords_cctv')
                s_list = item.get('coords_sat')
                data['cctv'] = tuple(c_list) if (c_list and len(c_list) == 2) else None
                data['sat'] = tuple(s_list) if (s_list and len(s_list) == 2) else None

        self._refresh_markers()

    def _load_images(self, obj):
        proj_root = getattr(self, 'project_root', None) or os.getcwd()
        loc_code = obj.get('meta', {}).get('location_code')
        if not loc_code: return

        # Undistort CCTV
        cctv_path = os.path.join(proj_root, 'location', loc_code, f'cctv_{loc_code}.png')
        if os.path.isfile(cctv_path):
            src = cv2.imread(cctv_path)
            if src is not None:
                und = obj.get('undistort', {})
                K_list = und.get('K')
                D_list = und.get('D', [0]*5)
                h, w = src.shape[:2]

                K = np.array(K_list, dtype=np.float64) if (K_list and len(K_list)==3) else \
                    np.array([[max(w,h), 0, w/2], [0, max(w,h), h/2], [0, 0, 1]], dtype=np.float64)
                D = np.array(D_list, dtype=np.float64)

                try:
                    # FIX: Use K.copy() instead of getOptimalNewCameraMatrix to match reference code logic
                    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
                    newcameramtx = K.copy()

                    self._current_cctv_cv = remap_with_supersample(src, K, D, newcameramtx)
                    qimg = self._cv_to_qimage(self._current_cctv_cv)
                    self.view_cctv.load_pixmap(QPixmap.fromImage(qimg))
                    self.view_cctv.fitToView()
                except Exception as e:
                    print(f"Undistort Error: {e}")

        # Satellite
        sat_path = os.path.join(proj_root, 'location', loc_code, f'sat_{loc_code}.png')
        if os.path.isfile(sat_path):
            self._current_sat_cv = cv2.imread(sat_path)
            if self._current_sat_cv is not None:
                qimg = self._cv_to_qimage(self._current_sat_cv)
                self.view_sat.load_pixmap(QPixmap.fromImage(qimg))
                self.view_sat.fitToView()

    def _next_or_new_slot(self, key):
        for data in self.anchors:
            if data[key] is None:
                return data
        idx = len(self.anchors)
        data = {'id': idx, 'cctv': None, 'sat': None, 'cctv_marker': None, 'sat_marker': None}
        self.anchors.append(data)
        return data

    def _find_anchor_for_marker(self, key, marker):
        for data in self.anchors:
            if data[f'{key}_marker'] is marker:
                return data
        return None

    def _prune_empty_anchors(self):
        self.anchors = [a for a in self.anchors if a['cctv'] is not None or a['sat'] is not None]
        for i, data in enumerate(self.anchors):
            data['id'] = i

    def _on_cctv_right_clicked(self, x, y, hit_marker):
        if hit_marker is not None:
            data = self._find_anchor_for_marker('cctv', hit_marker)
            if data is not None:
                # Must null the marker ref too, not just the coordinate --
                # _refresh_markers() starts with _sync_anchors_from_markers(),
                # which would otherwise read this now-stale marker's position
                # straight back into data['cctv'] and undo the removal.
                data['cctv'] = None
                data['cctv_marker'] = None
                self._prune_empty_anchors()
        else:
            data = self._next_or_new_slot('cctv')
            data['cctv'] = (x, y)
        self._refresh_markers()

    def _on_sat_right_clicked(self, x, y, hit_marker):
        if hit_marker is not None:
            data = self._find_anchor_for_marker('sat', hit_marker)
            if data is not None:
                data['sat'] = None
                data['sat_marker'] = None
                self._prune_empty_anchors()
        else:
            data = self._next_or_new_slot('sat')
            data['sat'] = (x, y)
        self._refresh_markers()

    def _sync_anchors_from_markers(self):
        """Markers are draggable (CrosshairMarker), so their on-screen
        position can be ahead of self.anchors[i]['cctv'/'sat'] -- pull the
        live position back before anything reads or rebuilds from the
        anchors list (compute, or _refresh_markers itself when it's about
        to tear down and recreate the marker items)."""
        for data in self.anchors:
            if data['cctv_marker'] is not None:
                data['cctv'] = data['cctv_marker'].scene_xy()
            if data['sat_marker'] is not None:
                data['sat'] = data['sat_marker'].scene_xy()

    def _refresh_markers(self):
        self._sync_anchors_from_markers()
        self._clear_scene_markers(self.view_cctv)
        self._clear_scene_markers(self.view_sat)

        for data in self.anchors:
            label = str(data['id'] + 1)
            data['cctv_marker'] = None
            data['sat_marker'] = None
            if data['cctv']:
                data['cctv_marker'] = self._add_marker(self.view_cctv, data['cctv'], label)
            if data['sat']:
                data['sat_marker'] = self._add_marker(self.view_sat, data['sat'], label)

    def _clear_scene_markers(self, viewer):
        scene = viewer.scene()
        for item in scene.items():
            if isinstance(item, QGraphicsPixmapItem):
                continue
            if item.parentItem() is not None:
                # A marker's label is a child item -- already removed along
                # with its parent marker; removing it again would double-pop.
                continue
            scene.removeItem(item)

    def _add_marker(self, viewer, xy, label):
        marker = CrosshairMarker(xy[0], xy[1], MARKER_COLOR, label, movable=True)
        viewer.scene().addItem(marker)
        return marker

    def _on_compute(self):
        if not HAS_CV2: return

        self._sync_anchors_from_markers()
        valid_anchors = [a for a in self.anchors if a['cctv'] is not None and a['sat'] is not None]

        if len(valid_anchors) < MIN_ANCHORS:
            QMessageBox.critical(
                self, "Critical Error",
                f"請在兩邊各放好至少 {MIN_ANCHORS} 個對應點（目前只有 {len(valid_anchors)} 組完整）。",
            )
            return

        use_ransac = self.chk_ransac.isChecked()

        try:
            src_pts = np.array([p['cctv'] for p in valid_anchors], dtype=np.float32).reshape(-1, 1, 2)
            dst_pts = np.array([p['sat'] for p in valid_anchors], dtype=np.float32).reshape(-1, 1, 2)

            method = cv2.RANSAC if use_ransac else 0
            H, mask = cv2.findHomography(src_pts, dst_pts, method, 5.0)

            if H is not None:
                host = getattr(self, 'host_tab', None) or self.parent()
                if host and getattr(host, 'inspect_obj', None):
                    obj = host.inspect_obj
                    hom = obj.setdefault('homography', {})
                    hom['H'] = H.tolist()

                    hom['anchors_list'] = [
                        {
                            "id": data['id'],
                            "coords_cctv": list(data['cctv']) if data['cctv'] else None,
                            "coords_sat": list(data['sat']) if data['sat'] else None,
                        }
                        for data in self.anchors
                    ]

                    host.inspect_obj = obj

                total = len(valid_anchors)
                msg_icon = QMessageBox.Information

                if use_ransac:
                    # Only meaningful with RANSAC -- without it every point
                    # is used as-is (mask.ravel().sum() would just equal
                    # total, since there's no outlier concept to report).
                    inliers = int(mask.ravel().sum()) if mask is not None else total
                    report_lines = [f"RANSAC Inliers: {inliers} / {total}"]
                    if inliers < MIN_ANCHORS:
                        report_lines.append("⚠️ WARNING: RANSAC found fewer than the minimum required good points.\nYour projection may be unstable.")
                        msg_icon = QMessageBox.Warning
                    elif inliers < total:
                        report_lines.append(f"ℹ️ Note: {total - inliers} points were ignored as outliers (likely imprecise clicks).")
                    else:
                        report_lines.append("✅ Excellent: All points fit the model.")
                else:
                    report_lines = [
                        f"Homography computed from all {total} points (RANSAC disabled).",
                        "⚠️ No outlier rejection -- a single mis-clicked point can skew the whole fit.",
                    ]

                report_text = "\n".join(report_lines)
                if msg_icon == QMessageBox.Warning:
                    QMessageBox.warning(self, "Homography Report", report_text)
                else:
                    QMessageBox.information(self, "Homography Report", report_text)

            else:
                QMessageBox.critical(self, "Calculation Failed", "❌ CRITICAL: Homography calculation failed.\nPoints might be collinear.")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_proceed(self):
        host = getattr(self, 'host_tab', None) or self.parent()
        if host and hasattr(host, 'current_step_index'):
            try:
                host.current_step_index = 5
                host._show_stage(5)
                host._update_progress_to_index(5)
            except: pass

    def _cv_to_qimage(self, cv_bgr):
        if cv_bgr is None: return None
        rgb = cv_bgr[:, :, ::-1]
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        return qimg.copy()
