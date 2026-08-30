"""Standalone car-template GCP calibration tool.

Combines Method 4 (車輛模板作為 PnP GCP) and Method 7 (多參考物最小二乘法) from
docs/height-correction-algorithm-survey.md: instead of hand-dragging a
head/foot pixel pair per reference object (reference_point_calibration_tool.py),
the user clicks 2 tire ground-contact points on a CCTV frame and labels which
2 wheels they are. A rigid similarity transform (rotation + scale +
translation, exact for 2 point correspondences -- see
trafficlab.projection.car_template_placement) places a full car keypoint
template onto the sat plane, and every template keypoint's true ground
position/height is paired with that same car's actual OpenPifPaf pixel
detections (from an existing keypoints-openpifpaf replay JSON's kp_cctv) to
produce a batch of reference-point observations in one step. These
accumulate across frames/cars ("placements") and are solved jointly via the
existing trafficlab.projection.reference_point_calibration.calibrate().

Independent of the CalibrationTab wizard; does not modify
G_projection_<code>.json unless the user explicitly presses "套用到
G_projection". See docs/car-template-calibration-tool-guide.md for usage.
"""
from __future__ import annotations

import copy
import json
import math
import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QImage, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsItem,
    QGraphicsLineItem,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSlider,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from trafficlab.gui.tabs.calibration_stage.undistort_stage import ImageViewer
from trafficlab.gui.tools.reference_point_calibration_tool import (
    CrosshairMarker,
    DraggableMarkerViewer,
)
from trafficlab.visualization.video_player import VideoPlayer
from trafficlab.projection.g_projection import GProjection
from trafficlab.projection.reference_point_calibration import ReferencePoint, calibrate
from trafficlab.projection.car_template_placement import (
    CarTemplate,
    TemplatePose,
    load_car_template,
    fit_two_point_pose,
    apply_pose_to_points,
    apply_pose_to_template,
    scaled_heights_m,
    anchor_fit_report,
    build_reference_points,
)
from trafficlab.projection.parallax_reprojection import (
    find_frame,
    compute_frame_records,
    compute_view_extent,
    iter_frame_records,
    plot_frame_pre_post_keypoints,
)
from trafficlab.trajectory.io import load_json
from trafficlab.trajectory.plotting import _kp_part_color
from trafficlab.motion.keypoints_openpifpaf import KP_NAMES

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMPLATE_PATH = REPO_ROOT / "cad_models" / "nissan_juke_nismo" / "keypoint_template_nissan_juke_nismo.json"

WHEEL_NAMES = [
    "front_wheel_center_left",
    "front_wheel_center_right",
    "rear_wheel_center_left",
    "rear_wheel_center_right",
]

ANCHOR_COLOR = Qt.cyan
FOOT_COLOR = QColor("#33cc66")
PRED_COLOR = QColor("#3399ff")
DETECTED_COLOR = QColor("#ff5533")
LINK_COLOR = QColor("#ffcc00")
COMPUTED_CAM_COLOR = Qt.yellow
EXISTING_CAM_COLOR = Qt.magenta
BOX_COLOR = QColor("#3399ff")
BOX_SELECTED_COLOR = Qt.yellow

TRANSLATE_STEP_M = 0.05
TRANSLATE_STEP_M_FAST = 0.5
ROTATE_STEP_DEG = 0.5
ROTATE_STEP_DEG_FAST = 5.0

MARK_TAB_INDEX = 0
NUDGE_TAB_INDEX = 1
REVIEW_TAB_INDEX = 2
RESULT_TAB_INDEX = 3
VERIFY_TAB_INDEX = 4


class TemplateOverlayItem(QGraphicsItem):
    """Renders a placed template's full keypoint set on top of the CCTV
    frame, in CCTV pixel space: ground-footprint dots (where the tool thinks
    each keypoint's contact point is), predicted apparent pixel positions
    (where that keypoint *should* appear given the currently loaded
    parallax params), and the actual kp_cctv detections, connected by a
    line -- that connector length is the residual the least-squares solve
    is trying to minimize."""

    def __init__(self):
        super().__init__()
        self.setZValue(6)
        self._foot_px: Optional[np.ndarray] = None
        self._pred_px: Optional[np.ndarray] = None
        self._det_px: Optional[np.ndarray] = None
        self._detected_mask: Optional[np.ndarray] = None

    def set_geometry(self, foot_px: np.ndarray, pred_px: np.ndarray,
                      det_px: np.ndarray, detected_mask: np.ndarray):
        self.prepareGeometryChange()
        self._foot_px = foot_px
        self._pred_px = pred_px
        self._det_px = det_px
        self._detected_mask = detected_mask
        self.update()

    def clear_geometry(self):
        self.prepareGeometryChange()
        self._foot_px = None
        self._pred_px = None
        self._det_px = None
        self._detected_mask = None
        self.update()

    def boundingRect(self) -> QRectF:
        if self._foot_px is None:
            return QRectF()
        pts = [self._foot_px]
        if self._pred_px is not None:
            pts.append(self._pred_px)
        if self._det_px is not None and self._detected_mask is not None and self._detected_mask.any():
            pts.append(self._det_px[self._detected_mask])
        all_pts = np.vstack(pts)
        margin = 20.0
        x0, y0 = all_pts.min(axis=0) - margin
        x1, y1 = all_pts.max(axis=0) + margin
        return QRectF(x0, y0, x1 - x0, y1 - y0)

    def paint(self, painter, option, widget=None):
        if self._foot_px is None:
            return
        r = 3.0

        pen_foot = QPen(FOOT_COLOR, 1)
        pen_foot.setCosmetic(True)
        painter.setPen(pen_foot)
        for x, y in self._foot_px:
            painter.drawEllipse(QPointF(x, y), r, r)

        if self._pred_px is None or self._det_px is None or self._detected_mask is None:
            return
        pen_link = QPen(LINK_COLOR, 1, Qt.DotLine)
        pen_link.setCosmetic(True)
        pen_pred = QPen(PRED_COLOR, 1)
        pen_pred.setCosmetic(True)
        pen_det = QPen(DETECTED_COLOR, 1)
        pen_det.setCosmetic(True)
        for i in range(len(self._foot_px)):
            if not self._detected_mask[i]:
                continue
            px, py = self._pred_px[i]
            dx, dy = self._det_px[i][0], self._det_px[i][1]
            painter.setPen(pen_link)
            painter.drawLine(QPointF(px, py), QPointF(dx, dy))
            painter.setPen(pen_pred)
            painter.drawEllipse(QPointF(px, py), r, r)
            painter.setPen(pen_det)
            painter.drawEllipse(QPointF(dx, dy), r, r)


class TopDownTemplateOverlayItem(QGraphicsItem):
    """Renders a placed car template as a simple bird's-eye body outline +
    wheel markers directly on the sat image -- looking straight down, the
    way the car actually sits on the map -- instead of the CCTV predicted
    vs. detected keypoint dots (TemplateOverlayItem). Used in the 微調 tab
    where the user nudges the fitted pose."""

    def __init__(self):
        super().__init__()
        self.setZValue(6)
        self._corners: Optional[np.ndarray] = None      # (4, 2) sat px, body outline
        self._front_center: Optional[np.ndarray] = None  # (2,) sat px
        self._wheel_pts: Optional[np.ndarray] = None      # (K, 2) sat px
        self._is_anchor: Optional[np.ndarray] = None      # (K,) bool

    def set_geometry(self, corners: np.ndarray, front_center: np.ndarray,
                      wheel_pts: np.ndarray, is_anchor: np.ndarray):
        self.prepareGeometryChange()
        self._corners = corners
        self._front_center = front_center
        self._wheel_pts = wheel_pts
        self._is_anchor = is_anchor
        self.update()

    def clear_geometry(self):
        self.prepareGeometryChange()
        self._corners = None
        self._front_center = None
        self._wheel_pts = None
        self._is_anchor = None
        self.update()

    def boundingRect(self) -> QRectF:
        if self._corners is None:
            return QRectF()
        pts = [self._corners]
        if self._wheel_pts is not None and len(self._wheel_pts):
            pts.append(self._wheel_pts)
        if self._front_center is not None:
            pts.append(self._front_center.reshape(1, 2))
        all_pts = np.vstack(pts)
        margin = 15.0
        x0, y0 = all_pts.min(axis=0) - margin
        x1, y1 = all_pts.max(axis=0) + margin
        return QRectF(float(x0), float(y0), float(x1 - x0), float(y1 - y0))

    def paint(self, painter, option, widget=None):
        if self._corners is None:
            return

        body_pen = QPen(QColor("#ffffff"), 2)
        body_pen.setCosmetic(True)
        painter.setPen(body_pen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 40)))
        painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in self._corners]))

        if self._front_center is not None:
            center = self._corners.mean(axis=0)
            front_pen = QPen(QColor("#ff5533"), 2)
            front_pen.setCosmetic(True)
            painter.setPen(front_pen)
            painter.drawLine(
                QPointF(float(center[0]), float(center[1])),
                QPointF(float(self._front_center[0]), float(self._front_center[1])),
            )

        if self._wheel_pts is not None:
            r = 4.0
            for (x, y), is_anchor in zip(self._wheel_pts, self._is_anchor):
                color = ANCHOR_COLOR if is_anchor else QColor("#33cc66")
                pen = QPen(color, 1)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QPointF(float(x), float(y)), r, r)


class WheelAnchorMarker(CrosshairMarker):
    """CrosshairMarker plus a full-length dashed vertical guide line behind
    the crosshair (both above and below, unlike CrosshairMarker's own
    downward-only guide meant for a separate head/foot pair) -- helps judge
    whether the click lines up with the wheel/wheel-arch silhouette visible
    above the tire's ground-contact point. The line is a child item, so it
    tracks the marker while it's dragged."""

    def __init__(self, x, y, color, label, guide_length, movable=True):
        super().__init__(x, y, color, label, movable=movable, guide_length=guide_length)
        pen = QPen(color, 1, Qt.DashLine)
        pen.setCosmetic(True)
        size = self._size
        guide_up = QGraphicsLineItem(0, -size, 0, -size - guide_length, self)
        guide_up.setPen(pen)
        guide_up.setZValue(-1)
        guide_up.setAcceptedMouseButtons(Qt.NoButton)


class TemplatePlacementViewer(DraggableMarkerViewer):
    """Adds keyboard nudge (translate/rotate/flip/refit) on top of
    DraggableMarkerViewer's anchor-dragging + wheel-zoom. The widget owning
    this viewer wires nudge_requested-style calls through the callbacks
    below rather than a Qt signal, since the amount of accompanying state
    (current pose, template, nominal px_per_m) lives on the widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.on_translate = None  # callback(dx_m, dy_m)
        self.on_rotate = None     # callback(d_theta_rad)
        self.on_flip = None       # callback()
        self.on_refit = None      # callback()

    def keyPressEvent(self, event):
        key = event.key()
        fast = bool(event.modifiers() & Qt.ShiftModifier)
        t_step = TRANSLATE_STEP_M_FAST if fast else TRANSLATE_STEP_M
        r_step = math.radians(ROTATE_STEP_DEG_FAST if fast else ROTATE_STEP_DEG)

        if key == Qt.Key_Left and self.on_translate:
            self.on_translate(-t_step, 0.0)
        elif key == Qt.Key_Right and self.on_translate:
            self.on_translate(t_step, 0.0)
        elif key == Qt.Key_Up and self.on_translate:
            self.on_translate(0.0, -t_step)
        elif key == Qt.Key_Down and self.on_translate:
            self.on_translate(0.0, t_step)
        elif key == Qt.Key_Q and self.on_rotate:
            self.on_rotate(-r_step)
        elif key == Qt.Key_E and self.on_rotate:
            self.on_rotate(r_step)
        elif key == Qt.Key_F and self.on_flip:
            self.on_flip()
        elif key == Qt.Key_R and self.on_refit:
            self.on_refit()
        else:
            super().keyPressEvent(event)
            return
        event.accept()


@dataclass
class Placement:
    id: int
    frame_index: int
    obj_id: int
    tracked_id: Optional[int]  # may be None -- this frame's detection wasn't tracked
    anchor_names: tuple[str, str]
    anchor_px: tuple[tuple[float, float], tuple[float, float]]
    pose: TemplatePose
    template_path: str
    kp_cctv: list
    nominal_px_per_m: float
    kp_conf: float
    min_height_m: float


class CarTemplateCalibrationWidget(QWidget):
    def __init__(self, initial_replay_json: Optional[str] = None,
                 initial_g_proj: Optional[str] = None, parent=None):
        super().__init__(parent)

        self._video_player: Optional[VideoPlayer] = None
        self._video_path: Optional[str] = None
        self._current_frame_index: Optional[int] = None

        self._replay_json_path: Optional[str] = None
        self._replay_data: Optional[dict] = None
        self._replay_location_code: Optional[str] = None

        self._template_path: Optional[str] = None
        self._template: Optional[CarTemplate] = None

        self._g_proj_path: Optional[Path] = None
        self._location_code: Optional[str] = None
        self._g_projection: Optional[GProjection] = None
        self._sat_image_path: Optional[str] = None
        self._sat_bg_loaded_path: Optional[str] = None

        # Object selection is keyed on the per-detection 'id' field, not
        # 'tracked_id' -- tracked_id can legitimately be None for multiple
        # objects in the same frame (untracked detections, e.g. --localizer
        # wheel_pair output or --yolo "" runs), so it can't identify which
        # object is selected. tracked_id is kept only for display/output.
        self._current_obj_id: Optional[int] = None
        self._current_tracked_id: Optional[int] = None
        self._current_kp_cctv: Optional[list] = None
        # Selecting a car from the dropdown only stages it -- boxes for
        # every detected car stay visible until the user presses "確定選擇",
        # so wheel-marking can't start against a car the user never
        # deliberately confirmed.
        self._selection_confirmed: bool = False

        self._pending: Optional[dict] = None
        self._current_pose: Optional[TemplatePose] = None
        self._fit_anchors: Optional[list] = None
        self._nudge_anchor_markers: list = []

        self.placements: list[Placement] = []
        self._next_placement_id = 1
        self._last_result: Optional[dict] = None

        self._build_ui()

        if initial_g_proj:
            self.edit_g_proj_path.setText(initial_g_proj)
            self._load_g_proj(initial_g_proj)
        self.edit_template_path.setText(str(DEFAULT_TEMPLATE_PATH))
        self._load_template(str(DEFAULT_TEMPLATE_PATH))
        if initial_replay_json:
            self.edit_replay_path.setText(initial_replay_json)
            self._on_load_replay()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.tabs.addTab(self._build_mark_tab(), "1. 標記")
        self.tabs.addTab(self._build_nudge_tab(), "2. 微調")
        self.tabs.addTab(self._build_review_tab(), "3. 確認")
        self.tabs.addTab(self._build_result_tab(), "4. 結果")
        self.tabs.addTab(self._build_verify_tab(), "5. 驗證")

    # -- Tab 1: mark -------------------------------------------------
    def _build_mark_tab(self) -> QWidget:
        page = QWidget()
        main_layout = QHBoxLayout(page)

        splitter = QSplitter(Qt.Horizontal)
        # This viewer is for clicking/dragging the 2 wheel anchors on the
        # CCTV frame only -- nudging the fitted pose happens on the sat-plane
        # viewer in the 微調 tab (viewer_nudge), not here, so no on_translate
        # etc. callbacks are wired on this one.
        self.viewer_mark = TemplatePlacementViewer()
        self._detection_overlay_items: list = []
        splitter.addWidget(self.viewer_mark)

        sidebar = QWidget()
        side_vbox = QVBoxLayout(sidebar)
        side_vbox.setSpacing(10)
        side_vbox.addWidget(self._build_replay_group())
        side_vbox.addWidget(self._build_template_group())
        side_vbox.addWidget(self._build_g_proj_group())
        side_vbox.addWidget(self._build_frame_group())
        side_vbox.addWidget(self._build_wheel_group())
        side_vbox.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(sidebar)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(440)

        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        main_layout.addWidget(splitter)
        return page

    def _build_replay_group(self) -> QGroupBox:
        grp = QGroupBox("Replay JSON")
        layout = QVBoxLayout(grp)
        lbl_hint = QLabel("需含 kp_cctv，即 run_keypoints_openpifpaf.py 的輸出：")
        lbl_hint.setWordWrap(True)
        layout.addWidget(lbl_hint)
        row = QHBoxLayout()
        self.edit_replay_path = QLineEdit()
        btn_browse = QPushButton("瀏覽…")
        btn_browse.clicked.connect(self._on_browse_replay)
        row.addWidget(self.edit_replay_path)
        row.addWidget(btn_browse)
        layout.addLayout(row)
        btn_load = QPushButton("載入 replay JSON")
        btn_load.clicked.connect(self._on_load_replay)
        layout.addWidget(btn_load)
        self.lbl_replay_status = QLabel("尚未載入。")
        self.lbl_replay_status.setWordWrap(True)
        layout.addWidget(self.lbl_replay_status)
        return grp

    def _build_template_group(self) -> QGroupBox:
        grp = QGroupBox("CAD 模板 JSON")
        layout = QVBoxLayout(grp)
        row = QHBoxLayout()
        self.edit_template_path = QLineEdit()
        btn_browse = QPushButton("瀏覽…")
        btn_browse.clicked.connect(self._on_browse_template)
        row.addWidget(self.edit_template_path)
        row.addWidget(btn_browse)
        layout.addLayout(row)
        btn_load = QPushButton("載入模板")
        btn_load.clicked.connect(lambda: self._load_template(self.edit_template_path.text().strip()))
        layout.addWidget(btn_load)
        self.lbl_template_status = QLabel("尚未載入。")
        self.lbl_template_status.setWordWrap(True)
        layout.addWidget(self.lbl_template_status)
        return grp

    def _build_g_proj_group(self) -> QGroupBox:
        grp = QGroupBox("G_projection.json")
        layout = QVBoxLayout(grp)
        row = QHBoxLayout()
        self.edit_g_proj_path = QLineEdit()
        btn_browse = QPushButton("瀏覽…")
        btn_browse.clicked.connect(self._on_browse_g_proj)
        row.addWidget(self.edit_g_proj_path)
        row.addWidget(btn_browse)
        layout.addLayout(row)
        self.lbl_g_proj_status = QLabel("尚未載入。")
        self.lbl_g_proj_status.setWordWrap(True)
        layout.addWidget(self.lbl_g_proj_status)
        return grp

    def _build_frame_group(self) -> QGroupBox:
        grp = QGroupBox("幀 / 車輛")
        layout = QVBoxLayout(grp)

        self.lbl_frame_info = QLabel("尚未載入影片")
        self.lbl_frame_info.setWordWrap(True)
        layout.addWidget(self.lbl_frame_info)

        frame_row = QHBoxLayout()
        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(0, 0)
        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setRange(0, 0)
        self.spin_frame.valueChanged.connect(self._sync_slider_from_spin)
        self.spin_frame.valueChanged.connect(self._on_frame_spin_changed)
        self.slider_frame.valueChanged.connect(self._sync_spin_from_slider)
        # _sync_spin_from_slider blocks spin_frame's own signals while it
        # syncs the displayed number (to avoid a signal loop), so dragging
        # the slider alone would never reach _on_frame_spin_changed -- wire
        # it directly here too. Order matters: this runs after the sync
        # above, so spin_frame.value() is already up to date by the time
        # _on_frame_spin_changed reads it.
        self.slider_frame.valueChanged.connect(self._on_frame_spin_changed)
        frame_row.addWidget(self.spin_frame)
        frame_row.addWidget(self.slider_frame, 1)
        layout.addLayout(frame_row)

        layout.addWidget(QLabel("這一幀的車輛（tracked_id）："))
        self.combo_tracked_id = QComboBox()
        self.combo_tracked_id.currentIndexChanged.connect(self._on_tracked_id_changed)
        layout.addWidget(self.combo_tracked_id)

        self.btn_confirm_car_selection = QPushButton("確定選擇")
        self.btn_confirm_car_selection.clicked.connect(self._on_confirm_car_selection)
        layout.addWidget(self.btn_confirm_car_selection)

        return grp

    def _build_wheel_group(self) -> QGroupBox:
        grp = QGroupBox("輪胎接地點標記")
        layout = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("點 1："))
        self.combo_wheel1 = QComboBox()
        self.combo_wheel1.addItems(WHEEL_NAMES)
        self.combo_wheel1.setMaximumWidth(230)
        row1.addWidget(self.combo_wheel1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("點 2："))
        self.combo_wheel2 = QComboBox()
        self.combo_wheel2.addItems(WHEEL_NAMES)
        self.combo_wheel2.setCurrentIndex(1)
        self.combo_wheel2.setMaximumWidth(230)
        row2.addWidget(self.combo_wheel2)
        layout.addLayout(row2)

        self.btn_add_wheels = QPushButton("標記兩個輪胎點")
        self.btn_add_wheels.clicked.connect(self._on_add_wheels_clicked)
        layout.addWidget(self.btn_add_wheels)

        row = QHBoxLayout()
        self.btn_done_wheels = QPushButton("完成（自動擬合）")
        self.btn_done_wheels.setEnabled(False)
        self.btn_done_wheels.clicked.connect(self._on_done_wheels_clicked)
        self.btn_cancel_wheels = QPushButton("取消")
        self.btn_cancel_wheels.setEnabled(False)
        self.btn_cancel_wheels.clicked.connect(self._on_cancel_wheels_clicked)
        row.addWidget(self.btn_done_wheels)
        row.addWidget(self.btn_cancel_wheels)
        layout.addLayout(row)

        self.lbl_wheel_status = QLabel("請先載入 replay JSON、模板、G_projection，選好這一幀的車輛，再按「標記兩個輪胎點」。")
        self.lbl_wheel_status.setWordWrap(True)
        layout.addWidget(self.lbl_wheel_status)

        return grp

    # -- Tab 2: nudge -----------------------------------------------------
    def _build_nudge_tab(self) -> QWidget:
        page = QWidget()
        main_layout = QHBoxLayout(page)

        splitter = QSplitter(Qt.Horizontal)
        # Read-only CCTV reference panel: the current frame with the
        # confirmed car's keypoints and the two locked wheel-anchor marks,
        # so the user has something to cross-check the sat-plane top-down
        # car icon against while nudging.
        self.viewer_nudge_cctv = ImageViewer()
        splitter.addWidget(self.viewer_nudge_cctv)

        # Sat-plane viewer: shows the placed template as a top-down car
        # icon (looking straight down), and owns keyboard nudging.
        self.viewer_nudge = TemplatePlacementViewer()
        self.viewer_nudge.on_translate = self._on_nudge_translate
        self.viewer_nudge.on_rotate = self._on_nudge_rotate
        self.viewer_nudge.on_flip = self._on_flip_180
        self.viewer_nudge.on_refit = self._on_refit
        self._topdown_overlay_item: Optional[TopDownTemplateOverlayItem] = None
        splitter.addWidget(self.viewer_nudge)

        sidebar = QWidget()
        side_vbox = QVBoxLayout(sidebar)
        side_vbox.setSpacing(10)
        side_vbox.addWidget(self._build_filter_group())
        side_vbox.addWidget(self._build_nudge_group())

        btn_row = QHBoxLayout()
        self.btn_confirm_placement = QPushButton("確認加入列表")
        self.btn_confirm_placement.clicked.connect(self._on_confirm_placement)
        self.btn_cancel_placement = QPushButton("取消此次放置")
        self.btn_cancel_placement.clicked.connect(self._on_cancel_wheels_clicked)
        btn_row.addWidget(self.btn_confirm_placement)
        btn_row.addWidget(self.btn_cancel_placement)
        side_vbox.addLayout(btn_row)

        side_vbox.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(sidebar)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(300)

        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        main_layout.addWidget(splitter)
        return page

    def _build_filter_group(self) -> QGroupBox:
        grp = QGroupBox("參考點篩選 / 擬合資訊")
        layout = QVBoxLayout(grp)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("關鍵點信心度閾值："))
        self.spin_kp_conf = QDoubleSpinBox()
        self.spin_kp_conf.setRange(0.0, 1.0)
        self.spin_kp_conf.setSingleStep(0.05)
        self.spin_kp_conf.setValue(0.2)
        self.spin_kp_conf.valueChanged.connect(self._on_kp_conf_changed)
        conf_row.addWidget(self.spin_kp_conf)
        layout.addLayout(conf_row)

        height_row = QHBoxLayout()
        height_row.addWidget(QLabel("最低真實高度 (m)："))
        self.spin_min_height = QDoubleSpinBox()
        self.spin_min_height.setRange(0.0, 5.0)
        self.spin_min_height.setSingleStep(0.05)
        self.spin_min_height.setValue(0.0)
        self.spin_min_height.valueChanged.connect(self._on_kp_conf_changed)
        height_row.addWidget(self.spin_min_height)
        layout.addLayout(height_row)

        self.lbl_fit_report = QLabel("")
        self.lbl_fit_report.setWordWrap(True)
        layout.addWidget(self.lbl_fit_report)

        self.lbl_ref_count = QLabel("")
        self.lbl_ref_count.setWordWrap(True)
        layout.addWidget(self.lbl_ref_count)

        return grp

    def _build_nudge_group(self) -> QGroupBox:
        grp = QGroupBox("微調（僅旋轉／平移，不含縮放）")
        layout = QVBoxLayout(grp)
        lbl_nudge_help = QLabel(
            "方向鍵：平移 0.05m（Shift+方向鍵 0.5m）　Q/E：旋轉 0.5°（Shift+Q/E 5°）\n"
            "F：翻轉 180°　R：重新自動擬合\n"
            "（點一下左邊的圖讓它取得鍵盤焦點，才能用鍵盤微調）"
        )
        lbl_nudge_help.setWordWrap(True)
        layout.addWidget(lbl_nudge_help)

        row1 = QHBoxLayout()
        for label, dx, dy in (("←", -TRANSLATE_STEP_M, 0), ("↑", 0, -TRANSLATE_STEP_M),
                               ("↓", 0, TRANSLATE_STEP_M), ("→", TRANSLATE_STEP_M, 0)):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, dx=dx, dy=dy: self._on_nudge_translate(dx, dy))
            row1.addWidget(btn)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        btn_ccw = QPushButton("↺ 旋轉")
        btn_ccw.clicked.connect(lambda: self._on_nudge_rotate(-math.radians(ROTATE_STEP_DEG)))
        btn_cw = QPushButton("旋轉 ↻")
        btn_cw.clicked.connect(lambda: self._on_nudge_rotate(math.radians(ROTATE_STEP_DEG)))
        row2.addWidget(btn_ccw)
        row2.addWidget(btn_cw)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        btn_flip = QPushButton("翻轉 180°")
        btn_flip.clicked.connect(self._on_flip_180)
        btn_refit = QPushButton("重新自動擬合")
        btn_refit.clicked.connect(self._on_refit)
        row3.addWidget(btn_flip)
        row3.addWidget(btn_refit)
        layout.addLayout(row3)

        return grp

    # -- Tab 3: review --------------------------------------------------
    def _build_review_tab(self) -> QWidget:
        page = QWidget()
        main_layout = QHBoxLayout(page)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_vbox = QVBoxLayout(left)
        left_vbox.addWidget(QLabel("已加入的放置（點選可預覽）："))
        self.list_placements = QListWidget()
        self.list_placements.currentItemChanged.connect(self._on_placement_selection_changed)
        left_vbox.addWidget(self.list_placements)

        self.btn_delete_placement = QPushButton("刪除選取的放置")
        self.btn_delete_placement.clicked.connect(self._on_delete_placement)
        left_vbox.addWidget(self.btn_delete_placement)

        self.btn_compute = QPushButton("計算")
        self.btn_compute.clicked.connect(self._on_compute)
        left_vbox.addWidget(self.btn_compute)

        left.setMaximumWidth(360)
        splitter.addWidget(left)

        self.viewer_review = ImageViewer()
        splitter.addWidget(self.viewer_review)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)
        return page

    # -- Tab 4: result ----------------------------------------------------
    def _build_result_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.text_result_summary = QTextEdit()
        self.text_result_summary.setReadOnly(True)
        self.text_result_summary.setMaximumHeight(180)
        layout.addWidget(self.text_result_summary)

        layout.addWidget(QLabel("Sat 平面：黃色叉叉＝這次計算出的相機位置，洋紅色叉叉＝目前 G_projection 裡既有的相機位置。"))
        self.viewer_result = ImageViewer()
        layout.addWidget(self.viewer_result, 1)

        btn_row = QHBoxLayout()
        self.btn_save_json = QPushButton("另存新檔（獨立 JSON）")
        self.btn_save_json.clicked.connect(self._on_save_json)
        self.btn_apply_g_proj = QPushButton("套用到 G_projection")
        self.btn_apply_g_proj.clicked.connect(self._on_apply_to_g_proj)
        btn_row.addWidget(self.btn_save_json)
        btn_row.addWidget(self.btn_apply_g_proj)
        layout.addLayout(btn_row)

        return page

    # -- Tab 5: verify ------------------------------------------------
    def _build_verify_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        settings_grp = QGroupBox("比較設定")
        settings_layout = QVBoxLayout(settings_grp)

        before_row = QHBoxLayout()
        before_row.addWidget(QLabel("修正前 G_projection JSON："))
        self.edit_verify_gproj_before = QLineEdit()
        before_row.addWidget(self.edit_verify_gproj_before)
        btn_browse_before = QPushButton("瀏覽…")
        btn_browse_before.clicked.connect(self._on_browse_verify_gproj_before)
        before_row.addWidget(btn_browse_before)
        settings_layout.addLayout(before_row)

        after_row = QHBoxLayout()
        after_row.addWidget(QLabel("修正後 G_projection JSON："))
        self.edit_verify_gproj_after = QLineEdit()
        self.edit_verify_gproj_after.setPlaceholderText("留空＝套用「確認」tab 本次計算出的結果")
        after_row.addWidget(self.edit_verify_gproj_after)
        btn_browse_after = QPushButton("瀏覽…")
        btn_browse_after.clicked.connect(self._on_browse_verify_gproj_after)
        after_row.addWidget(btn_browse_after)
        settings_layout.addLayout(after_row)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("比較幀："))
        self.spin_verify_frame = QSpinBox()
        self.spin_verify_frame.setRange(0, 0)
        filter_row.addWidget(self.spin_verify_frame)
        btn_auto_frame = QPushButton("自動挑選（參考點最多）")
        btn_auto_frame.clicked.connect(self._on_verify_auto_pick_frame)
        filter_row.addWidget(btn_auto_frame)
        filter_row.addSpacing(20)
        filter_row.addWidget(QLabel("篩選 tracked_id："))
        self.edit_verify_tid = QLineEdit("5")
        self.edit_verify_tid.setPlaceholderText("留空＝不篩選（全部車輛）")
        self.edit_verify_tid.setMaximumWidth(80)
        filter_row.addWidget(self.edit_verify_tid)
        filter_row.addStretch()
        settings_layout.addLayout(filter_row)

        layout.addWidget(settings_grp)

        top_row = QHBoxLayout()
        self.btn_generate_verify = QPushButton("產生比較圖")
        self.btn_generate_verify.clicked.connect(self._on_generate_verify)
        top_row.addWidget(self.btn_generate_verify)
        top_row.addStretch()
        layout.addLayout(top_row)

        self.lbl_verify_status = QLabel(
            "尚未產生。「修正後」留空時會用「確認」tab 本次計算出的結果；"
            "也可以直接指定兩個既有的 G_projection 檔案來比較。"
        )
        self.lbl_verify_status.setWordWrap(True)
        layout.addWidget(self.lbl_verify_status)

        images_row = QHBoxLayout()
        before_box = QVBoxLayout()
        before_box.addWidget(QLabel("修正前"))
        self.viewer_verify_before = ImageViewer()
        before_box.addWidget(self.viewer_verify_before)
        after_box = QVBoxLayout()
        after_box.addWidget(QLabel("修正後"))
        self.viewer_verify_after = ImageViewer()
        after_box.addWidget(self.viewer_verify_after)
        images_row.addLayout(before_box)
        images_row.addLayout(after_box)
        layout.addLayout(images_row, 1)

        return page

    # ------------------------------------------------------------------
    # Loading: replay JSON / template / G_projection
    # ------------------------------------------------------------------
    def _on_browse_replay(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇 replay JSON", "", "Replay JSON (*.json *.json.gz)")
        if path:
            self.edit_replay_path.setText(path)
            self._on_load_replay()

    def _on_load_replay(self):
        path = self.edit_replay_path.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "錯誤", "找不到 replay JSON 檔案。")
            return
        try:
            data = load_json(path)
        except Exception as e:
            QMessageBox.warning(self, "錯誤", f"無法讀取 replay JSON：{e}")
            return

        self._replay_json_path = path
        self._replay_data = data
        self._replay_location_code = data.get("location_code")

        mp4_path = data.get("mp4_path")
        video_path = None
        if mp4_path:
            candidate = Path(mp4_path)
            if not candidate.is_absolute():
                candidate = REPO_ROOT / candidate
            if candidate.is_file():
                video_path = str(candidate)

        if video_path:
            self._load_video(video_path)
            self.lbl_replay_status.setText(
                f"已載入（location={self._replay_location_code}，{len(data.get('frames', []))} 幀）。"
                f"已自動載入對應影片：{video_path}"
            )
        else:
            self.lbl_replay_status.setText(
                f"已載入（location={self._replay_location_code}，{len(data.get('frames', []))} 幀）。"
                f"找不到 replay JSON 裡記載的影片路徑（{mp4_path}），無法顯示畫面。"
            )
        self._check_consistency()

    def _on_browse_template(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇模板 JSON", "", "JSON (*.json)")
        if path:
            self.edit_template_path.setText(path)
            self._load_template(path)

    def _load_template(self, path: str):
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "錯誤", "找不到模板 JSON 檔案。")
            return
        try:
            template = load_car_template(path)
        except Exception as e:
            QMessageBox.warning(self, "錯誤", f"無法讀取模板：{e}")
            return
        self._template_path = path
        self._template = template
        self.lbl_template_status.setText(f"已載入模板（{len(template.kp_names)} 個關鍵點）：{Path(path).name}")
        self._check_consistency()

    def _on_browse_g_proj(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇 G_projection.json", "", "JSON (*.json)")
        if path:
            self.edit_g_proj_path.setText(path)
            self._load_g_proj(path)

    def _load_g_proj(self, path: str):
        try:
            with open(path) as f:
                g_data = json.load(f)
            g_projection = GProjection(g_data, base_dir=os.path.dirname(path))
        except Exception as e:
            QMessageBox.warning(self, "錯誤", f"無法讀取 G_projection：{e}")
            return
        self._g_proj_path = Path(path)
        self._g_projection = g_projection
        self._location_code = g_data.get("meta", {}).get("location_code")
        sat_rel = g_data.get("inputs", {}).get("sat_path")
        self._sat_image_path = str((Path(path).parent / sat_rel).resolve()) if sat_rel else None
        self.lbl_g_proj_status.setText(f"已載入 G_projection（location={self._location_code}）。")
        self.edit_verify_gproj_before.setText(path)
        self._check_consistency()

    def _check_consistency(self):
        """Cross-check replay JSON's location_code against the loaded
        G_projection, and the template's kp_names ordering against the
        production KP_NAMES constant (kp_cctv is always stored in KP_NAMES
        order -- a mismatched template would silently mis-map every
        keypoint). Hard-blocks the marking controls on mismatch rather than
        producing plausible-looking but wrong data."""
        problems = []
        if self._replay_location_code and self._location_code and self._replay_location_code != self._location_code:
            problems.append(
                f"replay JSON 的 location_code（{self._replay_location_code}）"
                f"跟載入的 G_projection（{self._location_code}）不一致。"
            )
        if self._template is not None and self._template.kp_names != list(KP_NAMES):
            problems.append("模板的 kp_names 順序跟目前版本的 KP_NAMES 不一致（模板可能是舊格式，或關鍵點順序有誤）。")

        ok = not problems
        for w in (self.btn_add_wheels, self.combo_wheel1, self.combo_wheel2):
            w.setEnabled(ok)
        if problems:
            self.lbl_wheel_status.setText("⚠ " + " ".join(problems))
        elif self._replay_data is not None and self._g_projection is not None and self._template is not None:
            self.lbl_wheel_status.setText("已就緒。選好這一幀的車輛，按「標記兩個輪胎點」開始。")
        return ok

    # ------------------------------------------------------------------
    # Video / frame loading
    # ------------------------------------------------------------------
    def _load_video(self, path: str):
        if self._video_player is not None:
            self._video_player.release()
        self._video_player = VideoPlayer(path)
        if not self._video_player.is_opened():
            QMessageBox.warning(self, "錯誤", "無法開啟影片。")
            self._video_player = None
            return
        self._video_path = path
        n = self._video_player.frame_count()
        if n <= 0:
            QMessageBox.warning(self, "錯誤", "無法讀取影片幀數（frame_count <= 0）。")
            return
        self.spin_frame.setRange(0, n - 1)
        self.slider_frame.setRange(0, n - 1)
        self.spin_frame.setValue(0)
        self.spin_verify_frame.setRange(0, n - 1)
        self.lbl_frame_info.setText(
            f"共 {n} 幀，fps={self._video_player.fps():.1f}，解析度={self._video_player.resolution()}"
        )
        self._on_load_frame()

    def _sync_slider_from_spin(self, value: int):
        if self.slider_frame.value() != value:
            self.slider_frame.blockSignals(True)
            self.slider_frame.setValue(value)
            self.slider_frame.blockSignals(False)

    def _sync_spin_from_slider(self, value: int):
        if self.spin_frame.value() != value:
            self.spin_frame.blockSignals(True)
            self.spin_frame.setValue(value)
            self.spin_frame.blockSignals(False)

    def _read_frame_robust(self, idx: int):
        frame = self._video_player.read_frame(idx)
        if frame is not None:
            return frame, False
        self._video_player.release()
        self._video_player = VideoPlayer(self._video_path)
        frame = None
        for _ in range(idx + 1):
            ret, f = self._video_player.read()
            if not ret:
                return None, True
            frame = f
        return frame, True

    def _on_frame_spin_changed(self, value):
        if self._video_player is None:
            return
        self._reload_current_frame()

    def _on_load_frame(self):
        if self._video_player is None:
            QMessageBox.warning(self, "錯誤", "請先載入 replay JSON（會自動載入對應影片）。")
            return
        self._reload_current_frame()

    def _reload_current_frame(self):
        try:
            self._reload_current_frame_unsafe()
        except Exception as e:
            QMessageBox.critical(
                self, "載入這一幀時發生錯誤",
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            )

    def _reload_current_frame_unsafe(self):
        self._discard_pending()
        idx = self.spin_frame.value()
        frame, used_fallback = self._read_frame_robust(idx)
        if frame is None:
            QMessageBox.warning(self, "錯誤", f"無法讀取第 {idx} 幀（seek 與循序讀取皆失敗）。")
            return

        self._current_frame_index = idx
        qimg = self._cv_to_qimage(frame)
        self.viewer_mark.load_pixmap(QPixmap.fromImage(qimg))
        self.viewer_mark.fitToView()
        self._detection_overlay_items = []
        self._selection_confirmed = False

        note = "（此影片 seek 不可靠，已改用循序讀取，可能較慢）" if used_fallback else ""
        self._refresh_tracked_id_combo()
        self.lbl_replay_status.setText(f"目前第 {idx} 幀。{note}")

    def _refresh_tracked_id_combo(self):
        self.combo_tracked_id.blockSignals(True)
        self.combo_tracked_id.clear()
        objs = self._objects_with_keypoints_in_current_frame()
        for obj in objs:
            n_conf = sum(
                1 for kp in (obj.get("kp_cctv") or [])
                if float(kp[2]) >= 0.2 and (float(kp[0]), float(kp[1])) not in {(0.0, 0.0), (0.0, -4.0)}
            )
            tracked_id = obj.get("tracked_id")
            tracked_label = f"tracked_id={tracked_id}" if tracked_id is not None else "未追蹤"
            self.combo_tracked_id.addItem(
                f"#{obj.get('id')}（{tracked_label}，{n_conf} 個高信心度關鍵點）", obj.get("id")
            )
        self.combo_tracked_id.blockSignals(False)
        if self.combo_tracked_id.count() > 0:
            self.combo_tracked_id.setCurrentIndex(0)
            self._on_tracked_id_changed(0)
        else:
            self._current_obj_id = None
            self._current_tracked_id = None
            self._current_kp_cctv = None

    def _objects_with_keypoints_in_current_frame(self) -> list[dict]:
        if self._replay_data is None or self._current_frame_index is None:
            return []
        frame = find_frame(self._replay_data, self._current_frame_index)
        if frame is None:
            return []
        return [o for o in (frame.get("objects") or []) if o.get("kp_cctv")]

    def _on_tracked_id_changed(self, index: int):
        if index < 0:
            self._current_obj_id = None
            self._current_tracked_id = None
            self._current_kp_cctv = None
            self._selection_confirmed = False
            self._draw_frame_detections_overlay()
            return
        self._current_obj_id = self.combo_tracked_id.itemData(index)
        objs = self._objects_with_keypoints_in_current_frame()
        obj = next((o for o in objs if o.get("id") == self._current_obj_id), None)
        self._current_tracked_id = obj.get("tracked_id") if obj else None
        self._current_kp_cctv = obj.get("kp_cctv") if obj else None
        # Picking a different car only stages it -- go back to box mode so
        # the user has to explicitly press "確定選擇" again before its
        # keypoints replace the boxes.
        self._selection_confirmed = False
        self._draw_frame_detections_overlay()

    def _on_confirm_car_selection(self):
        if self._current_obj_id is None:
            QMessageBox.warning(self, "尚未選擇車輛", "這一幀沒有帶 kp_cctv 的車輛可選。")
            return
        self._selection_confirmed = True
        self._draw_frame_detections_overlay()
        self.lbl_wheel_status.setText("已確定選擇車輛，按「標記兩個輪胎點」開始。")

    def _draw_frame_detections_overlay(self):
        """Before the user confirms a car (_selection_confirmed is False),
        draw every detected car's bbox_2d (highlighting whichever one the
        dropdown currently points at) so the user can see which cars are
        available to pick. After "確定選擇" is pressed, switch to the
        confirmed car's full color-coded kp_cctv keypoints instead -- the
        boxes are just clutter once a car is locked in. Runs underneath
        (lower zValue) the wheel-marking crosshairs and the fitted-template
        overlay, so it never blocks them."""
        scene = self.viewer_mark.scene()
        if scene is None:
            return
        for item in self._detection_overlay_items:
            if item.scene() == scene:
                scene.removeItem(item)
        self._detection_overlay_items = []

        if self._selection_confirmed and self._current_kp_cctv is not None:
            kp_conf = self.spin_kp_conf.value()
            for i, kp in enumerate(self._current_kp_cctv):
                x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
                if conf < kp_conf or (x, y) in {(0.0, 0.0), (0.0, -4.0)}:
                    continue
                name = KP_NAMES[i] if i < len(KP_NAMES) else ""
                r, g, b = _kp_part_color(name)
                kp_color = QColor.fromRgbF(r, g, b)
                pen_kp = QPen(kp_color, 1)
                pen_kp.setCosmetic(True)
                dot = scene.addEllipse(x - 3, y - 3, 6, 6, pen_kp, QBrush(kp_color))
                dot.setZValue(5)
                self._detection_overlay_items.append(dot)
            return

        for obj in self._objects_with_keypoints_in_current_frame():
            bbox = obj.get("bbox_2d")
            if not bbox:
                continue
            x1, y1, x2, y2 = (float(v) for v in bbox)
            is_selected = obj.get("id") == self._current_obj_id
            color = BOX_SELECTED_COLOR if is_selected else BOX_COLOR
            pen_box = QPen(color, 2 if is_selected else 1)
            pen_box.setCosmetic(True)
            rect = scene.addRect(QRectF(x1, y1, x2 - x1, y2 - y1), pen_box)
            rect.setZValue(5)
            self._detection_overlay_items.append(rect)

            tracked_id = obj.get("tracked_id")
            label_text = f"#{obj.get('id')}" + (f" tid={tracked_id}" if tracked_id is not None else "")
            label = scene.addSimpleText(label_text)
            label.setBrush(QBrush(color))
            label.setPos(x1, y1 - label.boundingRect().height())
            label.setZValue(5)
            self._detection_overlay_items.append(label)

    # ------------------------------------------------------------------
    # Tab 1/2: wheel-point placement + auto-fit (tab 1), nudge (tab 2)
    # ------------------------------------------------------------------
    def _on_add_wheels_clicked(self):
        if self._video_player is None or self._current_frame_index is None:
            QMessageBox.warning(self, "尚未載入影片幀", "請先載入 replay JSON 並選好一幀。")
            return
        if self._current_obj_id is None or self._current_kp_cctv is None:
            QMessageBox.warning(self, "尚未選擇車輛", "這一幀沒有帶 kp_cctv 的車輛可選。")
            return
        if not self._selection_confirmed:
            QMessageBox.warning(self, "尚未確定選擇", "請先選車並按「確定選擇」。")
            return
        if not self._check_consistency():
            return
        name1 = self.combo_wheel1.currentText()
        name2 = self.combo_wheel2.currentText()
        if name1 == name2:
            QMessageBox.warning(self, "輪胎標籤重複", "請選 2 個不同的輪子。")
            return
        if self._pending is not None:
            return

        center = self.viewer_mark.mapToScene(self.viewer_mark.viewport().rect().center())
        cx, cy = float(center.x()), float(center.y())
        # Long enough to span the whole frame regardless of where the
        # marker ends up after dragging.
        guide_length = self.viewer_mark.scene().sceneRect().height()
        marker1 = WheelAnchorMarker(cx - 60, cy, ANCHOR_COLOR, name1, guide_length, movable=True)
        marker2 = WheelAnchorMarker(cx + 60, cy, ANCHOR_COLOR, name2, guide_length, movable=True)
        self.viewer_mark.scene().addItem(marker1)
        self.viewer_mark.scene().addItem(marker2)
        self._pending = {"marker1": marker1, "marker2": marker2, "name1": name1, "name2": name2, "locked": False}

        self.btn_add_wheels.setEnabled(False)
        self.btn_done_wheels.setEnabled(True)
        self.btn_cancel_wheels.setEnabled(True)
        self.combo_wheel1.setEnabled(False)
        self.combo_wheel2.setEnabled(False)
        self.spin_frame.setEnabled(False)
        self.slider_frame.setEnabled(False)
        self.combo_tracked_id.setEnabled(False)
        self.lbl_wheel_status.setText(f"拖曳 2 個青色叉叉到「{name1}」與「{name2}」的實際輪胎接地點，完成後按「完成（自動擬合）」。")

    def _on_done_wheels_clicked(self):
        if self._pending is None or self._pending["locked"]:
            return
        self._pending["marker1"].set_movable(False)
        self._pending["marker2"].set_movable(False)
        self._pending["locked"] = True

        px1 = self._pending["marker1"].scene_xy()
        px2 = self._pending["marker2"].scene_xy()
        try:
            sat1 = self._g_projection.cctv_to_sat(px1[0], px1[1], h=0.0)
            sat2 = self._g_projection.cctv_to_sat(px2[0], px2[1], h=0.0)
            anchors = [(self._pending["name1"], sat1), (self._pending["name2"], sat2)]
            pose = fit_two_point_pose(self._template, anchors)
        except Exception as e:
            QMessageBox.warning(self, "擬合失敗", str(e))
            self._on_cancel_wheels_clicked()
            return

        self._fit_anchors = anchors
        self._current_pose = pose
        self.btn_done_wheels.setEnabled(False)
        self._refresh_pose_overlays()
        self.lbl_wheel_status.setText("已自動擬合，已切換到「微調」分頁。")
        self.tabs.setCurrentIndex(NUDGE_TAB_INDEX)
        self.viewer_nudge.setFocus()

    def _on_cancel_wheels_clicked(self):
        self._discard_pending()
        self._reset_wheel_state()
        self.tabs.setCurrentIndex(MARK_TAB_INDEX)

    def _discard_pending(self):
        if self._pending is not None:
            scene = self.viewer_mark.scene()
            for key in ("marker1", "marker2"):
                item = self._pending[key]
                if item.scene() == scene:
                    scene.removeItem(item)
            self._pending = None
        if self._topdown_overlay_item is not None:
            scene = self.viewer_nudge.scene()
            if scene is not None and self._topdown_overlay_item.scene() == scene:
                scene.removeItem(self._topdown_overlay_item)
            self._topdown_overlay_item = None
        if self._nudge_anchor_markers:
            scene = self.viewer_nudge.scene()
            for marker in self._nudge_anchor_markers:
                if scene is not None and marker.scene() == scene:
                    scene.removeItem(marker)
            self._nudge_anchor_markers = []
        self._current_pose = None
        self._fit_anchors = None
        self._reset_wheel_state()

    def _reset_wheel_state(self):
        self.btn_add_wheels.setEnabled(True)
        self.btn_done_wheels.setEnabled(False)
        self.btn_cancel_wheels.setEnabled(False)
        self.combo_wheel1.setEnabled(True)
        self.combo_wheel2.setEnabled(True)
        self.spin_frame.setEnabled(True)
        self.slider_frame.setEnabled(True)
        self.combo_tracked_id.setEnabled(True)
        self.lbl_fit_report.setText("")
        self.lbl_ref_count.setText("")
        # Each wheel-marking round needs its own explicit "確定選擇" --
        # go back to box mode so the next car has to be confirmed again.
        self._selection_confirmed = False
        self._draw_frame_detections_overlay()
        if self._replay_data is not None:
            self.lbl_wheel_status.setText("選好這一幀的車輛，按「標記兩個輪胎點」開始。")

    def _on_nudge_translate(self, dx_m: float, dy_m: float):
        if self._current_pose is None:
            return
        px_per_m = self._g_projection.px_per_m
        self._current_pose = self._current_pose.translated(dx_m * px_per_m, dy_m * px_per_m)
        self._refresh_pose_overlays()

    def _on_nudge_rotate(self, d_theta_rad: float):
        if self._current_pose is None:
            return
        self._current_pose = self._current_pose.rotated(d_theta_rad)
        self._refresh_pose_overlays()

    def _on_flip_180(self):
        if self._current_pose is None:
            return
        self._current_pose = self._current_pose.flipped_180()
        self._refresh_pose_overlays()

    def _on_refit(self):
        if self._fit_anchors is None:
            return
        self._current_pose = fit_two_point_pose(self._template, self._fit_anchors)
        self._refresh_pose_overlays()

    def _on_kp_conf_changed(self, _value):
        self._draw_frame_detections_overlay()
        if self._current_pose is not None:
            self._refresh_pose_overlays()

    def _refresh_pose_overlays(self):
        self._refresh_fit_report()
        self._refresh_topdown_overlay()
        self._populate_nudge_cctv_reference()

    def _refresh_fit_report(self):
        """Compute the anchor-fit quality report and reference-point-count
        preview (feeds lbl_fit_report/lbl_ref_count in the 微調 tab's filter
        group) -- no longer draws the CCTV predicted-vs-detected
        TemplateOverlayItem on viewer_mark; the 微調 tab's sat-plane
        top-down car icon plus CCTV reference panel replace it."""
        if self._current_pose is None or self._template is None or self._g_projection is None:
            return
        pose = self._current_pose
        placed_sat_xy = apply_pose_to_template(self._template, pose)

        kp_cctv = self._current_kp_cctv or []
        report = anchor_fit_report(pose, self._g_projection.px_per_m)
        report_line = (
            f"縮放比例：{report['scale_ratio']:.3f}"
            f"（解出 {report['scale_px_per_m']:.2f} px/m，此 location 標定值 {report['nominal_px_per_m']:.2f} px/m）"
        )
        if report["warning"]:
            report_line += f"\n⚠ {report['warning']}"
        self.lbl_fit_report.setText(report_line)

        refs_preview = build_reference_points(
            self._template, pose, placed_sat_xy, kp_cctv, self._current_frame_index,
            self._g_projection.px_per_m, kp_conf=self.spin_kp_conf.value(),
            min_height_m=self.spin_min_height.value(),
        )
        self.lbl_ref_count.setText(f"這次放置預計產生 {len(refs_preview)} 組參考點。")

    def _ensure_sat_background_loaded(self):
        if self._sat_bg_loaded_path == self._sat_image_path:
            return
        if not self._sat_image_path or not os.path.isfile(self._sat_image_path):
            return
        sat_img = cv2.imread(self._sat_image_path)
        if sat_img is None:
            return
        qimg = self._cv_to_qimage(sat_img)
        self.viewer_nudge.load_pixmap(QPixmap.fromImage(qimg))
        self.viewer_nudge.fitToView()
        # load_pixmap clears the whole scene -- drop the (now dangling)
        # Python references rather than touching them again.
        self._topdown_overlay_item = None
        self._nudge_anchor_markers = []
        self._sat_bg_loaded_path = self._sat_image_path

    def _refresh_topdown_overlay(self):
        """Draw the placed template as a bird's-eye car icon on the sat
        image -- what the car actually looks like parked on the map, not
        just a cloud of predicted/detected keypoints. Lives in the 微調 tab,
        alongside the nudge controls."""
        if self._current_pose is None or self._template is None or self._g_projection is None:
            return
        self._ensure_sat_background_loaded()
        if self._sat_bg_loaded_path is None:
            return
        pose = self._current_pose

        ground_xz = self._template.ground_xz
        margin = 0.15
        x_min, x_max = float(ground_xz[:, 0].min()) - margin, float(ground_xz[:, 0].max()) + margin
        z_min, z_max = float(ground_xz[:, 1].min()) - margin, float(ground_xz[:, 1].max()) + margin
        # z is -front/+rear (see car_template_placement.py), so the front
        # edge is at z_min.
        body_and_front = np.array([
            [x_min, z_min], [x_max, z_min], [x_max, z_max], [x_min, z_max],
            [(x_min + x_max) / 2.0, z_min],
        ])
        sat_pts = apply_pose_to_points(pose, body_and_front)
        corners_sat = sat_pts[:4]
        front_center_sat = sat_pts[4]

        wheel_names_present = [n for n in WHEEL_NAMES if n in self._template.kp_names]
        wheel_indices = [self._template.index_of(n) for n in wheel_names_present]
        all_sat = apply_pose_to_template(self._template, pose)
        wheel_sat = all_sat[wheel_indices]
        anchor_names = {self._pending["name1"], self._pending["name2"]} if self._pending else set()
        is_anchor = np.array([n in anchor_names for n in wheel_names_present])

        if self._topdown_overlay_item is None:
            self._topdown_overlay_item = TopDownTemplateOverlayItem()
            self.viewer_nudge.scene().addItem(self._topdown_overlay_item)
        self._topdown_overlay_item.set_geometry(corners_sat, front_center_sat, wheel_sat, is_anchor)

        # Fixed reference marks at the originally-clicked sat-plane wheel
        # positions -- the auto-fit places the template's wheel exactly on
        # these (zero residual), but nudging (translate/rotate/flip) moves
        # the template away from them, so these stay put as a "how far off
        # the original click am I now" reference. Created once per fit;
        # _ensure_sat_background_loaded resets the list when the scene gets
        # cleared out from under them.
        if not self._nudge_anchor_markers and self._fit_anchors:
            scene = self.viewer_nudge.scene()
            for name, (ax, ay) in self._fit_anchors:
                marker = CrosshairMarker(ax, ay, ANCHOR_COLOR, name, movable=False)
                scene.addItem(marker)
                self._nudge_anchor_markers.append(marker)

    def _populate_nudge_cctv_reference(self):
        """Read-only CCTV reference panel to the left of the sat view:
        current frame, the confirmed car's keypoints, and the two locked
        wheel-anchor marks -- lets the user cross-check the sat-plane
        top-down car icon against the original CCTV picture while nudging."""
        if self._video_player is None or self._current_frame_index is None:
            return
        frame, _ = self._read_frame_robust(self._current_frame_index)
        if frame is None:
            return
        qimg = self._cv_to_qimage(frame)
        self.viewer_nudge_cctv.load_pixmap(QPixmap.fromImage(qimg))
        self.viewer_nudge_cctv.fitToView()
        scene = self.viewer_nudge_cctv.scene()

        kp_conf = self.spin_kp_conf.value()
        for i, kp in enumerate(self._current_kp_cctv or []):
            x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
            if conf < kp_conf or (x, y) in {(0.0, 0.0), (0.0, -4.0)}:
                continue
            name = KP_NAMES[i] if i < len(KP_NAMES) else ""
            r, g, b = _kp_part_color(name)
            kp_color = QColor.fromRgbF(r, g, b)
            pen_kp = QPen(kp_color, 1)
            pen_kp.setCosmetic(True)
            dot = scene.addEllipse(x - 3, y - 3, 6, 6, pen_kp, QBrush(kp_color))
            dot.setZValue(5)

        if self._pending is not None:
            for key, name_key in (("marker1", "name1"), ("marker2", "name2")):
                x, y = self._pending[key].scene_xy()
                marker = CrosshairMarker(x, y, ANCHOR_COLOR, self._pending[name_key], movable=False)
                scene.addItem(marker)

    def _on_confirm_placement(self):
        if self._current_pose is None or self._pending is None:
            return
        placement = Placement(
            id=self._next_placement_id,
            frame_index=self._current_frame_index,
            obj_id=self._current_obj_id,
            tracked_id=self._current_tracked_id,
            anchor_names=(self._pending["name1"], self._pending["name2"]),
            anchor_px=(self._pending["marker1"].scene_xy(), self._pending["marker2"].scene_xy()),
            pose=self._current_pose,
            template_path=self._template_path,
            kp_cctv=list(self._current_kp_cctv or []),
            nominal_px_per_m=self._g_projection.px_per_m,
            kp_conf=self.spin_kp_conf.value(),
            min_height_m=self.spin_min_height.value(),
        )
        self._next_placement_id += 1
        self.placements.append(placement)

        self._discard_pending()
        self._refresh_placement_list()
        self.tabs.setCurrentIndex(MARK_TAB_INDEX)

    # ------------------------------------------------------------------
    # Tab 3: review placements
    # ------------------------------------------------------------------
    def _placement_n_refs(self, placement: Placement) -> int:
        placed_sat_xy = apply_pose_to_template(self._template_for(placement), placement.pose)
        refs = build_reference_points(
            self._template_for(placement), placement.pose, placed_sat_xy, placement.kp_cctv,
            placement.frame_index, placement.nominal_px_per_m,
            kp_conf=placement.kp_conf, min_height_m=placement.min_height_m,
        )
        return len(refs)

    def _template_for(self, placement: Placement) -> CarTemplate:
        # Placements normally all share the currently-loaded template; if the
        # user swapped templates mid-session, fall back to reloading the one
        # the placement was actually made with, so a stale review doesn't
        # silently mix templates.
        if self._template is not None and self._template.source_path == placement.template_path:
            return self._template
        return load_car_template(placement.template_path)

    def _refresh_placement_list(self):
        self.list_placements.blockSignals(True)
        self.list_placements.clear()
        for p in self.placements:
            report = anchor_fit_report(p.pose, p.nominal_px_per_m)
            n_refs = self._placement_n_refs(p)
            car_label = f"tracked_id={p.tracked_id}" if p.tracked_id is not None else f"obj#{p.obj_id}（未追蹤）"
            item = QListWidgetItem(
                f"#{p.id} · frame {p.frame_index} · {car_label} · "
                f"{n_refs} 組參考點 · scale_ratio {report['scale_ratio']:.2f}"
            )
            item.setData(Qt.UserRole, p.id)
            self.list_placements.addItem(item)
        self.list_placements.blockSignals(False)

    def _on_delete_placement(self):
        item = self.list_placements.currentItem()
        if item is None:
            return
        placement_id = item.data(Qt.UserRole)
        self.placements = [p for p in self.placements if p.id != placement_id]
        self._refresh_placement_list()

    def _on_placement_selection_changed(self, current, previous):
        if current is None or self._video_player is None:
            return
        placement_id = current.data(Qt.UserRole)
        placement = next((p for p in self.placements if p.id == placement_id), None)
        if placement is None:
            return

        frame, _ = self._read_frame_robust(placement.frame_index)
        if frame is None:
            QMessageBox.warning(self, "錯誤", f"無法讀取第 {placement.frame_index} 幀。")
            return
        qimg = self._cv_to_qimage(frame)
        self.viewer_review.load_pixmap(QPixmap.fromImage(qimg))
        self.viewer_review.fitToView()

        scene = self.viewer_review.scene()
        for name, (x, y) in zip(placement.anchor_names, placement.anchor_px):
            marker = CrosshairMarker(x, y, ANCHOR_COLOR, name, movable=False)
            scene.addItem(marker)

        template = self._template_for(placement)
        placed_sat_xy = apply_pose_to_template(template, placement.pose)
        heights = scaled_heights_m(template, placement.pose, placement.nominal_px_per_m)
        foot_px = np.array([self._g_projection.sat_to_cctv(x, y, h=0.0) for x, y in placed_sat_xy])
        pred_px = np.array([
            self._g_projection.sat_to_cctv(x, y, h=float(h))
            for (x, y), h in zip(placed_sat_xy, heights)
        ])
        n = len(template.kp_names)
        det_px = np.zeros((n, 2))
        detected_mask = np.zeros(n, dtype=bool)
        for i in range(min(n, len(placement.kp_cctv))):
            x, y, conf = float(placement.kp_cctv[i][0]), float(placement.kp_cctv[i][1]), float(placement.kp_cctv[i][2])
            det_px[i] = (x, y)
            if conf >= placement.kp_conf and (x, y) not in {(0.0, 0.0), (0.0, -4.0)}:
                detected_mask[i] = True
        overlay = TemplateOverlayItem()
        overlay.set_geometry(foot_px, pred_px, det_px, detected_mask)
        scene.addItem(overlay)

    def _on_compute(self):
        if self._g_projection is None:
            QMessageBox.warning(self, "尚未載入 G_projection", "請先在「標記」tab 載入 G_projection.json。")
            return
        if not self.placements:
            QMessageBox.warning(self, "沒有放置紀錄", "請先在「標記」tab 加入至少一次放置。")
            return

        refs: list[ReferencePoint] = []
        next_id = 1
        for p in self.placements:
            template = self._template_for(p)
            placed_sat_xy = apply_pose_to_template(template, p.pose)
            placement_refs = build_reference_points(
                template, p.pose, placed_sat_xy, p.kp_cctv, p.frame_index, p.nominal_px_per_m,
                kp_conf=p.kp_conf, min_height_m=p.min_height_m, start_id=next_id,
            )
            refs.extend(placement_refs)
            next_id += len(placement_refs) + 1

        if len(refs) < 2:
            QMessageBox.warning(self, "參考點不足", f"目前展開後只有 {len(refs)} 組參考點，至少需要 2 組。")
            return

        result = calibrate(refs, self._g_projection)
        self._last_result = result
        if not result["success"]:
            QMessageBox.warning(self, "計算未成功", result["message"] or "未知錯誤。")
            return
        self._populate_result_tab(result)
        self.tabs.setCurrentIndex(RESULT_TAB_INDEX)

    # ------------------------------------------------------------------
    # Tab 4: result
    # ------------------------------------------------------------------
    def _populate_result_tab(self, result: dict):
        cam_x, cam_y = result["cam_sat_xy"]
        px_per_m = self._g_projection.px_per_m if self._g_projection else 1.0
        rmse_m = result["rmse_m"] / px_per_m if result["rmse_m"] is not None else None
        lines = [
            f"計算結果 — cam_sat_xy: ({cam_x:.2f}, {cam_y:.2f})    z_cam_meters: {result['z_cam_meters']:.3f} m",
            f"RMSE: {rmse_m:.4f} m（換算自 sat 像素 / px_per_m）    "
            f"放置數: {len(self.placements)}    使用參考點數: {result['n_references']}    求解狀態: {result['message']}",
        ]

        existing_cam_sat = None
        existing_z_cam = None
        if self._g_projection is not None:
            existing_cam_sat = tuple(float(v) for v in self._g_projection.cam_sat)
            existing_z_cam = float(self._g_projection.z_cam)
            delta_xy = float(np.linalg.norm(np.array([cam_x, cam_y]) - np.array(existing_cam_sat)))
            delta_z = result["z_cam_meters"] - existing_z_cam
            delta_z_pct = (delta_z / existing_z_cam * 100.0) if existing_z_cam else float("nan")
            lines.append("")
            lines.append(
                f"既有 G_projection（{self._g_proj_path}）— "
                f"cam_sat_xy: ({existing_cam_sat[0]:.2f}, {existing_cam_sat[1]:.2f})    "
                f"z_cam_meters: {existing_z_cam:.3f} m"
            )
            lines.append(
                f"差異 — cam_sat 位移: {delta_xy:.2f} px    z_cam: {delta_z:+.3f} m ({delta_z_pct:+.1f}%)"
            )
        else:
            lines.append("")
            lines.append("（沒有已載入的 G_projection，無法比較。）")

        if not self._sat_image_path or not os.path.isfile(self._sat_image_path):
            lines.append(f"（找不到衛星影像：{self._sat_image_path}，無法畫出相機位置示意圖。）")
        self.text_result_summary.setPlainText("\n".join(lines))

        if self._sat_image_path and os.path.isfile(self._sat_image_path):
            self._draw_result_sat_view(result["cam_sat_xy"], existing_cam_sat)

    def _draw_result_sat_view(self, computed_cam_sat, existing_cam_sat):
        sat_img = cv2.imread(self._sat_image_path)
        if sat_img is None:
            return
        qimg = self._cv_to_qimage(sat_img)
        self.viewer_result.load_pixmap(QPixmap.fromImage(qimg))
        self.viewer_result.fitToView()

        scene = self.viewer_result.scene()
        if existing_cam_sat is not None:
            marker = CrosshairMarker(existing_cam_sat[0], existing_cam_sat[1], EXISTING_CAM_COLOR, "既有", size=14, movable=False)
            scene.addItem(marker)
        marker = CrosshairMarker(computed_cam_sat[0], computed_cam_sat[1], COMPUTED_CAM_COLOR, "計算", size=14, movable=False)
        scene.addItem(marker)

    def _build_calibrated_g_data(self, base_g_data: dict) -> dict:
        """Deep-copy `base_g_data` and patch only its parallax block
        (x_cam_coords_sat / y_cam_coords_sat / z_cam_meters) with this
        session's self._last_result -- everything else in the G_projection
        (homography, distortion, sat_path, ...) is carried over unchanged.
        Shared by 「另存新檔」(writes the copy to a new file), 「套用到
        G_projection」(overwrites the loaded file in place), and the 驗證
        tab's "修正後" fallback when no explicit file is given there."""
        g_data = copy.deepcopy(base_g_data)
        cam_x, cam_y = self._last_result["cam_sat_xy"]
        par = g_data.setdefault("parallax", {})
        par["x_cam_coords_sat"] = cam_x
        par["y_cam_coords_sat"] = cam_y
        par["z_cam_meters"] = self._last_result["z_cam_meters"]
        return g_data

    def _on_save_json(self):
        if self._last_result is None:
            QMessageBox.warning(self, "尚未計算", "請先在「確認」tab 按「計算」。")
            return
        if self._g_proj_path is None:
            QMessageBox.warning(self, "沒有 G_projection", "請先載入 G_projection.json。")
            return

        try:
            with open(self._g_proj_path) as f:
                base_g_data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "讀取失敗", f"{type(e).__name__}: {e}")
            return
        g_data = self._build_calibrated_g_data(base_g_data)

        default_dir = REPO_ROOT / "output" / "car_template_calibration" / (self._location_code or "unknown")
        default_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = str(default_dir / f"G_projection_{self._location_code or 'unknown'}_calibrated_{ts}.json")

        path, _ = QFileDialog.getSaveFileName(self, "另存新檔（G_projection 複本）", default_path, "JSON (*.json)")
        if not path:
            return

        with open(path, "w") as f:
            json.dump(g_data, f, indent=2, ensure_ascii=False)
        QMessageBox.information(self, "已儲存", f"已儲存至 {path}")

    def _on_apply_to_g_proj(self):
        if self._last_result is None:
            QMessageBox.warning(self, "尚未計算", "請先在「確認」tab 按「計算」。")
            return
        if self._g_proj_path is None:
            QMessageBox.warning(self, "沒有 G_projection", "請先載入 G_projection.json。")
            return

        reply = QMessageBox.question(
            self, "確認套用",
            f"即將覆寫 {self._g_proj_path} 的 parallax 區塊（x_cam_coords_sat / y_cam_coords_sat / z_cam_meters）。\n"
            "此動作無法復原（除非該檔案本身有版本控制）。是否繼續？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            with open(self._g_proj_path) as f:
                base_g_data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "讀取失敗", f"{type(e).__name__}: {e}")
            return
        g_data = self._build_calibrated_g_data(base_g_data)

        try:
            with open(self._g_proj_path, "w") as f:
                json.dump(g_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            QMessageBox.critical(self, "寫入失敗", f"{type(e).__name__}: {e}")
            return

        QMessageBox.information(self, "已套用", f"已寫入 {self._g_proj_path}")

    # ------------------------------------------------------------------
    # Tab 5: verify -- reuses trafficlab.projection.parallax_reprojection
    # pre/post overlay machinery (no changes to that module) to compare the
    # same frame's kp_cctv under a "修正前" G_projection vs. a "修正後" one,
    # without touching whichever G_projection is loaded in tab 1.
    # ------------------------------------------------------------------
    def _on_browse_verify_gproj_before(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇修正前 G_projection.json", "", "JSON (*.json)")
        if path:
            self.edit_verify_gproj_before.setText(path)

    def _on_browse_verify_gproj_after(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇修正後 G_projection.json", "", "JSON (*.json)")
        if path:
            self.edit_verify_gproj_after.setText(path)

    def _load_g_projection_from_path(self, path: str) -> Optional[GProjection]:
        try:
            with open(path) as f:
                g_data = json.load(f)
            return GProjection(g_data, base_dir=os.path.dirname(path))
        except Exception as e:
            QMessageBox.critical(self, "讀取失敗", f"{type(e).__name__}: {e}")
            return None

    def _resolve_verify_g_projections(self) -> Optional[tuple]:
        """(g_engine_before, g_engine_after), reading the "修正前" field
        (falls back to whatever G_projection is loaded in tab 1) and the
        "修正後" field -- if that one is left blank, derive it by applying
        this session's calibrated parallax params (self._last_result) on
        top of the "修正前" file, so a quick verify of the current run
        doesn't require saving a file first."""
        before_path = self.edit_verify_gproj_before.text().strip()
        if not before_path and self._g_proj_path is not None:
            before_path = str(self._g_proj_path)
        if not before_path or not os.path.isfile(before_path):
            QMessageBox.warning(self, "找不到修正前 G_projection", f"找不到檔案：{before_path or '(未指定)'}")
            return None
        g_engine_before = self._load_g_projection_from_path(before_path)
        if g_engine_before is None:
            return None

        after_path = self.edit_verify_gproj_after.text().strip()
        if after_path:
            if not os.path.isfile(after_path):
                QMessageBox.warning(self, "找不到修正後 G_projection", f"找不到檔案：{after_path}")
                return None
            g_engine_after = self._load_g_projection_from_path(after_path)
            if g_engine_after is None:
                return None
        else:
            if self._last_result is None:
                QMessageBox.warning(
                    self, "沒有可用的修正後結果",
                    "請先在「確認」tab 按「計算」，或在上面指定一個「修正後 G_projection」檔案。",
                )
                return None
            try:
                with open(before_path) as f:
                    base_g_data = json.load(f)
            except Exception as e:
                QMessageBox.critical(self, "讀取失敗", f"{type(e).__name__}: {e}")
                return None
            g_data_after = self._build_calibrated_g_data(base_g_data)
            g_engine_after = GProjection(g_data_after, base_dir=os.path.dirname(before_path))

        return g_engine_before, g_engine_after

    def _parse_verify_tid_filter(self) -> Optional[int]:
        text = self.edit_verify_tid.text().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            QMessageBox.warning(self, "tracked_id 格式錯誤", f"「{text}」不是有效的整數，請留空或輸入數字。")
            return None

    def _pick_verify_frame(self, g_engine, kp_conf: float, tid_filter: Optional[int]) -> Optional[int]:
        """Pick the frame with the most valid pre/post keypoint pairs (summed
        across objects), honoring the tracked_id filter."""
        best_idx, best_n = None, -1
        for frame_index, records in iter_frame_records(
            self._replay_data, g_engine, kp_conf=kp_conf, template=self._template.xhz,
        ):
            if tid_filter is not None:
                records = [r for r in records if r["tracked_id"] == tid_filter]
            n = sum(len(r["pairs"]) for r in records)
            if n > best_n:
                best_idx, best_n = frame_index, n
        return best_idx

    def _on_verify_auto_pick_frame(self):
        if self._replay_data is None or self._template is None:
            QMessageBox.warning(self, "尚未就緒", "請先載入 replay JSON 與模板。")
            return
        resolved = self._resolve_verify_g_projections()
        if resolved is None:
            return
        _, g_engine_after = resolved
        tid_filter = self._parse_verify_tid_filter()
        frame_index = self._pick_verify_frame(g_engine_after, self.spin_kp_conf.value(), tid_filter)
        if frame_index is None:
            QMessageBox.warning(self, "沒有可用的幀", "找不到任何符合條件（含 tracked_id 篩選）的有效參考點。")
            return
        self.spin_verify_frame.setValue(frame_index)

    def _on_generate_verify(self):
        if self._template is None:
            QMessageBox.warning(self, "沒有模板", "請先載入 CAD 模板。")
            return
        if self._replay_data is None:
            QMessageBox.warning(self, "沒有 replay JSON", "請先載入 replay JSON。")
            return
        if not self._sat_image_path or not os.path.isfile(self._sat_image_path):
            QMessageBox.warning(self, "找不到衛星影像", f"找不到衛星影像：{self._sat_image_path}")
            return
        resolved = self._resolve_verify_g_projections()
        if resolved is None:
            return
        g_engine_old, g_engine_new = resolved

        frame_index = self.spin_verify_frame.value()
        frame = find_frame(self._replay_data, frame_index)
        if frame is None:
            QMessageBox.warning(self, "找不到這一幀", f"replay JSON 裡沒有 frame_index={frame_index}。")
            return

        kp_conf = self.spin_kp_conf.value()
        template_xhz = self._template.xhz
        tid_filter = self._parse_verify_tid_filter()
        records_old = compute_frame_records(frame, g_engine_old, kp_conf=kp_conf, template=template_xhz)
        records_new = compute_frame_records(frame, g_engine_new, kp_conf=kp_conf, template=template_xhz)
        if tid_filter is not None:
            records_old = [r for r in records_old if r["tracked_id"] == tid_filter]
            records_new = [r for r in records_new if r["tracked_id"] == tid_filter]
        if not records_old or not records_new:
            QMessageBox.warning(
                self, "沒有可比較的資料",
                f"第 {frame_index} 幀（tracked_id 篩選：{tid_filter if tid_filter is not None else '全部'}）"
                "在修正前或修正後沒有任何有效參考點——可能是這一幀沒有這台車，或信心度閾值太高。",
            )
            return

        view_extent = compute_view_extent([records_old, records_new], pad=60.0)

        out_dir = REPO_ROOT / "output" / "car_template_calibration" / (self._location_code or "unknown")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        before_path = out_dir / f"verify_{ts}_before.png"
        after_path = out_dir / f"verify_{ts}_after.png"

        try:
            plot_frame_pre_post_keypoints(
                frame_index, records_old, self._sat_image_path, before_path,
                title=f"修正前 — frame {frame_index}",
                recomputed=True, view_extent=view_extent,
            )
            plot_frame_pre_post_keypoints(
                frame_index, records_new, self._sat_image_path, after_path,
                title=f"修正後 — frame {frame_index}",
                recomputed=True, view_extent=view_extent,
            )
        except Exception as e:
            QMessageBox.critical(
                self, "產生比較圖失敗",
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            )
            return

        self.viewer_verify_before.load_pixmap(QPixmap(str(before_path)))
        self.viewer_verify_before.fitToView()
        self.viewer_verify_after.load_pixmap(QPixmap(str(after_path)))
        self.viewer_verify_after.fitToView()
        self.lbl_verify_status.setText(
            f"已產生比較圖（frame {frame_index}，tracked_id 篩選："
            f"{tid_filter if tid_filter is not None else '全部'}）："
            f"{before_path.name} / {after_path.name}"
        )

    # ------------------------------------------------------------------
    def _cv_to_qimage(self, cv_bgr):
        if cv_bgr is None:
            return None
        rgb = cv_bgr[:, :, ::-1]
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).copy()


class CarTemplateCalibrationWindow(QMainWindow):
    def __init__(self, initial_replay_json: Optional[str] = None, initial_g_proj: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("TrafficLab Car-Template GCP Calibration")
        self.setWindowFlags(Qt.Window)
        self.resize(1600, 960)

        self.widget = CarTemplateCalibrationWidget(
            initial_replay_json=initial_replay_json, initial_g_proj=initial_g_proj
        )
        self.setCentralWidget(self.widget)
