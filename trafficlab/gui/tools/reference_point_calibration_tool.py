"""Standalone N-point least-squares reference-point calibration tool.

Implements Method 7 from docs/height-correction-algorithm-survey.md:
generalizes pars_stage.py's 2-object head/foot manual calibration to N >= 2
reference objects (of known height) collected across one or more video
frames, solved jointly via trafficlab.projection.reference_point_calibration
.calibrate(). Independent of the CalibrationTab wizard -- like
vp_calibration_tool.py, it never modifies G_projection_<code>.json unless
the user explicitly presses "apply".

See docs/reference-point-calibration-tool-guide.md for usage.
"""
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QBrush, QFont, QImage, QPen, QPixmap
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsSimpleTextItem,
    QGraphicsView,
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
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from trafficlab.gui.tabs.calibration_stage.undistort_stage import ImageViewer
from trafficlab.visualization.video_player import VideoPlayer
from trafficlab.projection.g_projection import GProjection
from trafficlab.projection.reference_point_calibration import ReferencePoint, calibrate
from trafficlab.projection.parallax_reprojection import (
    find_frame,
    compute_frame_records,
    compute_view_extent,
    plot_frame_pre_post_keypoints,
)
from trafficlab.trajectory.io import load_json
from trafficlab.motion.keypoints_openpifpaf import build_car_template, _FALLBACK_DIMS

REPO_ROOT = Path(__file__).resolve().parents[3]

HEAD_COLOR = Qt.red
FOOT_COLOR = Qt.cyan
LINK_COLOR = Qt.green
COMPUTED_CAM_COLOR = Qt.yellow
EXISTING_CAM_COLOR = Qt.magenta


class CrosshairMarker(QGraphicsItem):
    """Thin 'X' marker (deliberately not a circle, and a 1px cosmetic pen)
    for a reference-point head/foot pixel. Draggable via the standard Qt
    ItemIsMovable flag while a new reference is being placed; frozen
    (movable=False) once confirmed into the list or shown read-only in the
    review tab.

    When *guide_length* is given, a dashed vertical line is attached below
    the marker (as a child item, so it tracks the marker while dragged) to
    help the user line up the paired foot marker directly underneath the
    head marker while placing a new reference."""

    def __init__(self, x, y, color, label, size=9, movable=True, guide_length=None):
        super().__init__()
        self._size = size
        self._pen = QPen(color, 1)
        self._pen.setCosmetic(True)
        self.setPos(x, y)
        self.setZValue(10)
        self.setFlag(QGraphicsItem.ItemIsMovable, movable)

        self._label = QGraphicsSimpleTextItem(label, self)
        self._label.setBrush(QBrush(color))
        self._label.setFont(QFont("Arial", 9, QFont.Bold))
        self._label.setPos(size + 2, -size - 4)

        if guide_length:
            guide_pen = QPen(color, 1, Qt.DashLine)
            guide_pen.setCosmetic(True)
            guide = QGraphicsLineItem(0, size, 0, size + guide_length, self)
            guide.setPen(guide_pen)
            guide.setZValue(-1)
            # Purely a visual aid -- must not steal drag/click events meant
            # for the foot marker it may visually overlap.
            guide.setAcceptedMouseButtons(Qt.NoButton)

    def set_movable(self, movable: bool):
        self.setFlag(QGraphicsItem.ItemIsMovable, movable)

    def scene_xy(self):
        p = self.pos()
        return float(p.x()), float(p.y())

    def boundingRect(self):
        s = self._size + 2
        return QRectF(-s, -s, 2 * s, 2 * s)

    def paint(self, painter, option, widget=None):
        painter.setPen(self._pen)
        s = self._size
        painter.drawLine(QPointF(-s, -s), QPointF(s, s))
        painter.drawLine(QPointF(-s, s), QPointF(s, -s))


class DraggableMarkerViewer(ImageViewer):
    """Same pan/zoom base as the rest of the calibration stages, but with
    NoDrag instead of ScrollHandDrag. ScrollHandDrag intercepts every
    left-button press for panning before the scene/items ever see it (this
    is exactly why pars_stage.py / vp_calibration_tool.py had to move point
    placement to *right*-click), which would make CrosshairMarker's
    ItemIsMovable left-click-drag impossible. Panning here is via scrollbars
    instead; wheel-zoom (inherited from ImageViewer) is unaffected."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.NoDrag)


class ComparisonViewer(ImageViewer):
    """Two QGraphicsPixmapItem layers stacked in the same scene -- a bottom
    ("baseline") layer and a top ("new") layer whose opacity can be tuned
    live, for eyeballing how much a recomputed parallax correction differs
    from the baseline without having to compare two separate saved PNGs.
    Reuses ImageViewer's pan/zoom.

    The two pixmaps only line up if the caller rendered them with the same
    figsize/dpi and the same view_extent (see plot_frame_pre_post_keypoints
    and compute_view_extent in parallax_reprojection.py) -- this class
    just stacks whatever it's given, it doesn't verify alignment."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._top_pixmap_item = None

    def load_pixmaps(self, bottom: QPixmap, top: QPixmap):
        # Same "drop Python refs before scene.clear()" discipline as
        # ImageViewer.load_pixmap -- see its comment for why.
        _scene = self.scene()
        for attr in ("_pixmap_item", "_overlay_item", "_top_pixmap_item"):
            item = getattr(self, attr, None)
            if item is not None:
                try:
                    if item.scene() == _scene:
                        _scene.removeItem(item)
                except Exception:
                    pass
                setattr(self, attr, None)
        _scene.clear()

        self._pixmap_item = QGraphicsPixmapItem(bottom)
        self._pixmap_item.setZValue(0)
        _scene.addItem(self._pixmap_item)

        self._top_pixmap_item = QGraphicsPixmapItem(top)
        self._top_pixmap_item.setZValue(1)
        _scene.addItem(self._top_pixmap_item)

        self.setSceneRect(self._pixmap_item.boundingRect())
        self._zoom = 0

    def set_top_opacity(self, value: float):
        if self._top_pixmap_item is not None:
            self._top_pixmap_item.setOpacity(value)


class ReferencePointCalibrationWidget(QWidget):
    def __init__(self, initial_video: Optional[str] = None, initial_g_proj: Optional[str] = None, parent=None):
        super().__init__(parent)

        self._video_player: Optional[VideoPlayer] = None
        self._video_path: Optional[str] = None
        self._current_frame_index: Optional[int] = None
        self._review_frame_index: Optional[int] = None

        self._g_proj_path: Optional[Path] = None
        self._location_code: Optional[str] = None
        self._g_projection: Optional[GProjection] = None
        self._sat_image_path: Optional[str] = None

        self._verify_g_proj_path: Optional[Path] = None
        self._verify_g_projection: Optional[GProjection] = None
        self._verify_sat_image_path: Optional[str] = None
        self._verify_template: Optional[np.ndarray] = None

        self._verify_baseline_g_proj_path: Optional[Path] = None
        self._verify_baseline_g_projection: Optional[GProjection] = None
        self._verify_baseline_sat_image_path: Optional[str] = None

        self.references: list[ReferencePoint] = []
        self._next_ref_id = 1
        self._pending: Optional[dict] = None  # {"head": marker, "foot": marker, "locked": bool}
        self._last_result: Optional[dict] = None

        self._build_ui()

        if initial_g_proj:
            self.edit_g_proj_path.setText(initial_g_proj)
            self._load_g_proj(initial_g_proj)
            self.edit_verify_g_proj_path.setText(initial_g_proj)
            self._load_verify_g_proj(initial_g_proj)
            self.edit_verify_baseline_g_proj_path.setText(initial_g_proj)
            self._load_verify_baseline_g_proj(initial_g_proj)
        if initial_video:
            self.edit_video_path.setText(initial_video)
            self._on_load_video()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.tabs.addTab(self._build_add_tab(), "1. 新增參考點")
        self.tabs.addTab(self._build_review_tab(), "2. 確認")
        self.tabs.addTab(self._build_result_tab(), "3. 結果")
        self.tabs.addTab(self._build_verify_tab(), "4. 驗證")

    # -- Tab 1: add reference points -----------------------------------
    def _build_add_tab(self) -> QWidget:
        page = QWidget()
        main_layout = QHBoxLayout(page)

        splitter = QSplitter(Qt.Horizontal)
        self.viewer_add = DraggableMarkerViewer()
        splitter.addWidget(self.viewer_add)

        sidebar = QWidget()
        side_vbox = QVBoxLayout(sidebar)
        side_vbox.setSpacing(10)
        side_vbox.addWidget(self._build_video_group())
        side_vbox.addWidget(self._build_add_group())
        side_vbox.addWidget(self._build_frame_list_group())
        side_vbox.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(sidebar)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(400)

        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        main_layout.addWidget(splitter)
        return page

    def _build_video_group(self) -> QGroupBox:
        grp = QGroupBox("影片與幀")
        layout = QVBoxLayout(grp)

        row = QHBoxLayout()
        self.edit_video_path = QLineEdit()
        btn_browse_video = QPushButton("瀏覽…")
        btn_browse_video.clicked.connect(self._on_browse_video)
        row.addWidget(self.edit_video_path)
        row.addWidget(btn_browse_video)
        layout.addLayout(row)

        btn_load_video = QPushButton("開啟影片")
        btn_load_video.clicked.connect(self._on_load_video)
        layout.addWidget(btn_load_video)

        layout.addWidget(QLabel("G_projection.json（必要，提供既有地面 homography）："))
        row2 = QHBoxLayout()
        self.edit_g_proj_path = QLineEdit()
        btn_browse_g_proj = QPushButton("瀏覽…")
        btn_browse_g_proj.clicked.connect(self._on_browse_g_proj)
        row2.addWidget(self.edit_g_proj_path)
        row2.addWidget(btn_browse_g_proj)
        layout.addLayout(row2)

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
        frame_row.addWidget(self.spin_frame)
        frame_row.addWidget(self.slider_frame, 1)
        layout.addLayout(frame_row)

        self.btn_load_frame = QPushButton("載入這一幀")
        self.btn_load_frame.clicked.connect(self._on_load_frame)
        layout.addWidget(self.btn_load_frame)

        return grp

    def _build_add_group(self) -> QGroupBox:
        grp = QGroupBox("新增參考點")
        layout = QVBoxLayout(grp)

        self.lbl_add_status = QLabel("請先載入影片幀，再按「新增參考點」。")
        self.lbl_add_status.setWordWrap(True)
        layout.addWidget(self.lbl_add_status)

        self.btn_add_ref = QPushButton("新增參考點")
        self.btn_add_ref.clicked.connect(self._on_add_reference_clicked)
        layout.addWidget(self.btn_add_ref)

        row = QHBoxLayout()
        self.btn_done = QPushButton("完成")
        self.btn_done.setEnabled(False)
        self.btn_done.clicked.connect(self._on_done_clicked)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        row.addWidget(self.btn_done)
        row.addWidget(self.btn_cancel)
        layout.addLayout(row)

        height_row = QHBoxLayout()
        height_row.addWidget(QLabel("真實高度 (m)："))
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(0.1, 20.0)
        self.spin_height.setSingleStep(0.05)
        self.spin_height.setValue(1.6)
        self.spin_height.setVisible(False)
        height_row.addWidget(self.spin_height)
        layout.addLayout(height_row)

        self.btn_confirm = QPushButton("確認加入列表")
        self.btn_confirm.setVisible(False)
        self.btn_confirm.clicked.connect(self._on_confirm_clicked)
        layout.addWidget(self.btn_confirm)

        return grp

    def _build_frame_list_group(self) -> QGroupBox:
        grp = QGroupBox("這一幀已加入的參考點")
        layout = QVBoxLayout(grp)

        self.list_frame_refs = QListWidget()
        self.list_frame_refs.setMaximumHeight(160)
        layout.addWidget(self.list_frame_refs)

        self.btn_delete_frame_ref = QPushButton("刪除選取的參考點")
        self.btn_delete_frame_ref.clicked.connect(self._on_delete_frame_ref)
        layout.addWidget(self.btn_delete_frame_ref)

        return grp

    # -- Tab 2: review ----------------------------------------------------
    def _build_review_tab(self) -> QWidget:
        page = QWidget()
        main_layout = QHBoxLayout(page)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_vbox = QVBoxLayout(left)
        left_vbox.addWidget(QLabel("所有已加入的參考點（點選可預覽）："))
        self.list_all_refs = QListWidget()
        self.list_all_refs.currentItemChanged.connect(self._on_review_selection_changed)
        left_vbox.addWidget(self.list_all_refs)

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

    # -- Tab 3: result ------------------------------------------------
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

    # -- Tab 4: verify --------------------------------------------------
    def _build_verify_tab(self) -> QWidget:
        page = QWidget()
        main_layout = QHBoxLayout(page)

        splitter = QSplitter(Qt.Horizontal)
        self.viewer_verify = ComparisonViewer()
        splitter.addWidget(self.viewer_verify)

        sidebar = QWidget()
        side_vbox = QVBoxLayout(sidebar)
        side_vbox.setSpacing(10)

        lbl_intro = QLabel(
            "用一份已經跑過關鍵點推論的 replay JSON（帶 kp_cctv）比較兩個 G_projection："
            "「原版」讀 JSON 裡存的 kp_sat（不重算），「新版」用選的 G_projection 即時重算 parallax correction。"
            "兩張圖縮放倍率與座標範圍固定一致，方便疊圖比對——原版在下層、新版在上層，可用透明度滑桿調整。"
        )
        lbl_intro.setWordWrap(True)
        side_vbox.addWidget(lbl_intro)

        side_vbox.addWidget(self._build_verify_g_proj_group())
        side_vbox.addWidget(self._build_verify_json_group())
        side_vbox.addWidget(self._build_verify_output_group())

        self.btn_verify_run = QPushButton("確認並輸出對照圖")
        self.btn_verify_run.clicked.connect(self._on_verify_run)
        side_vbox.addWidget(self.btn_verify_run)

        self.lbl_verify_status = QLabel("")
        self.lbl_verify_status.setWordWrap(True)
        side_vbox.addWidget(self.lbl_verify_status)

        side_vbox.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(sidebar)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(400)

        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        main_layout.addWidget(splitter)
        return page

    def _build_verify_g_proj_group(self) -> QGroupBox:
        grp = QGroupBox("G_projection")
        layout = QVBoxLayout(grp)

        layout.addWidget(QLabel("新版（即時重算）："))
        new_row = QHBoxLayout()
        self.edit_verify_g_proj_path = QLineEdit()
        btn_browse_verify_g_proj = QPushButton("瀏覽…")
        btn_browse_verify_g_proj.clicked.connect(self._on_browse_verify_g_proj)
        new_row.addWidget(self.edit_verify_g_proj_path)
        new_row.addWidget(btn_browse_verify_g_proj)
        layout.addLayout(new_row)

        layout.addWidget(QLabel("原版（讀存好的 kp_sat）："))
        baseline_row = QHBoxLayout()
        self.edit_verify_baseline_g_proj_path = QLineEdit()
        btn_browse_verify_baseline_g_proj = QPushButton("瀏覽…")
        btn_browse_verify_baseline_g_proj.clicked.connect(self._on_browse_verify_baseline_g_proj)
        baseline_row.addWidget(self.edit_verify_baseline_g_proj_path)
        baseline_row.addWidget(btn_browse_verify_baseline_g_proj)
        layout.addLayout(baseline_row)

        return grp

    def _build_verify_json_group(self) -> QGroupBox:
        grp = QGroupBox("Replay JSON")
        layout = QVBoxLayout(grp)

        row = QHBoxLayout()
        self.edit_verify_json_path = QLineEdit()
        btn_browse_verify_json = QPushButton("瀏覽…")
        btn_browse_verify_json.clicked.connect(self._on_browse_verify_json)
        row.addWidget(self.edit_verify_json_path)
        row.addWidget(btn_browse_verify_json)
        layout.addLayout(row)

        return grp

    def _build_verify_output_group(self) -> QGroupBox:
        grp = QGroupBox("輸出設定")
        layout = QVBoxLayout(grp)

        layout.addWidget(QLabel("幀數："))
        self.spin_verify_frame = QSpinBox()
        self.spin_verify_frame.setRange(0, 1_000_000)
        layout.addWidget(self.spin_verify_frame)

        layout.addWidget(QLabel("篩選 tracked_id（選填，逗號分隔）："))
        self.edit_verify_ids = QLineEdit()
        self.edit_verify_ids.setPlaceholderText("例如 101,205（留空代表全部）")
        layout.addWidget(self.edit_verify_ids)

        self.lbl_verify_opacity = QLabel("上層（新版）透明度：100%")
        layout.addWidget(self.lbl_verify_opacity)
        self.slider_verify_opacity = QSlider(Qt.Horizontal)
        self.slider_verify_opacity.setRange(0, 100)
        self.slider_verify_opacity.setValue(100)
        self.slider_verify_opacity.valueChanged.connect(self._on_verify_opacity_changed)
        layout.addWidget(self.slider_verify_opacity)

        return grp

    def _on_verify_opacity_changed(self, value: int):
        self.lbl_verify_opacity.setText(f"上層（新版）透明度：{value}%")
        self.viewer_verify.set_top_opacity(value / 100.0)

    # ------------------------------------------------------------------
    # Video / G_projection loading
    # ------------------------------------------------------------------
    def _on_browse_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇影片", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.edit_video_path.setText(path)

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

        self.lbl_add_status.setText(f"已載入 G_projection（location={self._location_code}）。")

    @staticmethod
    def _read_g_projection_file(path: str) -> tuple[GProjection, Optional[str], Optional[str]]:
        """Open a G_projection_<code>.json and return (g_projection,
        sat_image_path, location_code). Shared by both verify-tab loaders
        (new/baseline) to avoid a third near-duplicate of this boilerplate."""
        with open(path) as f:
            g_data = json.load(f)
        g_projection = GProjection(g_data, base_dir=os.path.dirname(path))
        location_code = g_data.get("meta", {}).get("location_code")
        sat_rel = g_data.get("inputs", {}).get("sat_path")
        sat_image_path = str((Path(path).parent / sat_rel).resolve()) if sat_rel else None
        return g_projection, sat_image_path, location_code

    def _on_browse_verify_g_proj(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇 G_projection.json", "", "JSON (*.json)")
        if path:
            self.edit_verify_g_proj_path.setText(path)
            self._load_verify_g_proj(path)

    def _load_verify_g_proj(self, path: str):
        try:
            g_projection, sat_image_path, location_code = self._read_g_projection_file(path)
        except Exception as e:
            QMessageBox.warning(self, "錯誤", f"無法讀取 G_projection：{e}")
            return
        self._verify_g_proj_path = Path(path)
        self._verify_g_projection = g_projection
        self._verify_sat_image_path = sat_image_path

        dims = self._resolve_verify_car_dims(Path(path).parent)
        self._verify_template = build_car_template(dims)

        self.lbl_verify_status.setText(
            f"已載入新版 G_projection（location={location_code}）。彩色點會用這個 G_projection 重新計算。"
        )

    def _on_browse_verify_baseline_g_proj(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇 G_projection.json", "", "JSON (*.json)")
        if path:
            self.edit_verify_baseline_g_proj_path.setText(path)
            self._load_verify_baseline_g_proj(path)

    def _load_verify_baseline_g_proj(self, path: str):
        try:
            g_projection, sat_image_path, location_code = self._read_g_projection_file(path)
        except Exception as e:
            QMessageBox.warning(self, "錯誤", f"無法讀取 G_projection：{e}")
            return
        self._verify_baseline_g_proj_path = Path(path)
        self._verify_baseline_g_projection = g_projection
        self._verify_baseline_sat_image_path = sat_image_path

        self.lbl_verify_status.setText(
            f"已載入原版 G_projection（location={location_code}）。原版圖會讀 JSON 裡存的 kp_sat，不重新計算。"
        )

    @staticmethod
    def _resolve_verify_car_dims(g_proj_dir: Path) -> dict:
        # Dimension fallback chain: walk up from the G_projection's directory
        # looking for prior_dimensions.json, else the built-in defaults.
        d = g_proj_dir
        for _ in range(5):
            candidate = d / "prior_dimensions.json"
            if candidate.exists():
                with open(candidate) as f:
                    pj = json.load(f)
                return pj.get("measurements_visdrone", {}).get("car", dict(_FALLBACK_DIMS))
            d = d.parent
        return dict(_FALLBACK_DIMS)

    def _on_load_video(self):
        path = self.edit_video_path.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "錯誤", "找不到影片檔案。")
            return
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
        """Try VideoPlayer.read_frame(idx) (CAP_PROP_POS_FRAMES seek) first;
        fall back to sequential read-and-discard from the start if that
        fails -- same fallback vp_calibration_tool.py needs for some
        AV1-encoded footage where seeking silently fails."""
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

    # ------------------------------------------------------------------
    # Tab 1: add reference points
    # ------------------------------------------------------------------
    def _on_frame_spin_changed(self, value):
        # Auto-reload as the user scrubs, but stay silent (no dialog) if
        # no video is loaded yet -- this fires during widget construction
        # too (setRange/initial value), long before there's a video to warn
        # about. Explicit "載入這一幀" clicks still go through
        # _on_load_frame, which does warn.
        if self._video_player is None:
            return
        self._reload_current_frame()

    def _on_load_frame(self):
        if self._video_player is None:
            QMessageBox.warning(self, "錯誤", "請先開啟影片。")
            return
        self._reload_current_frame()

    def _reload_current_frame(self):
        # Any unexpected exception here must surface as a dialog, not
        # vanish -- PyQt5's default behavior for an uncaught exception
        # raised inside a slot is to print it to stderr and keep the event
        # loop running, which looks exactly like "clicked and nothing
        # happened" if nobody is watching the terminal.
        try:
            self._reload_current_frame_unsafe()
        except Exception as e:
            QMessageBox.critical(
                self,
                "載入這一幀時發生錯誤",
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            )

    def _reload_current_frame_unsafe(self):
        # Force-drop any in-progress placement first: load_pixmap() below
        # clears the whole QGraphicsScene (see undistort_stage.py's
        # load_pixmap comment on why stale Python references to
        # already-deleted C++ QGraphicsItems must never be touched again),
        # so any pending marker objects would become dangling otherwise.
        self._pending = None
        self._reset_add_state()

        idx = self.spin_frame.value()
        frame, used_fallback = self._read_frame_robust(idx)
        if frame is None:
            QMessageBox.warning(self, "錯誤", f"無法讀取第 {idx} 幀（seek 與循序讀取皆失敗）。")
            return

        self._current_frame_index = idx
        qimg = self._cv_to_qimage(frame)
        self.viewer_add.load_pixmap(QPixmap.fromImage(qimg))
        self.viewer_add.fitToView()

        for ref in self.references:
            if ref.frame_index == idx:
                self._draw_confirmed_overlay(self.viewer_add, ref)

        note = "（此影片 seek 不可靠，已改用循序讀取，可能較慢）" if used_fallback else ""
        self.lbl_add_status.setText(f"已載入第 {idx} 幀。{note} 按「新增參考點」開始。")
        self._refresh_frame_list()

    def _on_add_reference_clicked(self):
        if self._video_player is None or self._current_frame_index is None:
            QMessageBox.warning(self, "尚未載入影片幀", "請先開啟影片並載入一幀。")
            return
        if self._pending is not None:
            return

        center = self.viewer_add.mapToScene(self.viewer_add.viewport().rect().center())
        foot_xy = (float(center.x()), float(center.y()))
        head_xy = (float(center.x()), float(center.y()) - 80.0)

        # Spans the full frame height below the head marker so it stays
        # useful (as a vertical drag reference for the foot marker) no
        # matter where in the frame the head ends up after dragging.
        guide_length = self.viewer_add.scene().sceneRect().height()

        head_marker = CrosshairMarker(head_xy[0], head_xy[1], HEAD_COLOR, "頭", movable=True, guide_length=guide_length)
        foot_marker = CrosshairMarker(foot_xy[0], foot_xy[1], FOOT_COLOR, "腳", movable=True)
        self.viewer_add.scene().addItem(head_marker)
        self.viewer_add.scene().addItem(foot_marker)
        self._pending = {"head": head_marker, "foot": foot_marker, "locked": False}

        self.btn_add_ref.setEnabled(False)
        self.btn_done.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.spin_frame.setEnabled(False)
        self.slider_frame.setEnabled(False)
        self.btn_load_frame.setEnabled(False)
        self.lbl_add_status.setText("拖曳紅色「頭」與青色「腳」叉叉到正確位置，完成後按「完成」。")

    def _on_done_clicked(self):
        if self._pending is None or self._pending["locked"]:
            return
        self._pending["head"].set_movable(False)
        self._pending["foot"].set_movable(False)
        self._pending["locked"] = True
        self.btn_done.setEnabled(False)
        self.spin_height.setVisible(True)
        self.btn_confirm.setVisible(True)
        self.lbl_add_status.setText("輸入這個參考物的真實高度，按「確認加入列表」；或按「取消」放棄。")

    def _on_confirm_clicked(self):
        if self._pending is None or not self._pending["locked"]:
            return
        head_xy = self._pending["head"].scene_xy()
        foot_xy = self._pending["foot"].scene_xy()
        height_m = float(self.spin_height.value())

        # Drop the temporary markers first (they get replaced by a fresh
        # permanent overlay drawn from the ReferencePoint data), matching
        # the "never hold references past a scene mutation" rule above.
        scene = self.viewer_add.scene()
        for key in ("head", "foot"):
            item = self._pending[key]
            if item.scene() == scene:
                scene.removeItem(item)
        self._pending = None

        ref = ReferencePoint(
            id=self._next_ref_id,
            frame_index=self._current_frame_index,
            head_px=head_xy,
            foot_px=foot_xy,
            height_m=height_m,
        )
        self._next_ref_id += 1
        self.references.append(ref)
        self._draw_confirmed_overlay(self.viewer_add, ref)

        self._reset_add_state()
        self._refresh_frame_list()

    def _on_cancel_clicked(self):
        if self._pending is not None:
            scene = self.viewer_add.scene()
            for key in ("head", "foot"):
                item = self._pending[key]
                if item.scene() == scene:
                    scene.removeItem(item)
            self._pending = None
        self._reset_add_state()

    def _reset_add_state(self):
        self.btn_add_ref.setEnabled(True)
        self.btn_done.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.spin_height.setVisible(False)
        self.spin_height.setValue(1.6)
        self.btn_confirm.setVisible(False)
        self.spin_frame.setEnabled(True)
        self.slider_frame.setEnabled(True)
        self.btn_load_frame.setEnabled(True)
        if self._current_frame_index is not None:
            self.lbl_add_status.setText("按「新增參考點」開始。")

    def _draw_confirmed_overlay(self, viewer, ref: ReferencePoint):
        scene = viewer.scene()
        head = CrosshairMarker(ref.head_px[0], ref.head_px[1], HEAD_COLOR, f"#{ref.id} 頭", movable=False)
        foot = CrosshairMarker(ref.foot_px[0], ref.foot_px[1], FOOT_COLOR, f"#{ref.id} 腳", movable=False)
        scene.addItem(head)
        scene.addItem(foot)
        scene.addLine(ref.head_px[0], ref.head_px[1], ref.foot_px[0], ref.foot_px[1], QPen(LINK_COLOR, 1))

    def _refresh_frame_list(self):
        self.list_frame_refs.clear()
        if self._current_frame_index is None:
            return
        for ref in self.references:
            if ref.frame_index != self._current_frame_index:
                continue
            item = QListWidgetItem(f"#{ref.id} — 高度 {ref.height_m:.2f}m")
            item.setData(Qt.UserRole, ref.id)
            self.list_frame_refs.addItem(item)
        self._refresh_all_refs_list()

    def _on_delete_frame_ref(self):
        item = self.list_frame_refs.currentItem()
        if item is None:
            return
        ref_id = item.data(Qt.UserRole)
        self.references = [r for r in self.references if r.id != ref_id]
        # Redraw the current frame from scratch (safe: never touches the
        # now-invalid deleted overlay items directly, see load_pixmap note).
        self._on_load_frame()

    # ------------------------------------------------------------------
    # Tab 2: review
    # ------------------------------------------------------------------
    def _refresh_all_refs_list(self):
        self.list_all_refs.blockSignals(True)
        self.list_all_refs.clear()
        for ref in sorted(self.references, key=lambda r: (r.frame_index, r.id)):
            item = QListWidgetItem(f"Frame {ref.frame_index} · #{ref.id} · {ref.height_m:.2f}m")
            item.setData(Qt.UserRole, ref.id)
            self.list_all_refs.addItem(item)
        self.list_all_refs.blockSignals(False)

    def _on_review_selection_changed(self, current, previous):
        if current is None or self._video_player is None:
            return
        ref_id = current.data(Qt.UserRole)
        ref = next((r for r in self.references if r.id == ref_id), None)
        if ref is None:
            return

        frame, _ = self._read_frame_robust(ref.frame_index)
        if frame is None:
            QMessageBox.warning(self, "錯誤", f"無法讀取第 {ref.frame_index} 幀。")
            return
        self._review_frame_index = ref.frame_index
        qimg = self._cv_to_qimage(frame)
        self.viewer_review.load_pixmap(QPixmap.fromImage(qimg))
        self.viewer_review.fitToView()
        self._draw_confirmed_overlay(self.viewer_review, ref)

    def _on_compute(self):
        if self._g_projection is None:
            QMessageBox.warning(self, "尚未載入 G_projection", "請先在「新增參考點」tab 載入 G_projection.json。")
            return
        if len(self.references) < 2:
            QMessageBox.warning(self, "參考點不足", f"至少需要 2 個參考點，目前有 {len(self.references)} 個。")
            return

        result = calibrate(self.references, self._g_projection)
        self._last_result = result
        if not result["success"]:
            QMessageBox.warning(self, "計算未成功", result["message"] or "未知錯誤。")
            return
        self._populate_result_tab(result)
        self.tabs.setCurrentIndex(2)

    # ------------------------------------------------------------------
    # Tab 3: result
    # ------------------------------------------------------------------
    def _populate_result_tab(self, result: dict):
        cam_x, cam_y = result["cam_sat_xy"]
        lines = [
            f"計算結果 — cam_sat_xy: ({cam_x:.2f}, {cam_y:.2f})    z_cam_meters: {result['z_cam_meters']:.3f} m",
            f"RMSE: {result['rmse_m']:.4f} m    使用參考點數: {result['n_references']}    求解狀態: {result['message']}",
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

    def _on_save_json(self):
        if self._last_result is None:
            QMessageBox.warning(self, "尚未計算", "請先在「確認」tab 按「計算」。")
            return

        default_dir = REPO_ROOT / "output" / "reference_point_calibration" / (self._location_code or "unknown")
        default_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = str(default_dir / f"{ts}.json")

        path, _ = QFileDialog.getSaveFileName(self, "儲存結果 JSON", default_path, "JSON (*.json)")
        if not path:
            return

        output = self._build_output_json()
        with open(path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        QMessageBox.information(self, "已儲存", f"已儲存至 {path}")

    def _build_output_json(self) -> dict:
        meta = {
            "tool": "reference_point_calibration_tool",
            "version": 1,
            "timestamp": datetime.now().isoformat(),
            "location_code": self._location_code,
            "source_video": self._video_path,
            "g_projection_path": str(self._g_proj_path) if self._g_proj_path else None,
        }
        inputs = {
            "references": [
                {
                    "id": r.id,
                    "frame_index": r.frame_index,
                    "head_px": list(r.head_px),
                    "foot_px": list(r.foot_px),
                    "height_m": r.height_m,
                }
                for r in self.references
            ]
        }
        return {"meta": meta, "inputs": inputs, "results": self._last_result}

    def _on_apply_to_g_proj(self):
        if self._last_result is None:
            QMessageBox.warning(self, "尚未計算", "請先在「確認」tab 按「計算」。")
            return
        if self._g_proj_path is None:
            QMessageBox.warning(self, "沒有 G_projection", "請先載入 G_projection.json。")
            return

        reply = QMessageBox.question(
            self,
            "確認套用",
            f"即將覆寫 {self._g_proj_path} 的 parallax 區塊（x_cam_coords_sat / y_cam_coords_sat / z_cam_meters）。\n"
            "此動作無法復原（除非該檔案本身有版本控制）。是否繼續？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            with open(self._g_proj_path) as f:
                g_data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "讀取失敗", f"{type(e).__name__}: {e}")
            return

        cam_x, cam_y = self._last_result["cam_sat_xy"]
        par = g_data.setdefault("parallax", {})
        par["x_cam_coords_sat"] = cam_x
        par["y_cam_coords_sat"] = cam_y
        par["z_cam_meters"] = self._last_result["z_cam_meters"]

        try:
            with open(self._g_proj_path, "w") as f:
                json.dump(g_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            QMessageBox.critical(self, "寫入失敗", f"{type(e).__name__}: {e}")
            return

        QMessageBox.information(self, "已套用", f"已寫入 {self._g_proj_path}")

    # ------------------------------------------------------------------
    # Tab 4: verify
    # ------------------------------------------------------------------
    def _on_browse_verify_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇 replay JSON", "", "Replay JSON (*.json *.json.gz)"
        )
        if path:
            self.edit_verify_json_path.setText(path)

    def _on_verify_run(self):
        try:
            self._on_verify_run_unsafe()
        except Exception as e:
            QMessageBox.critical(
                self,
                "驗證時發生錯誤",
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            )

    def _on_verify_run_unsafe(self):
        json_path_str = self.edit_verify_json_path.text().strip()
        if not json_path_str or not os.path.isfile(json_path_str):
            QMessageBox.warning(self, "找不到檔案", "請先選擇一份存在的 replay JSON。")
            return
        if self._verify_g_projection is None:
            QMessageBox.warning(self, "沒有新版 G_projection", "請先在上方選擇「新版」G_projection.json。")
            return
        if not self._verify_sat_image_path or not os.path.isfile(self._verify_sat_image_path):
            QMessageBox.warning(self, "沒有衛星影像", f"找不到新版的衛星影像：{self._verify_sat_image_path}")
            return
        if self._verify_baseline_g_projection is None:
            QMessageBox.warning(self, "沒有原版 G_projection", "請先在上方選擇「原版」G_projection.json。")
            return
        if not self._verify_baseline_sat_image_path or not os.path.isfile(self._verify_baseline_sat_image_path):
            QMessageBox.warning(self, "沒有衛星影像", f"找不到原版的衛星影像：{self._verify_baseline_sat_image_path}")
            return

        json_path = Path(json_path_str)
        frame_index = self.spin_verify_frame.value()
        ids = self._parse_verify_ids(self.edit_verify_ids.text())

        data = load_json(json_path)
        frame = find_frame(data, frame_index)
        if frame is None:
            QMessageBox.warning(self, "找不到這一幀", f"frame_index {frame_index} 不存在於這份 JSON 裡。")
            return

        # "新版" always recomputes both pre and post points from kp_cctv via
        # self._verify_g_projection (template given). "原版" recomputes pre
        # the same way but reads post straight from the replay JSON's stored
        # kp_sat (template=None) -- i.e. what the pipeline actually produced
        # at inference time. Filtering by ids here (not just inside
        # plot_frame_pre_post_keypoints) so the shared view extent below is
        # computed from exactly what will be plotted.
        records_new = compute_frame_records(frame, self._verify_g_projection, kp_conf=0.2, template=self._verify_template)
        records_baseline = compute_frame_records(frame, self._verify_baseline_g_projection, kp_conf=0.2, template=None)
        if ids is not None:
            records_new = [r for r in records_new if r["tracked_id"] in ids]
            records_baseline = [r for r in records_baseline if r["tracked_id"] in ids]
        if not records_new:
            QMessageBox.warning(self, "沒有可用的關鍵點", f"frame {frame_index} 用新版 G_projection 沒有任何有效的 kp_cctv 關鍵點。")
            return
        if not records_baseline:
            QMessageBox.warning(self, "沒有可用的關鍵點", f"frame {frame_index} 用原版 G_projection 沒有任何有效的 kp_sat/kp_cctv 關鍵點。")
            return

        # Same view_extent for both plots (and a fixed, non-tight-cropped
        # layout inside plot_frame_pre_post_keypoints) is what makes the two
        # saved PNGs come out at identical pixel size/position, so they can
        # be stacked and overlaid.
        view_extent = compute_view_extent([records_new, records_baseline])

        name = json_path.name
        stem = name[:-8] if name.endswith(".json.gz") else json_path.stem
        baseline_out = json_path.with_name(f"{stem}.parallax_frame{frame_index}.baseline.png")
        new_out = json_path.with_name(f"{stem}.parallax_frame{frame_index}.recomputed.png")

        baseline_out, n_kp_baseline, n_obj_baseline = plot_frame_pre_post_keypoints(
            frame_index, records_baseline, self._verify_baseline_sat_image_path, baseline_out,
            dpi=200, recomputed=False, view_extent=view_extent,
            title=f"原版（frame {frame_index}）",
        )
        new_out, n_kp_new, n_obj_new = plot_frame_pre_post_keypoints(
            frame_index, records_new, self._verify_sat_image_path, new_out,
            dpi=200, recomputed=True, view_extent=view_extent,
            title=f"新版（frame {frame_index}）",
        )

        self.viewer_verify.load_pixmaps(QPixmap(str(baseline_out)), QPixmap(str(new_out)))
        self.viewer_verify.set_top_opacity(self.slider_verify_opacity.value() / 100.0)
        self.viewer_verify.fitToView()
        self.lbl_verify_status.setText(
            f"已顯示 frame {frame_index} — 新版：{n_kp_new} 個關鍵點/{n_obj_new} 個物件；"
            f"原版：{n_kp_baseline} 個關鍵點/{n_obj_baseline} 個物件。\n"
            f"已存 {baseline_out.name} 與 {new_out.name}"
        )

    @staticmethod
    def _parse_verify_ids(value: str):
        value = value.strip()
        if not value:
            return None
        return {int(item.strip()) for item in value.split(",") if item.strip()}

    # ------------------------------------------------------------------
    def _cv_to_qimage(self, cv_bgr):
        if cv_bgr is None:
            return None
        rgb = cv_bgr[:, :, ::-1]
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).copy()


class ReferencePointCalibrationWindow(QMainWindow):
    def __init__(self, initial_video: Optional[str] = None, initial_g_proj: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("TrafficLab Reference-Point Calibration (Method 7)")
        self.setWindowFlags(Qt.Window)
        self.resize(1600, 960)

        self.widget = ReferencePointCalibrationWidget(initial_video=initial_video, initial_g_proj=initial_g_proj)
        self.setCentralWidget(self.widget)
