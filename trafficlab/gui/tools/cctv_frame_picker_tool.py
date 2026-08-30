"""Standalone GUI tool: browse a location's footage, pick a still frame,
and save it as location/<code>/cctv_<code>.png at the source video's
native resolution.

See AGENTS.md 常用指令 13 (CCTV 影格選取工具).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from trafficlab.gui.tabs.calibration_stage.undistort_stage import ImageViewer
from trafficlab.visualization.video_player import VideoPlayer

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCATION_ROOT = REPO_ROOT / "location"


def list_locations_with_footage(location_root: Path = LOCATION_ROOT):
    if not location_root.is_dir():
        return []
    codes = []
    for d in sorted(location_root.iterdir()):
        footage_dir = d / "footage"
        if d.is_dir() and footage_dir.is_dir() and sorted(footage_dir.glob("*.mp4")):
            codes.append(d.name)
    return codes


def list_footage_videos(location_code: str, location_root: Path = LOCATION_ROOT):
    footage_dir = location_root / location_code / "footage"
    if not footage_dir.is_dir():
        return []
    return sorted(footage_dir.glob("*.mp4"))


class CCTVFramePickerWidget(QWidget):
    def __init__(self, initial_location_code: Optional[str] = None, initial_video: Optional[str] = None, parent=None):
        super().__init__(parent)
        self._player: Optional[VideoPlayer] = None
        self._video_path: Optional[str] = None
        self._current_frame_bgr = None
        self._current_frame_index = 0
        self._frame_count = 0
        self._fps = 25.0
        self._is_playing = False

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance_playback)

        self._build_ui()
        self._refresh_locations()

        if initial_location_code:
            idx = self.combo_location.findText(initial_location_code)
            if idx >= 0:
                self.combo_location.setCurrentIndex(idx)
        if initial_video:
            idx = self.combo_video.findData(str(Path(initial_video).resolve()))
            if idx >= 0:
                self.combo_video.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        row_select = QHBoxLayout()
        row_select.addWidget(QLabel("Location:"))
        self.combo_location = QComboBox()
        self.combo_location.currentIndexChanged.connect(self._on_location_changed)
        row_select.addWidget(self.combo_location, 1)

        row_select.addWidget(QLabel("影片:"))
        self.combo_video = QComboBox()
        self.combo_video.currentIndexChanged.connect(self._on_video_changed)
        row_select.addWidget(self.combo_video, 2)
        layout.addLayout(row_select)

        self.viewer = ImageViewer()
        self.viewer.setMinimumHeight(480)
        layout.addWidget(self.viewer, 1)

        row_transport = QHBoxLayout()
        self.btn_play = QPushButton("播放")
        self.btn_play.clicked.connect(self._toggle_play)
        row_transport.addWidget(self.btn_play)

        self.btn_prev = QPushButton("◀ 上一幀")
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        row_transport.addWidget(self.btn_prev)

        self.btn_next = QPushButton("下一幀 ▶")
        self.btn_next.clicked.connect(lambda: self._step(1))
        row_transport.addWidget(self.btn_next)

        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setRange(0, 0)
        self.slider_frame.valueChanged.connect(self._on_slider_changed)
        row_transport.addWidget(self.slider_frame, 1)

        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(0, 0)
        self.spin_frame.valueChanged.connect(self._on_spin_changed)
        row_transport.addWidget(self.spin_frame)

        self.lbl_frame_info = QLabel("0 / 0")
        row_transport.addWidget(self.lbl_frame_info)
        layout.addLayout(row_transport)

        row_save = QHBoxLayout()
        self.btn_save = QPushButton("存成 CCTV 圖片")
        self.btn_save.clicked.connect(self._on_save)
        row_save.addWidget(self.btn_save)
        self.lbl_status = QLabel("")
        row_save.addWidget(self.lbl_status, 1)
        layout.addLayout(row_save)

        self._set_controls_enabled(False)

    def _set_controls_enabled(self, enabled: bool):
        for w in (self.btn_play, self.btn_prev, self.btn_next, self.slider_frame, self.spin_frame, self.btn_save):
            w.setEnabled(enabled)

    # -- location / video population ------------------------------------
    def _refresh_locations(self):
        self.combo_location.blockSignals(True)
        self.combo_location.clear()
        self.combo_location.addItems(list_locations_with_footage())
        self.combo_location.blockSignals(False)
        if self.combo_location.count() > 0:
            self._on_location_changed(self.combo_location.currentIndex())
        else:
            self.lbl_status.setText("location/ 底下找不到任何含 footage/*.mp4 的 location。")

    def _on_location_changed(self, _index: int):
        location_code = self.combo_location.currentText()
        self.combo_video.blockSignals(True)
        self.combo_video.clear()
        if location_code:
            for mp4 in list_footage_videos(location_code):
                self.combo_video.addItem(mp4.name, str(mp4))
        self.combo_video.blockSignals(False)
        if self.combo_video.count() > 0:
            self._on_video_changed(0)
        else:
            self._close_player()

    def _on_video_changed(self, index: int):
        path = self.combo_video.itemData(index) if index >= 0 else None
        if not path:
            self._close_player()
            return
        self._load_video(path)

    # -- video loading ----------------------------------------------------
    def _close_player(self):
        self._pause()
        if self._player is not None:
            self._player.release()
        self._player = None
        self._video_path = None
        self._current_frame_bgr = None
        self.viewer.load_pixmap(QPixmap())
        self._frame_count = 0
        self.slider_frame.setRange(0, 0)
        self.spin_frame.setRange(0, 0)
        self.lbl_frame_info.setText("0 / 0")
        self._set_controls_enabled(False)

    def _load_video(self, path: str):
        self._pause()
        if self._player is not None:
            self._player.release()

        player = VideoPlayer(path)
        if not player.is_opened():
            QMessageBox.warning(self, "無法開啟影片", f"無法開啟：{path}")
            self._player = None
            self._video_path = None
            self._set_controls_enabled(False)
            return

        reported_n = player.frame_count()
        if reported_n <= 0:
            QMessageBox.warning(self, "錯誤", f"無法讀取影片幀數（frame_count <= 0）：{path}")
            player.release()
            self._player = None
            self._video_path = None
            self._set_controls_enabled(False)
            return

        self._player = player
        self._video_path = path
        self._fps = player.fps() or 25.0

        n = self._probe_usable_frame_count(reported_n)
        if n <= 0:
            QMessageBox.warning(
                self, "無法讀取影片",
                f"這支影片的容器 metadata 回報有 {reported_n} 幀，但連第 0 幀都無法解碼：{path}",
            )
            self._player.release()
            self._player = None
            self._video_path = None
            self._set_controls_enabled(False)
            return

        self._frame_count = n
        self.slider_frame.blockSignals(True)
        self.spin_frame.blockSignals(True)
        self.slider_frame.setRange(0, n - 1)
        self.spin_frame.setRange(0, n - 1)
        self.slider_frame.blockSignals(False)
        self.spin_frame.blockSignals(False)

        self._set_controls_enabled(True)
        self._goto_frame(0)

        if n < reported_n:
            self.lbl_status.setText(
                f"注意：容器 metadata 回報 {reported_n} 幀，但實際只有 {n} 幀可解碼"
                f"（可能是剪輯/remux 造成的中繼資料誤差），可用範圍已自動調整為 0–{n - 1}。"
            )
        else:
            self.lbl_status.setText(f"共 {n} 幀。")

    def _probe_usable_frame_count(self, reported_n: int) -> int:
        """CAP_PROP_FRAME_COUNT (container metadata) can overstate how many
        frames actually decode -- seen on this repo's kee-leyeh-mc.mp4
        (reports 199, only 126 decode; duration-based estimate agrees with
        126). Cheaply confirm the reported count by reading its last frame
        via _read_frame_robust (seek, falling back to sequential decode from
        the start); if that fails, binary-search the same way for the true
        last decodable frame instead of leaving a dead end the user only
        discovers by scrubbing into it.

        Must go through the sequential fallback, not a bare seek: some of
        this repo's AV1 footage (e.g. test21-4.mp4) seeks fail well before
        the real end even though every frame decodes fine sequentially --
        judging decodability by seek alone would wrongly truncate those."""
        if self._read_frame_robust(reported_n - 1) is not None:
            return reported_n

        last_good_idx = -1
        lo, hi = 0, reported_n - 2
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._read_frame_robust(mid) is not None:
                last_good_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return last_good_idx + 1

    # -- frame reading -----------------------------------------------------
    def _read_frame_robust(self, idx: int):
        """Try VideoPlayer.read_frame(idx) (CAP_PROP_POS_FRAMES seek) first;
        fall back to sequential read-and-discard from the start if that
        fails -- some of this repo's AV1-encoded footage silently fails to
        seek (see reference_point_calibration_tool.py's same fallback)."""
        frame = self._player.read_frame(idx)
        if frame is not None:
            return frame
        self._player.release()
        self._player = VideoPlayer(self._video_path)
        frame = None
        for _ in range(idx + 1):
            ret, f = self._player.read()
            if not ret:
                return None
            frame = f
        return frame

    def _goto_frame(self, idx: int):
        if self._player is None:
            return
        idx = max(0, min(idx, self._frame_count - 1))
        frame = self._read_frame_robust(idx)
        if frame is None:
            self._shrink_frame_count(idx)
            return
        self._current_frame_bgr = frame
        self._current_frame_index = idx
        self._display_frame(frame)
        self._sync_transport_controls(idx)

    def _shrink_frame_count(self, failed_idx: int):
        """Defensive fallback for a read failing mid-session despite the
        load-time probe in _probe_usable_frame_count -- binary-search (via
        _read_frame_robust, same seek-then-sequential-fallback reasoning as
        the load-time probe) for the last actually-decodable frame below
        failed_idx and shrink the usable range to match. Updates the status
        label only, no dialog -- matches _probe_usable_frame_count's
        load-time message for the same situation."""
        last_good_idx = -1
        last_good_frame = None
        lo, hi = 0, failed_idx - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            frame = self._read_frame_robust(mid)
            if frame is not None:
                last_good_idx, last_good_frame = mid, frame
                lo = mid + 1
            else:
                hi = mid - 1

        if last_good_idx < 0:
            self.lbl_status.setText("無法讀取影片：從第 0 幀開始就無法解碼。")
            self._set_controls_enabled(False)
            return

        usable = last_good_idx + 1
        self._frame_count = usable
        self.slider_frame.blockSignals(True)
        self.spin_frame.blockSignals(True)
        self.slider_frame.setRange(0, usable - 1)
        self.spin_frame.setRange(0, usable - 1)
        self.slider_frame.blockSignals(False)
        self.spin_frame.blockSignals(False)

        self._current_frame_bgr = last_good_frame
        self._current_frame_index = last_good_idx
        self._display_frame(last_good_frame)
        self._sync_transport_controls(last_good_idx)

        self.lbl_status.setText(
            f"注意：第 {failed_idx} 幀開始無法解碼，可用範圍已自動調整為 0–{usable - 1}。"
        )

    def _sync_transport_controls(self, idx: int):
        for w in (self.slider_frame, self.spin_frame):
            w.blockSignals(True)
            w.setValue(idx)
            w.blockSignals(False)
        self.lbl_frame_info.setText(f"{idx} / {self._frame_count - 1}")

    def _display_frame(self, frame_bgr):
        rgb = frame_bgr[:, :, ::-1]
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format_RGB888).copy()
        self.viewer.load_pixmap(QPixmap.fromImage(qimg))
        self.viewer.fitToView()

    def _on_slider_changed(self, value: int):
        if value != self._current_frame_index:
            self._pause()
            self._goto_frame(value)

    def _on_spin_changed(self, value: int):
        if value != self._current_frame_index:
            self._pause()
            self._goto_frame(value)

    def _step(self, delta: int):
        self._pause()
        self._goto_frame(self._current_frame_index + delta)

    # -- playback ------------------------------------------------------
    def _toggle_play(self):
        if self._is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        if self._player is None or self._current_frame_index >= self._frame_count - 1:
            return
        self._is_playing = True
        self.btn_play.setText("暫停")
        self._timer.start(max(1, int(1000 / self._fps)))

    def _pause(self):
        self._is_playing = False
        self.btn_play.setText("播放")
        self._timer.stop()

    def _advance_playback(self):
        if not self._is_playing or self._player is None:
            return
        # Sequential .read() during playback (no re-seek per frame) so
        # smooth forward playback doesn't pay the seek cost -- only
        # explicit jumps (slider/spin/step) go through _goto_frame's
        # seek-with-fallback path.
        ret, frame = self._player.read()
        if not ret:
            failed_idx = self._current_frame_index + 1
            self._pause()
            if failed_idx < self._frame_count:
                self._shrink_frame_count(failed_idx)
            return
        self._current_frame_bgr = frame
        self._current_frame_index += 1
        self._display_frame(frame)
        self._sync_transport_controls(self._current_frame_index)
        if self._current_frame_index >= self._frame_count - 1:
            self._pause()
            return
        self._timer.start(max(1, int(1000 / self._fps)))

    # -- save ------------------------------------------------------------
    def _on_save(self):
        if self._current_frame_bgr is None:
            QMessageBox.warning(self, "沒有畫面", "請先選擇一支影片並定位到要輸出的那一幀。")
            return
        location_code = self.combo_location.currentText()
        if not location_code:
            QMessageBox.warning(self, "沒有 location", "請先選擇 location。")
            return

        out_dir = LOCATION_ROOT / location_code
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"cctv_{location_code}.png"

        if out_path.exists():
            reply = QMessageBox.question(
                self,
                "確認覆寫",
                f"{out_path} 已存在，是否覆寫？\n此動作無法復原（除非該檔案本身有版本控制）。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        ok = cv2.imwrite(str(out_path), self._current_frame_bgr)
        if not ok:
            QMessageBox.critical(self, "儲存失敗", f"無法寫入 {out_path}")
            return

        h, w = self._current_frame_bgr.shape[:2]
        msg = f"已儲存 frame {self._current_frame_index} 至 {out_path}（{w}x{h}，與原影片解析度相同）"
        self.lbl_status.setText(msg)
        QMessageBox.information(self, "已儲存", msg)


class CCTVFramePickerWindow(QMainWindow):
    def __init__(self, initial_location_code: Optional[str] = None, initial_video: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("TrafficLab CCTV Frame Picker")
        self.setWindowFlags(Qt.Window)
        self.resize(1280, 800)
        self.widget = CCTVFramePickerWidget(
            initial_location_code=initial_location_code, initial_video=initial_video
        )
        self.setCentralWidget(self.widget)
