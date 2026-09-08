"""Standalone GUI tool: seg yolobox localization -> wheel_pair correction.

Three tabs, each handing off to the next:

  1. 推論      pick a location, run the seg tight-box inference config
  2. seg 結果  show the resulting trajectory plot, offer the wheel_pair correction
  3. 對照      show the before/after trajectory plots side by side

The correction step reuses the seg replay's own mask_contour polygons as the
car-segmenter source for the OpenPifPaf pass (trafficlab/trajectory/
seg_mask_adapter.py), so the segmentation model runs once for the whole
workflow instead of once per pass. Because both replays then carry the same
tracker ids, the correction is driven by an identity track_id_map rather than
sat_coords proximity guessing.

See AGENTS.md 常用指令 (seg yolobox + wheel pair 修正工具).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from trafficlab.gui.inference_session import InferenceSession
from trafficlab.gui.tabs.calibration_stage.undistort_stage import ImageViewer
from trafficlab.inference.pipeline import InferencePipeline
from trafficlab.trajectory.io import (
    default_wheelpair_corrected_output_path,
    load_json,
    write_json,
)
from trafficlab.trajectory.seg_mask_adapter import (
    default_seg_masks_output_path,
    replay_to_seg_mask_records,
)
from trafficlab.trajectory.wheel_pair_correction import correct_replay

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCATION_ROOT = REPO_ROOT / "location"
DEFAULT_CONFIG_PATH = REPO_ROOT / "inference_config.yaml"
# The seg tight-box profile this tool runs. Fixed rather than selectable
# because the whole point of the tool is one prepared seg -> wheel_pair path;
# override it from the launcher's --config-name if a different profile is
# wanted. yolo_seg_tight_box_cpu (not seg_default) because seg_default pins
# device: mps, which only exists on Apple Silicon.
DEFAULT_CONFIG_NAME = "yolo_seg_tight_box_cpu"
OUTPUT_ROOT = str(REPO_ROOT / "output")
# run_keypoints_openpifpaf.py tags a PifPaf fragment that matched no mask with
# its per-frame index + 500; keep this in sync with that offset.
LEFTOVER_ID_OFFSET = 500

INFER_TAB, SEG_RESULT_TAB, COMPARE_TAB = 0, 1, 2


# ----------------------------------------------------------------------
# location / path helpers


def resolve_g_projection(location_code: str, location_root: Path = LOCATION_ROOT) -> Optional[Path]:
    """location/<code>/G_projection[_svg]_<code>.json, same order as the CLI tools.

    Matched case-insensitively on the filename: several sites on disk use a
    capitalised code in the file name (G_projection_Hsinchu1.json) while the
    directory itself is lowercase.
    """
    loc_dir = location_root / location_code
    if not loc_dir.is_dir():
        return None
    wanted = [
        f"g_projection_{location_code}.json".lower(),
        f"g_projection_svg_{location_code}.json".lower(),
    ]
    by_lower = {p.name.lower(): p for p in loc_dir.glob("*.json")}
    for name in wanted:
        if name in by_lower:
            return by_lower[name]
    # Fall back to any G_projection_*.json in the directory (a site whose file
    # name does not match its directory name at all).
    for path in sorted(loc_dir.glob("*.json")):
        lower = path.name.lower()
        if lower.startswith("g_projection_"):
            return path
    return None


def list_footage_videos(location_code: str, location_root: Path = LOCATION_ROOT) -> list[Path]:
    footage_dir = location_root / location_code / "footage"
    if not footage_dir.is_dir():
        return []
    return sorted(footage_dir.glob("*.mp4"))


def list_runnable_locations(location_root: Path = LOCATION_ROOT) -> list[str]:
    """Locations that have both footage and a G_projection config."""
    if not location_root.is_dir():
        return []
    codes = []
    for d in sorted(location_root.iterdir()):
        if not d.is_dir():
            continue
        if list_footage_videos(d.name, location_root) and resolve_g_projection(d.name, location_root):
            codes.append(d.name)
    return codes


def seg_replay_path(location_code: str, video_path: Path, config_name: str) -> Optional[Path]:
    """Where InferencePipeline writes this location/video's seg replay."""
    import yaml

    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    configs = raw.get("configs") if isinstance(raw, dict) else None
    if not isinstance(configs, dict) or config_name not in configs:
        return None
    cfg = configs[config_name]
    weights = cfg.get("model", {}).get("weights", "Unknown")
    tracker_type = cfg.get("tracking", {}).get("tracker_type", "default")
    is_seg = cfg.get("model", {}).get("type") == "seg"
    config_dir = InferencePipeline.config_output_dir(
        OUTPUT_ROOT, weights, tracker_type, config_name, is_seg=is_seg
    )
    return Path(config_dir) / location_code / f"{video_path.stem}.json.gz"


def trajectory_png_path(replay_path: Path) -> Path:
    """<stem>.trajectories.png next to the replay — trajectory_tools.py's default."""
    name = replay_path.name
    for suffix in (".json.gz", ".json"):
        if name.endswith(suffix):
            return replay_path.with_name(name[: -len(suffix)] + ".trajectories.png")
    return replay_path.with_name(replay_path.stem + ".trajectories.png")


# ----------------------------------------------------------------------
# workers


class PlotWorker(QObject):
    """Render replays to trajectory PNGs off the GUI thread.

    trafficlab.trajectory.plotting forces the Agg backend at import, so this
    never touches Qt from a worker thread.
    """

    sig_log = pyqtSignal(str)
    sig_finished = pyqtSignal(dict)  # {label: png path or ""}
    sig_error = pyqtSignal(str)

    def __init__(self, jobs: list[tuple[str, Path, str]]):
        """jobs: (label, replay_path, plot_title)."""
        super().__init__()
        self.jobs = jobs

    @pyqtSlot()
    def run(self):
        from trafficlab.trajectory.plotting import TrajectoryPlotter

        results: dict[str, str] = {}
        try:
            for label, replay_path, title in self.jobs:
                out_png = trajectory_png_path(replay_path)
                try:
                    plotter = TrajectoryPlotter.from_file(replay_path, project_root=REPO_ROOT)
                    plotter.plot(out_png, show_id_labels=True, title=title)
                    results[label] = str(out_png)
                    self.sig_log.emit(f"[plot] {label}: {out_png}")
                except ValueError as exc:
                    # "No trajectories matched" — a real, reportable outcome
                    # for a short clip, not a crash.
                    results[label] = ""
                    self.sig_log.emit(f"[plot] {label}: 無法繪圖 — {exc}")
        except Exception:
            self.sig_error.emit(traceback.format_exc())
            return
        self.sig_finished.emit(results)


class WheelPairWorker(QObject):
    """seg replay -> seg-mask records -> OpenPifPaf wheel_pair -> correction."""

    sig_log = pyqtSignal(str)
    sig_finished = pyqtSignal(dict)
    sig_error = pyqtSignal(str)

    def __init__(self, target_replay: Path, video_path: Path, g_proj_path: Path):
        super().__init__()
        self.target_replay = Path(target_replay)
        self.video_path = Path(video_path)
        self.g_proj_path = Path(g_proj_path)
        self._proc: Optional[subprocess.Popen] = None
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    # -- steps ---------------------------------------------------------
    def _write_seg_masks(self, target_data: dict) -> Path:
        out_data, stats = replay_to_seg_mask_records(target_data)
        location_code = target_data.get("location_code")
        masks_path = REPO_ROOT / default_seg_masks_output_path(self.target_replay, location_code)
        write_json(masks_path, out_data)
        self.sig_log.emit(f"[1/3] 重用 seg mask：{stats.summary()}")
        self.sig_log.emit(f"      -> {masks_path}")
        return masks_path

    def _run_openpifpaf(self, masks_path: Path) -> Path:
        # .from-seg-masks marks the provenance and, more importantly, keeps
        # this tool from clobbering a wheel_pair replay produced by a manual
        # run_keypoints_openpifpaf.py run at the same canonical path.
        wp_out = REPO_ROOT / "output" / "wheel_pair" / str(
            self.target_replay.parent.name
        ) / f"{self.video_path.stem}.from-seg-masks.json.gz"
        # No --cad-template here on purpose: run_keypoints_openpifpaf.py defaults
        # --localizer wheel_pair to the committed nissan_juke_nismo template
        # (real wheel-hub height), and this tool follows that default.
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_keypoints_openpifpaf.py"),
            "--video", str(self.video_path),
            "--g-proj", str(self.g_proj_path),
            "--method", "segmentation",
            "--seg-masks-json", str(masks_path),
            "--localizer", "wheel_pair",
            "--out", str(wp_out),
        ]
        self.sig_log.emit(f"[2/3] wheel_pair 推論（--method segmentation，重用上面的 mask）")
        self.sig_log.emit("      " + " ".join(cmd))

        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["PYTHONUNBUFFERED"] = "1"
        # The child prints non-ASCII (e.g. the "→" in its progress line). Pin
        # both ends to UTF-8: without this, text=True decodes with the console
        # locale codec (cp950 on a zh-TW Windows) and dies on the first such
        # line. errors="replace" so a stray byte degrades one character rather
        # than aborting a long inference run.
        env["PYTHONIOENCODING"] = "utf-8"
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                self.sig_log.emit("      " + line.rstrip())
            code = self._proc.wait()
        except BaseException:
            # Never leave a multi-minute OpenPifPaf run orphaned behind a
            # failure in this loop.
            if self._proc.poll() is None:
                self._proc.kill()
                self._proc.wait()
            raise
        if self._stop_requested:
            raise RuntimeError("使用者中止了 wheel_pair 推論。")
        if code != 0:
            raise RuntimeError(f"run_keypoints_openpifpaf.py 失敗（exit code {code}），詳見上方輸出。")
        if not wp_out.exists():
            raise RuntimeError(f"wheel_pair 推論回報成功，但找不到輸出檔 {wp_out}。")
        return wp_out

    def _px_per_meter(self) -> float:
        g_data = load_json(self.g_proj_path)
        value = (g_data.get("parallax") or {}).get("px_per_meter")
        if not value or value <= 0:
            raise RuntimeError(
                f"{self.g_proj_path} 沒有可用的 parallax.px_per_meter，無法把公尺門檻換算成像素。"
            )
        return float(value)

    def _correct(self, target_data: dict, wp_data: dict) -> tuple[Path, str]:
        px_per_m = self._px_per_meter()

        # Both replays' tracked_ids came from the *same* seg mask records, so
        # the correspondence is exact identity — no need to guess it from
        # sat_coords proximity (correct_replay's default). Restricting the map
        # to ids actually present on both sides leaves any track the
        # wheel_pair pass never localized completely untouched.
        target_ids = _tracked_ids(target_data)
        wp_ids = _tracked_ids(wp_data)
        shared = sorted(target_ids & wp_ids)

        # run_keypoints_openpifpaf.py tags a PifPaf fragment that fell inside
        # no mask with its per-frame index offset by 500 (LEFTOVER_ID_OFFSET).
        # Those ids are not tracks at all, so an identity map is only sound
        # while no real track id can collide with them.
        collisions = sorted(t for t in shared if t >= LEFTOVER_ID_OFFSET)
        if collisions:
            self.sig_log.emit(
                f"[3/3] ⚠ target 有 tracked_id >= {LEFTOVER_ID_OFFSET}（{collisions}），"
                "會跟 PifPaf 未匹配碎片的合成 id 撞號，無法安全使用 identity map — "
                "退回 sat_coords 鄰近比對。"
            )
            track_id_map = None
        else:
            track_id_map = {tid: tid for tid in shared} or None

        if track_id_map:
            self.sig_log.emit(
                f"[3/3] 修正：兩份 replay 共用 tracker id，"
                f"以 identity track_id_map 對應 {len(shared)} 條 track {shared}"
            )
        elif not collisions:
            self.sig_log.emit(
                "[3/3] 修正：兩份 replay 沒有共同的 tracked_id，"
                "退回 sat_coords 鄰近比對（correct_replay 預設行為）"
            )

        report = correct_replay(
            target_data,
            wp_data,
            max_match_distance_px=7.0 * px_per_m,
            min_fit_span_px=2.0 * px_per_m,
            track_id_map=track_id_map,
        )

        lines = [f"總 target track 數：{report.total_target_tracks}"]
        for r in sorted(report.results, key=lambda r: r.tracked_id):
            if r.status == "corrected":
                lines.append(
                    f"  tid={r.tracked_id:>4}  已修正  fit={r.n_fit_points:3d}  "
                    f"frozen={r.n_frozen_frames:3d}  missing_anchor={r.n_missing_anchor_frames:3d}  "
                    f"平移={r.translation_px:7.2f}px  旋轉={r.rotation_deg:+6.2f}°"
                )
            else:
                lines.append(f"  tid={r.tracked_id:>4}  {r.status}  fit={r.n_fit_points}")
        lines.append(
            f"\n{len(report.corrected_tracks)} 條已修正，"
            f"{len(report.skipped_tracks)} 條跳過（共 {report.total_target_tracks} 條）。"
        )
        summary = "\n".join(lines)
        for line in lines:
            self.sig_log.emit("      " + line)

        out_path = default_wheelpair_corrected_output_path(self.target_replay)
        write_json(out_path, target_data)
        self.sig_log.emit(f"      -> {out_path}")
        return Path(out_path), summary

    @pyqtSlot()
    def run(self):
        try:
            target_data = load_json(self.target_replay)
            masks_path = self._write_seg_masks(target_data)
            if self._stop_requested:
                raise RuntimeError("使用者中止。")
            wp_path = self._run_openpifpaf(masks_path)
            wp_data = load_json(wp_path)
            corrected_path, summary = self._correct(target_data, wp_data)
        except Exception as exc:
            self.sig_error.emit(f"{exc}\n\n{traceback.format_exc()}")
            return
        self.sig_finished.emit(
            {
                "wheel_pair": str(wp_path),
                "corrected": str(corrected_path),
                "summary": summary,
            }
        )


def _tracked_ids(data: dict) -> set[int]:
    ids: set[int] = set()
    for frame in data.get("frames") or []:
        for obj in frame.get("objects") or []:
            tid = obj.get("tracked_id")
            if tid is not None:
                ids.add(int(tid))
    return ids


# ----------------------------------------------------------------------
# widget


class SegWheelPairWidget(QWidget):
    def __init__(self, initial_location_code: Optional[str] = None,
                 config_name: str = DEFAULT_CONFIG_NAME, parent=None):
        super().__init__(parent)
        self.config_name = config_name
        self._thread: Optional[QThread] = None
        self._worker = None
        self._video_path: Optional[Path] = None
        self._g_proj_path: Optional[Path] = None
        self._seg_replay: Optional[Path] = None
        self._corrected_replay: Optional[Path] = None

        self._build_ui()
        self._refresh_locations()
        if initial_location_code:
            idx = self.combo_location.findText(initial_location_code)
            if idx >= 0:
                self.combo_location.setCurrentIndex(idx)
        self._on_location_changed()

    # -- ui ------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_infer_tab(), "1. 推論")
        self.tabs.addTab(self._build_seg_result_tab(), "2. seg yolobox 結果")
        self.tabs.addTab(self._build_compare_tab(), "3. 修正前後對照")
        self.tabs.setTabEnabled(SEG_RESULT_TAB, False)
        self.tabs.setTabEnabled(COMPARE_TAB, False)
        layout.addWidget(self.tabs)

    def _build_infer_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        box = QGroupBox("選擇 location")
        box_layout = QVBoxLayout(box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Location:"))
        self.combo_location = QComboBox()
        self.combo_location.currentIndexChanged.connect(self._on_location_changed)
        row.addWidget(self.combo_location, 1)
        box_layout.addLayout(row)

        self.lbl_resolved = QLabel("—")
        self.lbl_resolved.setWordWrap(True)
        self.lbl_resolved.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box_layout.addWidget(self.lbl_resolved)
        layout.addWidget(box)

        self.btn_infer = QPushButton("執行 seg yolobox 推論")
        self.btn_infer.clicked.connect(self._on_run_inference)
        layout.addWidget(self.btn_infer)

        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100)
        layout.addWidget(self.pbar)

        self.log_infer = QPlainTextEdit()
        self.log_infer.setReadOnly(True)
        layout.addWidget(self.log_infer, 1)
        return page

    def _build_seg_result_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.lbl_seg_result = QLabel("—")
        self.lbl_seg_result.setWordWrap(True)
        self.lbl_seg_result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.lbl_seg_result)

        self.view_seg = ImageViewer()
        self.view_seg.setMinimumHeight(400)
        layout.addWidget(self.view_seg, 1)

        self.btn_wheelpair = QPushButton("執行 wheel pair 修正")
        self.btn_wheelpair.clicked.connect(self._on_run_wheelpair)
        layout.addWidget(self.btn_wheelpair)

        self.log_wp = QPlainTextEdit()
        self.log_wp.setReadOnly(True)
        self.log_wp.setMaximumHeight(220)
        layout.addWidget(self.log_wp)
        return page

    def _build_compare_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        splitter = QSplitter(Qt.Horizontal)

        before = QWidget()
        before_layout = QVBoxLayout(before)
        before_layout.addWidget(QLabel("修正前（seg yolobox）"))
        self.view_before = ImageViewer()
        before_layout.addWidget(self.view_before, 1)
        splitter.addWidget(before)

        after = QWidget()
        after_layout = QVBoxLayout(after)
        after_layout.addWidget(QLabel("修正後（wheel pair corrected）"))
        self.view_after = ImageViewer()
        after_layout.addWidget(self.view_after, 1)
        splitter.addWidget(after)

        splitter.setSizes([600, 600])
        layout.addWidget(splitter, 1)

        self.txt_report = QPlainTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setMaximumHeight(220)
        layout.addWidget(self.txt_report)
        return page

    # -- location resolution -------------------------------------------
    def _refresh_locations(self):
        self.combo_location.blockSignals(True)
        self.combo_location.clear()
        for code in list_runnable_locations():
            self.combo_location.addItem(code)
        self.combo_location.blockSignals(False)

    def _on_location_changed(self, *_):
        code = self.combo_location.currentText()
        self._video_path = None
        self._g_proj_path = None
        if not code:
            self.lbl_resolved.setText("找不到同時具備 footage/*.mp4 與 G_projection 的 location。")
            self.btn_infer.setEnabled(False)
            return

        videos = list_footage_videos(code)
        g_proj = resolve_g_projection(code)
        self._video_path = videos[0] if videos else None
        self._g_proj_path = g_proj

        lines = [f"config：{self.config_name}"]
        if self._video_path:
            lines.append(f"影片：{self._video_path}")
            if len(videos) > 1:
                lines.append(
                    f"⚠ 這個 location 有 {len(videos)} 支影片，只會處理第一支："
                    + "、".join(v.name for v in videos)
                )
        else:
            lines.append("⚠ 找不到 footage/*.mp4")
        lines.append(f"G_projection：{g_proj if g_proj else '⚠ 找不到'}")
        self.lbl_resolved.setText("\n".join(lines))
        self.btn_infer.setEnabled(bool(self._video_path and g_proj))

    # -- logging -------------------------------------------------------
    def _log(self, pane: QPlainTextEdit, msg: str):
        pane.appendPlainText(msg)
        pane.verticalScrollBar().setValue(pane.verticalScrollBar().maximum())

    def _show_png(self, viewer: ImageViewer, png_path: str, what: str) -> bool:
        if not png_path or not Path(png_path).exists():
            return False
        pixmap = QPixmap(png_path)
        if pixmap.isNull():
            self._log(self.log_wp, f"[plot] 無法讀取 {what} 圖片：{png_path}")
            return False
        viewer.load_pixmap(pixmap)
        viewer.fitInView(viewer.sceneRect(), Qt.KeepAspectRatio)
        return True

    # -- thread plumbing -----------------------------------------------
    def _start_worker(self, worker: QObject, on_finished, log_pane: QPlainTextEdit,
                      *, progress_slot=None, big_stack: bool = False):
        thread = QThread()
        if big_stack:
            # 64MB — same reason as tab_inference.py: OpenBLAS's parallel LU
            # decomposition overflows the default worker stack.
            thread.setStackSize(64 * 1024 * 1024)
        worker.moveToThread(thread)
        worker.sig_log.connect(lambda m: self._log(log_pane, m))
        worker.sig_error.connect(lambda m: self._on_worker_error(m, log_pane))
        if progress_slot is not None and hasattr(worker, "sig_progress"):
            worker.sig_progress.connect(progress_slot)
        if hasattr(worker, "sig_finished"):
            worker.sig_finished.connect(on_finished)
        thread.started.connect(worker.run)
        self._thread, self._worker = thread, worker
        thread.start()

    def _stop_thread(self):
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

    def _on_worker_error(self, msg: str, log_pane: QPlainTextEdit):
        self._log(log_pane, msg)
        self._stop_thread()
        self.btn_infer.setEnabled(True)
        self.btn_wheelpair.setEnabled(True)
        QMessageBox.critical(self, "執行失敗", msg.strip().splitlines()[0] if msg.strip() else "未知錯誤")

    # -- step 1: inference ---------------------------------------------
    def _on_run_inference(self):
        if not (self._video_path and self._g_proj_path):
            return
        code = self.combo_location.currentText()
        self.btn_infer.setEnabled(False)
        self.pbar.setValue(0)
        self.log_infer.clear()
        self._log(self.log_infer, f"=== seg yolobox 推論：{code} / {self._video_path.name} ===")

        worker = InferenceSession(
            location_code=code,
            footage_path=str(self._video_path),
            config_path=str(DEFAULT_CONFIG_PATH),
            output_root=OUTPUT_ROOT,
            g_proj_path=str(self._g_proj_path),
            config_name=self.config_name,
        )
        self._start_worker(
            worker,
            self._on_inference_finished,
            self.log_infer,
            progress_slot=self.pbar.setValue,
            big_stack=True,
        )

    @pyqtSlot()
    def _on_inference_finished(self):
        self._stop_thread()
        self.btn_infer.setEnabled(True)

        code = self.combo_location.currentText()
        replay = seg_replay_path(code, self._video_path, self.config_name)
        if replay is None or not replay.exists():
            msg = f"推論結束，但找不到預期的輸出檔：{replay}"
            self._log(self.log_infer, msg)
            QMessageBox.warning(self, "找不到輸出", msg)
            return

        self._seg_replay = replay
        self._log(self.log_infer, f"輸出：{replay}")
        self.lbl_seg_result.setText(f"seg yolobox replay：{replay}\n繪圖中…")
        self.tabs.setTabEnabled(SEG_RESULT_TAB, True)
        self.tabs.setCurrentIndex(SEG_RESULT_TAB)

        self.btn_wheelpair.setEnabled(False)
        worker = PlotWorker([("seg", replay, f"seg yolobox — {code}")])
        self._start_worker(worker, self._on_seg_plot_finished, self.log_wp)

    @pyqtSlot(dict)
    def _on_seg_plot_finished(self, results: dict):
        self._stop_thread()
        self.btn_wheelpair.setEnabled(True)
        png = results.get("seg", "")
        if self._show_png(self.view_seg, png, "seg yolobox"):
            self.lbl_seg_result.setText(f"seg yolobox replay：{self._seg_replay}\n軌跡圖：{png}")
        else:
            self.lbl_seg_result.setText(
                f"seg yolobox replay：{self._seg_replay}\n（沒有可繪製的軌跡，詳見下方 log）"
            )

    # -- step 2: wheel_pair correction ---------------------------------
    def _on_run_wheelpair(self):
        if not (self._seg_replay and self._video_path and self._g_proj_path):
            return
        self.btn_wheelpair.setEnabled(False)
        self.log_wp.clear()
        self._log(self.log_wp, "=== wheel pair 修正 ===")
        worker = WheelPairWorker(self._seg_replay, self._video_path, self._g_proj_path)
        self._start_worker(worker, self._on_wheelpair_finished, self.log_wp, big_stack=True)

    @pyqtSlot(dict)
    def _on_wheelpair_finished(self, results: dict):
        self._stop_thread()
        self.btn_wheelpair.setEnabled(True)
        self._corrected_replay = Path(results["corrected"])
        self.txt_report.setPlainText(
            f"wheel_pair replay：{results['wheel_pair']}\n"
            f"修正後 replay：{results['corrected']}\n\n{results['summary']}"
        )
        self.tabs.setTabEnabled(COMPARE_TAB, True)
        self.tabs.setCurrentIndex(COMPARE_TAB)

        code = self.combo_location.currentText()
        worker = PlotWorker(
            [
                ("before", self._seg_replay, f"seg yolobox（修正前）— {code}"),
                ("after", self._corrected_replay, f"wheel pair corrected（修正後）— {code}"),
            ]
        )
        self._start_worker(worker, self._on_compare_plot_finished, self.log_wp)

    @pyqtSlot(dict)
    def _on_compare_plot_finished(self, results: dict):
        self._stop_thread()
        self._show_png(self.view_before, results.get("before", ""), "修正前")
        self._show_png(self.view_after, results.get("after", ""), "修正後")

    # -- shutdown ------------------------------------------------------
    def closeEvent(self, event):
        if self._worker is not None and hasattr(self._worker, "request_stop"):
            self._worker.request_stop()
        self._stop_thread()
        super().closeEvent(event)


class SegWheelPairWindow(QMainWindow):
    def __init__(self, initial_location_code: Optional[str] = None,
                 config_name: str = DEFAULT_CONFIG_NAME):
        super().__init__()
        self.setWindowTitle("TrafficLab seg yolobox + wheel pair 修正")
        self.setWindowFlags(Qt.Window)
        self.resize(1400, 900)
        self.widget = SegWheelPairWidget(
            initial_location_code=initial_location_code, config_name=config_name
        )
        self.setCentralWidget(self.widget)

    def closeEvent(self, event):
        self.widget.close()
        super().closeEvent(event)
