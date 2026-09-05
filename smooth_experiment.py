#!/usr/bin/env python3
"""
獨立腳本：實驗 2D 軌跡後處理平滑化
載入 inference 輸出的 JSON.gz，比較原始軌跡 vs 平滑後軌跡。
確認效果滿意後再整合進視覺化層。

用法：
  python smooth_experiment.py <json_gz_path> [--sat <sat_image_path>] [--window 21] [--poly 2]

範例：
  python smooth_experiment.py \
    output/model-yolov8s-seg_tracker-bytetrack/seg_default/Yilan-Wujie/Yilan-Wujie.json.gz \
    --sat location/Yilan-Wujie/sat_Yilan-Wujie.png
"""

import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import savgol_filter


MIN_TRACK_LEN = 5  # 幀數太少的 track 跳過不顯示


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_json_gz(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def extract_trajectories(data: dict) -> dict:
    """
    回傳 {tid: {"cls": str, "frames": [int, ...], "coords": [[px, py], ...]}}
    coords 是 inference 寫入的 sat_coords（若有開 Kalman 則已是平滑後座標）
    """
    tracks = {}
    for frame in data.get("frames", []):
        fi = frame["frame_index"]
        for obj in frame.get("objects", []):
            tid = obj.get("tracked_id")
            coord = obj.get("sat_coords")
            cls = obj.get("class", "?")
            if tid is None or coord is None:
                continue
            if tid not in tracks:
                tracks[tid] = {"cls": cls, "frames": [], "coords": []}
            tracks[tid]["frames"].append(fi)
            tracks[tid]["coords"].append(coord)
    return tracks


# ── 平滑演算法 ─────────────────────────────────────────────────────────────────

def smooth_savgol(coords: list, window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay filter：對 x, y 分別做多項式擬合平滑。"""
    arr = np.array(coords, dtype=float)
    n = len(arr)
    # window 必須是奇數且 > poly，不能超過資料長度
    w = min(window, n if n % 2 == 1 else n - 1)
    w = max(w, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)
    if w > n:
        return arr
    return np.stack([
        savgol_filter(arr[:, 0], w, poly),
        savgol_filter(arr[:, 1], w, poly),
    ], axis=1)


def smooth_moving_average(coords: list, window: int) -> np.ndarray:
    """移動平均：對 x, y 分別做等權重滑動平均。"""
    arr = np.array(coords, dtype=float)
    w = min(window, len(arr))
    kernel = np.ones(w) / w
    sx = np.convolve(arr[:, 0], kernel, mode="same")
    sy = np.convolve(arr[:, 1], kernel, mode="same")
    return np.stack([sx, sy], axis=1)


# ── 視覺化 ─────────────────────────────────────────────────────────────────────

def load_sat_image(path: str):
    if path and Path(path).exists():
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def plot_comparison(tracks: dict, sat_img, window: int, poly: int):
    skipped = {tid: len(t["coords"]) for tid, t in tracks.items() if len(t["coords"]) < MIN_TRACK_LEN}
    if skipped:
        print(f"跳過（幀數 < {MIN_TRACK_LEN}）：{skipped}")

    valid = {tid: t for tid, t in tracks.items() if len(t["coords"]) >= MIN_TRACK_LEN}
    if not valid:
        print("沒有足夠長度的 track 可顯示。")
        return

    colors = cm.tab20(np.linspace(0, 1, len(valid)))

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    titles = [
        "原始軌跡（Kalman only）",
        f"Savitzky-Golay（window={window}, poly={poly}）",
        f"移動平均（window={window}）",
    ]

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        if sat_img is not None:
            ax.imshow(sat_img, origin="upper")
        else:
            ax.invert_yaxis()  # 沒有背景圖時手動翻轉 y 軸（像素座標向下）

    for (tid, track), color in zip(valid.items(), colors):
        coords = track["coords"]
        raw = np.array(coords, dtype=float)
        sg  = smooth_savgol(coords, window, poly)
        ma  = smooth_moving_average(coords, window)
        label = f"ID {tid} ({track['cls']})"

        for ax, arr in zip(axes, [raw, sg, ma]):
            ax.plot(arr[:, 0], arr[:, 1], color=color, linewidth=1.8, label=label)
            ax.scatter(arr[0, 0],  arr[0, 1],  color=color, s=60, marker="o", zorder=5)  # 起點
            ax.scatter(arr[-1, 0], arr[-1, 1], color=color, s=60, marker="x", zorder=5)  # 終點

    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("軌跡後處理平滑比較（o=起點 x=終點）", fontsize=14)
    plt.tight_layout()
    plt.show()


def print_summary(tracks: dict, window: int, poly: int, fps: float):
    """印出每個 track 的平均速度估算（平滑後路徑長 / 時間）。"""
    from trafficlab.projection.g_projection import GProjection  # 只在有需要時 import
    print("\n── 平滑後平均速度估算（參考用）──")
    print(f"{'TID':>5}  {'Class':<12}  {'Frames':>6}  {'Path(px)':>10}  {'SavGol Path(px)':>16}")
    for tid, track in tracks.items():
        if len(track["coords"]) < MIN_TRACK_LEN:
            continue
        raw = np.array(track["coords"], dtype=float)
        sg  = smooth_savgol(track["coords"], window, poly)
        raw_len = float(np.sum(np.linalg.norm(np.diff(raw, axis=0), axis=1)))
        sg_len  = float(np.sum(np.linalg.norm(np.diff(sg,  axis=0), axis=1)))
        print(f"{tid:>5}  {track['cls']:<12}  {len(track['coords']):>6}  {raw_len:>10.1f}  {sg_len:>16.1f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="2D 軌跡後處理平滑實驗")
    parser.add_argument("json_gz", help="inference 輸出的 .json.gz 路徑")
    parser.add_argument("--sat",    default=None,  help="衛星圖路徑（選填，疊在背景）")
    parser.add_argument("--window", type=int, default=21, help="平滑視窗大小（奇數，預設 21）")
    parser.add_argument("--poly",   type=int, default=2,  help="SavGol 多項式次數（預設 2）")
    args = parser.parse_args()

    print(f"載入 {args.json_gz} ...")
    data   = load_json_gz(args.json_gz)
    tracks = extract_trajectories(data)
    fps    = data.get("meta", {}).get("fps", 30)
    print(f"偵測到 {len(tracks)} 個 track，影片 fps={fps}")

    sat_img = load_sat_image(args.sat)
    if sat_img is not None:
        print(f"衛星圖：{args.sat}  ({sat_img.shape[1]}×{sat_img.shape[0]})")
    else:
        print("未載入衛星圖，以空白背景顯示。")

    print_summary(tracks, args.window, args.poly, fps)
    plot_comparison(tracks, sat_img, args.window, args.poly)


if __name__ == "__main__":
    main()
