"""軌跡後處理平滑工具。

GUI 的 SAT 顯示（trafficlab/gui/tabs/tab_visualization.py）與離線實驗腳本
（smooth_experiment.py）共用這裡的實作，確保兩邊看到的平滑結果一致。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


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
