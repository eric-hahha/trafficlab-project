# Bbox 幾何朝向估計

## 是什麼

`scripts/archive/bbox_heading.py` 的 `estimate_heading_from_bbox()`：純幾何、不需要關鍵點或額外模型，只用 YOLO bbox 的長寬比＋相機到車輛的視線角度（line-of-sight）來猜車輛朝向。

- 相機→車輛連線角度 `los_angle`
- bbox 長寬比 ≥ 1.8（扁）→ 側面朝鏡頭，行進方向垂直於視線，兩個候選角（`los_angle ± 90°`）
- 長寬比 ≤ 1.3（接近方形）→ 正面/背面朝鏡頭，行進方向平行於視線，兩個候選角（同向／反向 180°）
- 介於中間 → 四個候選角都給，信心值固定 0.1
- 有 `road_heading`（路網方向提示）時選最接近路網方向的候選；沒有的話預設選「背離相機」那個候選
- 回傳 `(heading_deg, confidence)`，`confidence` 落在 0–1

跟 h-aware / CarFusion 的關鍵點定位法是完全不同、更輕量的手段——不需要姿態模型，任何有 bbox 的偵測都能用。

## 目前狀態：已歸檔，pipeline 不再依賴

原本有兩個呼叫端，現在都已移除：

- `trafficlab/inference/pipeline.py` 原本在 motion-based heading 算不出來時（新軌跡、還沒有足夠歷史）拿 `estimate_heading_from_bbox()` 當 fallback；已移除該 fallback，heading 算不出來時就維持 `None`（`have_heading=False`），不再退回這個較不可靠的幾何猜測。
- `trafficlab/motion/wheel_localization.py` 原本 import 它的小工具函式 `_closest_candidate()`（配 180° 消歧義）；已把 `_closest_candidate()`／`_angular_distance()` 直接內聯進 `wheel_localization.py` 自己，不再依賴這個模組。

因此整個模組（連同它的評估腳本）一起移到 `scripts/archive/bbox_heading.py`，不再是 `trafficlab/motion/` 底下的套件模組。

## 視覺化測試腳本

`scripts/archive/eval_bbox_heading.py`——拿一段影片跑 YOLO 偵測車輛 bbox，逐框呼叫 `estimate_heading_from_bbox()`，把估出來的朝向箭頭畫在影格上輸出成 PNG，用來肉眼檢查這個方法準不準。純評估用途，pipeline 不會呼叫它，因此歸檔。

```bash
python scripts/archive/eval_bbox_heading.py \
    --video location/test21/footage/test21-3sf.mp4 \
    --config location/test21/G_projection_test21.json \
    --out /tmp/bbox_heading_eval \
    [--frames 20] [--conf 0.35] [--weights models/yolo11s-visdrone-v2-ft.pt]
```

輸出：每個取樣影格一張 PNG，magenta 箭頭 = 信心 ≥0.30，灰色箭頭 = 低信心。

更早的實測結果與待辦（road_heading 消歧義尚未接上）記錄在 `docs/method-survey.md`。
