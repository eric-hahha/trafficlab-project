# Two-Pass PifPaf 實作紀錄

## 背景問題

PifPaf Apollo-24 訓練資料以側面視角為主，用在俯瞰 CCTV 時有明顯 domain gap。
test21-4（184 幀）的實際偵測率約 4.7%，許多車輛只被偵測到 1–5 幀，即使車輛清晰可見。

初步嘗試降低 `--pifpaf-threshold`（0.2 → 0.01）和 `--seed-threshold`（0.2 → 0.05）：
- detections 從 505 增加到 4606
- 但 failed（無足夠 keypoint）同步大增，最終有效 sat_coords 僅從 1137 → 1241
- 結論：閾值調整對 domain gap 幾乎無效，因為 PifPaf 根本看不出車輛骨架

## Two-Pass 的動機

原本 pipeline 的做法：YOLO 先定位車輛 bbox → 裁切 → PifPaf 在裁切圖上偵測 keypoints。
裁切後車輛佔畫面比例增大，PifPaf 能偵測到更多 keypoints。

h-aware replay 腳本沒有 YOLO，但 PifPaf Pass-1 本身已輸出 `ann.bbox()`（instance 的外接矩形）。
因此可以：
1. Pass-1：全幀跑 PifPaf，拿到所有 instance 及其 `ann.bbox()`
2. Pass-2：對每個 instance bbox 加 padding 裁切，對裁切圖重跑 PifPaf
3. 選 Pass-2 中 confident keypoints 最多的 annotation，座標還原回全幀後送入定位器

## 實作

檔案：`scripts/eval_haware_replay.py`

### 新增 CLI 旗標

```python
parser.add_argument('--two-pass', action='store_true',
    help='Crop each Pass-1 bbox with padding and re-run PifPaf')
parser.add_argument('--crop-padding', type=float, default=0.5,
    help='Fractional padding around bbox for two-pass crop (default 0.5)')
```

### 幀迴圈中的 two-pass 邏輯（`scripts/eval_haware_replay.py:285-304`）

```python
kp_24 = ann.data
if args.two_pass:
    bx, by, bw, bh = ann.bbox()
    pad_x, pad_y = bw * args.crop_padding, bh * args.crop_padding
    x0 = max(0, int(bx - pad_x))
    y0 = max(0, int(by - pad_y))
    x1 = min(W, int(bx + bw + pad_x))
    y1 = min(H, int(by + bh + pad_y))
    if x1 > x0 and y1 > y0:
        try:
            crop_preds, _, _ = predictor.pil_image(pil.crop((x0, y0, x1, y1)))
        except Exception:
            crop_preds = []
        if crop_preds:
            best = max(crop_preds,
                       key=lambda a: sum(1 for kp in a.data if kp[2] >= args.kp_conf))
            kp_24 = best.data.copy()
            kp_24[:, 0] += x0
            kp_24[:, 1] += y0

result = localizer.localize(kp_24)
```

**關鍵細節：**
- `ann.bbox()` 是 method，不是 property，必須加 `()`（修過一次 bug）
- 裁切後 Pass-2 的 keypoint 座標是相對裁切圖的，需加回 `(x0, y0)` 才還原到全幀座標
- 若 Pass-2 找不到任何 annotation，fallback 使用 Pass-1 的 `ann.data`
- 選擇標準：confident keypoints 最多的 annotation（`kp[2] >= args.kp_conf`）

### 輸出欄位

`kp_cctv` 欄位儲存最終使用的 keypoints（Pass-2 或 Pass-1 fallback），供 GUI overlay 渲染：

```python
'kp_cctv': kp_24.tolist()
```

## 修復過的 Bug

### 1. `ann.bbox` 是 method 不是 property

```python
# 錯誤（TypeError: cannot unpack non-iterable method object）
bx, by, bw, bh = ann.bbox

# 正確
bx, by, bw, bh = ann.bbox()
```

### 2. Linter 亂加不存在的 import

Linter 自動加了：
```python
from trafficlab.motion.haware_localization import kp_bbox_xyxy, match_by_bbox_iou
```

但這兩個函數不存在於 `haware_localization.py`，導致 ImportError。
解法：移除這兩個 import，在腳本內定義本地 helper：

```python
def _kp_bbox_xyxy(kp_24: np.ndarray, kp_conf: float): ...
def _match_by_bbox_iou(pifpaf_boxes, yolo_boxes, yolo_tids, iou_threshold=0.3): ...
```

### 3. `openpifpaf.decoder.configure()` 的 Namespace 問題

直接傳 kwarg 會 TypeError；直接建 `Namespace(instance_threshold=...)` 會 AttributeError（缺少 decoder 子項目）。
解法：先用 `openpifpaf.decoder.cli()` 建完整預設值再覆蓋：

```python
import argparse as _ap
_dec_p = _ap.ArgumentParser()
openpifpaf.decoder.cli(_dec_p)
_dec_args = _dec_p.parse_args([])
_dec_args.instance_threshold = args.pifpaf_threshold
_dec_args.seed_threshold     = args.seed_threshold
openpifpaf.decoder.configure(_dec_args)
```

## 效能考量

Two-pass 的代價：若一幀有 N 個 instance，實際推論次數為 1（全幀）+ N（各別裁切）。
test21-4 平均每幀約 6 個 instance → 推論量約 7 倍。

裁切圖比全幀小，單次推論較快，但總體仍明顯較慢（估計 2–4× 全幀時間）。
因為是離線處理，吞吐量影響可接受，但尚未在完整影片上實測驗證效果。

## 使用方式

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && \
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/eval_haware_replay.py \
  --video location/test21/footage/test21-4.mp4 \
  --g-proj location/test21/G_projection_test21.json \
  --pifpaf-threshold 0.01 \
  --seed-threshold 0.05 \
  --kp-conf 0.1 \
  --two-pass \
  --crop-padding 0.5
```
