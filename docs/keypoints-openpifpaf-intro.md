# H-aware 車輛定位法介紹

**狀態**：已實作（`trafficlab/motion/haware_localization.py`），評估中  
**評估腳本**：`scripts/eval_haware_replay.py`  
**完整技術文件**：`docs/3d-keypoint-template-localization.md`

---

## 這個方法解決什麼問題

現行 pipeline 預設以 YOLO bbox 底部中心投影至衛星座標。這個點對應的是車輛靠近相機的邊緣，不是幾何中心，導致系統性偏移。

第一代改良（輪胎定位法 `WheelLocalizer`）改用 OpenPifPaf 偵測輪胎 keypoint，在地面平面投影後取幾何中心，精度有所提升，但覆蓋率很低（成功率約 4.7%），因為：
1. 只認輪胎：俯角場景輪胎被遮蔽時直接失敗，即使車頂清楚可見也用不上
2. 逐 YOLO 框 crop 再偵測，框歪或框漏一關就整台車沒有 keypoint
3. 需要對上 YOLO track ID 才算數，對不上就丟棄

H-aware 同時解決這兩類問題：用全部 24 個 keypoint（精度），並對整張影像一次偵測不 crop 不綁 ID（覆蓋率）。

---

## 核心想法

把車輛建模為一個 3D 立體 keypoint 樣板，而不是地面上的長方形。

**3D 樣板**（`build_car_template()` in `haware_localization.py`）：24 個 keypoint，每個都有 `(x, y, z)` 座標，其中 `y` 是高度（0 = 地面，正值 = 向上）。輪胎高度為 0、車頂高度約 1.45m、車燈約 0.65m。

**定位流程**（4 步驟）：

```
1. 偵測：OpenPifPaf 對整張影像輸出各台車的 24 個 keypoint 2D 像素座標 (u_i, v_i)

2. 重建 3D 座標：每個 keypoint 以樣板高度 h_i 透過 cctv_to_sat 投影
   (sat_x_i, sat_y_i) = cctv_to_sat(u_i, v_i, h=h_i)
   → 高度越高的點（車頂），投影時視差校正量越大

3. 樣板擬合：把所有投影點與縮放後的 3D 樣板 (x,z) 平面做固定尺度 Procrustes 擬合
   T + R(θ) · template_i ≈ P_i   for all detected i
   一次 SVD 同時解車輛中心 T 和朝向 θ

4. 輸出：T 的 (x,y) 分量即為車輛幾何中心的衛星座標
```

**為什麼不用 PnP**：PnP 需要相機內參 K 矩陣。本專案的 K 是預設估算值（`fx = image_width = 1280`），未經實際標定，誤差可達 30–100%。此外高俯角（60–80°）下透視縮放效果微弱，PnP 深度估算不可靠。H-aware 用高度先驗取代透視深度，完全不依賴 K。

---

## 實作細節

### Procrustes 擬合（`HawareLocalizer.localize()`）

```python
s  = g_engine.px_per_m
Q  = template[idx][:, [0, 2]] * s   # 樣板 (x,z) → sat 像素
P  = np.stack(P_i_list)             # cctv_to_sat(u_i, v_i, h_i) 落點
qb, pb = Q.mean(0), P.mean(0)
Hc = (Q - qb).T @ (P - pb)
U, _, Vt = np.linalg.svd(Hc)
det_sign = np.sign(np.linalg.det(Vt.T @ U.T))
R = Vt.T @ np.diag([1.0, det_sign]) @ U.T   # 防鏡射
T_sat = pb - R @ qb                           # 車輛中心
heading = degrees(atan2(R[1,1], -R[0,1])) % 360
```

`s = px_per_m` 為已知常數（從 homography 取得），所以這是**固定尺度**的擬合，只解旋轉 R 和平移 T（3 DoF：cx, cy, θ）。

### 輸出欄位

| 欄位 | 說明 |
|------|------|
| `sat_coords` | `(x, y)` 車輛中心衛星像素座標；失敗時為 None |
| `heading` | 朝向角（0=East, 90=North）；朝向不明時為 None |
| `confidence` | 0–1，與偵測到的 keypoint 數量和擬合殘差有關 |
| `n_keypoints` | 參與擬合的 keypoint 數量 |
| `status` | `ok` / `ambiguous_heading` / `failed_insufficient_kp` |
| `p_sat` | 各 keypoint 的衛星座標 `{kp_idx: (sat_x, sat_y)}`（可視化用）|

### 失敗與模糊

- **`failed_insufficient_kp`**：信心值 ≥ 0.2 的 keypoint 少於 2 個 → `sat_coords=None`
- **`ambiguous_heading`**：所有偵測到的 kp 都在車輛同一縱向端（全在車頭或全在車尾），無法判斷朝向 → `heading=None`，但 `sat_coords` 仍有效

---

## 3D 樣板設計

樣板由 `build_car_template(dims)` 從車輛規格尺寸建構，24 個 keypoint 對應 Apollo-24 索引：

| 類別 | keypoint | 高度來源 | 精度 |
|------|---------|----------|------|
| 輪胎（7, 8, 18, 19） | h = 0 m | 地面，精確 | ✅ |
| 車頂（0, 1, 6, 10, 11, 16） | h ≈ 1.45 m | 車高實測值 | ✅ |
| 車燈（2, 3, 12, 13） | h ≈ 0.65 m | 估算 | ⚠️ |
| 保險桿（4, 5, 14, 15） | h ≈ 0.20 m | 估算 | ⚠️ |
| 後視鏡（22, 23） | h ≈ 1.05 m | 估算 | ⚠️ |

**現行實際使用的尺寸**（`eval_haware_replay.py` 預設，未傳 `--spec-csv`）：

| 優先順序 | 來源 | 說明 |
|---------|------|------|
| 1 | `--spec-csv engines.csv` | 規格庫實測值（18,715 台 sedan 中位數：L=4.509m W=1.781m H=1.448m TW=1.522m WB=2.670m）|
| 2 | `prior_dimensions.json` | 湊整估算值（腳本在 g-proj 同目錄自動搜尋）|
| 3 | `_FALLBACK_DIMS`（內建）| L=3.8m W=1.8m H=1.55m TW=1.53m WB=2.55m |

> 目前的 eval 結果皆使用 **prior_dimensions.json 或 `_FALLBACK_DIMS`**，並非規格庫實測值。要啟用實測尺寸，需下載 `engines.csv` 並加 `--spec-csv` 參數（見下方執行指令）。

設計上規格庫（[ilyasozkurt/automobile-models-and-specs](https://github.com/ilyasozkurt/automobile-models-and-specs) `engines.csv`）是首選，但目前尚未下載到本地。

---

## Track ID 橋接（Method B）

OpenPifPaf 對整張影像偵測，不經過 YOLO tracker，輸出的 `tracked_id` 原本全為 `null`。

Method B 用 bbox IoU 橋接：
1. 每幀同時跑 YOLO tracker（BoT-SORT / ByteTrack）
2. 從每個 PifPaf 偵測的 confident keypoint 計算 tight bounding box
3. 與 YOLO bbox 做 IoU 配對（閾值 0.3），指派 YOLO track ID

**實作**：`kp_bbox_xyxy()` + `match_by_bbox_iou()` in `haware_localization.py`  
**詳細說明**：`docs/haware-id-matching.md`

---

## 如何執行

### 基本執行（不帶 YOLO，tracked_id 全為 null）

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/eval_haware_replay.py \
  --video location/test21/footage/test21-4.mp4 \
  --g-proj location/test21/G_projection_test21.json
```

### 帶 YOLO track ID 橋接（推薦）

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/eval_haware_replay.py \
  --video location/test21/footage/test21-4.mp4 \
  --g-proj location/test21/G_projection_test21.json \
  --yolo models/yolo11s-visdrone-v2-ft.pt \
  --yolo-classes 2
```

### 使用車輛規格庫樣板

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/eval_haware_replay.py \
  --video location/test21/footage/test21-4.mp4 \
  --g-proj location/test21/G_projection_test21.json \
  --spec-csv path/to/engines.csv
```

**輸出**：`output/haware/<location_code>/<video_stem>.json.gz`（標準 replay JSON 格式，可直接載入 GUI）

---

## 與輪胎法的比較

| | 輪胎法（WheelLocalizer）| H-aware（HawareLocalizer）|
|---|---|---|
| 偵測方式 | YOLO crop → OpenPifPaf | 整張影像 → OpenPifPaf |
| 使用 keypoint | 4 個輪胎（h=0）| 任意 2 個以上（24 kp 任選）|
| 失敗門檻 | 輪胎被遮即失敗 | 2 個任意 kp 即可 |
| 需要 YOLO bbox | 是（crop 用）| 否（定位不需要；track ID 橋接需要）|
| 輸出 heading | 有（幾何推算）| 有（SVD 解 R，若不模糊）|
| 現行成功率（test21）| 4.7% | 測試中（估計更高）|

H-aware 是輪胎法的超集：輪胎法可視為只用 4 個 h=0 keypoint 的簡化特例。

---

## PifPaf Instance 分裂問題

### 問題描述

PifPaf 可能把同一台車偵測為兩個（或更多）獨立 instance，各自得到不同的 `id`。成因是 CIF/CAF 機制用 keypoint 鏈接的方式組裝 instance：當車頭與車尾在影像上距離遠、中間 CAF 連接場弱時，鏈接在中途斷掉，形成兩個 fragment。

### 對定位品質的影響

`HawareLocalizer.localize()` 逐個 instance 獨立執行，每個 instance 只能用自己那份 keypoints 做 Procrustes 擬合。若同一台車分裂成兩個 fragment：

- **方向估計失效**：fragment 往往只覆蓋車輛縱向的一端（全車頭或全車尾），`z_vals` 全偏向同側 → `symmetric=True` → `heading=None`（`ambiguous_heading`）
- **位置精度下降**：keypoints 少、分佈集中，SVD 欠約束，`sat_coords` 偏移
- **鬼車**：同一實體車在衛星圖上出現兩個座標點

### 與 ID 匹配的關係

分裂和 ID 匹配是不同層的問題，但相互影響：

| | 問題層次 | 發生時機 |
|---|---|---|
| **Instance 分裂** | 單幀內偵測品質 | PifPaf 推論後 |
| **ID 匹配（Method B）** | 跨幀追蹤 | YOLO bbox IoU 配對時 |

Method B 對分裂有部分過濾效果：一台車的兩個 fragment 通常只有一個與 YOLO bbox IoU 夠高，另一個得 `tracked_id=None`。但 ID 匹配**不能修復** fragment 內部 keypoints 不足造成的位置/方向錯誤，修復必須在 ID 匹配之前。

### 為什麼調 threshold 無效

- **`--seed-threshold` 調高**：減少種子數量，但若車頭和車尾各自有高信心 keypoint，兩個種子照樣存在，分裂不消失
- **`--pifpaf-threshold`（instance-threshold）調高**：fragment 分數通常比完整 instance 低，調高可能過濾掉其中一個，但這是副作用，不是針對性修復——遠距或部分遮擋的完整車也會一起被過濾，代價高而效益不穩定

Threshold 調整是設計來控制偵測靈敏度的，不是用來消除分裂的。

### 正確的修復方向

分裂是 post-decoding 問題，應在 PifPaf 輸出後處理：

**Keypoint NMS（推薦）**：計算每對 predictions 的 keypoint bbox IoU，若重疊超過閾值，合併為一個 instance（保留信心較高的 keypoints）。可在 `eval_haware_replay.py` 的 frame loop 內、送入 `localizer.localize()` 之前加一層合併。

---

## 已知限制與下一步

**限制（無法靠換定位方法解決）**：
- PifPaf 在俯角 footage 的偵測率天花板受 domain gap 限制（ApolloCar3D 前視訓練）
- 需要 fine-tune PifPaf 或改用路側訓練的 keypoint 模型（見 `docs/method-survey.md`）

**下一步**：
- [ ] 與輪胎法在同一段影片做定量比較（定位成功率 + 中心點位置誤差）
- [ ] 整合進主 pipeline（新增 `localization_method: haware` 選項）
- [ ] PifPaf fine-tune 改善 domain gap（與定位演算法正交，不改動 h-aware 流程）
- [ ] Keypoint NMS：合併同幀內重疊的 PifPaf fragment，解決 instance 分裂問題
