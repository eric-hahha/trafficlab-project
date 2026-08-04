# H-aware 車輛定位法介紹

**狀態**：已實作（`trafficlab/motion/keypoints_openpifpaf.py`），評估中
**推論腳本**：`scripts/run_keypoints_openpifpaf.py`
**完整技術文件**：`docs/3d-keypoint-template-localization.md`

---

## 這個方法解決什麼問題

現行 pipeline 預設以 YOLO bbox 底部中心投影至衛星座標。這個點對應的是車輛靠近相機的邊緣，不是幾何中心，導致系統性偏移。

---

## 核心想法

把車輛建模為一個 3D 立體 keypoint 樣板，而不是地面上的長方形。

**3D 樣板**（`build_car_template()` in `keypoints_openpifpaf.py`）：24 個 keypoint，每個都有 `(x, y, z)` 座標，其中 `y` 是高度（0 = 地面，正值 = 向上）。輪胎高度為 0，車頂（roof）相關的六個點固定用 1.65m（不隨車輛規格的 `height` 參數變動——一般房車的車頂尖峰位置比車身整體高度規格要高），車燈、保桿、後照鏡等其餘位置用估算高度。

**兩種擬合方法**（`OpenPifPafKeypointsLocalizer.localize()` / `.localize_reprojection()`，見下方各自小節）都遵循同樣的第一步：

```
1. 偵測：OpenPifPaf 對整張影像輸出各台車的 24 個 keypoint 2D 像素座標 (u_i, v_i)

2. 重建 3D 座標：每個 keypoint 以樣板高度 h_i 透過 cctv_to_sat 投影
   (sat_x_i, sat_y_i) = cctv_to_sat(u_i, v_i, h=h_i)
   → 高度越高的點（車頂），投影時視差校正量越大
```

再往下，兩種方法用不同方式把這些投影點組合成車輛中心 + 朝向：`localize()` 是對所有點一次做固定尺度 Procrustes（SVD）；`localize_reprojection()` 是不做整體最小平方擬合，改用配對關鍵點的幾何關係直接求交點。

---

## 方法一：Procrustes 擬合（`localize()`）

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
heading = degrees(atan2(-R[1,1], -R[0,1])) % 360
```

`s = px_per_m` 為已知常數（從 homography 取得），所以這是**固定尺度**的擬合，把所有偵測到的 keypoint 一次丟進同一個最小平方問題，只解旋轉 R 和平移 T（3 DoF：cx, cy, θ）。至少需要 2 個信心值 ≥ `kp_conf` 的 keypoint，否則回傳 `failed_insufficient_kp`。除此之外沒有其他失敗/模糊狀態——`status` 只會是 `ok` 或 `failed_insufficient_kp`。

heading 公式用 `atan2(dy, dx)`（不取負號），跟 `trafficlab/motion/kinematics.py`（正式 pipeline 的 heading 來源）與 `sat_renderer.py` 畫箭頭的慣例一致；`wheel_localization.py` 文件裡記載的 `atan2(−dy, dx)` 是另一套慣例，跟這裡不同，兩者不能直接混用比較。

---

## 方法二：幾何中線交點法（`localize_reprojection()`）

不對所有點做整體最小平方擬合，而是直接利用關鍵點兩兩之間已知的幾何關係（模板上哪些點左右對稱、哪些點是同側前後輪胎）求出兩條通過車輛中心的直線，交點即為車輛位置。依照偵測到哪些關鍵點組合，分三種情況：

**情況一：同時偵測到左右對稱點 + 同側前後輪胎點**（`_LR_PAIRS` 任一組 + `_FR_WHEEL_PAIRS` 任一組）
兩組各自的中垂線都通過車輛中心（一條沿車軸方向、一條垂直車軸方向），取交點定位置。朝向由「左右對稱點中垂線」和「前後輪胎連線」兩個獨立方向做圓周平均得到。

**情況二：只有其中一種配對可見**
單靠這一組配對的兩個點，就能唯一決定一個旋轉（不會有鏡射歧義，因為是已知對應關係的兩點，不是任意對稱點）。用這個旋轉把其他所有可見關鍵點依照各自的模板座標，沿著中垂線方向平移，平移後的位置平均起來就是第二條（垂直方向的）中線，兩線交點即為位置。

朝向：如果配對是左右對稱點，車頭朝哪一端無法只靠這組點自己判斷（模板本身的前後假設可能因為偵測器整組標籤前後顛倒而失準），改用其他所有模板 z 值不為零的可見關鍵點做多數決——檢查每個點實際落在哪一側、是否跟模板記錄的前後符合，多數同意的一側才是車頭方向。如果配對是前後輪胎，方向本來就有明確語意（前輪、後輪是不同的關鍵點），不需要這個多數決。

**情況三：兩種配對都沒有偵測到** → 幾何約束不足，回傳 `failed_insufficient_kp`。

`OpenPifPafKeypointsResult.method` 欄位記錄實際用了哪一種情況（1/2/3）。目前尚未決定的地方：同一幀內如果偵測到多組左右對稱點或多組輪胎配對，要以哪一組為準還沒有規則，程式碼目前固定取 `_LR_PAIRS`/`_FR_WHEEL_PAIRS` 清單裡第一個符合的；`confidence` 也只是依情況給的固定值（0.6 / 0.4），還沒有跟擬合品質掛鉤。

---

## 輸出欄位

| 欄位 | 說明 |
|------|------|
| `sat_coords` | `(x, y)` 車輛中心衛星像素座標；失敗時為 None |
| `heading` | 朝向角（0=East, 90=North）；失敗時為 None |
| `confidence` | 0–1 |
| `n_keypoints` | 參與擬合的 keypoint 數量 |
| `status` | `ok` / `failed_insufficient_kp` |
| `p_sat` | 各 keypoint 的衛星座標 `{kp_idx: (sat_x, sat_y)}`（可視化用）|
| `method` | 只有 `localize_reprojection()` 會填：1 / 2 / 3；`localize()` 恆為 None |

---

## 3D 樣板設計

樣板由 `build_car_template(dims)` 從車輛規格尺寸建構，24 個 keypoint 對應 Apollo-24 索引：

| 類別 | keypoint | 高度 | 精度 |
|------|---------|----------|------|
| 輪胎（7, 8, 18, 19） | h = 0 m | 地面，精確 | ✅ |
| 車頂（0, 1, 6, 10, 11, 16） | h = 1.65 m（固定值） | 估算，不隨車輛規格變動 | ⚠️ |
| 車燈（2, 3, 12, 13） | h ≈ 0.65 m | 估算 | ⚠️ |
| 保險桿（4, 5, 14, 15） | h ≈ 0.20 m | 估算 | ⚠️ |
| 後視鏡（22, 23） | h ≈ 1.05 m | 估算 | ⚠️ |
| 後車身角/車牌（9, 17, 20, 21） | h ≈ 0.50 m | 估算 | ⚠️ |

車輛的長寬高/輪距/軸距尺寸來源，依優先順序：

| 優先順序 | 來源 | 說明 |
|---------|------|------|
| 1 | `--spec-csv engines.csv` | [ilyasozkurt/automobile-models-and-specs](https://github.com/ilyasozkurt/automobile-models-and-specs) 規格庫，過濾房車合理範圍後取中位數 |
| 2 | `prior_dimensions.json` | 腳本在 `--g-proj` 同目錄自動搜尋 |
| 3 | `_FALLBACK_DIMS`（內建）| L=3.8m W=1.8m H=1.55m TW=1.53m WB=2.55m |

---

## Track ID 橋接（Method B）

OpenPifPaf 對整張影像偵測，不經過 YOLO tracker，輸出的 `tracked_id` 原本全為 `null`。

`--method geometric` 用 bbox IoU 橋接：
1. YOLO 框來源二選一：即時模型（`--yolo`）或預先算好的 replay JSON（`--yolo-boxes-json`）
2. 從每個 PifPaf 偵測的 confident keypoint 計算 tight bounding box
3. 與 YOLO bbox 做 IoU 配對（預設閾值 0.3，`--iou-threshold`），指派 YOLO track ID；沒配對到的偵測給一個當幀內的合成 id（非跨幀追蹤，只為了畫圖時顏色能區分）
4. 只有一個信心關鍵點、bbox IoU 天生算不出來的偵測碎片，如果那個點剛好落在唯一一個 YOLO 框內，會被吸收合併進同一幀已配對到那個 track 的偵測，補上更多關鍵點

`--method crop`（crop-and-redetect）是另一條互斥路徑：裁切 Pass-1 bbox 加 50% padding 後重新偵測，不做 YOLO IoU 配對。

`kp_bbox_xyxy()` / `match_by_bbox_iou()`（`keypoints_openpifpaf.py` 模組層級的公開版本）跟這裡實際用的邏輯是重複的兩份程式碼——`run_keypoints_openpifpaf.py` 用的是自己內部 `_` 開頭的版本，公開版本目前沒有呼叫者。

---

## 如何執行

### 基本執行（不帶 YOLO，tracked_id 全為 null）

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_keypoints_openpifpaf.py \
  --video location/test21/footage/test21-4.mp4 \
  --g-proj location/test21/G_projection_test21.json \
  --method geometric --yolo ""
```

### 帶 YOLO track ID 橋接 + 幾何中線交點法

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_keypoints_openpifpaf.py \
  --video location/test21/footage/test21-4.mp4 \
  --g-proj location/test21/G_projection_test21.json \
  --method geometric \
  --yolo models/best.pt \
  --localizer reprojection
```

### 使用預先算好的 YOLO 框 json + 車輛規格庫樣板

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_keypoints_openpifpaf.py \
  --video location/test21/footage/test21-4.mp4 \
  --g-proj location/test21/G_projection_test21.json \
  --method geometric \
  --yolo-boxes-json path/to/replay.json.gz \
  --spec-csv path/to/engines.csv
```

**輸出**：`output/haware/<location_code>/<video_stem>.json.gz`（標準 replay JSON 格式，可直接載入 GUI）

---


## PifPaf Instance 分裂問題

### 問題描述

PifPaf 可能把同一台車偵測為兩個（或更多）獨立 instance，各自得到不同的 `id`。成因是 CIF/CAF 機制用 keypoint 鏈接的方式組裝 instance：當車頭與車尾在影像上距離遠、中間 CAF 連接場弱時，鏈接在中途斷掉，形成兩個 fragment。

### 對定位品質的影響

兩種擬合方法都是逐個 instance 獨立執行，每個 instance 只能用自己那份 keypoints。若同一台車分裂成兩個 fragment：

- **位置精度下降 / 方向可能算不出來**：fragment 往往只覆蓋車輛縱向的一端（全車頭或全車尾），關鍵點數量少、分佈集中，兩種方法都可能因為找不到足夠的幾何約束而失敗或精度變差
- **鬼車**：同一實體車在衛星圖上出現兩個座標點

### 與 ID 匹配的關係

分裂和 ID 匹配是不同層的問題，但相互影響：

| | 問題層次 | 發生時機 |
|---|---|---|
| **Instance 分裂** | 單幀內偵測品質 | PifPaf 推論後 |
| **ID 匹配（Method B）** | 跨幀追蹤 | YOLO bbox IoU 配對時 |

Method B 對分裂有部分過濾效果：一台車的兩個 fragment 通常只有一個與 YOLO bbox IoU 夠高，另一個得 `tracked_id=None`（或落到單點吸收合併邏輯裡）。但 ID 匹配**不能修復** fragment 內部 keypoints 不足造成的位置/方向錯誤，修復必須在 ID 匹配之前。

### 為什麼調 threshold 無效

- **`--seed-threshold` 調高**：減少種子數量，但若車頭和車尾各自有高信心 keypoint，兩個種子照樣存在，分裂不消失
- **`--pifpaf-threshold`（instance-threshold）調高**：fragment 分數通常比完整 instance 低，調高可能過濾掉其中一個，但這是副作用，不是針對性修復——遠距或部分遮擋的完整車也會一起被過濾，代價高而效益不穩定

Threshold 調整是設計來控制偵測靈敏度的，不是用來消除分裂的。

### 正確的修復方向

分裂是 post-decoding 問題，應在 PifPaf 輸出後處理：

**Keypoint NMS（推薦）**：計算每對 predictions 的 keypoint bbox IoU，若重疊超過閾值，合併為一個 instance（保留信心較高的 keypoints）。可在 `run_keypoints_openpifpaf.py` 的 frame loop 內、送入 `localizer.localize()`/`localize_reprojection()` 之前加一層合併。

---

## 已知限制與下一步

**限制（無法靠換定位方法解決）**：
- PifPaf 在俯角 footage 的偵測率天花板受 domain gap 限制（ApolloCar3D 前視訓練）
- 需要 fine-tune PifPaf 或改用路側訓練的 keypoint 模型（見 `docs/pose-model-alternatives.md`）

**幾何中線交點法（`localize_reprojection`）尚待決定的地方**：
- 同一幀多組左右對稱點/多組輪胎配對時如何選擇，目前固定取清單第一組
- `confidence` 只是依情況給的固定值，還沒有跟實際擬合品質掛鉤
- 沒有偵測器標籤整體錯誤（例如前後端點名稱系統性顛倒）以外的模糊/錯誤偵測機制

**下一步**：
- [ ] 整合進主 pipeline（新增 `localization_method` 選項）
- [ ] PifPaf fine-tune 改善 domain gap（與定位演算法正交，不改動 h-aware 流程）
- [ ] Keypoint NMS：合併同幀內重疊的 PifPaf fragment，解決 instance 分裂問題
