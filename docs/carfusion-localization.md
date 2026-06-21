# CarFusion 車輛定位：完整程式邏輯

## 概覽

CarFusion 定位流程分三層：

```
影片幀
  ↓ YOLOv8-Pose (CarFusion 權重)
車輛 bbox + 14 個關鍵點 (x_img, y_img, conf)
  ↓ G-projection：每個關鍵點用各自的高度先驗投影到衛星座標
14 個衛星座標點 (sat_x, sat_y)
  ↓ Procrustes SVD：把觀測點與 3D 車輛模板配準
車輛衛星中心 + 朝向角
```

---

## 第一層：YOLO 偵測

### 使用的模型

```
Habib0905/Vehicle-Pose-Estimation
權重：last.pt（YOLOv8-Pose）
```

YOLOv8-Pose 是 one-stage 模型，一次推論同時輸出：
- **bbox**：(x1, y1, x2, y2) + confidence
- **keypoints**：每個 bbox 對應 14 個關鍵點，每點三個值 `[x_img, y_img, conf]`

### 呼叫方式

```python
from ultralytics import YOLO

model = YOLO('last.pt')
results = model(frame, conf=0.25, verbose=False)
result  = results[0]

# 取第 i 輛車的關鍵點
kp_xy   = result.keypoints.xy[i].cpu().numpy()    # shape (14, 2)
kp_conf = result.keypoints.conf[i].cpu().numpy()  # shape (14,)
kp_14   = np.column_stack([kp_xy, kp_conf])       # shape (14, 3)
```

### 14 個關鍵點定義

| index | 名稱 | 語意 |
|-------|------|------|
| 0 | wheel_fl | 前左輪 |
| 1 | wheel_fr | 前右輪 |
| 2 | wheel_rl | 後左輪 |
| 3 | wheel_rr | 後右輪 |
| 4 | light_fl | 前左燈 |
| 5 | light_fr | 前右燈 |
| 6 | light_rl | 後左燈 |
| 7 | light_rr | 後右燈 |
| 8 | roof_fl | 車頂前左角 |
| 9 | roof_fr | 車頂前右角 |
| 10 | roof_rl | 車頂後左角 |
| 11 | roof_rr | 車頂後右角 |
| 12 | exhaust | 排氣管 |
| 13 | center | 車身中心 |

---

## 第二層：關鍵點投影到衛星座標

### 核心函式

```python
sat_xy = g_engine.cctv_to_sat(x_img, y_img, h=h_k)
```

`h` 是該關鍵點在現實中距離地面的高度（公尺）。高度不同，視差補正量就不同，投影結果也不同。

### 各關鍵點的高度先驗

```python
_BASE_KP_HEIGHTS = [
    0.00, 0.00, 0.00, 0.00,   # wheel_fl/fr/rl/rr：輪子接地，h=0
    0.65, 0.65, 0.65, 0.65,   # light_fl/fr/rl/rr：頭燈/尾燈高度
    H,    H,    H,    H,       # roof_fl/fr/rl/rr：車頂，h=車身高度
    0.20,                      # exhaust：排氣管，接近地面
    0.70,                      # center：車身中線高度
]
```

`H` 來自車輛尺寸先驗（預設從 `prior_dimensions.json` 的 `measurements_visdrone.car.height` 讀取，約 1.55m）。

### 篩選條件

每個關鍵點要同時滿足以下條件才投影：
1. `conf >= kp_conf`（預設 0.2）
2. 座標不是 `(0, 0)`（YOLO 用 (0,0) 表示不可見點）

```python
for i in range(14):
    x, y, c = kp_14[i]
    if c < kp_conf or (x == 0.0 and y == 0.0):
        continue
    kp_sat[i] = g_engine.cctv_to_sat(x, y, h=kp_heights[i])
```

---

## 第三層：Procrustes SVD 配準

### 車輛 3D 模板

在車身座標系定義 14 個點的位置（單位：公尺）：

```
座標軸：
  x：側向，正方向 = 車輛左側
  y：高度，0 = 地面
  z：縱向，負方向 = 前方，正方向 = 後方
```

```python
t[0]  = [ htw,  0.0,  -hwb]   # wheel_fl：前左輪（x=+軌距/2, z=-軸距/2）
t[1]  = [-htw,  0.0,  -hwb]   # wheel_fr
t[2]  = [ htw,  0.0,  +hwb]   # wheel_rl
t[3]  = [-htw,  0.0,  +hwb]   # wheel_rr

t[4]  = [ hw*0.85, 0.65, -hl]  # light_fl
t[5]  = [-hw*0.85, 0.65, -hl]  # light_fr
t[6]  = [ hw*0.85, 0.65, +hl]  # light_rl
t[7]  = [-hw*0.85, 0.65, +hl]  # light_rr

t[8]  = [ hw*0.70, H, -hl*0.50]  # roof_fl
t[9]  = [-hw*0.70, H, -hl*0.50]  # roof_fr
t[10] = [ hw*0.70, H, +hl*0.40]  # roof_rl
t[11] = [-hw*0.70, H, +hl*0.40]  # roof_rr

t[12] = [-hw*0.15, 0.20, +hl]    # exhaust
t[13] = [0.0,      0.70,  0.0]   # center
```

尺寸來源（預設值）：
- `L=3.8m`、`W=1.8m`、`H=1.55m`
- `TW=1.53m`（軌距）、`WB=2.55m`（軸距）

### SVD 配準演算法

目標：找到一個旋轉矩陣 R 和平移向量 T，讓模板點（`Q`）盡量對齊衛星觀測點（`P`）。

**步驟：**

```python
idx = list(kp_sat.keys())            # 有效關鍵點的 index 清單

# Q：從模板取 (x, z) 兩欄，乘以衛星像素/公尺比例
Q = template[idx][:, [0, 2]] * px_per_m   # shape (n, 2)

# P：實際衛星座標
P = np.array([kp_sat[i] for i in idx])    # shape (n, 2)

# 去中心化
qb, pb = Q.mean(0), P.mean(0)

# 計算交叉共變異數矩陣
Hc = (Q - qb).T @ (P - pb)   # shape (2, 2)

# SVD 分解
U, _, Vt = np.linalg.svd(Hc)

# 旋轉矩陣（防止反射解）
det_sign = np.sign(np.linalg.det(Vt.T @ U.T))
R = Vt.T @ np.diag([1.0, det_sign]) @ U.T   # shape (2, 2)

# 車輛中心在衛星座標系的位置
T_sat = pb - R @ qb
```

`det_sign` 的用途：SVD 有時會找到包含鏡像的解，`diag([1, det_sign])` 強制只取純旋轉。

### 朝向角演算

車輛前進方向在模板座標系中為 `-z`，即向量 `(0, -1)`。旋轉後：

```
forward_sat = R @ [0, -1]^T = [-R[0,1], -R[1,1]]
```

朝向角定義（與 haware_localization.py 一致）：

```python
heading = math.degrees(math.atan2(R[1, 1], -R[0, 1])) % 360.0
```

- `0°` = 東（衛星圖右方）
- `90°` = 北（衛星圖上方）

### 朝向不明（ambiguous）判斷

如果所有有效關鍵點都在車輛的同一個縱向半邊（全在前半或全在後半），則無法判斷前後：

```python
z_vals = template[idx, 2]
if np.all(z_vals >= 0) or np.all(z_vals <= 0):
    status  = 'ambiguous_heading'
    heading = None
```

### 信心值計算

```python
P_pred = (Q - qb) @ R.T + pb          # 模板點旋轉後的預測衛星位置
rms    = sqrt(mean(||P - P_pred||^2)) # 觀測點與預測點的 RMS 誤差（像素）

confidence = min(1.0, n/8.0) * max(0.0, 1.0 - rms / (5.0 * px_per_m))
```

- `n/8.0`：用越多關鍵點，信心越高（飽和點為 8 個）
- `rms` 越小，信心越高；RMS 超過 5 公尺時信心歸零

---

## 輸出欄位

```python
@dataclass
class CarFusionResult:
    sat_coords:  Optional[tuple]   # (sat_x, sat_y) 衛星像素座標；失敗時為 None
    heading:     Optional[float]   # 0–360 度；朝向不明時為 None
    confidence:  float             # 0–1
    n_keypoints: int               # 實際參與配準的關鍵點數
    status:      str               # 'ok' | 'ambiguous_heading' | 'failed_insufficient_kp'
    kp_sat:      dict              # {kp_idx: (sat_x, sat_y)}，所有投影成功的關鍵點
```

---

## 評估腳本使用方式

### 視覺化評估（CCTV + 衛星並排）

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && \
python scripts/eval_carfusion_sat.py \
    --video   location/test21/footage/test21-4.mp4 \
    --weights /private/tmp/carfusion_weights/weights/last.pt \
    --g-proj  location/test21/G_projection_test21.json \
    --sat     location/test21/sat_test21.png \
    --out     /private/tmp/carfusion_sat/ \
    --start-frame 120
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--video` | test21-4.mp4 | 輸入影片 |
| `--weights` | carfusion last.pt | YOLOv8-Pose 權重 |
| `--g-proj` | G_projection_test21.json | 投影參數 |
| `--sat` | sat_test21.png | 衛星底圖 |
| `--out` | /private/tmp/carfusion_sat/ | 輸出目錄 |
| `--start-frame` | 0 | 開始處理的幀號 |
| `--frames` | -1 (全部) | 最多處理幾幀 |
| `--conf` | 0.25 | YOLO 偵測閾值 |
| `--kp-conf` | 0.2 | 關鍵點信心閾值 |

### 相關程式碼位置

| 檔案 | 用途 |
|------|------|
| `trafficlab/motion/carfusion_localization.py` | `CarFusionLocalizer` 類別、`build_carfusion_template()` |
| `scripts/eval_carfusion_sat.py` | 評估腳本，並排輸出 |
| `scripts/eval_carfusion.py` | 純關鍵點視覺化（不計算衛星座標） |
| `trafficlab/motion/haware_localization.py` | 相同演算法的 Apollo-24 版本（參考） |

---

## 與 haware 的比較

`haware_localization.py` 和 `carfusion_localization.py` 使用完全相同的核心演算法（Procrustes SVD），差異在偵測前段和關鍵點定義。

### 架構差異

| 面向 | haware | CarFusion |
|------|--------|-----------|
| **偵測模型** | OpenPifPaf（`shufflenetv2k16-apollo-24`） | YOLOv8-Pose（CarFusion `last.pt`） |
| **推論方式** | 對整張幀做 pose estimation，bbox 由 PifPaf 自行估計 | 一次推論同時輸出 bbox + keypoints（one-stage） |
| **關鍵點數** | 24（Apollo-24 規格） | 14（CarFusion 規格） |
| **訓練資料域** | ApolloCar3D（地面前視街道攝影） | CarFusion Pittsburgh 十字路口 CCTV + Bangladesh 資料 |
| **適合場景** | 前視、近距、清晰車身 | 斜角 CCTV、俯角、遠距 |

### 關鍵點定義差異

Apollo-24 的 24 個點涵蓋更多車身細節（保險桿底部、後車牌、後視鏡等）。CarFusion 的 14 個點語意更簡潔，集中在幾何上最穩定的位置：

| 語意類型 | Apollo-24（24kp） | CarFusion（14kp） |
|----------|-------------------|-------------------|
| 車輪 | 4 個（索引 7,8,18,19） | 4 個（索引 0-3） |
| 車燈 / 前後燈 | 4 個 | 4 個（索引 4-7） |
| 車頂 | 2 個（前左右） | 4 個（前後左右，索引 8-11） |
| 保險桿底部 | 4 個 | — |
| 車頂側面 / 後車角 | 4 個 | — |
| 後車牌 | 2 個 | — |
| 後視鏡邊緣 | 2 個 | — |
| 排氣管 / 中心 | — | 2 個（索引 12-13） |

CarFusion 多了 **4 個車頂角點**（roof_fl/fr/rl/rr），覆蓋前後兩端，對朝向判斷有利。

### 高度先驗差異

兩者的高度先驗策略相同（每個關鍵點各自帶 `h` 投影），但因關鍵點語意不同，具體數值也不同：

| 關鍵點類型 | haware (Apollo-24) | CarFusion |
|-----------|-------------------|-----------|
| 車輪 | 0.0 m | 0.0 m |
| 頭燈 / 尾燈 | 0.65 m | 0.65 m |
| 車頂 | dims.height (~1.55 m) | dims.height |
| 保險桿底部 | 0.20 m | — |
| 後車角 | 0.50 m | — |
| 後視鏡邊緣 | 1.05 m | — |
| 排氣管 | — | 0.20 m |
| 車身中心 | — | 0.70 m |

### 核心演算法：完全相同

SVD 配準、朝向公式、ambiguous 判斷、信心值計算，兩個 localizer 的程式碼邏輯一模一樣。CarFusion localizer 直接繼承 `haware_localization.py` 匯出的 `_FALLBACK_DIMS`、`_TRACK_RATIO`、`_WHEELBASE_RATIO`。

```python
# carfusion_localization.py 繼承自 haware_localization.py
from trafficlab.motion.haware_localization import _FALLBACK_DIMS, _TRACK_RATIO, _WHEELBASE_RATIO
```


---

## 測試結果（test21-4.mp4，frame 120–175）

| 指標 | 數值 |
|------|------|
| 處理幀數 | 56 |
| 有偵測的幀 | 41（73.2%）|
| 偵測車輛數 | 80 |
| 定位成功（ok）| 80（100%）|
| 朝向不明 | 0 |
| 定位失敗 | 0 |
