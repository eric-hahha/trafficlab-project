# 錨點修正：依接地形狀決定支撐距離

**狀態：已實作後還原**（實作與還原皆為 2026-08-16）
**目前程式碼裡沒有這項修正** —— 錨點修正對所有類別一律當矩形，本文記錄的橫向過度修正仍然存在。
**未決事項：** 軸距 vs 全長、汽車是否有同樣問題、行人縱向項的處置、碰撞幀間隙不穩定

還原時動到的檔案（改動已全部移除，其餘 session 改動保留）：
`trafficlab/io/prior_dimensions.py`（`GROUND_CONTACT_SHAPE`、`contact_shape()`）／
`trafficlab/projection/g_projection.py`（`footprint_anchor_to_center()` 的 `contact_shape` 參數）／
`trafficlab/inference/pipeline.py`（import 與呼叫引數）

還原後重跑 Hsinchu1 驗證：與改動前的輸出**逐物件逐幀完全一致**（最大座標差 0.000000 m）。

---

## 一句話

錨點修正原本假設所有物件的接地輪廓都是完整矩形，但兩輪車的前後輪都在中心線上、行人雙腳在
身體正下方。改成依類別套不同的接地形狀後，機車不再被橫向推開 W/2、行人不再被無故推移。

---

## 背景：錨點修正在做什麼

`_get_mask_bottom_center()` 取 mask 在影像上的最低點當接地參考點。在掠射視角下（本專案各場地
仰角 5°~23°），那個點落在物件**接地輪廓上離相機最近的邊**，不是幾何中心。直接拿它當 floor box
中心會讓整個框往相機方向偏。

`footprint_anchor_to_center()` 沿視線往內推回一段**支撐距離**來還原中心：

```
h(u) = |u·前| × L/2  +  |u·右| × W/2       u = 由物件指向相機的單位向量
        ─────────      ─────────
         縱向項          橫向項
```

這是矩形的支撐函數（中心沿方向 u 走到邊界的距離）。

---

## 問題：矩形假設只對汽車成立

```
汽車俯視                        機車俯視
  ┌────────────┐                    ●          ← 前輪
  ●          ●                      │
  │            │                     │          接地 = 中心線上的線段
  ●          ●                      │
  └────────────┘                    ●          ← 後輪
  接地 = 矩形四角                    寬度方向沒有任何接地物
```

- **兩輪車**：前後輪都在中心線上，接地是一條**零寬度線段** → 橫向項應為 0
- **行人**：雙腳在身體正下方，接地基本上是一個**點** → 兩項都應為 0

把 W（或 L、W）設為 0，同一個支撐函數自動退化成線段／點的版本。

### 為什麼與視角無關

線段的支撐函數就是 `|u·前| × L/2`，對**任意** u 都成立，不需要分角度討論。兩個極端的物理檢查：

- **θ=90°（正側面）**：最低點是前後兩個輪胎接地點，離相機一樣遠、影像上一樣低，
  取平均後落在軸距中點 = 幾何中心 → **不該推**。線段模型給 0，舊公式給 W/2，全錯。
- **θ=0°（正前／正後）**：最低點是近端輪胎接地點 → **該往前推**。兩個模型給相同的縱向值。

誤差隨 θ 單調上升（速克達 1.90 × 0.68 m，`θ` 為視線與車身方向的夾角）：

```
  θ      舊 h(u)   線段模型   橫向誤差
   0°     0.950    0.950     0.000
  30°     0.993    0.823     0.170
  45°     0.912    0.672     0.240
  60°     0.769    0.475     0.294
  90°     0.340    0.000     0.340
```

**Hsinchu1 的 track 2 落在 θ≈84°**（`|u·前|=0.099`、`|u·右|=0.994`），幾乎正側面 ——
剛好是誤差最大的角度。改動前該機車的修正拆解：

```
track   類別          縱向分量   橫向分量   總修正
  1     car           1.348     0.721    2.069
  2     motorcycle    0.094     0.338    0.433   ← 橫向佔 78%，整段是錯的
  5     person        0.216     0.125    0.332
 18     person        0.053     0.244    0.298
```

---

## 實作內容（已還原，以下為當時的作法記錄）

`prior_dimensions.py` 新增類別 → 接地形狀的對照表，`g_projection.py` 加 `contact_shape` 參數，
`pipeline.py` 依類別傳入。

| 形狀 | 類別 | 縱向項 | 橫向項 | 等向 fallback（heading 未知時） |
|---|---|---|---|---|
| `rect` | car / van / truck / bus / three_wheeler | `L/2` | `W/2` | `(L+W)/π` |
| `segment` | motorcycle / motorbike / motor / two_wheeler / bike / bicycle | `L/2` | 0 | `L/π` |
| `point` | person / people / pedestrian | 0 | 0 | 0 |

未列在表上的類別一律當 `rect`（維持既有行為）。
等向 fallback 的係數來自 `E[|cos θ|] = 2/π`。

---

## 實測影響（Hsinchu1，`seg_default`）

以下是**改動生效時**量到的數字。程式碼已還原，所以現況等同下表的「舊／全部矩形」那一側；
這些數字保留下來的用意是：日後若重做，可以直接對照驗證是否複現。

### 位置變化

```
類別          n     中位      最大
car         106    0.000    0.000     ← 完全不受影響
motorcycle  288    0.191    0.340     ← 上限精確等於 W/2 = 0.340
person      106    0.318    0.354
```

track 2 單獨看：中位 0.337 m、最大 0.340 m。**新增跳動 0 幀**（逐幀位移惡化 > 0.3 m 的幀數）。

### 碰撞幾何（相對白車車身座標系）

```
  f=50   舊 縱向 -0.16 橫向 +1.27 → 右側    新 縱向 -0.40 橫向 +1.51 → 右側
  f=51   舊 縱向 -0.57 橫向 +1.35 → 右側    新 縱向 -0.81 橫向 +1.59 → 右側
```

接觸面判定不變。白車沒動，所以相對幾何的變化量就等於機車自己移動的 0.34 m。

### 連帶影響

- `sat_floor_box`、`bbox_3d` 跟著平移（**大小不變**，只有位置）
- 軌跡線（trail 畫的就是 `sat_coords`）跟著移動
- 位移法平均速度：track 1 `10.99 → 10.99`、track 2 `18.62 → 18.41` km/h
- `speed_kmh`（Kalman 存檔值）**完全不變** —— 錨點修正在 smoother 之後才套用

---

## 未決事項

### 1. 縱向項該用軸距而非全長

接地點是輪胎，不是車體的頭尾兩端。1.90 m 的速克達軸距大約 1.32 m，車體前後各懸出 0.3 m，
所以縱向項在正前／正後視角下高估約 0.29 m —— 量級和已修掉的橫向誤差相同。

```
  θ=0°   用全長 0.950 m   vs   用軸距 0.660 m
```

對 Hsinchu1 影響很小（θ≈84°，縱向項只有 0.094 m，改用軸距是 0.065 m，差 0.03 m），
但換一個從正後方拍的路口就會變成主要誤差。

要做的話需要在 `prior_dimensions.json` 和 `dimensions_<loc>.json` 加選填的 `wheelbase`，
沒填則退回 `length × 0.7`（兩輪車的典型比例）。**使用者 2026-08-16 決定先不做。**

### 2. 汽車是否有同樣問題（未驗證）

原則上汽車的輪胎接地點也在軸距四角，不在保險桿四角。但汽車有裙板、保險桿下緣離地僅 0.15 m
且涵蓋整個車身輪廓，影像上的最低點是輪胎接地點還是近側裙線，取決於視角與車體造型 ——
**沒有驗證過**。目前汽車維持用全長全寬，屬保守選擇。

驗證方式：把 `reference_point` 畫在 CCTV 畫面上，逐幀看它落在輪胎接地處還是裙線／保險桿下緣。

### 3. 行人該歸 `point` 還是 `segment`

當時的指示是「先做橫向項關掉」，行人被歸為 `point`（縱橫都關）是實作時自行的判斷 ——
理由和橫向一樣確定（雙腳在身體正下方，縱向同樣沒有東西可以讓錨點偏移），且不涉及軸距的
未定問題。**這個判斷始終沒有經過確認，改動就先被還原了。** 日後重做時要先確認：
行人歸 `point`（縱橫都關）還是 `segment`（只關橫向）。

### 4. 碰撞幀間隙變得不穩定（觀察，未解釋）

車框與機車框的最近距離：

```
  f=48   0.08 → 0.15
  f=50   0.14 → 0.10
  f=51   0.05 → 0.19
```

改動**前**看起來更像「持續接觸」，改動**後**跳動變大。但這三幀本身有兩個已知干擾：
f=51 起白車 mask 被污染（見 [遮擋造成的 mask 污染](occlusion-mask-contamination.md)），
且 f=50→51 之間才有一次真實動作（來源影片只有 8.5 fps 的內容）。三個樣本點不足以判斷
哪個版本比較準。

**這項改動的正當性來自幾何論證，不是來自間隙有沒有變小。** 若日後取得干擾較少的案例，
應重新檢視這個指標。

---

## 重現方式

### 修正量的縱橫分解

```python
import gzip, json, math
import numpy as np
PX = 17.723320166486495
CAM = np.array([519.4423828125, 364.55853271484375])   # G_projection 的 x/y_cam_coords_sat
d = json.load(gzip.open('output/.../Hsinchu1/Hsinchu.json.gz'))
byf = {f['frame_index']: {o['tracked_id']: o for o in f['objects']
                          if o.get('tracked_id') is not None} for f in d['frames']}
o = byf[51][2]                       # track 2 = 機車
W, L = 0.68, 1.90
p = np.array(o['sat_coords']); r = math.radians(o['heading'])
fwd = np.array([math.cos(r), math.sin(r)]); rgt = np.array([-math.sin(r), math.cos(r)])
u = CAM - p; u = u / np.linalg.norm(u)
print('縱向項', abs(u @ fwd) * L / 2, '橫向項', abs(u @ rgt) * W / 2)
```

### A/B 比對

不改動 `output/` 的作法：跑推論時把 `output_root` 指到暫存目錄，兩個版本各跑一次再逐幀比對。
若要在**不改原始碼**的情況下產生對照組，可以 monkeypatch 掉修正本身：

```python
from trafficlab.projection.g_projection import GProjection
GProjection.footprint_anchor_to_center = lambda self, *a, **k: None   # 停用錨點修正
# 之後再 import InferencePipeline 並跑到另一個 output_root
```

重做這項改動時，同樣的手法可以用來取得「全部當矩形」的對照組
（把 `contact_shape` 強制成 `'rect'`）。比對腳本：

```python
import gzip, json
import numpy as np
def bf(p):
    d = json.load(gzip.open(p))
    return {f['frame_index']: {o['tracked_id']: o for o in f['objects']
                               if o.get('tracked_id') is not None} for f in d['frames']}
A, B = bf('.../out_a/....json.gz'), bf('.../out_b/....json.gz')
worst = max(np.linalg.norm(np.array(o['sat_coords']) - np.array(A[i][tid]['sat_coords']))
            for i in B for tid, o in B[i].items() if tid in A[i])
print('最大座標差 (sat px)', worst)
```