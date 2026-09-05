# 遮擋造成的 mask 污染

**狀態：擱置**（現象通用且會影響其他場景，但對 Hsinchu1 的碰撞判定實測沒有影響）
**調查日期：2026-08-16**
**相關程式：** `trafficlab/inference/pipeline.py` 的 `_get_mask_bottom_center()` 與 `mask_ref` 路徑

---

## 一句話結論

實例分割在遮擋下會把 B 的像素判給 A 的 mask，這在**四段影片全部都有**（9%~40% 的重疊配對）。
但 Hsinchu1 的碰撞幀裡，污染發生在白車 mask 的**右上方**，而 `_get_mask_bottom_center()` 取的是
**最低點**（在車輛左下角），兩者不重疊 —— 所以扣除重疊區後參考點只移動 **0.7 px**（< 0.04 m），
不值得為這個案子先做。

---

## 現象

`_get_mask_bottom_center()` 取 mask 輪廓最低點附近的平均 x 當作接地參考點，再投影到衛星圖。
當兩個物件在畫面上重疊時，分割器可能把其中一方的像素併進另一方的 mask，使輪廓範圍失真。

Hsinchu1 的 track 1（白車）與 track 2（機車）在 f=51 開始明顯發生：

```
  f     車 mask 右緣   機車 bbox 左緣   越界     車 conf   機車 conf
 50         644            545        98 px     0.89      0.79
 51         722            524       197 px     0.85      0.53   ← 突變
 59         722            538       183 px     0.85      0.54
```

白車 mask 的右緣在一幀內外擴 78 px、越界量翻倍，同時機車信心值掉到 0.53 並鎖住。
目視疊圖可確認：白車的輪廓把機車的後半與後輪包了進去。

---

## 跨影片盛行率

指標：對每一組 bbox 有重疊的物件配對 (A, B)，計算 A 的輪廓點落在 B 的 mask 多邊形內的比例，
超過 15% 記為一次污染。

```
影片                      幀數   重疊對   污染對    比率     最嚴重案例
Hsinchu / Hsinchu1         60     646     153    23.7%   motorcycle→person 67% (f=21)
Yilan-Wujie               181     370     116    31.4%   car→truck        100% (f=119)
Yilan-babycar             599     440      40     9.1%   car→bus           90% (f=375)
test2-Yii                  50     740     293    39.6%   car→car           56% (f=31)
```

`car→truck 100%` 表示整台車的輪廓完全落在卡車的 mask 內 —— 這種情況下該車的底部參考點幾乎確定是錯的。

**這是實例分割的固有失效模式，不是特定影片的問題**，而且對事故影片必然發生：碰撞的定義就是
兩個物件在畫面上重疊。

### 指標的保留

這個數字量的是 mask **重疊**，不等於**污染**。真實遮擋下兩個輪廓本來就會交疊一部分，
15% 的門檻也是隨手取的。只有 67%、90%、100% 那種量級才是明確病態。
要嚴謹區分需要人工標註或另外的判準。

---

## 為什麼對 Hsinchu1 沒有影響（實測）

把白車 mask 光柵化後扣掉機車（track 2）與騎士（track 18）的重疊區域，再重新取底部中心：

```
  f      原參考點            扣除後           差異
 48   [507.0 475.5]    [507.0 475.0]      0.5 px
 51   [528.0 463.5]    [527.5 463.0]      0.7 px
 59   [528.8 463.5]    [528.5 463.0]      0.6 px
```

差異全部小於 1 px（< 0.04 m）。而且逐幀檢查，**白車的原參考點從來沒有落在機車的 mask 內**。

原因：污染發生在白車 mask 的右上方（機車車身、後輪的位置），但最低點在白車的**左下角**
（後保險桿 / 左後輪一帶），距離污染區很遠。mask 確實被污染，但污染到的部位不參與底部中心的計算。

> 註：調查初期曾把「f=51 之後參考點凍結」歸因於這個污染，該推論是錯的。
> 凍結的真正原因是**原始影片在 f=51 之後就沒有動作**（見 [[video-duplicate-frames]]）。

---

## 如果要做，怎麼做

### 修法 A：扣除其他實例的 mask（推薦）

在 `_get_mask_bottom_center()` 之前，把該物件 mask 與同幀其他物件 mask 的交集區域移除，
再取剩餘區域的最低點。

- 優點：原理乾淨、**沒有任何調參**、對所有場地一體適用
- 成本：需要光柵化，每幀每物件一次 `fillPoly`；只在 bbox 有重疊時才需要做，開銷可控
- 已驗證：在 Hsinchu1 上可正常執行並得出結果（只是結果幾乎不變）

### 修法 B：標記低信心而非猜測（互補，可能更重要）

當「扣除後的可見區域」小於原 mask 的某個比例、或最低點所在的列被其他實例覆蓋時，
在輸出加上不可靠旗標（例如 `ref_point_occluded: true`），讓下游知道這幾幀不要採信。

### 治不了的上限

當物件的接地部位**真的被擋住**時（車尾藏在另一台車後面），扣除 mask 只會讓可見區域更小，
不會變出看不見的接地點。這時正確的做法不是猜一個位置，而是誠實標記 —— 對事故重建而言，
一個「這裡量不準」比一個漂亮但錯誤的猜測有價值。這也是為什麼修法 B 可能比 A 重要。

### 另一個未解的問題

目前的指標是有方向性的（A 的點落進 B），但**無法判斷是誰的 mask 錯了** ——
「白車的 mask 錯誤吸收了機車」和「機車確實在白車前方、輪廓合理交疊」在數據上長得一樣。
要自動決定該扣誰，需要額外的判準（例如深度排序、或以離相機遠近推斷遮擋順序）。

---

## 重現方式

### 掃描所有輸出的污染率

```python
import gzip, json, glob
import cv2, numpy as np
for p in sorted(glob.glob('output/model-yolov8s-seg*/**/*.json.gz', recursive=True)):
    d = json.load(gzip.open(p))
    npair = ncont = 0
    for f in d['frames']:
        objs = [o for o in f['objects'] if o.get('mask_contour') and len(o['mask_contour']) >= 3]
        for a in range(len(objs)):
            for b in range(len(objs)):
                if a == b: continue
                A, B = objs[a], objs[b]
                ba, bb = A['bbox_2d'], B['bbox_2d']
                if ba[2] < bb[0] or bb[2] < ba[0] or ba[3] < bb[1] or bb[3] < ba[1]: continue
                npair += 1
                poly = np.array(B['mask_contour'], np.float32)
                pts = np.array(A['mask_contour'], np.float32)
                inside = sum(1 for q in pts
                             if cv2.pointPolygonTest(poly, (float(q[0]), float(q[1])), False) >= 0)
                if inside / len(pts) > 0.15: ncont += 1
    print(p, npair, ncont, f'{ncont/npair*100:.1f}%' if npair else '—')
```

### 測試扣除重疊後參考點的變化

```python
def raster(poly, x0, y0, w, h):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32) - [x0, y0]], 1)
    return m

# 對目標物件 c 與同幀其他物件 others：
P = np.array(c['mask_contour'], float)
x0, y0 = int(P[:,0].min()) - 2, int(P[:,1].min()) - 2
w, h = int(P[:,0].max()) + 3 - x0, int(P[:,1].max()) + 3 - y0
mc = raster(c['mask_contour'], x0, y0, w, h)
for o in others:
    mc = cv2.bitwise_and(mc, 1 - raster(o['mask_contour'], x0, y0, w, h))
ys, xs = np.nonzero(mc)
my = ys.max()
new_ref = np.array([xs[ys >= my - 3].mean() + x0, my + y0])   # 對比 c['reference_point']
```

### 畫出碰撞幀的 mask 疊圖

用 `cv2.polylines` 畫 `mask_contour`、`cv2.drawMarker` 畫 `reference_point`，
裁切到兩物件 bbox 的聯集範圍再放大。Hsinchu1 看 f=48~59。

---

## 重新檢視的條件

出現以下任一情況時值得回頭做：

- 分析的案子裡，受遮擋物件的**接地部位**（車輪、保險桿下緣）本身被擋住
- 遇到 `car→truck 100%` 那種整台被吞的案例
- 開始需要輸出「這一幀的量測可不可信」給下游 3D 重建端