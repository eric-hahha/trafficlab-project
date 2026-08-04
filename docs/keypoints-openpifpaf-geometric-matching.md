# H-aware Track ID 配對（geometric matching）

## 背景

OpenPifPaf 對整張影像偵測，不經過任何 tracker，所以 h-aware 輸出的 `tracked_id` 欄位原本全為 `null`。
Geometric matching（原稱 Method B）用 bbox IoU 橋接：讓 YOLO tracker 同幀偵測，再把 YOLO 的 `track_id` 指派給 PifPaf 偵測結果。

`eval_haware_replay.py` 現在有 `--method {geometric,crop}` 二選一（互斥）：本文件講的是 `--method geometric` 這條路徑；另一條 `--method crop`（crop-and-redetect，原稱 two-pass，裁切 Pass-1 bbox 重新偵測以取得更多信心關鍵點）不在本文件範圍內。

涉及檔案：
- `scripts/eval_haware_replay.py` — 主要實作
- `trafficlab/motion/haware_localization.py` — 有一份同邏輯的公開函式 `kp_bbox_xyxy`、`match_by_bbox_iou`，但 `eval_haware_replay.py` 實際上用的是自己內部 `_` 開頭的版本，兩者是平行、重複的程式碼，公開版本目前沒有任何呼叫者（死碼）
- `scripts/export_cctv_sat_composite.py` — 把 replay json 逐幀渲染成 CCTV+SAT 對照圖，用來肉眼檢查配對結果（重用 `trafficlab/visualization/` 底下既有的 GUI renderer，沒有新的繪圖邏輯）

---

## 目前預設值

```
--method          geometric           （必填，二選一；crop 模式下 YOLO 完全不會載入）
--yolo            models/best.pt      （VisDrone classes，car=3；空字串可關閉 YOLO 配對）
--yolo tracker    bytetrack.yaml      （顯式指定；ultralytics 套件本身的預設是 botsort.yaml，不指定就是 BoT-SORT）
--pifpaf-threshold 0.01
--seed-threshold   0.01
--iou-threshold    0.3
--yolo-classes     None（全類別，需要的話手動加 --yolo-classes 3 只留車）
```

⚠️ `models/best.pt` 的 class index 跟舊模型（`yolo11s-visdrone-v2-ft.pt`，car=2）不一樣，car 是 index **3**。換模型要記得檢查 `--yolo-classes` 傳的數字對不對，可以用 `YOLO('models/best.pt').names` 查。

## 執行流程（每幀）

```
PifPaf 偵測 (ann.data → 24 keypoints)
    ↓
YOLO tracker（僅 --method geometric 才載入；--yolo 有傳才跑；每幀無條件執行，不依賴 PifPaf 是否有偵測）
    ↓
PifPaf bbox：對每個偵測，取 confident keypoints (conf ≥ kp_conf, 預設 0.2) 的 tight xyxy
YOLO bbox：從 r.boxes.xyxy + r.boxes.id 取得
    ↓
IoU 配對：每個 PifPaf bbox 找 IoU 最高的 YOLO bbox
    若 IoU >= iou_threshold → 指派該 YOLO track_id，bbox_2d 用 YOLO 的框
    否則 → tracked_id = 該 PifPaf instance 在這幀的 index + 500（合成 id，見下）
            bbox_2d 退回用 PifPaf 關鍵點算出的框
```

### PifPaf bbox 計算（`_kp_bbox_xyxy`）

```python
pts = kp_24[(kp_24[:, 2] >= kp_conf) & ~((kp_24[:, 0] == 0) & (kp_24[:, 1] == 0))]
return (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
```

**永遠是軸對齊矩形**（min/max 直接取 x、y，跟車身實際朝向無關，不會旋轉貼合車身）。

### IoU 配對（`_match_by_bbox_iou`）

- Greedy argmax：每個 PifPaf 框獨立找 IoU 最高的 YOLO 框，**沒有互斥機制**——同一個 YOLO 框可能同時被兩個 PifPaf 框搶到（設計上就是簡單橋接，不是匈牙利法全域最優配對）
- 現在同時回傳配對到的 YOLO 框本身（`matched_boxes`），供 `bbox_2d` 使用

### bbox_2d 的來源

- **配對成功**：用 YOLO 的框（xyxy），不是 PifPaf 自己的框
- **配對失敗**：退回用 PifPaf 關鍵點算出的框（`_kp_bbox_xyxy`，跟 IoU 計算當下用的是同一個，不重算）
- 兩種情況 `bbox_2d` 都不再是 `None`（舊版本恆為 `None`）

### 未配對到 YOLO 的合成 id

配對失敗的 PifPaf 偵測，`tracked_id` 設為「該 instance 在這一幀的 index + 500」（例如這幀第 3 個 instance → `tracked_id=503`）。**這不是真的追蹤 id**：
- 純粹是這一幀內的位置索引，跨幀完全沒有連續性——同一台車下一幀很可能拿到不同的合成 id，顏色會跳
- 唯一目的：讓同一幀內多個配對失敗的偵測，在下游視覺化（`export_cctv_sat_composite.py`）裡彼此顏色能區分開來，不會全部疊成同一色

---

## 已知限制：退化框結構性配不上

`_kp_bbox_xyxy` 只有 0~1 個信心關鍵點時，min/max 相等，框退化成一個點（寬高皆 0）；就算有 2 個信心關鍵點，只要恰好共線（x 或 y 座標相同），一樣會退化成零面積。

IoU 公式的交集只要寬或高為 0 就必然是 0：

```python
inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)   # 寬或高為0 → inter恆為0 → IoU恆為0
```

**這是數學上的必然，不是機率低**：退化框無論跟哪個 YOLO 框比較、不管座標實際上離哪台車多近，IoU 永遠算出 0，永遠配不到 `iou_threshold=0.3`。`--pifpaf-threshold 0.01` 這個低門檻會大量產生只有 1~2 個信心關鍵點的低品質 instance，這些幾乎都會落入退化框、拿到 500+ 合成 id。

### 這些低品質 instance 值得救嗎？—— 進行中的討論，尚未實作

**假設**：部分低品質 instance 其實是 PifPaf 把同一台車重複偵測成兩個 instance（over-segmentation），不是真的漏偵測或雜訊，應該想辦法讓它們也能拿到跟「正確 instance」相同的 YOLO id。這個假設有實際觀察支持，是值得繼續深入的方向。

**已檢查的具體案例**（test21-6.mp4，`output/haware/test21/test21-6.json.gz`，這份輸出後來被清掉了，數字記錄於此）：frame 37、`id=13`（→ `tracked_id=513`）只有 1 個信心關鍵點（座標 `(321.79, 352.53)`，conf=0.35），該座標落在畫面左側卡車的 YOLO 框內（`(0,229,328,412)`），不是任何已配對成功車輛的重複偵測，比較像雜訊。**這只是一個樣本**，不能用來論證「大部分低品質 instance 都是雜訊」，需要多抽查幾個才知道比例。

**討論過的配對方法（都還沒實作）**：

| 方法 | 優點 | 缺點 |
|---|---|---|
| IoU（現況） | 對正常大小的框，能正確用「大小是否吻合」判斷是否同一台車 | 對退化框結構性恆為 0，完全無法使用 |
| Padding 成小框 + IoU | 直覺、實作簡單 | Padding 固定值只對遠景小車有效；近景大車/卡車需要的 padding 大到會產生誤配對，且效果隨畫面深度變化，固定值無法通用（實測：`±15px` padding 對 frame 37 案例的卡車框只能算出 IoU≈0.011，遠低於門檻） |
| Overlap coefficient（`交集 / min(兩框面積)`） | 對大小不成比例的框不懲罰，能正確處理「小點被大框包住」的情況；固定小 padding 即可（不需要動態調整） | 對**正常、有面積的框**套用反而危險：兩台車在畫面上重疊/遮擋時（路口常見），可能誤判為同一台車，因為它不懲罰大小差距 |
| 點到框最短距離（歐氏距離 ≤ 容忍值） | 概念最直接，不需要人工造一個 padding 框；在兩框不重疊、只是靠近的情況下判斷平滑自然 | 一旦點同時落在兩個重疊框內，兩者距離都是 0，完全沒有訊號可以分辨該給哪個——比 overlap coefficient 在這個情境下更沒有資訊量 |

**結論方向**（尚未定案）：
- 不建議對所有 instance 統一使用同一種方法——IoU 適合處理「大小是否吻合」有意義的正常框，overlap coefficient / 距離測試只該用在退化框（用 `n_keypoints` 或共線判斷去分流）
- 兩台車靠近/重疊時，任何方法都有可能因為單一低信心關鍵點資訊量不足而無法可靠判斷——建議的安全網是「候選分數太接近時直接判定無法分辨、不分配」，而不是強行選一個，避免把雜訊誤標成真實 id
- 下一步應該是**先多抽查幾個低品質 instance 的實際案例**，確認「重複偵測同一台車」這個假設在多少比例的案例中成立，再決定要不要投入實作
