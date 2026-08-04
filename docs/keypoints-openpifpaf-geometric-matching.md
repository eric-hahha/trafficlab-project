# H-aware Track ID 配對（geometric matching）

## 背景

OpenPifPaf 對整張影像偵測，不經過任何 tracker，所以 h-aware 輸出的 `tracked_id` 欄位原本全為 `null`。
Geometric matching（原稱 Method B）用 bbox IoU 橋接：讓 YOLO tracker 同幀偵測，再把 YOLO 的 `track_id` 指派給 PifPaf 偵測結果。

`run_keypoints_openpifpaf.py` 現在有 `--method {geometric,crop}` 二選一（互斥）：本文件講的是 `--method geometric` 這條路徑；另一條 `--method crop`（crop-and-redetect，原稱 two-pass，裁切 Pass-1 bbox 重新偵測以取得更多信心關鍵點）不在本文件範圍內。

涉及檔案：
- `scripts/run_keypoints_openpifpaf.py` — 主要實作
- `trafficlab/motion/keypoints_openpifpaf.py` — 有一份同邏輯的公開函式 `kp_bbox_xyxy`、`match_by_bbox_iou`，但 `run_keypoints_openpifpaf.py` 實際上用的是自己內部 `_` 開頭的版本，兩者是平行、重複的程式碼，公開版本目前沒有任何呼叫者（死碼）
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
    否則 → 進入單點回收（見下）
    ↓
單點回收：僅處理 IoU 配對失敗、且恰好只有 1 個信心關鍵點的偵測
    → 若能唯一定位到一個 YOLO 框，且該 track 這幀已有 main 偵測 → 併入 main，本偵測丟棄
    → 若能唯一定位但沒有 main → 直接指派該 track_id，偵測照常輸出
    → 其餘情況（無單點 / 座標落在 0 或 2+ 個框內 / 無 track_id）→ 不處理
    ↓
仍未取得 tracked_id 的偵測 → tracked_id = 該 PifPaf instance 在這幀的 index + 500（合成 id，見下）
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

### 單點回收（`_single_confident_kp` / `_point_in_box`）

只有 0~1 個信心關鍵點時，`_kp_bbox_xyxy` 的 min/max 相等，框退化成一個點（寬高皆 0），跟任何 YOLO 框的 IoU 恆為 0（交集寬或高為 0），永遠配不到 `iou_threshold`。這類偵測不再直接落入合成 id，而是多一次判斷：

- 只處理**恰好 1 個**信心關鍵點的偵測——用信心關鍵點數直接數，不是用退化框的形狀反推（兩個不同 index 的關鍵點剛好落在同一像素，也會產生零面積框，但那是 n=2，不該當成單點處理）
- 該關鍵點座標檢查落在哪些 YOLO 框內：**沒有做 padding**，直接用原始 YOLO 框做包含判斷
- 座標同時落在 0 個或 2 個以上 YOLO 框內時，訊號不足以分辨該給哪個，直接跳過不處理，不強行選一個
- 若唯一落在一個 YOLO 框內：
  - 該 YOLO track 這一幀若已有其他偵測透過 IoU 配對成功（main）→ 把這個關鍵點依 apollo-24 index 併入 main 的 `kp_24`（main 已有信心值的 index 不覆蓋），本偵測**丟棄**，不再單獨輸出成一個物件
  - 若這一幀沒有 main 可以併 → 直接把該 track_id 指派給這個偵測，偵測本身照常輸出

### 未配對到 YOLO 的合成 id

IoU 配對失敗、且沒有被單點回收機制救到的 PifPaf 偵測，`tracked_id` 設為「該 instance 在這一幀的 index + 500」（例如這幀第 3 個 instance → `tracked_id=503`）。**這不是真的追蹤 id**：
- 純粹是這一幀內的位置索引，跨幀完全沒有連續性——同一台車下一幀很可能拿到不同的合成 id，顏色會跳
- 唯一目的：讓同一幀內多個配對失敗的偵測，在下游視覺化（`export_cctv_sat_composite.py`）裡彼此顏色能區分開來，不會全部疊成同一色
