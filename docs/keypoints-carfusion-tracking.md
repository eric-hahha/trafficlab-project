# CarFusion 跨幀 ID 追蹤（ByteTrack）實作紀錄

## 背景

`scripts/eval_carfusion_sat.py` 原本每幀對 CarFusion YOLOv8-Pose 模型做**單張獨立推論**（`yolo_model(frame, ...)`），輸出的 `tracked_id` 其實是 `det_id`——每幀從 1 重新遞增的流水號，跟上一幀的同一台車完全沒有對應關係。`tracked_id` 這個欄位在下游被廣泛消費（`trafficlab/visualization/sat_renderer.py`、`cctv_renderer.py`、`trafficlab/trajectory/plotting.py`、`smoothing.py`、`scripts/postprocess/postprocess.py` 等），全部假設它是跨幀不變的整數，原本的假 ID 會讓這些工具全部失效。

使用者確認 CarFusion 自己的 bbox 是準的，因此不需要自建 world-space tracker，改用 ultralytics 內建的 tracker（ByteTrack）。

---

## 實作

### 涉及檔案

- `scripts/eval_carfusion_sat.py`：主要改動
- `trafficlab/inference/bytetrack.yaml`：新增，pinned tracker 設定檔

### 推論方式改動

```python
# 原本
results = yolo_model(frame, conf=conf_thresh, verbose=False)

# 改為
results = yolo_model.track(frame, conf=conf_thresh, persist=True,
                            tracker=tracker, verbose=False)
```

`result.boxes.id` 是跨幀持續的 track id（`None` 表示這幀沒有任何框被 tracker 接受）；`tid = int(track_ids[i]) if track_ids is not None else None`，寫入 `det_records['tracked_id']`，取代原本的 `det_id` 計數器。

跟現有慣例一致：`trafficlab/inference/pipeline.py` 的主 pipeline 已經是這樣用 `model.track()` + `boxes.id`；`scripts/eval_haware_replay.py` 的 YOLO 橋接也是逐幀呼叫 `.track(frame, persist=True, ...)`。

### 為什麼把 `bytetrack.yaml` 複製進專案

`--tracker` 原本傳的是裸檔名 `'bytetrack.yaml'`，會用 ultralytics 套件安裝路徑內建的版本——套件升級後這份預設值可能改變，追蹤行為會在不知情下跟著變。做法比照 `pipeline.py._build_tracker_config()` 已經在做的事：把設定明確固定進專案（`trafficlab/inference/bytetrack.yaml`），可重現、可在 git history 追蹤變更。

`--tracker` 預設值改指向 `trafficlab/inference/bytetrack.yaml`。

### 每 ID 專屬顏色（視覺化）

新增 `_id_color_rgb01(tid)`（golden-ratio hue spacing，`hue = (tid * 0.618...) % 1.0`），cv2 畫框（`_id_color()`，轉 BGR int）跟 matplotlib 散點圖（`_save_scatter()`，直接用 RGB float）共用同一組色相邏輯，同一個 id 在 CCTV bbox 疊圖跟衛星散點圖上是同一個顏色。`tid=None` 時 fallback 成原本的青色（`_C_BOX_NONE = _C_BOX`）。

`sat_coords_out` 從 `(sat_coords, heading)` 改成 `(sat_coords, heading, tid)` 三元組，傳給 `_save_scatter()` 上色用。

---

## 踩過的坑：ByteTrack 內部機制

用 test21-6.mp4（conf=0.01）反覆測試時發現，光是 YAML 調參數不足以解釋所有現象，追進 ultralytics 原始碼（`ultralytics/trackers/byte_tracker.py`）後確認三層機制：

### 1. `.track()` 會把配不到 track 的框直接丟棄，不是留著標 `tracked_id=None`

`BYTETracker.update()` 最後一行 `return [x.result for x in self.tracked_stracks if x.is_activated]`——只有 `is_activated=True` 的 track 才會出現在 `result.boxes` 裡；配不到、也還沒確認的偵測框，整個從輸出消失，不會以任何形式出現（包括 `tracked_id=None`）。因此換成 `.track()` 後，同一個 `--conf` 值畫面上看到的框可能比單張推論少很多。

驗證：test21-6 frame 10，conf=0.01 時單張推論有 10 框，`.track()` 只剩 1 框。

### 2. `conf` 參數在 `.track()` 模式下的作用範圍縮小了

`conf` 只決定 YOLO 交給 tracker 多少候選框（第一道過濾，YOLO 自己套用），最終畫面上留下多少框由 tracker 自己的門檻（`track_high_thresh`/`new_track_thresh`，預設 0.25）決定。同一幀測過 `conf=0.01/0.05/0.15/0.25`，單張推論框數隨之變化（10→6→3→1），但 `.track()` 輸出永遠是 1——`conf` 調再低也沒用，因為瓶頸在 tracker 門檻，不在 `conf`。

### 3. 新 track 要連續兩幀「確認」才會輸出，且確認門檻是 `IOU × 偵測信心值`，不是純 IOU

`STrack.activate()`：新 track 建立時 `is_activated=False`（除非是影片第 1 幀）；要等下一幀被 `unconfirmed` 匹配到 `.update()` 才變 `True`。

匹配用的 cost 來自 `get_dists()`：

```python
dists = iou_distance(tracks, detections)      # = 1 - IOU
if fuse_score:
    dists = fuse_score(dists, detections)     # = 1 - (IOU × 偵測信心值)
```

`fuse_score()` 原始碼（`ultralytics/trackers/utils/matching.py`）：`fuse_cost = 1 - IOU * score`。未確認 track 的匹配門檻是 `thresh=0.7`，等於要求 `IOU × score >= 0.3`。

**實際案例**：test21-6 有台車（frame 70 附近，bbox 起點 `x=0`，貼著畫面左緣、疑似因為框不完整導致信心值長期偏低）信心值長期在 0.02～0.16，但幀間 IOU 其實有 0.8～1.0（用使用者提供的、未經 tracker 處理的原始 JSON `carfusion_test21_6_conf001.json` 反算驗證）。`IOU × score` 幾乎從未到 0.3，導致這台車每一幀都建立新 track、下一幀確認失敗、被丟棄、再重建——track_id 編號因此跳得很快（404→408→413→417...），但從未真正輸出過。

`is_activated` 這道過濾是 ultralytics 寫死在程式碼裡的邏輯，`bytetrack.yaml` 完全管不到；但 `fuse_score` 本身就是 yaml 現有欄位，可以直接關閉。

---

## 目前設定（`trafficlab/inference/bytetrack.yaml`）

```yaml
track_high_thresh: 0.01   # 原預設 0.25
track_low_thresh: 0.01    # 原預設 0.1
new_track_thresh: 0.01    # 原預設 0.25
track_buffer: 30          # 未變
match_thresh: 0.8         # 未變
fuse_score: False         # 原預設 True
```

門檻降到 0.01：讓 `conf=0.01` 篩出來的候選框都有機會進入 tracker 的評估流程（見坑 2）。
`fuse_score: False`：改成純 IOU 匹配，不再讓低信心值拖累幾何位置本來就穩定的偵測（見坑 3）。

**已知代價（相對於 ultralytics 預設值）**：
- 更多重疊/雜訊候選框會被單獨建立 track（門檻降低的直接後果）。
- Stage 1 匹配不再用信心值當 tie-breaker，理論上可能讓既有的高信心 track（例如已經穩定追蹤的車）在極少數情況下被幾何上更接近、但其實是雜訊/重複框的候選誤配對走——目前測試影片沒有觀察到明顯案例，但要留意。

---

## 測試結果

### test21-4.mp4（frame 120-175，56 幀，`--conf` 預設 0.25，門檻改動前）

| 指標 | 數值 |
|------|------|
| 處理幀數 | 56 |
| 有偵測的幀 | 41 |
| 總偵測數 | 71 |
| 定位成功 | 71（100%）|
| unique tracked_id | 5 |
| 最長連續 track | id=1，37 幀無斷點（128-164）|
| tracked_id=None | 2/71，皆為新車第一次出現（下一幀即拿到真實 id，屬預期行為）|

### test21-6.mp4（96 幀，`--conf 0.01`）

| 設定 | unique id | tracked_id=None | 總偵測數 | 備註 |
|------|-----------|------------------|----------|------|
| 門檻 0.25（預設）| 9 | 123/265（46%）| 265 | None 集中在 17 幀「整幀都低於門檻」的情況 |
| 門檻 0.01，fuse_score=True | 25 | 94/250 | 250 | track_id 編號跳到 400+，大量建立即銷毀 |
| 門檻 0.01，fuse_score=False | 25 | 0/640 | 640 | id 編號回到 1-29，無單幀即消失的 track；使用者指定要追蹤的車（frame 70 附近）穩定拿到 `id=23`，連續出現 62-95 幀（33/34）|

---

## 相關程式碼位置

| 檔案 | 用途 |
|------|------|
| `scripts/eval_carfusion_sat.py` | 主要改動：`.track()` 呼叫、`tracked_id` 寫入、per-id 上色 |
| `trafficlab/inference/bytetrack.yaml` | pinned tracker 設定，門檻與 `fuse_score` 已調整 |
| `trafficlab/inference/pipeline.py` | 主 pipeline 既有的 tracker 用法，本次改動參考的慣例來源 |
| `scripts/eval_haware_replay.py` | haware 的 YOLO 橋接，逐幀 `.track()` 呼叫的既有先例 |
