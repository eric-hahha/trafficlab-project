# H-aware Instance 分裂修正（segmentation matching）

## 背景

PifPaf instance 分裂問題見 `docs/keypoints-openpifpaf-intro.md`「PifPaf Instance 分裂問題」一節：CIF/CAF decoder 在前後車身距離較遠、association field 較弱時，會把同一台車拆成兩個（或更多）獨立 annotation，各自的 keypoints 都不完整，導致定位/heading 不準，甚至在衛星視角變成兩個「幽靈車」。

`--method geometric` 的單點回收（見 `docs/keypoints-openpifpaf-geometric-matching.md`）只解決「其中一個 fragment 退化成單一 keypoint」的特例；兩個 fragment 各自都有多個 keypoint 的一般分裂情況，該機制完全處理不了——因為兩個 fragment 各自的 bbox 都不是退化框，IoU 判斷不出它們其實是同一台車。

`docs/keypoints-openpifpaf-intro.md` 原本設想的解法是「Keypoint NMS」（比較同幀內 PifPaf fragment 彼此的 bbox IoU 來合併），但這個做法本質上仍是自我參照的啟發式——兩個 fragment 分裂得夠開時，bbox IoU 一樣是 0，跟現在 IoU 配對失效是同一個原因。`--method segmentation` 改用外部、獨立的 instance segmentation 模型（car-segmenter，YOLO11-seg）給出的真實車體 mask 當作 ground truth：不管 PifPaf 把 keypoints 切成幾個 fragment，只要這些 keypoints 落在同一個 mask 裡，就判定屬於同一台車。

涉及檔案：
- `scripts/run_keypoints_openpifpaf.py` — `--method segmentation` 分支
- `scripts/record_car_masks.py` — 獨立記錄 car-segmenter 輸出到檔案，見下方「兩種 mask 來源」
- `trafficlab/motion/segmentation_car.py` — `CarMaskSource`（包一層 car-segmenter）、`assign_predictions_to_masks`/`assign_predictions_to_polygons`、`merge_keypoints_by_group`、`detections_to_records`、`mask_to_polygon`
- `trafficlab/vendor/car_segmenter/` — vendored 進來的 car-segmenter 原始碼（來源：`/Users/ericc/Code/supervision/car-segmenter`），只保留 `predict()` 用得到的部分，沒有複製 `cli.py`

---

## 執行流程（每幀）

```
PifPaf 偵測 (predictions: 每個 annotation .data → 24 keypoints)
    ↓
car-segmenter 偵測 (CarMaskSource.detect(frame) → sv.Detections：.mask (M,H,W) / .xyxy / .tracker_id)
    ↓
assign_predictions_to_masks：每個 PifPaf annotation，用它所有信心關鍵點的像素座標
    對 M 個 mask 投票（keypoint 落在哪個 mask 裡就投那個 mask 一票），取票數最多的 mask index
    → 若沒有任何一個信心 keypoint 落在任何 mask 內 → None（不強行合併）
    ↓
merge_keypoints_by_group：group_of 相同（非 None）的 annotation 合併成一個 (24,3) 陣列
    → 每個 keypoint slot 取信心度較高的那個值
    → 回傳 {mask_index: 合併後 kp_24} + 沒配到任何 mask 的 annotation index（leftover）
    ↓
每個 merged 群組 → tracked_id 用 car-segmenter 自己的 tracker_id（ByteTrack，非 YOLO）
                    bbox_2d 用 car-segmenter 的 .xyxy（mask 導出的框，非 PifPaf 關鍵點框）
                    → 呼叫既有 localizer.localize()/.localize_reprojection()（完全不變）
每個 leftover → tracked_id = 該 annotation 在這幀的 index + 500（合成 id，跟 geometric 同慣例）
                bbox_2d 退回用 PifPaf 關鍵點算出的框（_kp_bbox_xyxy）
```

### 跟 `--method geometric` 的差異

- **不用 YOLO/Method B**：tracking 完全交給 car-segmenter 內建的 ByteTrackTracker，`--yolo*` 系列參數在這個 method 下不會被讀取。
- **分組依據是 mask membership，不是 bbox IoU**：兩個分裂的 fragment 只要 keypoints 落在同一個 mask 內就會被合併，不管兩個 fragment 各自的 bbox 離多遠——這正是 geometric 的 IoU 判斷做不到的。
- car-segmenter `tracker_id` 在 track 剛建立的前幾幀會回傳 `-1`（warm-up，見 vendored `segmenter.py` 的 `annotate()` 判斷慣例），這裡沿用同樣的判斷：`tracker_id < 0` 一律視為 `None`。

### 兩種 mask 來源：即時跑模型 vs 讀記錄檔（`--seg-masks-json`）

同一支影片常常要分別用 `--localizer procrustes` 和 `--localizer reprojection` 各跑一次比較結果——這種情況下車-segmenter 沒理由跑兩次，因為兩次的 mask/tracker_id 應該完全一樣。`scripts/record_car_masks.py` 把整支影片跑過 car-segmenter，將每幀的 `{tracker_id, bbox_xyxy, confidence, polygon}` 記錄成檔案（`output/car_masks/<location>/seg-mask_<video_stem>.json.gz`）；`run_keypoints_openpifpaf.py --method segmentation --seg-masks-json <path>` 讀這份檔案取代即時跑模型，car-segmenter 完全不會被載入。

**為什麼記錄的是 polygon，不是 dense mask**：`sv.Detections.mask` 是逐 pixel 的 `(N,H,W)` bool array，整支影片存下來太肥。改用 `cv2.findContours` + `cv2.approxPolyDP` 簡化成幾十個點的多邊形（跟 `trafficlab/gui/tabs/calibration_stage/homf_stage.py` 算 FOV polygon 的手法一樣），檔案大小差好幾個數量級。

**兩條路徑用兩份不同的比對實作，不是共用一份**（刻意的決定，不是遺漏）：
- 即時跑模型：`assign_predictions_to_masks`，對 dense mask array 直接 `masks[m, y, x]` 索引判斷。這條路徑先做過、已經 smoke test 驗證過，之後沒有再動。
- 讀記錄檔：`assign_predictions_to_polygons`，對 polygon 用 `cv2.pointPolygonTest` 判斷。只服務 `--seg-masks-json` 這條路徑。

兩者判斷邏輯對稱（同樣是「信心 keypoint 對每個 instance 投票，取最高票，沒票的維持 `None`」），只是輸入表示法（dense mask vs 簡化 polygon）不同，理論上只有 polygon 簡化造成的邊界誤差，不會有分組邏輯本身的差異。之後如果要做全面重構把兩者合併成一份實作，先確認 polygon 版在邊界像素上的行為跟 dense mask 版是否有肉眼可見差異。

### 目前不處理的情況

- **只有 segmentation 偵測到、PifPaf 完全沒偵測到的車**：這類車輛不會輸出成 detection（沒有 keypoints，現有 localizer 沒東西可用）。數量會累積進 `n_seg_no_pifpaf`，在跑完後的統計印出來，不是靜默丟棄。
- **car 以外的車種**（truck / bus / motorcycle）：`CarMaskSource` 目前寫死 `classes=('car',)`，之後要擴充车种範圍需要同時評估其他車種的 mask 品質。

## 目前預設值

```
--method     segmentation   （必填，三選一之一）
--seg-model  models/yolo11n-seg.pt （首次執行自動下載到這個路徑，需要網路）
--seg-conf   0.3
--seg-device None（交給 ultralytics 自動選擇 cuda/mps/cpu）
```
