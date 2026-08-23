# TrafficLab AI 指示文件

本檔案是給在這個 repository 裡工作的 AI coding agent 看的操作細節；README 保持面向使用者、簡潔扼要。

## 工作原則

除非明確override，否則這些規則適用於此專案中的每一項任務。
原則：在非小事的工作上，謹慎優先於速度。小事則依判斷行事。

### 規則 1 — 動手前先思考
明確說明假設。若不確定，應提問而非用猜的。
當存在歧義時，提出多種可能的解讀。
若有更簡單的做法，應提出來討論。
感到困惑時就停下來，並具體指出不清楚的地方。

### 規則 2 — 遵循該程式碼庫的慣例，即使你不認同
在程式碼庫裡，「一致性」優先於「個人品味」。
如果你真心認為某個慣例有害，就提出來討論，不要私自另起爐灶。

### 規則 3 — 手術式修改
只動你必須動的地方。只清理自己造成的髒亂。
不要「順便改善」旁邊的程式碼、註解或格式。
不要重構沒壞的東西。match現有的風格。

### 規則 4 — 以目標為導向的執行
先定義成功標準，反覆執行直到驗證通過。
不要只是照步驟做，要定義「成功」是什麼並反覆迭代。
明確的成功標準能讓你獨立地反覆執行。

### 規則 5 — 出問題要大聲說出來
如果任何東西被悄悄跳過，那「已完成」這個說法就是錯的。
如果任何測試被跳過，那「測試通過」這個說法就是錯的。
預設要主動揭露不確定性，而不是隱藏它。

### 規則 6 — 只在需要判斷力時才用到我（模型）
可以用我做：分類、草擬、摘要、擷取資訊。
不要用我做：routing、重試、確定性的轉換。
如果程式碼能處理，就交給程式碼處理。

### 規則 7 — 衝突要攤開來講，不要折衷處理它們或混合使用
若兩種模式互相矛盾，選一個（較新的／較經過驗證的）。
說明理由，並標記另一個待清理。
不要把互相衝突的模式混在一起用。

### 規則 8 — 先讀再寫
在新增程式碼之前，先讀過相關的 exports、直接呼叫方、共用的 utilities。
「看起來無關」是很危險的想法。若不確定程式碼為何這樣設計，就提問。

### 規則 9 — 測試要驗證意圖，不只是驗證行為
測試必須表達出「為什麼這個行為重要」，不能只是「做了什麼」。
如果商業邏輯改變了，測試卻不會失敗，那這個測試就是錯的。

### 規則 10 — 每完成一個重要步驟就checkpoint一次
摘要已完成的事、已驗證的事，以及還剩下什麼。
不要從一個你無法複述清楚的狀態繼續做下去。
如果你失去脈絡，就停下來重新整理狀態。

## Repository 用途

本專案把車禍影片重建成地圖上的 2D 行車軌跡。以資深貢獻者的標準工作：精準、有效率、重視可維護性與清晰度。

## Repository 結構

```text
TrafficLab-3D/
├── main.py                     # GUI 進入點
├── trafficlab/
│   ├── gui/                    # PyQt5 GUI 實作
│   ├── inference/               # GUI 與 CLI 共用的推論 pipeline
│   ├── projection/              # G-projection 與 SVG projection 的輔助工具
│   ├── visualization/           # replay 載入與渲染
│   ├── io/                      # replay/config I/O 輔助工具
│   ├── motion/                  # 運動學（kinematics）工具
│   └── trajectory/              # 推論後的軌跡平滑化與靜態繪圖工具
├── scripts/                     # CLI 輔助工具與維護用工具
├── location/<location_code>/    # 校正資產、衛星／CCTV 影像、投影檔案與影片素材
├── output/                      # 產生出來的 model/tracker/config/location replay 輸出結果
└── models/                      # 本地端的 detector/tracker checkpoint
```

- h-aware / OpenPifPaf：邏輯在 `trafficlab/motion/keypoints_openpifpaf.py`，CLI 入口在 `scripts/run_keypoints_openpifpaf.py`。
- CarFusion：邏輯在 `trafficlab/motion/keypoints_carfusion.py`，CLI 入口在 `scripts/run_keypoints_carfusion.py`。

## 指令規則

- 所有 Python 指令都要先啟動 `trafficlab` conda 環境、並從 repository root 執行：

```bash
source /opt/anaconda3/bin/activate trafficlab
```

- 優先直接啟動環境，不要用 `conda run -n trafficlab ...`（sandbox 裡可能因暫存檔寫入權限失敗，除非直接啟動行不通才用）。
- 從 repo root 執行時 `import trafficlab` 會正常解析；若指令必須在 repo root 以外執行，設定 `PYTHONPATH=<repo_root>`（例如 `PYTHONPATH=$(pwd)`，在 repo root 下執行時）。`scripts/run_inference.py` 是例外，即使在 repo root 也不會自動加進 `sys.path`，一律要明確帶 `PYTHONPATH`（見下方「YOLO 框定位」）。
- 只有跑 YOLO 模型時，才需要在 Apple Silicon 上加 `PYTORCH_ENABLE_MPS_FALLBACK=1`（GUI 若可能牽涉到推論也算）；`device: mps` 時不要假設 `half: true` 一定安全。
- 不要依賴系統的 `python` binary；缺 dependency 時先確認 `trafficlab` 環境已啟動再下結論。

## 常用指令

### 1. 開啟 GUI

```bash
source /opt/anaconda3/bin/activate trafficlab && PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py
```

### 2. YOLO 框定位

透過 `scripts/run_inference.py`（`trafficlab.inference.pipeline.InferencePipeline`）執行 YOLO 偵測＋追蹤＋bbox 投影，是 TrafficLab 原生的定位方法，不使用關鍵點。這是三種定位方法之一（另外兩種是下方的「keypoints-openpifpaf」與「keypoints-carfusion」）— 依情境挑選合適的方法，不要把它當成預設要跑的方法。

```bash
source /opt/anaconda3/bin/activate trafficlab && PYTHONPATH=$(pwd) PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_inference.py --config-name <config_name> <selector>
```

- `--all-pending`：處理所有待處理影片
- `--location <location_code>`：處理單一 location
- `--mp4 <video_path>`：處理單一 mp4
- 加 `--force`：即使輸出已存在也強制重跑
- `<config_name>`：YAML（預設 `inference_config.yaml`，可用 `--config-path` 換）裡定義的 config key，不填則用檔案裡第一個 config

Config 的 `model` 區塊預設是偵測模型（regression box 直接拿去投影）。設定 `model.type: "seg"` 可以改用 segmentation checkpoint（如 `models/yolo11m-seg.pt`）：box 來源改成從分割 mask 的最大 contour 算出的 tight box，其餘流程（追蹤、投影、kinematics）不變。seg 模式下 `model.classes` 為必填，是 COCO 類別名稱 → 專案內部類別名稱的對照表（例如 `{car: car, motorcycle: two_wheeler}`）；分割模型偵測到的每個物件都會算 tight box，但只有在這張表裡的類別會被定位、寫進輸出，其餘（如 `person`）算完 box 後就跳過。seg 模式的輸出固定寫到 `output/yolobox-seg/model-<model_name>_<tracker_type>/<config_name>/<location_code>/`，忽略一般模式用的 `output/` 輸出根目錄。

### 3. 後處理（Postprocess）

```bash
source /opt/anaconda3/bin/activate trafficlab && python postprocess.py --help
```

### 4. keypoints-openpifpaf

在每一幀上執行 OpenPifPaf Apollo-24 偵測，並透過 height-aware 關鍵點模板比對來定位每輛車。輸出是標準的 TrafficLab replay JSON，可以在 GUI 中載入（衛星座標位置、車頭朝向、車輛外框，以及 CCTV 關鍵點疊圖）。`--method` 是必填參數，用來選擇三種互斥的逐幀策略之一 — 只有被選中策略自己的 flag 才會生效。

```bash
source /opt/anaconda3/bin/activate trafficlab && \
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_keypoints_openpifpaf.py \
  --video <video_path> \
  --g-proj <g_proj_path> \
  --method <geometric|crop|segmentation>
```

輸出：`output/haware/<location_code>/<video_stem>.json.gz` — 但 `--localizer wheel_pair` 例外，會寫到 `output/wheel_pair/<location_code>/<video_stem>.json.gz`（獨立資料夾，避免跟同一支影片的 `procrustes`/`reprojection` 輸出互相覆蓋）。

`--method` 選項：

| Method | 何時選它 | 相關 flag |
|--------|---------------|-----------------|
| `geometric` | 沒有明顯 PifPaf instance-splitting 時，用 bbox IoU 把關鍵點對到 YOLO track id | `--yolo` / `--yolo-conf` / `--yolo-classes` / `--iou-threshold` / `--yolo-boxes-json` / `--yolo-boxes-class` |
| `crop` | 想從裁切圖重跑 PifPaf、找回更多高信心度關鍵點時用 | `--crop-redetect` / `--crop-padding` |
| `segmentation` | PifPaf 把同一台車拆成兩塊多關鍵點碎片、`geometric` 修不了時用 — 詳見 `docs/keypoints-openpifpaf-segmentation-matching.md` | `--seg-model` / `--seg-conf` / `--seg-device` |

共用選項：

| Flag | 預設值 | 說明 |
|------|---------|-------|
| `--checkpoint` | `shufflenetv2k16-apollo-24` | PifPaf model |
| `--spec-csv` | *(無)* | automobile-models-and-specs 的 `engines.csv`；找不到就退回 `prior_dimensions.json`，再找不到就用內建預設值 |
| `--body-type` | `Sedan` | 與 `--spec-csv` 搭配使用 |
| `--kp-conf` | `0.2` | 關鍵點信心度閾值 |
| `--frames` | `-1`（全部） | 限制幀數以便快速測試 |
| `--start-frame` | `0` | 開始處理的第一幀；在此之前的幀會被讀取後丟棄，而不是用 seek |
| `--localizer` | `procrustes` | `procrustes`（closed-form 擬合，一般情況）、`reprojection`（非線性最小平方，對 PifPaf 像素位置擬合），或 `wheel_pair`（只用同側前後輪關鍵點，適合僅輪胎清楚可見時 — 詳見 `docs/keypoints-openpifpaf-wheel-pair-localizer.md`） |
| `--out` | 自動 | 覆寫輸出路徑 |

`--method geometric` 選項：

| Flag | 預設值 | 說明 |
|------|---------|-------|
| `--yolo` | `models/best.pt` | 用於 track-ID 比對的 YOLO model（ByteTrack）；傳入 `--yolo ""` 可停用並讓 `tracked_id=None` |
| `--yolo-conf` | `0.25` | YOLO 偵測信心度閾值 |
| `--yolo-classes` | *(全部)* | 要保留的 YOLO class index，以逗號分隔 — 依 model 而異，請檢查 `--yolo` 所指 model 的 `.names` |
| `--iou-threshold` | `0.3` | 接受 PifPaf↔YOLO 比對所需的最小 bbox IoU |
| `--yolo-boxes-json` | *(無)* | 使用預先算好的 replay JSON（例如 `pipeline.py` 的輸出）作為 YOLO box 來源，而不是即時跑 model；設定時優先於 `--yolo` |
| `--yolo-boxes-class` | `car` | `--yolo-boxes-json` 中要視為汽車的 `class` 值 |

`--method crop` 選項：

| Flag | 預設值 | 說明 |
|------|---------|-------|
| `--crop-redetect` | 關閉 | 以 50% padding 裁切每個 Pass-1 bbox 並重新跑 PifPaf；沒有這個 flag 時，`--method crop` 等同於單純的 Pass-1 PifPaf |
| `--crop-padding` | `0.5` | 裁切時 bbox 周圍的 padding 比例 |

`--method segmentation` 選項：

| Flag | 預設值 | 說明 |
|------|---------|-------|
| `--seg-model` | `models/yolo11m-seg.pt` | car-segmenter 用的 Ultralytics `*-seg` checkpoint；若本地端沒有，首次使用時會自動下載到該路徑 |
| `--seg-conf` | `0.3` | car-segmenter 偵測信心度閾值 |
| `--seg-device` | *(自動)* | `cuda` / `mps` / `cpu`，或不設定讓 ultralytics 自行選擇 |
| `--seg-masks-json` | *(無)* | 指向 `record_car_masks.py` 輸出檔案的路徑，提供逐幀的汽車 instance（tracker_id/bbox/mask polygon），用它取代即時執行 car-segmenter。設定此參數時，`--seg-model`/`--seg-conf`/`--seg-device` 會被忽略，car-segmenter 也完全不會被載入。 |

在標準 14 個欄位之外，每個物件會新增的欄位：`kp_cctv`（GUI 疊圖用的原始 `[x, y, conf] × 24`）、`n_keypoints`、`status`（`ok` / `ambiguous_heading` / `failed_insufficient_kp`）。

#### 一次錄製 car-segmenter mask 供重複使用

`--method segmentation` 通常每次執行都會即時跑一次 car-segmenter。如果同一支影片需要同時跑 `--localizer procrustes` 和 `--localizer reprojection` 兩個 pass，就會對相同的輸出跑兩次 YOLO11-seg，改成只錄製一次、兩個 pass 共用：

1. 錄製一次：

```bash
python scripts/record_car_masks.py --video <video_path>
```

輸出：`output/car_masks/<location_code>/seg-mask_<video_stem>.json.gz`（純影像空間的紀錄 — 沒有 G-projection，也沒有 PifPaf）。

2. 兩個 pass 都指向同一份 mask 檔：

```bash
python scripts/run_keypoints_openpifpaf.py --video <video_path> \
  --g-proj <g_proj_path> --method segmentation \
  --seg-masks-json <mask_json_path> --localizer procrustes

python scripts/run_keypoints_openpifpaf.py --video <video_path> \
  --g-proj <g_proj_path> --method segmentation \
  --seg-masks-json <mask_json_path> --localizer reprojection
```

### 5. 重投影關鍵點視覺化

把 replay JSON 裡單一 frame 的逐車輛 `kp_sat`/`sat_coords` 重投影結果畫到衛星影像上，用於檢查重投影結果。

```bash
source /opt/anaconda3/bin/activate trafficlab && \
python scripts/plot_reprojection_keypoints.py \
  <replay_json_path> \
  --frame-index <N>
```

選項：

| Flag | 預設值 | 說明 |
|------|---------|-------|
| `--frame-index` | 自動 | 要繪製的 frame；預設會自動挑選有效關鍵點最多的那一幀 |
| `--ids` | *(全部)* | 要納入的 `tracked_id` 值，以逗號分隔 |
| `--location-code` | 自動推斷 | 覆寫推斷出的 location code |
| `--sat-image` | 自動推斷 | 覆寫衛星影像路徑 |
| `-o` / `--out` | 與輸入檔同層 | 輸出 PNG 路徑 |
| `--dpi` | `200` | 輸出解析度 |

輸出：預設把 PNG 存在輸入 JSON 旁邊（`<stem>.keypoints_frame<N>.png`）。

### 6. CCTV + SAT 合成輸出

把整份 replay JSON 批次渲染成 PNG：預設每幀左 CCTV、右 SAT 疊圖，並排輸出。

```bash
QT_QPA_PLATFORM=offscreen python scripts/export_cctv_sat_composite.py \
  --replay <replay_json_path> \
  --out-dir <out_dir>
```

選項：

| Flag | 預設值 | 說明 |
|------|---------|-------|
| `--replay` | *(必填)* | replay `.json`/`.json.gz` 的路徑 |
| `--out-dir` | *(必填)* | 逐幀 PNG 的輸出目錄 |
| `--cctv-video` | replay 裡的 `mp4_path` | 覆寫影片路徑 |
| `--sat-image` | `location/<code>/sat_<code>.png` | 覆寫 SAT 背景圖路徑 |
| `--kp-conf` | `0.2` | 僅在缺少 `bbox_2d` 時，用來推導 2D box 的關鍵點信心度閾值 |
| `--sat-only` | 關閉 | 完全跳過 CCTV 面板與水平拼接 — 只輸出 SAT 疊圖，也因為不需要而跳過開啟影片檔 |
| `--kp-color-mode` | `track` | `track` 依每輛車的 track 顏色來上色（預設，與 GUI 一致）；`part` 改為依車輛部位上色（wheel/light/plate/mirror/corner/low/up），同部位跨車輛使用相同顏色 |

輸出：`--out-dir` 裡每幀一張 `frame_<NNNN>.png`。要把這些幀轉成影片：

```bash
ffmpeg -framerate 25 -i <out_dir>/frame_%04d.png \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p \
  <out_dir>/composite.mp4
```

（只有在合成出來的寬／高是奇數、`libx264` 不接受時，才需要 `pad` filter。）

### 7. keypoints-carfusion

在每一幀上執行 CarFusion 的 YOLOv8-Pose model（單階段 bbox + 14 個關鍵點），並啟用 ByteTrack 跨幀追蹤，把關鍵點投影到衛星座標定位每輛車。輸出是每幀一張的 CCTV+SAT 並排合成圖，外加一份標準的 TrafficLab replay JSON 可載入 GUI。

```bash
source /opt/anaconda3/bin/activate trafficlab && \
python scripts/run_keypoints_carfusion.py \
  --video <video_path> \
  --weights models/carfusion_last.pt \
  --g-proj <g_proj_path> \
  --sat <sat_image_path> \
  --out <out_dir>
```

輸出：合成 JPG + `scatter.png` + `detections.json`，都在 `--out` 底下。

選項：

| Flag | 預設值 | 說明 |
|------|---------|-------|
| `--weights` | `models/carfusion_last.pt` | YOLOv8-Pose 權重 |
| `--start-frame` | `0` | 開始處理的第一幀 |
| `--frames` | `-1`（全部） | 限制幀數以便快速測試 |
| `--conf` | `0.25` | YOLO 偵測閾值 — 只決定 YOLO 交給 tracker 的內容，不決定最後留下什麼（見下方） |
| `--kp-conf` | `0.2` | 每個關鍵點的信心度閾值 |
| `--tracker` | `trafficlab/inference/bytetrack.yaml` | 固定版本的 ByteTrack config；不是原生 ultralytics 套件的預設值 |

追蹤相關注意事項：
- `trafficlab/inference/bytetrack.yaml` 目前把 `track_high_thresh` / `track_low_thresh` / `new_track_thresh` 調低到 `0.01`，`fuse_score: False`，是針對低信心度／部分遮蔽車輛調整過的。在假設套用 ultralytics 預設值之前，先檢查這個檔案目前的實際數值。

在標準欄位之外，每個物件額外新增的欄位：`bbox_2d`、`bbox_cctv`、`kp_cctv`（原始 `[x, y, conf] × 14`）、`n_keypoints`、`status`（`ok` / `ambiguous_heading` / `failed_insufficient_kp`）、`have_heading`、`have_measurements`、`sat_floor_box`、`bbox_3d`。頂層欄位：`mp4_path`、`meta`、`location_code`、`mp4_frame_count`、`animation_frame_count`。

要在不重新跑偵測／追蹤的情況下，把這些欄位回填到這個 schema 出現之前產生的 `detections.json`：

```bash
source /opt/anaconda3/bin/activate trafficlab && \
python scripts/archive/patch_carfusion_replay_fields.py \
  --json <detections_json_path> \
  --video <video_path> \
  --g-proj <g_proj_path>
```

### 8. 軌跡繪圖

涵蓋 `plot`（整條軌跡所有點畫一張圖）與 `frames`（對選定 id 用同一個固定裁切窗口逐幀輸出 PNG，適合接 ffmpeg 轉影片、畫面不跳動）：

```bash
source /opt/anaconda3/bin/activate trafficlab && python scripts/trajectory_tools.py <plot|frames> <replay_json_path> [flags]
```

| Flag | 適用 | 說明 |
|---|---|---|
| `--location-code` / `--sat-image` | 兩者皆可 | replay 缺 `location_code` metadata 時擇一必填；否則預設找 `location/<location_code>/sat_<location_code>.png` |
| `--ids` | 兩者皆可 | 篩選 `tracked_id`，逗號分隔 |
| `--zoom-margin`（`plot` 需搭配 `--zoom-to-fit`） | 兩者皆可 | 預設 `200`，衛星影像像素；裁切窗口是對軌跡 bounding box 加 padding 算出來的，會視需要放大到跟原圖同比例避免 `imshow` 變形 |
| `--show-heading-arrows` / `--show-keypoints`（`frames` 預設顯示，用 `--hide-heading-arrows`/`--hide-keypoints` 關閉） | 兩者皆可 | `--show-keypoints` 依部位（wheel/light/plate/mirror/corner/low/up）上色，跟「CCTV + SAT 合成輸出」`--kp-color-mode part` 同一套配色 |
| `--show-id-labels` | 僅 `plot` | 可見軌跡旁渲染同色 `tracked_id` 標籤 |
| `--min-points` | 僅 `plot` | 預設跳過少於 5 點的 track |
| `--include-out-of-bounds` | 僅 `plot` | 預設跳過完全在衛星影像外的 track |
| `--out-dir` | 僅 `frames` | 逐幀 PNG 輸出目錄（必填） |

`plot` 跟 `frames` 差別一句話：`plot` 把整條軌跡所有點畫在同一張圖；`frames` 用同一個固定裁切窗口，對選定 id 出現過的每一幀各輸出一張 PNG。

顯示完整說明：

```bash
source /opt/anaconda3/bin/activate trafficlab && python scripts/trajectory_tools.py --help
```

### 9. 軌跡平滑化

只涵蓋 `smooth`（與其合併捷徑 `smooth-and-plot`）：

```bash
source /opt/anaconda3/bin/activate trafficlab && python scripts/trajectory_tools.py smooth <replay_json_path>
```

| Flag | 說明 |
|---|---|
| `-o`/`--output` | 覆寫輸出路徑；預設 `<stem>.smoothed.json[.gz]`，存在輸入檔旁邊 |
| `--window-length` | Savitzky-Golay 窗口長度；短於這個長度的 track 不會被改動 |

平滑化用 Savitzky-Golay 濾波、依 `tracked_id` 分組。

要平滑化＋繪圖一次做完：

```bash
source /opt/anaconda3/bin/activate trafficlab && python scripts/trajectory_tools.py smooth-and-plot <replay_json_path> [smooth flags] [plot flags]
```

flag 是 `smooth` 與上一節「軌跡繪圖」`plot` 的合集，繪圖相關 flag 見上一節。

### 10. 對 script 做語法檢查

```bash
source /opt/anaconda3/bin/activate trafficlab && python -m py_compile scripts/run_inference.py
```

## 推論相關注意事項

- GUI 的 inference 分頁與 `scripts/run_inference.py` 都使用 `trafficlab.inference.pipeline.InferencePipeline`。
- Location 的輸入應該放在 `location/<location_code>/footage/*.mp4` 底下。
- G-projection 檔案應該放在以下其中一個位置：
  - `location/<location_code>/G_projection_<location_code>.json`
  - `location/<location_code>/G_projection_svg_<location_code>.json`
- 輸出會寫到：

```text
output/model-<model_name>_tracker-<tracker_name>/<config_name>/<location_code>/*.json.gz
```

- 比較 GUI 與 CLI 的推論行為時，要用相同的 config name、mp4、環境與工作目錄。

## Replay JSON 格式預期

大部分推論後工具預期的是標準的 replay 格式：

```json
{
  "frames": [
    {
      "frame_index": 0,
      "objects": [
        {
          "tracked_id": 101,
          "class": "Car",
          "sat_coords": [1234.5, 678.9],
          "sat_center": [1234.5, 678.9]
        }
      ]
    }
  ]
}
```

- `tracked_id` 用來連接跨幀的同一個物件。
- `sat_coords` 是平滑化與繪圖使用的標準衛星座標點。
- 當某個工具刻意移動衛星座標點時，`sat_center` 應該同步更新為 `sat_coords`。
- `class` 對繪圖來說是選填的，但對摘要與過濾很有用。
- `.json.gz` 是推論輸出的標準儲存格式；工具應該要保留 gzip 支援。

## 驗證檢查清單

對程式碼變更，執行範圍最小、但有效的檢查：

- 對改動過的 Python 檔案做語法檢查：

```bash
source /opt/anaconda3/bin/activate trafficlab && python -m py_compile path/to/file.py
```

- 對改動過的 script 檢查 CLI help：

```bash
source /opt/anaconda3/bin/activate trafficlab && python scripts/trajectory_tools.py --help
```

## 產生檔案與 Git 衛生守則

- 預設輸出放進 `output/`；只有純語法檢查／smoke-test 這類用後即丟的產出，才寫到 `/private/tmp`。驗證軌跡工具的改動時，省略 `-o`/`--output`/`--plot-output`，讓輸出留在預設路徑（輸入 JSON 旁邊）就好。
- 回報完成之前，先檢查 `git status --short`，把自己的變更和使用者原本就有的變更區分開來。
- 絕對不要還原使用者無關的變更。

