# TrafficLab AI 指示文件

本檔案是給在這個 repository 裡工作的 AI coding agent 看的操作細節；README 保持面向使用者、簡潔扼要。

## Repository 用途

本專案把車禍影片重建成地圖上的 2D 行車軌跡。以資深貢獻者的標準工作：精準、有效率、重視可維護性與清晰度。

## Repository 結構

```text
TrafficLab-3D/
├── main.py                     # GUI 進入點
├── trafficlab/
│   ├── gui/                    # PySide6 GUI 實作
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
source /Users/eric/opt/anaconda3/bin/activate trafficlab
```

- 優先直接啟動環境，不要用 `conda run -n trafficlab ...`（sandbox 裡可能因暫存檔寫入權限失敗，除非直接啟動行不通才用）。
- 從 repo root 執行時 `import trafficlab` 會正常解析；若指令必須在 repo root 以外執行，設定 `PYTHONPATH=/Users/eric/code/TrafficLab-3D-main`。`scripts/run_inference.py` 是例外，即使在 repo root 也不會自動加進 `sys.path`，一律要明確帶 `PYTHONPATH`（見下方「不透過 GUI 執行推論」）。
- 只有跑 YOLO 模型時，才需要在 Apple Silicon 上加 `PYTORCH_ENABLE_MPS_FALLBACK=1`；`device: mps` 時不要假設 `half: true` 一定安全。
- 不要依賴系統的 `python` binary；缺 dependency 時先確認 `trafficlab` 環境已啟動再下結論。
- 修改後想快速檢查語法，用 `python -m py_compile <files>`。
- 驗證用的輸出檔案優先寫到 `/private/tmp` 或其他可拋棄路徑，除非本來就要留存。

## 常用指令

### 開啟 GUI

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py
```

### 不透過 GUI 執行推論

用選定的 config 處理所有待處理的影片：

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && PYTHONPATH=/Users/eric/code/TrafficLab-3D-main PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_inference.py --config-name slight_smoothing_best --all-pending
```

處理單一 location：

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && PYTHONPATH=/Users/eric/code/TrafficLab-3D-main PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_inference.py --config-name slight_smoothing_best --location test1
```

處理單一 mp4：

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && PYTHONPATH=/Users/eric/code/TrafficLab-3D-main PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_inference.py --config-name slight_smoothing_best --mp4 location/test1/footage/test1_8_100_s.mp4
```

即使輸出已存在也強制重跑：

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && PYTHONPATH=/Users/eric/code/TrafficLab-3D-main PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_inference.py --config-name slight_smoothing_best --location test1 --force
```

### 後處理（Postprocess）

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python postprocess.py --help
```

### H-aware 3D 關鍵點定位

在每一幀上執行 OpenPifPaf Apollo-24 偵測，並透過 height-aware 關鍵點模板比對來定位每輛車。輸出是標準的 TrafficLab replay JSON，可以在 GUI 中載入（衛星座標位置、車頭朝向、車輛外框，以及 CCTV 關鍵點疊圖）。`--method` 是必填參數，用來選擇三種互斥的逐幀策略之一 — 只有被選中策略自己的 flag 才會生效。

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && \
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_keypoints_openpifpaf.py \
  --video location/test21/footage/test21-4.mp4 \
  --g-proj location/test21/G_projection_test21.json \
  --method geometric
```

輸出：`output/haware/<location_code>/<video_stem>.json.gz`

`--method` 選項：

| Method | 功能說明 | 相關 flag |
|--------|---------------|-----------------|
| `geometric` | 透過 bbox IoU，把 PifPaf 偵測結果橋接到 YOLO track ID。只有一個高信心度關鍵點的 PifPaf 偵測，其 bbox 面積為零，靠自己永遠過不了 IoU 閾值 — 第二輪處理會把這些找回來：如果那個孤立的關鍵點剛好落在唯一一個 YOLO box 裡面，就把它併入這一幀裡已經透過 IoU 比對到該 track 的偵測結果（併入後這個碎片就會被丟棄，避免它同時又以自己的物件身分出現）；如果這一幀裡沒有這樣的偵測結果，就直接把這個碎片標上該 track id。 | `--yolo` / `--yolo-conf` / `--yolo-classes` / `--iou-threshold` / `--yolo-boxes-json` / `--yolo-boxes-class` |
| `crop` | 把每個 Pass-1 的 bbox 裁切出來，在裁切圖上重新跑一次 PifPaf，藉此找回更多高信心度的關鍵點。 | `--crop-redetect` / `--crop-padding` |
| `segmentation` | 改用 car-segmenter（YOLO11-seg）的 instance mask，而不是 bbox-IoU 的啟發式方法，來合併屬於同一輛車的 PifPaf 碎片 — 修正的是 `geometric` 的單一關鍵點復原機制處理不了的一般性 PifPaf instance-splitting 情況（同一輛車拆成兩個多關鍵點碎片）。每個 PifPaf annotation 會被指派給包含它最多高信心度關鍵點的那個 mask；共用同一個 mask 的 annotation 會被合併（每個 slot 取信心度較高的關鍵點）；沒有匹配到任何 mask 的碎片則保持不合併。`tracked_id`/`bbox_2d` 來自 car-segmenter 自己的 instance（ByteTrack tracker id、由 mask 推導出的 box）— 這個方法完全不使用 `--yolo`/Method B。詳見 `docs/keypoints-openpifpaf-segmentation-matching.md`。 | `--seg-model` / `--seg-conf` / `--seg-device` |

共用選項：

| Flag | 預設值 | 說明 |
|------|---------|-------|
| `--checkpoint` | `shufflenetv2k16-apollo-24` | PifPaf model |
| `--spec-csv` | *(無)* | automobile-models-and-specs 的 `engines.csv`；找不到就退回 `prior_dimensions.json`，再找不到就用內建預設值 |
| `--body-type` | `Sedan` | 與 `--spec-csv` 搭配使用 |
| `--kp-conf` | `0.2` | 關鍵點信心度閾值 |
| `--frames` | `-1`（全部） | 限制幀數以便快速測試 |
| `--start-frame` | `0` | 開始處理的第一幀；在此之前的幀會被讀取後丟棄，而不是用 seek |
| `--localizer` | `procrustes` | `procrustes`（closed-form 2D Procrustes）或 `reprojection`（相對於 PifPaf 像素位置做非線性最小平方擬合） |
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
| `--seg-model` | `models/yolo11n-seg.pt` | car-segmenter 用的 Ultralytics `*-seg` checkpoint；若本地端沒有，首次使用時會自動下載到該路徑 |
| `--seg-conf` | `0.3` | car-segmenter 偵測信心度閾值 |
| `--seg-device` | *(自動)* | `cuda` / `mps` / `cpu`，或不設定讓 ultralytics 自行選擇 |
| `--seg-masks-json` | *(無)* | 指向 `record_car_masks.py` 輸出檔案的路徑，提供逐幀的汽車 instance（tracker_id/bbox/mask polygon），用它取代即時執行 car-segmenter。設定此參數時，`--seg-model`/`--seg-conf`/`--seg-device` 會被忽略，car-segmenter 也完全不會被載入。 |

在標準 14 個欄位之外，每個物件會新增的欄位：`kp_cctv`（GUI 疊圖用的原始 `[x, y, conf] × 24`）、`n_keypoints`、`status`（`ok` / `ambiguous_heading` / `failed_insufficient_kp`）。

#### 一次錄製 car-segmenter mask 供重複使用

`--method segmentation` 通常每次執行都會即時跑一次 car-segmenter。如果同一支影片需要同時跑 `--localizer procrustes` 和 `--localizer reprojection` 兩個 pass，就會對相同的輸出跑兩次 YOLO11-seg。改成只錄製一次，讓兩個 pass 都指向同一個檔案：

```bash
python scripts/record_car_masks.py --video location/test21/footage/test21-4.mp4
```

輸出：`output/car_masks/<location_code>/seg-mask_<video_stem>.json.gz`（純影像空間的紀錄 — 沒有 G-projection，也沒有 PifPaf）。接著：

```bash
python scripts/run_keypoints_openpifpaf.py --video location/test21/footage/test21-4.mp4 \
  --g-proj location/test21/G_projection_test21.json --method segmentation \
  --seg-masks-json output/car_masks/test21/seg-mask_test21-4.json.gz --localizer procrustes

python scripts/run_keypoints_openpifpaf.py --video location/test21/footage/test21-4.mp4 \
  --g-proj location/test21/G_projection_test21.json --method segmentation \
  --seg-masks-json output/car_masks/test21/seg-mask_test21-4.json.gz --localizer reprojection
```

### 重投影關鍵點視覺化

把 h-aware（或其他 replay 格式）JSON 裡單一 frame 的逐車輛 `kp_sat` 關鍵點重投影結果與 `sat_coords` 位置，畫在衛星影像上。每個關鍵點都會有一條黑色的指引線連到 `tracked_id-keypoint_name` 標籤；車輛位置點會畫得比較大、並帶黑色邊框（關鍵點則用白色邊框），且不加標籤。

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && \
python scripts/plot_reprojection_keypoints.py \
  output/haware/test21/test21-6_yolo_reprojection.json.gz \
  --frame-index 76
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

### CCTV + SAT 合成輸出

批次把整份 h-aware（或其他 replay 格式）JSON 渲染成 PNG，重複利用 GUI 自己的 headless renderer（`CCTRenderer`、`SatRenderer`）— 沒有新增任何繪圖邏輯。預設模式會每幀輸出一張 PNG，左邊是 CCTV 畫面，右邊是 SAT 疊圖（框線／箭頭／標籤／關鍵點），兩者縮放到同樣高度、並排放置。

```bash
QT_QPA_PLATFORM=offscreen python scripts/export_cctv_sat_composite.py \
  --replay output/haware/test21/test21-4.json.gz \
  --out-dir output/haware/test21/composite
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
ffmpeg -framerate 25 -i output/haware/test21/composite/frame_%04d.png \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p \
  output/haware/test21/composite.mp4
```

（只有在合成出來的寬／高是奇數、`libx264` 不接受時，才需要 `pad` filter。）

### CarFusion 車輛姿態與追蹤

在每一幀上執行 CarFusion 的 YOLOv8-Pose model（單階段 bbox + 14 個關鍵點），並啟用 ByteTrack 跨幀追蹤，把關鍵點投影到衛星座標，再透過 Procrustes SVD 擬合車輛中心點／朝向。輸出是每幀一張的 CCTV+SAT 並排合成圖，外加一份標準的 TrafficLab replay JSON 可載入 GUI。

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && \
python scripts/run_keypoints_carfusion.py \
  --video location/test21/footage/test21-4.mp4 \
  --weights models/carfusion_last.pt \
  --g-proj location/test21/G_projection_test21.json \
  --sat location/test21/sat_test21.png \
  --out /private/tmp/carfusion_sat/
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
- 如果 ByteTrack 從未把某個偵測結果確認為一個 track，`.track()` 模式可能會把整筆偵測結果從輸出中直接丟掉（而不是留著、把 `tracked_id` 設成 `None`）。這與 `--conf` 無關 — 背後的 `is_activated` / `fuse_score` 機制詳見 `docs/keypoints-carfusion-tracking.md`。
- `trafficlab/inference/bytetrack.yaml` 目前把 `track_high_thresh` / `track_low_thresh` / `new_track_thresh` 調低到 `0.01`，`fuse_score: False`，是針對低信心度／部分遮蔽車輛調整過的。在假設套用 ultralytics 預設值之前，先檢查這個檔案目前的實際數值。

在標準欄位之外，每個物件額外新增的欄位：`bbox_2d`、`bbox_cctv`、`kp_cctv`（原始 `[x, y, conf] × 14`）、`n_keypoints`、`status`（`ok` / `ambiguous_heading` / `failed_insufficient_kp`）、`have_heading`、`have_measurements`、`sat_floor_box`、`bbox_3d`。頂層欄位：`mp4_path`、`meta`、`location_code`、`mp4_frame_count`、`animation_frame_count`。

要在不重新跑偵測／追蹤的情況下，把這些欄位回填到這個 schema 出現之前產生的 `detections.json`：

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && \
python scripts/archive/patch_carfusion_replay_fields.py \
  --json /path/to/detections.json \
  --video location/test21/footage/test21-4.mp4 \
  --g-proj location/test21/G_projection_test21.json
```

### 軌跡平滑化與繪圖

整合過的軌跡工具放在 `trafficlab/trajectory/`，並透過 `scripts/trajectory_tools.py` 對外提供。

顯示說明：

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python scripts/trajectory_tools.py --help
```

平滑化單一 replay JSON：

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python scripts/trajectory_tools.py smooth output/example.json.gz
```

繪製單一 replay JSON：

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python scripts/trajectory_tools.py plot output/example.smoothed.json.gz --location-code test1
```

平滑化並繪製選定的 track：

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python scripts/trajectory_tools.py smooth-and-plot output/example.json.gz --ids 7,373 --zoom-to-fit
```

如果 replay 檔案不含 `location_code` metadata，要傳入 `--location-code <code>` 或 `--sat-image <path>` 其中之一。

放大到指定車輛、並疊出關鍵點：

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python scripts/trajectory_tools.py plot output/example.json.gz \
  --ids 0 --zoom-to-fit --zoom-margin 100 --show-heading-arrows --show-keypoints
```

`--zoom-margin`（預設 `200`）是裁切窗口在選定軌跡 bounding box 四周留的 padding，單位是衛星影像本身的像素；裁切窗口會視需要放大到跟原始衛星影像同比例，避免 `imshow` 變形。`--show-keypoints` 會把 `kp_sat` 依關鍵點名稱（wheel/light/plate/mirror/corner/low/up）上色印出來，跟「CCTV + SAT 合成輸出」的 `--kp-color-mode part` 是同一套配色。

逐幀輸出、裁切窗口全程固定不動（適合逐幀比較或轉成影片）：

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python scripts/trajectory_tools.py frames output/example.json.gz \
  --ids 0 --zoom-margin 100 --out-dir output/frames_id0
```

`frames` 跟 `plot` 的差別：`plot` 是把整條軌跡的所有點畫在同一張圖上；`frames` 是先從選定 id 的完整軌跡（跨所有幀）算出「一個」固定的裁切窗口，然後對這個 id 出現過的每一幀各輸出一張 PNG（位置點、heading 箭頭、關鍵點＋帶引線的標籤），所有幀共用同一個裁切窗口，不會逐幀跳動，方便用 `ffmpeg` 接成影片。用 `--hide-heading-arrows`/`--hide-keypoints` 可以關掉對應圖層。

### 對 script 做語法檢查

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python -m py_compile scripts/run_inference.py
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

## 軌跡工具注意事項

- `trafficlab/trajectory/io.py` 負責 `.json` / `.json.gz` 的 I/O 與路徑解析。
- `trafficlab/trajectory/smoothing.py` 用 Savitzky-Golay 濾波、依 `tracked_id` 分組來平滑化 `sat_coords`。
- `trafficlab/trajectory/plotting.py` 用 matplotlib 的非互動式 `Agg` backend，在衛星影像上渲染軌跡點。
- `scripts/trajectory_tools.py` 刻意設計成一個很薄的 CLI wrapper。
- 繪圖預設會跳過少於 5 個點的 track。用 `--min-points` 可以覆寫這個行為。
- 繪圖預設會跳過完全落在衛星影像範圍之外的 track。用 `--include-out-of-bounds` 可以覆寫這個行為。
- 用 `--show-id-labels` 可以在可見的軌跡旁邊渲染同色的 `tracked_id` 標籤。
- 用 `--zoom-to-fit` 搭配 `--zoom-margin`（預設 `200`，衛星影像像素）可以放大到選定的軌跡；裁切窗口是直接對軌跡 bounding box 加 padding 算出來的，不是用某種跟原圖尺寸相關的比例公式（`margin_px` 加大就是視野變大，不會出現怎麼加大都幾乎沒差的情況）。
- 用 `--show-keypoints` 可以在軌跡圖上疊出 `kp_sat` 關鍵點，依部位（wheel/light/plate/mirror/corner/low/up）上色。
- `frames` 子指令（對應 `TrajectoryPlotter.compute_zoom_transform()` + `.plot_frame()`）用來畫「同一個固定視角、逐幀」的圖，裁切窗口只算一次、共用給每一幀，不會像單獨對每一幀各自 zoom-to-fit 那樣跳動；關鍵點標籤共用 `plot()` 的 `show_id_labels` 那套「避免文字重疊」排版邏輯（`_label_placement`/`_estimate_label_box`/新增的 `_draw_kp_label`），不是另外重寫一套。
- 短於 `--window-length` 的 track 不會被改動。
- 除非有提供 `--sat-image`，plotter 會去找 `location/<location_code>/sat_<location_code>.png`。
- 不要把舊的獨立 `/Users/eric/code/traffic-trajectory-smooth` 檔案結構重新帶回這個 repository。要把可重複使用的邏輯整合進 `trafficlab/trajectory/`，除非有明確要求，否則範例資料要放在 repo 之外。

## 外部整合原則

要整合另一個本地端專案或 script 時：

- 先檢視來源專案，找出可重複使用的邏輯、進入點、資料檔案、產生的輸出，以及環境設定檔。
- 只複製或搬移可重複使用的原始碼與必要的文件。
- 除非使用者明確要求，否則不要複製 `.git/`、`.DS_Store`、`__pycache__/`、產生出來的 PNG/JSON 輸出、範例資料集，或獨立的環境設定檔。
- 把可重複使用的函式庫程式碼放在清楚的 `trafficlab/<domain>/` package 底下。
- 把可執行的 wrapper 放在 `scripts/` 底下。
- 當某次整合建立了一個新的子系統時，加上一份簡短的 domain README。
- 優先讓程式碼去配合既有的 TrafficLab I/O 格式，而不是另外建立平行格式。
- 保留 working tree 裡使用者既有的變更。如果有不相關的檔案已經被修改過，不要還原或重新格式化它們。

## 驗證檢查清單

對程式碼變更，執行範圍最小、但有效的檢查：

- 對改動過的 Python 檔案做語法檢查：

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python -m py_compile path/to/file.py
```

- 對改動過的 script 檢查 CLI help：

```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab && python scripts/trajectory_tools.py --help
```

- 對軌跡相關的變更，省略 `-o` / `--output` / `--plot-output`，讓輸出預設放在輸入 JSON 旁邊。只有在純語法／smoke test、不會產出有意義成果物的情況下，才把輸出導向 `/private/tmp`。
- 對推論相關的變更，在比較 GUI 與 CLI 行為時，要使用相同的 config name、mp4、環境與工作目錄。
- 對 GUI 相關的變更，如果可能牽涉到推論，啟動時要帶上 MPS fallback。

## 產生檔案與 Git 衛生守則

- 不要 commit 或刻意加入 `__pycache__/`、`.pyc`、`.DS_Store`、暫時性的圖表，或用後即丟的 JSON 輸出。
- `/private/tmp` 只用於純粹的 smoke-test 產出物（語法檢查、用後即丟的輸出）。軌跡繪圖與平滑化的輸出應該使用預設路徑（輸入 JSON 旁邊）。
- 只有在使用者想要保留產生出來的推論輸出時，才把它們放進 `output/`。
- 回報完成之前，先檢查 `git status --short`，把自己的變更和使用者原本就有的變更區分開來。
- 絕對不要還原使用者無關的變更。

