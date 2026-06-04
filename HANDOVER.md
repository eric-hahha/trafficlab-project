# TrafficLab 3D — 工作交接文件

**交接日期：** 2026-06-04
**分支：** `codex-evaluate-monorun`
**撰寫人：** eric (hahha)

---

## 專案背景

TrafficLab 3D 是一個 CCTV → 衛星圖 交通分析工具。輸入是監視器 MP4 影片和 Google Maps 對應位置，輸出是可在 GUI 中播放的 3D replay JSON。

核心流程：**Calibration → Inference → Postprocess → Visualization**

詳細說明請見 `README.md` 和 `AGENTS.md`。

---

## 本次開發內容（依時間順序）

### 1. Kinematics — 航向估算改進 (2026-04-29)

**commit:** `needle1`, `rearrange`, `off`

**異動檔案：**
- `trafficlab/motion/kinematics.py`
- `inference_config.yaml`

**做了什麼：**
- 重寫 heading 更新邏輯（adaptive EMA、cosine weighting、sat coords jitter）
- 加入 `heading_freeze_on_lost` 機制（追蹤丟失時凍結航向）
- 調整 config 參數順序和預設值

---

### 2. Postprocess 模組 (2026-04-29)

**commit:** `postprocess`, `postprocess-10`

**新增檔案：**
- `postprocess.py` — 主程式（約 1100 行）
- `postprocess_config.yaml` — 後處理設定
- `POSTPROCESS_REQUIREMENTS.md` — 功能需求文件

**做了什麼：**
- 提供 inference output (`.json.gz`) 的後處理流程
- 包含軌跡修正、插值、雜訊過濾等功能
- 用 `postprocess_config.yaml` 控制各步驟開關

**如何執行：**
```bash
python postprocess.py --config postprocess_config.yaml --location test1
```

---

### 3. 新增 Location 資料 (2026-05-15)

**commit:** `3 new location`

**新增：**
- `location/test3/` — G_projection + CCTV/SAT 圖
- `location/test4/` — 同上
- `location/test21/` — 同上

每個 location 都需要有 `G_projection_{code}.json`、`cctv_{code}.png`、`sat_{code}.png`。

---

### 4. Output 檢查 & 輸出工具 (2026-05-15)

**commit:** `output-check and output2blender program`

**新增檔案：**
- `scripts/filter_and_enrich_output.py` — 過濾並補充 output JSON，可匯出 Blender 用格式
- `scripts/plot_missing_motion_ranges.py` — 畫出 motion 資料缺失分佈圖
- `scripts/filter_and_enrich_output.md` — 使用說明
- `scripts/filtered_output.json` — 範例輸出（大檔，約 86k 行）
- `scripts/output_missing_motion_ranges.png` — 缺失分析圖

**如何執行：**
```bash
python scripts/filter_and_enrich_output.py --input output/.../file.json.gz --out filtered_output.json
python scripts/plot_missing_motion_ranges.py --input filtered_output.json
```

---

### 5. GUI 視窗功能 & CLI 推論腳本 (2026-05-21)

**commit:** `視窗功能`

**新增/修改檔案：**
- `scripts/run_inference.py` — 不開 GUI 直接跑 inference 的 CLI 腳本
- `scripts/open_visualization.py` — 直接開啟特定 replay 的視窗
- `trafficlab/gui/tabs/tab_visualization.py` — GUI 視覺化頁面大幅改寫
- `trafficlab/gui/visualization_window.py` — 視覺化視窗新功能
- `trafficlab/inference/pipeline.py` — pipeline 修改（未提交的更動仍在 working tree）

**CLI inference 用法：**
```bash
source /Users/eric/opt/anaconda3/bin/activate trafficlab
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_inference.py \
    --config-name mild_smoothing \
    --location test21 \
    --all-pending
```

**CLI 開視覺化視窗：**
```bash
python scripts/open_visualization.py --location test21 --config mild_smoothing
```

---

### 6. Trajectory 模組 — 平滑 & 繪圖 (2026-05-25)

**commit:** `plot-tool`, `inference_config整理`

**新增 package：** `trafficlab/trajectory/`

| 檔案 | 功能 |
|------|------|
| `io.py` | 讀寫 replay JSON，trajectory 資料擷取 |
| `smoothing.py` | Savitzky-Golay、EMA 等平滑方法 |
| `plotting.py` | 靜態軌跡圖（CCTV/SAT 疊圖，速度著色，heading 箭頭） |
| `__init__.py` | 公開 API |
| `README.md` | 模組使用說明 |

**CLI 工具：** `scripts/trajectory_tools.py`
```bash
python scripts/trajectory_tools.py plot --location test21 --config mild_smoothing --track 42
python scripts/trajectory_tools.py smooth --location test21 --config mild_smoothing
```

**inference_config 整理：**
- 移除重複、過時的 config 欄位（大幅縮減約 170 行）
- 新增 `inference_config_best_tmp.yaml`（目前最佳參數備份）

---

### 7. AI Agent 文件 (2026-05-21)

**commit:** `agents文件`

- 新增 `AGENTS.md` — 給 AI coding agent 看的 repo 說明
- 更新 `README.md` — 補充說明

---

## 未提交但已存在的工作（Working Tree / Untracked）

這些檔案**還沒 commit**，但已開發完成或進行中：

### Keypoint 模組 — `trafficlab/keypoint/`

| 檔案 | 功能 |
|------|------|
| `model.py` | Pirazh keypoint detection 模型定義 |
| `inference.py` | 單張圖 keypoint inference（20 個關鍵點 + 方向分類） |
| `veri_mean_std.pth.tar` | 模型統計資料 |

支援 20 個車輛關鍵點（車輪、頭燈、車牌、車頂角等）及 8 方向分類（front/rear/left/right 等組合）。

### Bbox Heading 估算 — `trafficlab/motion/bbox_heading.py`

- 純幾何方法（不需要額外 ML 模型）
- 利用 YOLO bbox 長寬比 + 相機對車輛的視線方向估算 heading
- 可配合 `road_heading` hint 消除歧義

### 評估腳本 — `scripts/`

| 腳本 | 功能 |
|------|------|
| `eval_keypoints.py` | 對影片評估 keypoint detection，輸出疊圖 PNG |
| `eval_bbox_heading.py` | 對影片評估純 bbox 幾何 heading，畫出箭頭 |
| `eval_yaen.py` | （內容待確認） |

**評估腳本用法：**
```bash
# Keypoint 評估
python scripts/eval_keypoints.py \
    --checkpoint models/best_fine_kp_checkpoint.pth.tar \
    --video location/test21/footage/test21-3sf.mp4 \
    --yolo models/yolov8n.pt \
    --out /private/tmp/kp_eval

# BBox heading 評估
python scripts/eval_bbox_heading.py \
    --video location/test21/footage/test21-3sf.mp4 \
    --config location/test21/G_projection_test21.json \
    --out /tmp/bbox_heading_eval
```

### 新模型 — `models/best_fine_kp_checkpoint.pth.tar`
- Keypoint 模型的 fine-tuned checkpoint

### 新 Location — `location/test5/`, `location/testt/`
- 尚未 commit 的兩個新地點

### Spec 文件 — `kiro/specs/vehicle-keypoint-heading/`
- vehicle keypoint heading 的需求規格（設計中）

---

## 未提交的 Config 變更

- `inference_config.yaml` — 有未提交修改
- `inference_config_best_tmp.yaml` — 有未提交修改
- `trafficlab/inference/pipeline.py` — 有未提交修改

---

## 環境設定

```bash
# 啟動環境
source /Users/eric/opt/anaconda3/bin/activate trafficlab

# Apple Silicon 必加
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 若從 repo 根目錄以外執行
export PYTHONPATH=/Users/eric/code/TrafficLab-3D-main
```

---

## 主要目錄結構（本次開發後）

```
trafficlab/
├── inference/      pipeline.py 有修改
├── motion/
│   ├── kinematics.py     heading 估算重寫
│   └── bbox_heading.py   幾何 heading（未 commit）
├── keypoint/             新增，整個 package（未 commit）
└── trajectory/           新增，平滑 + 繪圖 package

scripts/
├── run_inference.py          CLI inference
├── open_visualization.py     CLI 開視覺化視窗
├── trajectory_tools.py       軌跡平滑 + 繪圖 CLI
├── filter_and_enrich_output.py
├── plot_missing_motion_ranges.py
├── eval_keypoints.py         （未 commit）
├── eval_bbox_heading.py      （未 commit）
└── eval_yaen.py              （未 commit）

location/
├── test3/, test4/, test21/   本次新增
├── test5/, testt/            未 commit

postprocess.py               後處理主程式
postprocess_config.yaml      後處理設定
```

---

## 待接手的工作

1. **commit 未提交的檔案** — keypoint 模組、bbox_heading、eval 腳本、新 location、config 變更
2. **整合 keypoint heading 進 inference pipeline** — `kiro/specs/vehicle-keypoint-heading/` 有規格文件
3. **評估 heading 方法** — 比較 `bbox_heading` 和 keypoint-based heading 的精度
4. **pipeline.py 的未提交修改** — 確認是否為完整功能或中途修改
