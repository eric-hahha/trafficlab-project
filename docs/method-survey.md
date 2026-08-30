# 車輛朝向估算方法調查紀錄

**目標**：知道車輛 heading（朝向角），用於把 YOLO bbox 底部中心投影點從車輛近側邊緣補偏移至幾何中心。  
**場景**：路側交叉口監視相機，高度約 4m，俯角 30–60°，每幀 15–25 台車。  
**座標慣例**：0° = East，90° = North（sat y 向下，atan2 取 `-dy`）

---

## 狀態總覽

| 方法 | 類型 | 狀態 | 結論 |
|------|------|------|------|
| Bbox 幾何法 | 幾何 | ✅ 已整合 | 作為 fallback，靜止車有 180° 歧義 |
| Pirazh | Keypoint | 🔧 fine-tuning | 原始 weights 效果差；用 SKoPe3D fine-tune 中 |
| YAEN | Yaw 分類 | ✅ 已測試 | 效果差，行車記錄器視角不符 |
| OpenPifPaf Apollo-24 | Keypoint | ✅ 已測試 | Keypoint 位置準確，偵測率低（2–4 台 vs 15–25 台）；YOLO crop 已實作於 WheelLocalizer |
| 輪胎定位法（WheelLocalizer） | 幾何 + Keypoint | ✅ 整合進 Pipeline | OpenPifPaf 輪胎 kp → 地面投影 → 幾何中心；偵測率 4.7%，多數 fallback bbox；精度待目視驗證 |
| YOLOv8 pose CarFusion | Keypoint | ⛔ 已評估後棄用 | 14 個車輛 keypoints，CarFusion 路口視角；曾短暫整合，後移除（下方分析保留作外部 repo 記錄）|
| Vehicle_Orientation_Detect | 參考實作 | 🔍 已分析 | 提供「YOLO crop → OpenPifPaf → homography → BEV 角度」完整流程參考 |
| EgoNet | Monocular 3D | 🔍 已分析 | 根本不適合，side-view keypoint 在俯視角不存在 |
| BEVHeight / BEVHeight++ | BEV Detection | 🔍 已分析 | 架構最適合，待 GPU 環境測試 |
| HeightFormer | BEV Detection | 🔍 已分析 | 架構優於 BEVHeight，但 weights 未公開 |
| SGV3D | BEV Detection | 📋 待 GPU 測試 | BEVHeight 延伸，路側跨場景泛化；CUDA 阻擋同 BEVHeight |
| CoBEV | BEV Detection | 📋 待 GPU 測試 | 深度+高度互補 BEV，80% Vehicle AP；CUDA 阻擋同 BEVHeight |
| R³-Net | 3D Detection | ❌ 排除 | 無公開 code/weights |
| MonoGAE | Monocular 3D | 📋 待 GPU 測試 | 無 pretrained weights，需 GPU 在 Rope3D 訓練；排除理由同 HeightFormer B，不是標注成本 |
| AIC/CityFlow | 資料集 | ❌ 排除 | 無朝向標注 |
| SKoPe3D | 訓練資料 | 🔧 用於 fine-tune | Pirazh fine-tune 的訓練資料，已建 Dataset adapter；官方 Keypoint R-CNN baseline weights 未公開 |
| Rope3D | Benchmark 資料集 | 📋 參考 | BEVHeight/HeightFormer 主要訓練+eval 資料集 |
| DAIR-V2X-I | Benchmark 資料集 | 📋 參考 | BEVHeight/HeightFormer 輔助資料集，直路場景為主 |
| WARM-3D | 弱監督遷移 | 📋 待評估 | 可搭配 BEVHeight 做 domain adaptation |

---

## 已整合至 Pipeline

### Bbox 幾何法

**論文/來源**：無，專案內自行開發  
**相關檔案**：`bbox_heading.py` / `eval_bbox_heading.py`（`aa4ad70` 從 `trafficlab/motion` 解耦歸檔，後續已從 repo 移除；`pipeline.py` 的 bbox-heading fallback 分支也已在 `aa4ad70` 拿掉）

**原理**：
1. 用 YOLO bbox 長寬比判斷車輛姿態（aspect > 1.8 = 側向，< 1.3 = 正背面）
2. 由相機位置與車輛 sat_coords 計算視線角，推出兩個候選朝向
3. 用 SVG 路網方向消歧義（若 `use_svg=true`）

**適合的地方**：
- 零訓練成本，無需 GPU
- 在現有 pipeline 裡已整合為 fallback（`pipeline.py` line ~275）
- 對移動中車輛效果合理

**不適合的地方**：
- 靜止車輛有 180° 歧義（無法分辨車頭/車尾方向）
- 長寬比 1.3–1.8 的車 confidence = 0.1，低於閾值不採用
- 目前 eval 腳本沒有傳入 SVG heading，消歧義尚未完整驗證

**Pipeline 觸發條件**（`pipeline.py` line ~275）：`heading is None`（新建 track 或無 tid）時觸發，confidence >= 0.3 才採用。aspect 1.3–1.8 的車 confidence = 0.1，低於閾值，實際上不會被採用。

**測試結果**：`eval_bbox_heading.py` 在 `location/test21/footage/test21-3sf.mp4` 正常執行，每幀 11–16 支箭頭。結果圖存 `/tmp/bbox_heading_eval/`（未持久化）。eval 腳本目前傳 `road_heading=None`，箭頭方向靠「遠離相機」啟發式，不可靠；pipeline 整合時若 `use_svg=true` 效果更好。SVG 消歧義尚未獨立驗證。

---

### 輪胎定位法（WheelLocalizer）

**論文依據**：Radmehr et al. (2019) ICSM；Roch et al. (2021) ISVC；MDPI 2025 Precise Position Estimation（報告精度提升 66.5%）  
**相關檔案**：`trafficlab/motion/wheel_localization.py`、`scripts/eval_wheel_localization.py`

**原理**：
1. YOLO bbox + 10% padding 裁切 → OpenPifPaf `shufflenetv2k16-apollo-24` 偵測 24 個車輛 keypoints
2. 取 4 個輪胎 kp（FL=7, FR=19, RL=8, RR=18），呼叫 `g_engine.cctv_to_sat(x, y, h=0)` 投影至地面
3. 依可見輪胎數量套用對應幾何案例，計算車輛幾何中心與 heading

**輪胎幾何案例**：

| Case | 需要輪胎 | center 計算 | heading | confidence |
|------|---------|------------|---------|-----------|
| 4wheel | FL+FR+RL+RR | 四點平均 | front_mid→rear_mid | 1.0 |
| front_axle | FL+FR | axle_mid 後移 wheelbase/2 | 軸垂直 + 相機消歧 | 0.75 |
| rear_axle | RL+RR | axle_mid 前移 wheelbase/2 | 軸垂直 + 相機消歧 | 0.75 |
| left_side | FL+RL | side_mid 右移 track_width/2 | FL→RL 方向 | 0.60 |
| right_side | FR+RR | side_mid 左移 track_width/2 | FR→RR 方向 | 0.60 |
| single | 任一 | 輪胎位置 | None | 0.25 |
| none | 無 | fallback bbox | — | 0.0 |

**180° 消歧義**：有 `svg_heading` 則用 `_closest_candidate()`；無則優先遠離相機方向（`g_engine.cam_sat`）。

**車輛尺寸來源**：`prior_dimensions.json` 各類別加入 `track_width` / `wheelbase`（計算依據：`width × 0.85` / `length × 0.67`）。

**Pipeline 整合**（`inference_config.yaml`）：
```yaml
localization_method: bbox   # "bbox" | "wheel"
wheel_localization:
  checkpoint: shufflenetv2k16-apollo-24
  kp_conf: 0.2
  min_confidence: 0.6          # 低於此值（single/none）fallback bbox
  vehicle_classes: [car, van, truck]
```
`bbox` 模式完全不載入 OpenPifPaf；`wheel` 模式對符合類別的每個 bbox 都跑 OpenPifPaf，失敗時 fallback bbox 保持 tracker 連續性。

**Eval 結果**（`test21-3sf.mp4`，40 frames，`yolo11s-visdrone-v2-ft.pt`）：
- YOLO 偵測（car/van/truck）：106 個 bbox
- ≥ 2 輪胎偵測成功：5 個（4.7%）
- Displacement 中位數：144 px（10.2 m）— 數量極少，樣本不可靠
- **根本原因**：ApolloCar3D 訓練於前視街道，路側俯角 domain gap 嚴重

**已知問題**：
- 偵測率 4.7% 遠低於可用門檻，多數偵測仍走 bbox fallback
- crop 可能包含鄰近車輛的輪胎（背景污染），待修正：過濾座標落在 bbox 外的 kp

**比較推論執行記錄**（`test21-2.mp4`，`inference_config_test.yaml`，2026-06-10）：
- bbox 模式（`slight_smoothing_default`）：`output/model-best_tracker-default/slight_smoothing_default/test21/test21-2.json.gz`
- wheel 模式（`slight_smoothing_default_wheel`）：`output/model-best_tracker-default/slight_smoothing_default_wheel/test21/test21-2.json.gz`
- wheel 模式耗時約 8 分鐘（bbox 約 1 分鐘）

---

## 已測試 / Fine-tuning 中

### Pirazh（Vehicle_Key_Point_Orientation_Estimation）

**論文**：Vehiclenet: Large-scale 3D Vehicle Instance Segmentation in Urban Scenes (ICCV 2019 workshop)  
**Repo**：`Vehicle_Key_Point_Orientation_Estimation`  
**相關檔案**：`trafficlab/keypoint/model.py`、`trafficlab/keypoint/inference.py`、`scripts/eval_pirazh.py`、`scripts/finetune_pirazh.py`、`trafficlab/keypoint/skope3d_dataset.py`  
**Checkpoint**：`models/best_fine_kp_checkpoint.pth.tar`（Stage 2 原始）、`models/pirazh_skope3d_finetuned.pth.tar`（fine-tune 後，待產出）

**架構**：
```
KeyPointModel
├── CoarseRegressor  (VGG16-BN backbone，輸出 21-channel heatmap @ 56×56)
└── FineRegressor    (輸入 CoarseRegressor 輸出 + 原圖 crop)
    ├── heatmap 輸出：21 channels（20 keypoints + background）
    └── orientation 輸出：8-class FC（front/rear/left/right + 4 斜角）
```

**20 個 Keypoints 的俯角可見性**：

| 類型 | keypoints | 俯角可見性 |
|------|-----------|-----------|
| 車頂四角 | kp12–15 | **高** |
| 後視鏡 | kp10–11 | **高** |
| 前/後 logo、車牌 | kp8–9, 18–19 | 中 |
| 四輪、車燈 | kp0–7, 16–17 | 低 |

**原理**：輸出 20 個車輛 keypoints，同時做 8 方向分類（正面/背面/左/右/左前/左後/右前/右後）。

**適合的地方**：
- 訓練資料 VeRi-776 是路側監視相機，概念上視角接近
- 輸出 keypoint 可用車軸方向推導連續 heading 角（無需 8 方向分類）
- Stage 2 benchmark（VeRi-776 test set）：keypoint MSE = 1.56 px，方向分類準確率 84.44%
- 已成功 port 至 Python 3 / 現代 PyTorch

**不適合的地方**：
- VeRi-776 以較低仰角側向拍攝為主，與本專案高俯角不符
- 原始 weights 在俯角下 heatmap argmax 全集中在左上角 (0,0)

**Baseline 測試結果**（`test21-3sf.mp4`，274 frames）：
- Roof corners（kp12–15）全部預測在 (0,0)，即 heatmap 在俯角下完全失準
- Orientation label 亂猜
- `heading_from_keypoints` 因兩組 keypoint 均 `allclose(0)` 而 return None，heading arrow 不顯示
- **結論**：原始 VeRi-776 weights 在俯角下不可用，必須 fine-tune

**已修正的 Bug**：
- `trafficlab/keypoint/inference.py:137`：`heading_from_keypoints` 的 wheel unpacking 順序錯誤（rl/fr 對調），導致 heading 偏 90°。俯角下影響低（wheel 不可見會 fallback），但已修正。

**Fine-tune 計劃**：
- **訓練資料**：SKoPe3D（路側俯角合成資料，見下方）
- **策略**：凍結 CoarseRegressor，只更新 FineRegressor（FineRegressor 第一層 `Conv2d(24, 64, 7)` 可學會忽略錯誤的 coarse heatmap，只靠 3ch 圖像特徵）
- **有 GT 的 channels**：9 個（車頂四角 kp12–15、四輪 kp0–3、前 logo kp8），其餘 11 個 mask 掉
- **Loss**：`masked_heatmap_loss（9 channels）+ 0.1 × cross_entropy（orientation）`
- **設定**：batch 32，epoch 20，LR 1e-4（每 5 epoch × 0.5），train: scene 0–23，val: scene 24–27

**Fine-tune 進度**：
- [x] Baseline eval 完成
- [x] `SKoPe3DDataset` 已寫（`trafficlab/keypoint/skope3d_dataset.py`）
- [x] `finetune_pirazh.py` 已寫，含 Colab debug 修正（I/O 瓶頸、cache 路徑、AVI race condition、tqdm 進度條）
- [ ] 完整訓練尚未完成（Colab 環境 debugging 中）
- [ ] before/after eval 比較

---

### SKoPe3D（Pirazh Fine-tune 訓練資料）

**論文**：SKoPe3D: A Synthetic Dataset for Keypoint-based 3D Object Detection (arXiv:2309.01324, 2024)  
**資料位置**：`location/skope3d/scene_{0..27}.zip`（已下載）  
**Dataset adapter**：`trafficlab/keypoint/skope3d_dataset.py`

**資料集規格**：

| 項目 | 內容 |
|------|------|
| 來源 | CARLA 模擬器，路側/俯角監視器視角 |
| 規模 | 28 scenes，24,425 幀，156,394 車輛實例 |
| 解析度 | 1920×1080，30fps |
| Keypoints | 33 個（車頂、輪子、車燈等） |
| 授權 | CC BY-NC-SA |

**與 Pirazh 的 Keypoint 對應（9 個有 GT）**：

| Pirazh channel | Pirazh label | SKoPe3D kp |
|---|---|---|
| 0 | left-front wheel | kp16 |
| 1 | left-back wheel | kp17 |
| 2 | right-front wheel | kp18 |
| 3 | right-back wheel | kp19 |
| 8 | front auto logo | kp32 |
| 12 | right-front roof corner | kp0 |
| 13 | left-front roof corner | kp1 |
| 14 | left-back roof corner | kp3 |
| 15 | right-back roof corner | kp2 |

kp0–3 順序由 scene_0 視覺化確認；kp2/kp3 為交叉順序（非直觀）。

**適合的地方**：
- 視角（路側俯角）與 test21 footage 最接近，解決 VeRi-776 視角不符的根本問題
- 規模大（15 萬實例），足以 fine-tune FineRegressor
- 車頂四角（kp0–3）和輪子（kp16–19）在俯角下可見性高，對應 Pirazh 最重要的 channels

**不適合的地方**：
- 合成資料（CARLA），sim-to-real gap 大小尚未量化
- 部分 keypoints（kp8–11, 28–31）含義未完全確認，已 mask 掉不用

**使用狀態**：Dataset adapter 已完成，fine-tune script 已寫，完整訓練尚未跑完。

---

### YAEN（Yaw-angle-estimation-network）

**Repo**：`Hurri-cane/Yaw-angle-estimation-network`  
**相關檔案**：`scripts/eval_yaen.py`  
**Checkpoints**：`models/car_car_part_model.pt`（零件偵測）、`models/yaw_angle_estimation.pt`（yaw 解碼）

**原理**：兩階段：先偵測車輛零件（前輪、後輪、車頭、車尾），再從零件相對位置估算 yaw 角。

**適合的地方**：
- 設計目標明確（yaw 估算），輸出格式符合需求
- 有 pretrained weights 可直接使用

**不適合的地方**：
- 訓練資料是行車記錄器視角（水平前視），與本專案俯視角差距極大
- 零件偵測在本專案的俯視影像上幾乎全部失敗
- 每幀只偵測 1–2 台車（YOLO11s 偵測 15–25 台），吞吐量不足

**測試結果**：在 test21 footage 上執行，零件偵測失敗率高，yaw 輸出不可用。

---

### OpenPifPaf ApolloCar3D（shufflenetv2k16-apollo-24）

**Repo**：`openpifpaf/openpifpaf` + `openpifpaf_plugin_apollocar3d`  
**訓練資料**：ApolloCar3D（百度 Apollo 街道影像，24個車輛 keypoints）  
**相關檔案**：`scripts/run_keypoints_openpifpaf.py`、`trafficlab/motion/keypoints_openpifpaf.py`  
**安裝**：`pip install openpifpaf`（trafficlab env 已裝，torch 2.2.2 相容）

**架構**：ShuffleNetV2k16 backbone + PifPaf fields，端對端輸出每台車 24 個 keypoints + confidence。

**24 個 Keypoints**：

| 類型 | 包含 |
|------|------|
| 輪胎 | 前左/前右/後左/後右輪（4個）|
| 車燈 | 前左/前右/後左/後右燈（4個）|
| 後照鏡 | 左/右鏡邊（2個）|
| 車頂角 | 前左/前右/後左/後右（4個）|
| 保險桿角 | 前低左/前低右/後低左/後低右（4個）|
| 其他 | 車牌左/右、中央上左/右（4個）|

Heading 計算用**水平配對**（左右對稱點，共 12 組），連線方向 +90° = 前進方向，仍有 180° 歧義。

**適合的地方**：
- Keypoint 位置在近景大車上非常準確（目視確認）
- Weights 自動下載，整合成本低
- 輪子、車燈均有定義，涵蓋本專案需要的零件

**不適合的地方**：
- 訓練資料 ApolloCar3D 是前視街道視角，與路側俯角有 domain gap
- 偵測率極低：`test21-3sf.mp4` 每幀只偵測 2–5 台（YOLO11s 偵測 15–25 台）
- 遠景小車完全漏掉；機車不偵測（car-only model）

**目前腳本的 heading 計算問題**：
- 現在在影像座標系算垂直方向，未考慮透視變形，箭頭方向不合理
- 正確做法：keypoint → homography → sat 座標 → arctan2 → 角度（尚未實作）

**測試結果**（`test21-3sf.mp4`，40 frames 取樣）：
- 偵測到的車輛 keypoint 位置準確（目視）
- 每幀 2–5 台，偵測率約 15–25%
- 箭頭方向不可靠（待修正）
- 部分車輛只偵測到單側（例如側面朝相機），水平配對失敗，無箭頭（預期行為）

**已完成的後續工作**：YOLO crop → OpenPifPaf 已實作於 `WheelLocalizer`（見下方輪胎定位法章節）；h-aware 定位改由 `scripts/run_keypoints_openpifpaf.py` 在 sat 座標系計算 heading。

**能否用 SKoPe3D 訓練 OpenPifPaf？**

技術上可行，但需要額外工作：

| 問題 | 說明 |
|------|------|
| Keypoint schema 不同 | SKoPe3D 有 33 個 keypoints，Apollo-24 只有 24 個；無法直接用，需寫新的 OpenPifPaf plugin 定義 33 個點和骨架連線 |
| 格式相容 | SKoPe3D 用 COCO keypoint 格式，OpenPifPaf 原生支援，格式轉換工作量低 |
| 訓練方式 | 需從頭訓練（或從 Apollo-24 做 transfer，但 schema 不同效果不確定）|

更直接的替代路：SKoPe3D 論文本身的 baseline 是 **Keypoint R-CNN**（torchvision 內建），不需要寫 plugin，`num_keypoints=33` 直接接資料集訓練，實作成本比 OpenPifPaf 自訂 plugin 低。但兩條路目前都沒有現成 weights，都需要自行訓練。

---

### Vehicle_Orientation_Detect（參考實作）

**Repo**：`hanlanqian/Vehicle_Orientation_Detect`  
**訓練資料**：中山大學廣州東校區 CCTV footage（固定路側監視器）  
**架構**：兩階段，Stage 1 Vanishing Point 標定 + Stage 2 OpenPifPaf keypoints → BEV 角度  
**Weights**：YOLOv5 TensorRT engine（repo 內）+ OpenPifPaf apollo-24（自動下載）

**核心流程**（Stage 2，對本專案有用的部分）：
```python
# 1. YOLO 裁切 bbox → OpenPifPaf → 取 keypoints
flag, points = get_keypoints(frame[y1:y2, x1:x2])
# 2. 位移回全圖座標
points[:, 0] += x1;  points[:, 1] += y1
# 3. 套透視矩陣（本專案可直接用 homography 替換）
pts_h = np.c_[points[:, :2], np.ones(len(points))]
pts_bev = pts_h @ perspective.T;  pts_bev /= pts_bev[:, 2:]
# 4. 取水平配對連線角度
diff = pts_bev[i] - pts_bev[j]
orientation = np.degrees(np.arctan2(diff[1], diff[0]))
```

**對本專案的價值**：
- Stage 1（Vanishing Point 標定）**不需要**，本專案已有 homography（`G_projection.json`）
- `self.perspective` 直接對應本專案的 homography 矩陣
- Stage 2 的 keypoint→BEV→角度 邏輯可直接參考移植

**限制**：
- 只在單一校園 CCTV 上驗證，泛化能力未知
- 水平配對仍有前後 180° 歧義，需 SVG 路網消歧義

---

## 詳細分析，尚未測試

### EgoNet

**論文**：Exploring Intermediate Representation for Monocular Vehicle Pose Estimation (CVPR 2021, HKUST)  
**Repo**：`Nicholasli1995/EgoNet`（公開，有 KITTI pretrained weights）

**原理**：
1. YOLO crop → 2D keypoint heatmaps（車身骨架/角點，side-view 設計）
2. MLP Lifter：將 2D keypoints 提升至 3D（隱式學習 KITTI 幾何）
3. SVD 求解 yaw + 3D bbox

**適合的地方**：
- 完整開源，有可用 weights
- 輸出 yaw 角，sin/cos 雙通道 regression 無 180° 歧義

**不適合的地方（根本問題）**：
- 訓練資料：KITTI，水平前視，相機高度 1.65m
- **核心中間表示是 side-view keypoints**：在俯視角下，車側面被壓縮或遮蔽，這些 keypoints 在幾何上不存在
- MLP Lifter 隱式學習 KITTI 的相機-地面幾何，俯視角下系統性錯誤

**Fine-tune 可行性分析（結論：不值得）**：

EgoNet 有兩段需要分別處理，各有獨立障礙：

*障礙 1：Keypoint 定義必須整個換掉*  
Stage 1 的 heatmap 是 side-view 設計。俯角下要改用車頂四角等可見點，但這樣 keypoint 定義整個換掉，Stage 1 等於從頭重訓，不是 fine-tune。

*障礙 2：MLP Lifter 需要 3D 標注*  
Stage 2 Lifter 的訓練目標是 `2D keypoint → 3D 世界座標`，必須有每個 keypoint 的 3D ground truth。  
SKoPe3D 有 3D bbox 8 角點，理論上可推導車頂角的 3D 座標作為 GT，但需要：
1. 重訓 Stage 1（用 SKoPe3D 2D keypoints）
2. 重訓 Stage 2 Lifter（用 SKoPe3D 3D bbox 角點）
3. 確保 SKoPe3D 相機幾何能代表 test21 相機（焦距、俯角不一定相符）

*障礙 3：Lifter 對特定相機幾何過擬合*  
Lifter 隱式學了訓練集相機的 pitch、高度、焦距。即使用 SKoPe3D 重訓，也只能泛化到 SKoPe3D 的相機分佈，不保證在 test21（fx=1280, z_cam=4.218m）上準確。

**與 Pirazh fine-tune 的比較**：

| | EgoNet fine-tune | Pirazh fine-tune（現在做的）|
|---|---|---|
| 需要 3D 標注 | 是（Lifter） | 否（只需 2D keypoints）|
| 需要重設計 keypoints | 是（兩段都要改）| 否（沿用定義）|
| SKoPe3D 可直接用 | 勉強（需推導 3D GT）| 直接可用 |
| 實作複雜度 | 高 | 低 |

**測試結果**：未實際執行，分析後判定 fine-tune 成本遠高於 Pirazh，且輸出品質沒有理由更好。不值得投入。

---

### BEVHeight / BEVHeight++

**論文**：BEVHeight: A Robust Framework for Vision-based Roadside 3D Object Detection (CVPR 2023)；BEVHeight++（arXiv 2309.16179）  
**Repo**：`ADLab-AutoDrive/BEVHeight`（原版）、`yanglei18/BEVHeight_Plus`（++版）  
**Weights**：Tsinghua Cloud 直連可下載（Rope3D R50/R101、DAIR-V2X-I R50/R101）

**原理（LSS 架構）**：
```
圖片 → ResNet-50/101 + FPN → HeightNet（估算每像素物體離地高度）
  → Voxel Pooling（自訂 CUDA op）→ BEV Feature Map
  → CenterPoint Head → 7-DOF bbox [x, y, z, L, W, H, yaw]
```
關鍵：用「物體離地高度」取代傳統 BEVDepth 的「深度估算」，路側俯視幾何下更準確。  
Yaw 用 sin/cos 雙通道 regression，無 180° 歧義。

**適合的地方**：
- 專為路側相機設計，在 Rope3D（交叉口 pole camera）和 DAIR-V2X-I（路側）上訓練
- 有 pretrained weights 可直接下載測試
- 相機高度 H 被顯式用於投影公式，可從 `z_cam = 4.218m` 直接填入
- 從現有 homography + K 可推導所需的 `sensor2ego_mats`，不需要重新標定

**不適合的地方**：
- 需要 NVIDIA GPU（自訂 CUDA op `voxel_pooling_ext`，無 CPU fallback）
- 依賴舊版本：PyTorch 1.9、mmdet3d 0.18.1、mmcv-full 1.4.0（與現有 YOLO11 環境衝突）
- **無單圖 inference script**，官方只有 Lightning eval loop，需要自己寫
- BEVHeight 做全圖偵測，不接受 YOLO bbox 輸入，需要額外做兩套結果配對
- 本專案相機 FOV 約 53°（fx=1280 at 1280px），DAIR-V2X-I 相機較廣角，有 domain gap

**測試結果**：尚未測試。**阻擋原因：本機為 Apple M4，無 NVIDIA GPU，CUDA op 無法編譯。**  
下一步：使用 Google Colab（T4 GPU）或遠端 GPU server 測試。

**測試步驟備忘**：
1. `conda create -n bevheight python=3.8`
2. 安裝 PyTorch 1.9 + mmdet3d 0.18.1（或用 Docker `yanglei2024/op-bevheight:base`）
3. 下載 Rope3D R50 weights：`https://cloud.tsinghua.edu.cn/f/fa3e2d07d62a44b7a337/?dl=1`
4. 把 test21 一幀轉 DAIR 格式（K 需乘以 1.2 縮放到 1536×864）
5. 跑 eval，視覺化 yaw 輸出疊加在 SAT 底圖上
6. 與 bbox 幾何法結果對比

---

### HeightFormer

**論文 A**：HeightFormer: A Semantic Alignment Monocular 3D Object Detection Method from Roadside Perspective（arXiv 2410.07758，HKUST Guangzhou + Li Auto，2024）  
**論文 B**：HeightFormer: Learning Height Prediction in Voxel Features for Roadside Vision Centric 3D Object Detection via Transformer（arXiv 2503.10777，北京理工 + ETH，2025）  
**Weights**：兩篇均尚未公開（2026/06 確認）

**原理（在 BEVHeight 基礎上改進）**：
- 論文 A：加入 DMSC（Deformable Multi-Scale Cross-Attention）解決高度特徵與外觀特徵的空間錯位；加入 VPF（Voxel Pooling Former）以 patch 自注意力取代簡單 voxel pooling
- 論文 B：在 voxel 特徵的垂直列方向做自注意力，直接在 voxel 空間學習高度分佈

**性能（Rope3D，IoU=0.5）**：
| 方法 | Car AP | Big-vehicle AP |
|------|--------|----------------|
| BEVHeight++ | 76.1% | 50.1% |
| HeightFormer A | 78.5% (+2%) | 60.7% (+10%) |
| HeightFormer B | 84.9% (+9%) | 53.6% (+3%) |

**適合的地方**：
- BEVHeight 的所有優點全部保留
- Rope3D benchmark 上 Car AP 和 Big-vehicle AP 均優於 BEVHeight++
- 論文 B 的 GitHub 已建立（`zhangzhang2024/HeightFormer`），有 MIT License

**不適合的地方**：
- Weights 未公開，目前無法直接使用
- 同樣需要 NVIDIA GPU 環境
- 兩篇均為 arXiv preprint，尚未過 peer review

**從頭訓練的可行性（Weights 遲遲未公開時的替代方案）**：

論文 B 有 MIT license 且 code repo 已建立，可以自己用 Rope3D 或 DAIR-V2X-I 訓練。但有兩個獨立的不確定性：

*不確定性 1：訓練成本 vs 收益是否值得*  
BEVHeight++ 有現成 weights，測試成本接近零。HeightFormer B 自訓需要幾天 GPU（論文用 4× RTX-3090）。  
你只需要 yaw 角，不需要精確的 3D bbox IoU。BEVHeight++ 的 76% AP 和 HeightFormer B 的 85% AP，在 yaw 角估算上的實際差距可能遠小於 AP 差距暗示的。**先確認 BEVHeight++ 的 yaw 夠不夠用，再決定要不要投入訓練成本。**

*不確定性 2：Domain gap 對兩者影響一樣*  
HeightFormer B 架構更好，是在 Rope3D/DAIR 的 in-distribution 測試上成立的。test21 相機（FOV 53°、z_cam 4.218m）與 Rope3D 訓練分佈有 domain gap，這個 gap 對 BEVHeight++（pretrained）和 HeightFormer B（自訓於同樣的 Rope3D）的影響是一樣的。  
所以「HeightFormer B 在 Rope3D benchmark 比較好」不等於「HeightFormer B 在 test21 場景比較好」，需要實際測試才能確認。

**建議順序**：
```
BEVHeight++（有 weights，先測）
    → 若 yaw 品質不足 → WARM-3D 弱監督遷移
    → 若還不足，且 weights 仍未公開 → HeightFormer B 自訓（高成本）
```

**測試結果**：未測試。待 weights 公開後，可作為 BEVHeight++ 的升級替換。

---

### SGV3D

**論文**：SGV3D: Towards Scenario Generalization for Vision-based Roadside 3D Object Detection  
**Repo**：`yanglei18/SGV3D`

**輸入**：單張路側相機影像 + K 矩陣 + 相機外參  
**輸出**：7-DoF 3D bbox（x, y, z, L, W, H, **yaw**）

**原理**：在 BEVHeight++ 架構上加入兩個模組：
- **Background-suppressed Module（BSM）**：抑制路側相機常見的背景雜訊（建築立面、天空），提高前景偵測率
- **半監督資料生成**：用現有 weights 在新場景生成 pseudo-label，做 source-free domain adaptation，不需要新場景的 3D 標注

**為何相關**：直接針對 domain gap 問題——BEVHeight 在已見場景（Rope3D training scenes）表現好，SGV3D 讓同一套 weights 在未見過的新路口也維持效果。論文報告在 new scenes 上 +42.57% Vehicle AP over BEVHeight。

**適合的地方**：
- 路側相機特化，與 BEVHeight 同樣的高度先驗優勢
- 跨場景泛化設計對 test21 這類新路口有直接幫助

**阻擋原因**：同 BEVHeight——建在 mmdet3d + 自訂 CUDA op 上，Apple M4 無法執行。

**建議**：先測 BEVHeight++ out-of-box yaw 品質；若 domain gap 明顯（test21 vs Rope3D training），SGV3D 的半監督遷移是比 WARM-3D 更輕量的替代選項。

---

### CoBEV

**論文**：CoBEV: Elevating Roadside 3D Object Detection with Depth and Height Complementarity (IEEE TIP 2024)  
**Repo**：`MasterHow/CoBEV`

**輸入**：單張路側相機影像 + K 矩陣 + 相機外參  
**輸出**：7-DoF 3D bbox（x, y, z, L, W, H, **yaw**）

**原理**：
```
圖片 → ResNet backbone
  ├── DepthNet（估算深度分佈）
  └── HeightNet（估算離地高度分佈）
  → 互補融合（depth 高仰角退化，height 低仰角退化；兩者互補）
  → BEV Feature Map → 偵測頭 → 7-DoF bbox
```

**核心觀察**：深度估算在高俯角（路側）退化；高度估算在低俯角（前視）退化。兩者互補後，單一模型可在更大的俯角範圍穩定運作。本專案俯角 30–60° 正好落在兩者各自有盲區的範圍，互補融合理論上最有幫助。

**性能**：Vehicle AP 80% on DAIR-V2X-I（vs BEVHeight 63%）

**適合的地方**：
- 路側相機特化
- 有 pretrained weights 可下載

**阻擋原因**：同 BEVHeight——需要 NVIDIA GPU，Apple M4 無法執行。

**建議**：與 BEVHeight++、SGV3D 同批在 GPU 環境測試，重點看 yaw 角品質，不需要精確 3D bbox IoU。

---

## 訓練資料 / Benchmark 資料集

### Rope3D

**論文**：Rope3D: The Roadside Perception Dataset for Autonomous Driving and Monocular 3D Object Detection Task (CVPR 2022)

| 項目 | 內容 |
|------|------|
| 資料來源 | 真實拍攝，中國城市路口 |
| 規模 | ~50,000 張，>150 萬個標注物體 |
| 場景數 | 26 個路口 |
| 相機位置 | 路側桿架，高度和仰角刻意多樣化 |
| 解析度 | 1920×1080 |
| 標注格式 | 7-DOF 3D bbox（x, y, z, L, W, H, **yaw**），LiDAR 校準 |
| 標定資訊 | K 矩陣 + 地面平面方程式 + 外參 |
| 場景條件 | 日間/夜間/黃昏，多種天氣 |

**對本專案的適合度**：高。交叉口場景、多仰角相機，與 test21 最接近。  
**已知 domain gap**：Rope3D 路側相機通常較廣角（FOV > 70°），test21 FOV 約 53°，焦距偏長，可能造成距離估算偏差。

---

### DAIR-V2X-I

**論文**：DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object Detection (CVPR 2022)

| 項目 | 內容 |
|------|------|
| 資料來源 | 真實拍攝，中國高速公路 + 城市路段 |
| 規模 | ~10,000 幀（路側部分） |
| 相機位置 | 路側桿架，搭配 LiDAR |
| 解析度 | 1920×1080，25fps |
| 標注格式 | 7-DOF 3D bbox，含車輛/行人/自行車 |
| 特點 | 相機與 LiDAR 同步，標注由 LiDAR 點雲校準 |

**對本專案的適合度**：中。直路/高速場景多，交叉口少，場景與 test21 差距較大。  
**用途**：BEVHeight / HeightFormer 的輔助訓練集和 benchmark。

---

### 三個資料集比較

| | SKoPe3D | Rope3D | DAIR-V2X-I |
|---|---|---|---|
| **資料來源** | 合成（CARLA） | 真實拍攝 | 真實拍攝 |
| **規模（實例數）** | 156,394 | >1,500,000 | 較小 |
| **標注類型** | 2D keypoints + 3D bbox 8角點 | 7-DOF 3D bbox | 7-DOF 3D bbox |
| **LiDAR GT** | 無 | 有 | 有 |
| **場景** | 路側/俯角（模擬）| 真實交叉口 | 直路/高速 |
| **用於** | Pirazh fine-tune | BEVHeight/HeightFormer | BEVHeight/HeightFormer |

三個資料集針對不同方法，無法互換：Rope3D/DAIR 有 7-DOF 3D bbox 但沒有 keypoints；SKoPe3D 有 keypoints 但格式不直接相容 BEVHeight。

---

## 排除（快速調查後）

### R³-Net

**論文**：R³-Net: A Deep Network for Multi-oriented Vehicle Detection in Aerial Images and Videos（IEEE TGRS 2019, arXiv 1808.05560）  
**作者**：Qingpeng Li, Lichao Mou 等（DLR / TU Munich）  
**Benchmark**：DLR 3K Munich、VEDAI（均為空拍資料集）

**架構**：
```
空拍影像 → ResNet-101 → R-RPN（可旋轉 anchor 提案）
  → R-PS Pooling（旋轉位置敏感池化）
  → 輸出：rotatable bbox + orientation（半角座標系，0–180°）
```

**表面上的優點**：
- 輸出含車輛方向角，無需另外估算 heading
- 針對俯視車輛設計，車頂可見性高
- 設計用於同幀多車，與本專案每幀 15–25 台的密度吻合

**排除理由（根本問題，非僅缺 weights）**：

| 問題 | 說明 |
|------|------|
| **視角根本不符** | 針對空拍/無人機 nadir view 設計，本專案是路側相機斜視角 30–60°，這是架構出發點，fine-tune 無法修正 |
| **輸出是 0–180° 半角** | 空拍下前/後方向不可分，架構故意只輸出半角。本專案需要完整 360° heading，此設計根本無法滿足 |
| **無相機高度/depth 模組** | 路側 3D 估計的核心困難是從斜視角估算 z 軸，R³-Net 無此模組，無法輸出世界座標 |
| **無公開 code / weights** | 無 official GitHub repo，無 pretrained weights |
| **已被後續方法超越** | 2019 年發表，BEVHeight（CVPR 2023）專為路側設計，在對應 benchmark 上全面更好 |

**結論**：視角（nadir vs. 斜視角）和輸出格式（0–180° vs. 360°）均根本不符，即使有 weights 也不可用。不考慮。

---

### MonoGAE

**Repo**：`HIYYJX/MonoGAE`（[arXiv 2310.00400](https://arxiv.org/abs/2310.00400)，IROS 2024 / IEEE T-ITS 2024）  
**結論**：無 pretrained weights，需要 GPU 在 Rope3D 或 DAIR-V2X-I 上從頭訓練。這兩個資料集已有 7-DOF 3D bbox 標注，**不需要自行標注任何資料**。實際阻擋原因與 HeightFormer B 相同（無 weights + 需 GPU），應與 BEVHeight++、HeightFormer B 並列在「待 GPU 環境測試」，不應以標注成本排除。  
**備注**：架構上以 cross-attention 把地面平面 prior 嵌入影像特徵，固定攝影機 + 已知地面平面的場景最有優勢——homography 可直接推導地面平面方程式；K matrix 可從 homography + 正交約束估算，不需另外標定。

---

### AIC / CityFlow

**結論**：AIC 沒有朝向 track；CityFlow 沒有朝向標注。作為訓練資料來源不可用。

---

## 待評估

### YOLOv8 Pose CarFusion（Habib0905/Vehicle-Pose-Estimation）

> 註：曾實作並短暫整合進本專案（`keypoints_carfusion.py` + `run_keypoints_carfusion.py`），後決定棄用、程式碼已移除。以下為原始外部 repo 調查記錄，保留供日後參考。

**Repo**：`Habib0905/Vehicle-Pose-Estimation`  
**訓練資料**：CarFusion（CVPR 2018，Pittsburgh 路口攝影機）+ 孟加拉交通資料集  
**架構**：YOLOv8 pose（ultralytics），偵測 + keypoint 一體  
**Weights**：Google Drive 連結在 repo 內（已公開）

**14 個 Keypoints**：
- 4 輪胎（前左/前右/後左/後右）
- 4 車燈（前左/前右/後左/後右）
- 4 車頂角（前左/前右/後左/後右）
- 排氣管、車輛中心點

**適合的地方**：
- CarFusion 是路口攝影機視角，與本專案場景最接近（比 ApolloCar3D 前視更符合）
- 本專案已用 ultralytics，直接載入 YOLOv8 pose weights，整合成本最低
- 一次推論同時輸出 bbox + keypoints，不需要兩段式
- 輪胎 + 車燈 keypoints 可構成水平配對，算 heading

**不確定因素**：
- CarFusion 視角比本專案偏低（Pittsburgh 路口桿架，高度未知）
- Bangladesh 混合資料對泛化的影響未知
- Weights 品質未實際測試

**建議測試步驟**：
1. 下載 weights（Google Drive，repo 內連結）
2. `model = YOLO('path/to/weights.pt')` 跑 `test21-3sf.mp4`
3. 比對每幀偵測台數（vs OpenPifPaf 的 2–5 台）
4. 目視確認輪胎/車燈 keypoints 位置是否合理

**整合難度**：低（ultralytics 已在 trafficlab env）

---

### WARM-3D（弱監督遷移）

**論文**：WARM-3D: A Weakly-Supervised Sim2Real Domain Adaptation Framework for Roadside Monocular 3D Detection（arXiv 2407.20818）  
**特點**：
- 僅需 2D pseudo-labels（不需要 3D 標注）做 domain adaptation
- 在 BEVHeight 基礎上做遷移，可直接使用現有 YOLO11 偵測結果作為 2D 監督訊號
- 論文報告 +12.4% mAP3D 改善（vs 直接用 source domain weights）

**使用時機**：先測試 BEVHeight++ pretrained weights 的 out-of-the-box 效果，若 domain gap 明顯再用 WARM-3D 做遷移。

---

## 下一步選項

### 選項 A：驗證 bbox 幾何法 + 開啟 SVG

> 過時：bbox 幾何朝向估計的程式碼與評估腳本已從 repo 移除，本選項僅存記錄。

1. 確認 `location/test21/G_projection_test21.json` 有 `"use_svg": true` 且 SVG 路網檔案存在
2. 修改 eval 腳本傳入 SVG heading 做消歧義
3. 目視比對有無 SVG 的箭頭差異

### 選項 B：Space Box 幾何修正

`specs/vehicle-keypoint-heading/README.md` Phase 1 尚未實作：
- `apply_space_box_correction()` — 用 heading + vehicle_length 把 sat_coords 從邊緣移到中心
- 整合點：`pipeline.py`，在 `TrackSmoother.update()` 之後、3D lifting 之前

### 選項 C：標注少量真實資料 fine-tune Pirazh

- 在 test21 footage 標注 200 張車輛 keypoints（真實資料，補 SKoPe3D 合成資料的 sim-to-real gap）
- Fine-tune `trafficlab/keypoint/model.py` 的 FineRegressor
- 與 SKoPe3D fine-tune 方向互補，不互斥

---

## 測試環境備忘

| 環境 | 用途 |
|------|------|
| `trafficlab` conda env | 現有 pipeline，YOLO11，PyTorch ≥ 2.0 |
| `bevheight` conda env（待建） | BEVHeight / BEVHeight++，PyTorch 1.9，mmdet3d 0.18.1 |
| Google Colab T4 / 遠端 GPU | BEVHeight 測試執行（本機 Apple M4 無 NVIDIA GPU） |

**本機限制**：Apple M4，無 NVIDIA GPU。所有需要 CUDA op 的模型（BEVHeight、HeightFormer）須在遠端環境執行。

**test21 相機參數**：
- `cam_sat = (788.86, 1235.76)`，高度 `z_cam = 4.218m`（`4.22m`）
- `px_per_m = 14.07`
- `fx = 1280`，FOV ≈ 53°

---

## 相關檔案索引

| 檔案 | 說明 |
|------|------|
| `trafficlab/motion/wheel_localization.py` | 輪胎定位法，`WheelLocalizer`、`WheelLocResult` |
| `scripts/eval_wheel_localization.py` | 輪胎定位法 vs bbox 並排視覺化評估 |
| `trafficlab/motion/kinematics.py` | TrackSmoother，運動向量朝向 |
| `trafficlab/keypoint/model.py` | Pirazh 模型（已 port） |
| `trafficlab/keypoint/inference.py` | Pirazh 推論，`PirazhDetector`、`heading_from_keypoints()` |
| `trafficlab/keypoint/skope3d_dataset.py` | SKoPe3D Dataset adapter |
| `trafficlab/projection/g_projection.py` | 座標投影，`cam_sat` 屬性 |
| `trafficlab/inference/pipeline.py` | 主 pipeline |
| `scripts/eval_pirazh.py` | Pirazh keypoint 視覺化評估 |
| `scripts/eval_yaen.py` | YAEN 朝向評估 |
| `scripts/finetune_pirazh.py` | Pirazh fine-tune（SKoPe3D） |
| `models/best_fine_kp_checkpoint.pth.tar` | Pirazh Stage 2 checkpoint |
| `models/car_car_part_model.pt` | YAEN 零件偵測器 |
| `models/yaw_angle_estimation.pt` | YAEN 朝向解碼器 |
| `prior_dimensions.json` | 車輛類別尺寸 |
| `location/test21/G_projection_test21.json` | test21 相機標定 |
| `specs/vehicle-keypoint-heading/README.md` | 問題定義、keypoint 方法設計 |
