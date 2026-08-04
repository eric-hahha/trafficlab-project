# 車輛定位方案列表

**問題定義**：從 CCTV 影像取得車輛幾何中心的衛星座標 `sat_coords = (x, y)`，供軌跡追蹤與視覺化使用。

**本文範圍**：僅討論「如何定位車輛中心」，不含朝向估算（heading）——那是 `docs/method-survey.md` 的範圍。

---

## 方案總覽

| 方案 | 輸入 | 原理 | 狀態 | 精度 | 覆蓋率 |
|------|------|------|------|------|--------|
| **A. Bbox 底部中心** | YOLO bbox | 底部中心投影至 h=0 | ✅ Pipeline 預設 | 低（近側邊緣偏移）| 高（同 YOLO）|
| **B. 輪胎定位法** | YOLO bbox + OpenPifPaf | 4 輪 kp → 地面投影 → 幾何中心 | ✅ Pipeline 選項 | 中 | 低（4.7% 成功率）|
| **C. H-aware 3D Template** | OpenPifPaf（全圖）| 24 kp + 各自高度先驗 → SVD 擬合 | 🔧 評估中 | 高（理論）| 中（受 PifPaf domain gap）|
| **D. PnP（Perspective-n-Point）** | OpenPifPaf + 3D 樣板 + K | 2D–3D 對應 → cv2.solvePnP → 6-DoF 姿態 | ❌ 不採用 | 低（高俯角退化）| 同 PifPaf |
| **E. BEVHeight / BEVHeight++** | 單張影像（需 K）| BEV 特徵 + height-aware voxel pooling → 7-DoF bbox | 📋 待測試 | 高（路側設計）| 高（端對端偵測）|
| **F. CarFusion YOLOv8 Pose** | 整張影像 | YOLOv8 pose，14 kp，路口視角訓練 | ✅ 已評估、已整合跨幀追蹤 | 中高（同 SVD 擬合法）| 高（one-stage，未見明顯 domain gap）|

---

## A. Bbox 底部中心（現行 Pipeline 預設）

**相關檔案**：`trafficlab/inference/pipeline.py`（`localization_method: bbox`）

**原理**：取 YOLO bbox 底邊中心點 `(x_mid, y_bottom)` 投影至衛星座標（高度 h=0）。

**問題**：底部中心對應的是車輛靠近相機那側邊緣，不是幾何中心。俯角越低（相機越接近水平），偏差越大（可達數公尺）。

**適用場景**：快速測試、不要求精確位置時的 fallback。

---

## B. 輪胎定位法 WheelLocalizer

**相關檔案**：`trafficlab/motion/wheel_localization.py`

**原理**：
1. YOLO bbox crop → OpenPifPaf 偵測 4 個輪胎 keypoint（FL=7, FR=19, RL=8, RR=18）
2. 各輪胎 kp 投影至地面（h=0）得衛星座標
3. 依可見輪胎數量套用幾何案例（4 輪 → 直接平均；2 輪 → 軸距/輪距補偏移）

**優點**：輪胎在地面（h=0），投影無高度誤差；幾何直覺明確。

**問題**：
- 成功率 4.7%（`test21-3sf.mp4`，40 frames），因 ApolloCar3D 訓練於前視街道，路側俯角 domain gap 嚴重
- 逐 YOLO 框 crop → 框歪或漏偵測 → 整台車沒有 kp
- 只用 4 個輪胎，車頂等高可見的 kp 完全棄用

**Config 設定**：
```yaml
localization_method: wheel
wheel_localization:
  checkpoint: shufflenetv2k16-apollo-24
  kp_conf: 0.2
  min_confidence: 0.6
```

---

## C. H-aware 3D Keypoint Template（主要候選）

**相關檔案**：`trafficlab/motion/keypoints_openpifpaf.py`、`scripts/run_keypoints_openpifpaf.py`  
**詳細說明**：`docs/keypoints-openpifpaf-intro.md`、`docs/3d-keypoint-template-localization.md`

**原理**：
- 對整張影像（不 crop）跑 OpenPifPaf，取得所有 24 Apollo-24 keypoint
- 每個偵測到的 kp 以樣板高度 `h_i` 投影到衛星座標
- 所有投影點透過 SVD 固定尺度 Procrustes 擬合，同時解車輛中心 `T` 與朝向 `θ`

**相對於輪胎法的改進**：
- 不 crop、不綁 YOLO track ID → 消除架構層的偵測損耗
- 使用全部 24 個 kp（任意 2 個以上即可定位）
- 車頂（高可見）、車燈、後視鏡均可參與擬合

**3D 樣板來源**（兩種方式，互斥）：

| 來源 | 尺寸精度 | 中間高度 kp | 落地成本 | 現行狀態 |
|------|---------|------------|---------|---------|
| 規格庫實測值（`--spec-csv engines.csv`）| 外框/輪胎實測 | 估算（spec sheet 不含）| 低（下載 CSV 即可）| ⚠️ 尚未下載 |
| CAD 模型轉換（幾何啟發式 / Blender 手標）| 全 kp 同源 | 實測 | 高（語義映射工作量大）| 📋 未實作 |

目前實際使用 `prior_dimensions.json`（湊整估算值）或內建 `_FALLBACK_DIMS`，兩者均非實測。詳見 `docs/keypoints-openpifpaf-intro.md`。

**現存限制**：
- PifPaf domain gap（ApolloCar3D 前視視角訓練）仍使偵測率偏低
- Track ID 需後處理橋接（Method B：bbox IoU 配對 YOLO tracker，見 `docs/haware-id-matching.md`）
- 尚未整合進主 pipeline，目前以 `run_keypoints_openpifpaf.py` 獨立評估

**執行指令**：
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_keypoints_openpifpaf.py \
  --video location/test21/footage/test21-4.mp4 \
  --g-proj location/test21/G_projection_test21.json \
  --yolo models/yolo11s-visdrone-v2-ft.pt --yolo-classes 2
```

---

## D. PnP（Perspective-n-Point）

**詳細說明**：`docs/3d-keypoint-template-localization.md` § 3A

**原理**：與 h-aware 共用同一組 2D keypoint 偵測與 3D 樣板，但求解方式不同。最小化每個 keypoint 的重投影誤差（reprojection error），透過 `cv2.solvePnP()` 同時解旋轉 R 和平移 T（6 DoF）：

```
minimize  Σᵢ || project(template_i, R, T, K) - (u_i, v_i) ||²
```

深度資訊來源：兩個已知 3D 距離的 keypoint 在影像中的像素距離比例（需要 K 矩陣）。

**不採用的原因**：

| 問題 | 說明 |
|------|------|
| **K 矩陣不可靠** | 本專案 K 用 `fx = image_width`（未實測），誤差可達 30–100%；PnP 深度估算 ∝ fx，fx 差多少 depth 就差多少 |
| **高俯角退化** | PnP 靠前後端透視比例差推深度，俯角 75° 時比例差僅 ~5%，估算誤差極大 |
| **h-aware 更適合** | 用高度先驗取代透視深度，完全不依賴 K，高俯角下不退化 |

**學術參考**：

- **Deep MANTA（CVPR 2017）**：此路線的代表實作——250 個 CAD 車型庫 + 36 語義 kp + EPnP，專為前視相機設計。說明 PnP 在「有精確 K + 低俯角」的條件下可行，但這兩個條件本專案都不具備。論文發表時未釋出官方代碼，搜尋不到後續官方開源，即使幾何假設成立也無法直接取用。
- **RTM3D（ECCV 2020）**：最精簡版本——不用語義零件，只偵測 3D bbox 的 9 個幾何角點 heatmap，再 PnP 反推，達 ~14 FPS。說明即使 keypoint 極簡化，PnP 仍需要精確 K；在前視場景效果好，俯角場景同樣退化。

> 若要強行使用 PnP，需先校正 K（棋盤格標定、深度學習單張估算 GeoCalib、或消失點幾何法），詳見 `docs/3d-keypoint-template-localization.md` § 3A。

---

## E. BEVHeight / BEVHeight++  <!-- 原 D -->

**論文**：BEVHeight: A Robust Framework for Vision-based Roadside 3D Object Detection (CVPR 2023)  
**Repo**：`ADLab-AutoDrive/BEVHeight`

**原理**：
- 用「物體離地高度」取代深度估算，直接輸出 7-DoF 3D bbox（含 yaw）
- 路側相機特化設計，在 Rope3D / DAIR-V2X-I benchmark 上表現優異

**優點**：端對端輸出位置 + 朝向，不需要額外估算步驟；有 pretrained weights 可直接下載。

**阻擋原因**：
- 需要 NVIDIA GPU（自訂 CUDA op），本機 Apple M4 無法執行
- 舊版依賴（PyTorch 1.9、mmdet3d 0.18.1），與現有環境衝突
- 未提供單圖 inference script，需要額外工作

**測試環境**：Google Colab T4 或遠端 GPU。  
**建議順序**：先測 BEVHeight++（有 weights）；若 yaw 品質不足，再考慮 WARM-3D domain adaptation。

---

## F. CarFusion YOLOv8 Pose

**Repo**：`Habib0905/Vehicle-Pose-Estimation`  
**訓練資料**：CarFusion（Pittsburgh 路口攝影機，路側視角）
**詳細說明**：`docs/keypoints-carfusion-intro.md`（原理、演算法）、`docs/keypoints-carfusion-tracking.md`（跨幀追蹤整合、除錯過程、測試數字）

**原理**：YOLOv8 pose 一階段輸出 14 個車輛 keypoint（4 輪 + 4 燈 + 4 車頂角 + 排氣管 + 中心），核心配準演算法跟 C. H-aware 完全相同（Procrustes SVD），差別只在偵測前段（one-stage YOLO vs OpenPifPaf）跟 keypoint 定義。

**已確認**：訓練視角比 ApolloCar3D 更接近路側交叉口，實測未見明顯 domain gap；已整合跨幀 `tracked_id`（ultralytics ByteTrack）與 GUI 播放器相容輸出（`bbox_2d`/`sat_floor_box`/`bbox_3d` 等）。

**現存限制**：跟 C. H-aware 一樣依賴 `prior_dimensions.json`/`_FALLBACK_DIMS` 的車輛尺寸估算值，非實測；ByteTrack 門檻需要針對低信心/部分入鏡車輛調整（見 `docs/keypoints-carfusion-tracking.md`）。

---

## 方案選擇建議

```
目前生產用（快速/穩定）
  └─ A. Bbox 底部中心（精度差但覆蓋率高，作為 fallback）

近期改良目標（主線）
  └─ C. H-aware 3D Template
       ├─ 已實作，評估中
       └─ 改善 domain gap → fine-tune PifPaf（見 docs/method-survey.md Pirazh 章節）

中期目標（需 GPU 環境）
  └─ E. BEVHeight++
       ├─ 先在 Colab 跑 out-of-box 看 yaw 品質
       └─ 若不足 → WARM-3D 弱監督遷移

已評估、與 C 並列的候選（one-stage、覆蓋率較高）
  └─ F. CarFusion YOLOv8 Pose
       └─ 已整合跨幀追蹤（ByteTrack）+ GUI 播放器輸出
```

---

## 相關文件

| 文件 | 內容 |
|------|------|
| `docs/keypoints-openpifpaf-intro.md` | H-aware 方法介紹（原理、流程、使用方式）|
| `docs/3d-keypoint-template-localization.md` | H-aware 完整技術文件（概念、PnP vs h-aware、樣板設計）|
| `docs/haware-id-matching.md` | Track ID 橋接實作（Method B，bbox IoU 配對）|
| `docs/method-survey.md` | 車輛**朝向估算**方法調查（Pirazh / BEVHeight / YAEN 等）|
| `docs/keypoints-carfusion-intro.md` | CarFusion 方法介紹、演算法細節（同 SVD 擬合，14 kp）|
| `docs/keypoints-carfusion-tracking.md` | CarFusion 跨幀追蹤（ByteTrack）整合過程、除錯細節、測試數字 |
