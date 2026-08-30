# CAD 關鍵點模板流程（Blender → `--cad-template`）

## 背景

`build_car_template(dims)`（`trafficlab/motion/keypoints_openpifpaf.py`）目前是靠車輛整體尺寸（軸距／輪距／車長／車高）用幾何規則生出 24 個關鍵點的模板位置，關鍵點之間的相對形狀是估算出來的，不是實測值。這套流程改成反過來：在 Blender 裡對一個車輛 CAD mesh 手動標出關鍵點，再把標出來的世界座標轉成這個專案的 `(x, h, z)` 模板慣例（`x`=側向、+左；`h`=高度、0=地面；`z`=前後向），直接拿這個「量出來的」模板取代 `build_car_template()` 的幾何估算。

涉及三個檔案，依執行順序：

1. `scripts/blender_extract_cad_keypoints.py` — 在 Blender 裡跑，把手標的 marker 座標倒出成一份中繼 JSON。
2. `scripts/build_cad_keypoint_template.py`（core 邏輯在 `trafficlab/motion/cad_keypoint_template.py`）— 在 `trafficlab` conda 環境跑，讀中繼 JSON、自動判斷 Blender 座標軸對應關係、鏡射成完整 24 點模板，輸出最終模板 JSON + 尺寸健檢報告。
3. `scripts/run_keypoints_openpifpaf.py` 的 `--cad-template` — 讀最終模板 JSON，取代 `build_car_template(dims)` 直接使用。

三步是分開的環境／分開的執行時機：第 1 步只能在 Blender 自帶的 Python（有 `bpy`）跑；第 2、3 步只能在 `trafficlab` conda 環境跑（`bpy` 拿不到、也不需要）。

## 第 1 步：`scripts/blender_extract_cad_keypoints.py`

**只能在 Blender 的 Scripting workspace 裡執行**（Text > Open 開啟這個檔案，然後 Run Script），且要先在同一個 scene 裡標好 marker，不能用 `python`/`trafficlab` 環境跑（`import bpy` 只存在 Blender 自帶的直譯器裡）。

標記方式：在已開啟的 `.blend` 檔案場景中放 13 個具名的 marker 物件（Empty 或 Mesh 皆可，讀的是物件自己的 origin/`matrix_world.translation`，所以 marker 的 origin 一定要精準對到目標頂點）：

- 12 個 Apollo-24 `*_left` 關鍵點（左右成對關鍵點只需標左側那一個，右側靠鏡射生成）：`front_glass_top_left`、`front_light_left`、`front_low_fog_light_left`、`front_door_top_left`、`front_wheel_center_left`、`rear_wheel_center_left`、`rear_corner_left`、`rear_glass_up_left`、`rear_light_left`、`rear_bumper_left`、`rear_plate_left`、`front_door_base_left`
- 第 13 個 `ground_ref`：吸附到前輪最低／接地點的頂點，用來在 Blender 原始座標系裡建立真正的 `h=0` 地面基準

`EXPECTED_NAMES`（腳本內硬編碼列表）必須跟 `trafficlab.motion.keypoints_openpifpaf.KP_NAMES`/`_LR_PAIRS` 的 `*_left` 那一半保持同步——因為 `bpy` 的 Python 拿不到這個專案的 package，這份名單沒辦法直接 import，`KP_NAMES` 改名時要手動同步這裡。

執行時的檢查：
- 缺 marker 或 marker 型別不是 EMPTY/MESH → 收集全部問題後一次性 `RuntimeError`，不會標一個報一個。
- 場景 `unit_settings.scale_length != 1.0` → 印警告（後續數學假設 1 Blender unit = 1 公尺）。
- 偵測到疑似意外重複的 marker（Blender 自動加 `.001` 之類的後綴）→ 印警告，提醒檢查是不是打錯名字多產生了一個物件。

輸出：預設寫到目前 `.blend` 檔案同層的 `blender_keypoints_raw.json`（`.blend` 必須先存檔過，否則會丟例外要求先存檔；也可以在腳本開頭把 `OUTPUT_JSON_PATH` 設成別的路徑）。內容是純 Blender 世界座標（`coordinate_frame: "blender_world_meters (raw, unremapped)"`），不做任何座標軸或正負號的轉換——那是下一步的工作。

## 第 2 步：`scripts/build_cad_keypoint_template.py`

```bash
source /opt/anaconda3/bin/activate trafficlab
python scripts/build_cad_keypoint_template.py --input <blender_keypoints_raw.json> --out <template_out.json>
```

| Flag | 說明 |
|------|------|
| `--input` | 第 1 步輸出的中繼 JSON 路徑 |
| `--out` | 最終模板 JSON 的輸出路徑 |
| `--self-test` | 跑合成往返自我測試（不需要 `--input`/`--out`，也不需要真的 CAD/Blender 資料），驗證軸偵測＋模板重建邏輯本身是對的 |

核心邏輯在 `trafficlab/motion/cad_keypoint_template.py`：

- **`detect_axes(raw_points)`** — 自動判斷 Blender 原始座標系裡哪個軸是高度／前後向／側向，以及需要哪些正負號翻轉才能對上這個專案的 `(x=+左, h=0地面, z=-前/+後)` 慣例。不是硬編碼假設（因為跟 Blender importer 行為有關，不驗證不能假設），而是從資料本身反推：
  - 高度軸：期待 Blender world Z（index 2）勝出（`front_glass_top_left - ground_ref` 在該軸最大），且 `ground_ref` 是全部 13 點裡該軸座標最小者；不是就直接 `ValueError`（多半代表 local-vs-world 座標 bug、有奇怪的 parent transform，或 `ground_ref` 沒放對）。
  - 前後向／側向軸：剩下兩軸裡，`front_wheel_center_left`/`rear_wheel_center_left` 的座標差落在軸距合理範圍（`2.0–3.1` 公尺）的那一軸是前後向，差距 `<0.2` 公尺（前後輪距幾乎不變）的那一軸是側向；兩個條件都要唯一滿足，否則 `ValueError`（多半是這兩個 marker 標反了）。
  - 前後向正負號＋原點：由前後輪座標差的正負決定，原點取前後輪中點（跟 `build_car_template()` 的慣例一致）。
  - 側向正負號：因為全部 12 個手標點依構造都在左側，用「多數決」自行判斷正負號；跟多數不同號但差距 `<0.05` 公尺的點視為靠近中心線的點擊誤差（印警告、仍採多數號），差距更大則直接 `ValueError`（可能是標到車輛另一側去了）。
  - 地面基準：`ground_ref` 在高度軸的座標即為 `h=0` 原點；額外做一個非強制性的健檢——12 個手標點裡最低點理論上應該比 `ground_ref` 高約一個輪胎半徑（`0.15–0.50` 公尺），落在範圍外只印警告，不會擋掉輸出。
- **`build_template(raw_points, axes)`** — 用偵測到的軸對應關係把 12 個手標點轉成 `(x, h, z)`，並依 `KP_NAMES`/`_LR_PAIRS` 鏡射出對應的右側點（`x` 取負、`h`/`z` 不變），組成完整 `(24, 3)` 模板，列順序跟 `KP_NAMES` 一致。
- **`compute_dimensions(template)`** — 從模板算軸距/輪距/車長/車高，供健檢比對；車高用 `max(h)`（不是 `max-min`），因為 `h=0` 已經是校正過的真地面、輪胎關鍵點帶著真實非零的輪轂高度，用 range 會系統性低估車高（大約差一個輪胎半徑）。
- **`check_dimensions(computed, reference, tolerances)`** — 跟已知參考尺寸比對，超出容許誤差百分比標記 `WARN`（仍會寫檔，但要求人工複查）。
- **`self_test()`** — 合成往返測試：用 `build_car_template(_FALLBACK_DIMS)` 產生已知的 ground-truth 模板，反推出一組帶任意軸翻轉/位移的假 Blender 原始資料，餵給 `detect_axes`+`build_template`，斷言能精準復原原始模板（誤差 `<1e-9`）。不需要真的 CAD/Blender 資料，純驗證這套轉換數學本身是對的。

目前 `scripts/build_cad_keypoint_template.py` 裡的參考尺寸與容許誤差（`_NISSAN_JUKE_NISMO_REFERENCE`/`_NISSAN_JUKE_NISMO_TOLERANCES`）是針對 `cad_models/nissan_juke_nismo/nissan_juke_nismo.glb` 這一個車型硬編碼的（來源：該 glTF 的 chassis-only bounding box 解析，不含非車身的「glow」decal mesh）；未來如果要支援更多 CAD 車型，這兩組數字要改成 CLI flag。

輸出 JSON 的關鍵欄位：`kp_names`（`KP_NAMES` 快照，供下游一致性檢查）、`template`（`(24, 3)` 陣列）、`points_by_name`（同一份資料，用具名 dict 方便人工檢查）、`axis_mapping`（記錄本次偵測到的軸對應與正負號，供追溯）、`validation`（尺寸健檢結果）。

**這一步不會自動接進 `scripts/run_keypoints_openpifpaf.py`／`build_car_template()`**（腳本 docstring 明講），要用產出的模板必須手動帶 `--cad-template`（見下）。

## 第 3 步：`scripts/run_keypoints_openpifpaf.py --cad-template`

```bash
python scripts/run_keypoints_openpifpaf.py \
  --video <video_path> --g-proj <g_proj_path> --method <geometric|segmentation> \
  --cad-template cad_models/nissan_juke_nismo/keypoint_template_nissan_juke_nismo.json
```

設定 `--cad-template <template.json>` 時，直接使用該 JSON 裡的 `(24, 3)` 模板取代 `build_car_template(dims)`——`prior_dimensions.json` 會被忽略。載入時會做兩項一致性檢查：

- `template.json` 的 `kp_names` 必須跟目前這份程式碼的 `KP_NAMES` 順序完全一致，否則 `ValueError`（提示用 `scripts/build_cad_keypoint_template.py` 重新產生）——防止 `KP_NAMES` 改過順序後，舊模板被默默套用到錯位的關鍵點上。
- `template` 陣列形狀必須是 `(24, 3)`，否則 `ValueError`。

`car_template_calibration_tool.py`（見 AGENTS.md 第 10 節）也做了同樣的 `KP_NAMES` 一致性檢查，兩處邏輯彼此獨立但目的相同。

## 檔案對照表

| 檔案 | 執行環境 | 角色 |
|------|---------|------|
| `scripts/blender_extract_cad_keypoints.py` | Blender 內建 Python（`bpy`） | 手標 marker → 原始世界座標 JSON |
| `scripts/build_cad_keypoint_template.py` | `trafficlab` conda 環境 | 中繼 JSON → 最終 24 點模板 JSON（CLI 入口） |
| `trafficlab/motion/cad_keypoint_template.py` | `trafficlab` conda 環境 | 上述腳本的核心邏輯（軸偵測、鏡射、尺寸健檢、自我測試），純 numpy |
| `scripts/run_keypoints_openpifpaf.py --cad-template` | `trafficlab` conda 環境 | 消費最終模板 JSON，取代 `build_car_template(dims)` |
