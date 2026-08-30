# 多參考物最小二乘校正工具使用說明

`scripts/reference_point_calibration_tool.py` 是一個獨立的 GUI 工具，實作方向 7（多參考物最小二乘法）：蒐集 N 個（N ≥ 2）已知真實高度的參考物件的頭/腳像素座標，套用非線性最小二乘同時解出相機的 sat 平面位置（`cam_sat_xy`）與高度（`z_cam_meters`）。這是現有 `pars_stage.py` 兩物件手動標定法的推廣——差別只在參考物件數量從固定 2 個放寬到任意多個，數學模型完全相同（`factor=(z_cam-h)/z_cam` 的徑向縮放）。背景與公式見 [docs/height-correction-algorithm-survey.md](height-correction-algorithm-survey.md) 方向 7。

**這個工具完全獨立於現有的 4 階段校正精靈（CalibrationTab）**，預設不會動到 `G_projection_<code>.json`，只會另外存一份自己的結果 JSON；如果你確認結果可信，可以額外按「套用到 G_projection」才會覆寫。

跟消失點標定工具（`run_vp_calibration_tool.py`）不同，這個工具一定要有既有的 `G_projection_<code>.json` 才能運作（見下方「啟動」一節）。

## 啟動

不帶任何參數也能執行，預設會載入 `location/test21/footage/test21-4.mp4` 與 `location/test21/G_projection_test21.json`：

```bash
source /opt/anaconda3/bin/activate trafficlab
python scripts/reference_point_calibration_tool.py
```

要換成別的影片／location，用 `--video`／`--g-proj`（或 `--location-code`）覆寫預設值：

```bash
python scripts/reference_point_calibration_tool.py \
  --video location/taipei-cm/footage/taipei-cm.mp4 \
  --g-proj location/taipei-cm/G_projection_taipei-cm.json
```

| Flag | 說明 |
|------|------|
| `--video` | 開啟時直接載入這支影片（也可以在視窗裡用「瀏覽」選），預設 `location/test21/footage/test21-4.mp4` |
| `--g-proj` | 提供既有的 `G_projection_<code>.json`（undistort + homography），若給了會優先於 `--location-code` |
| `--location-code` | `--g-proj` 的替代寫法，會自動找 `location/<code>/G_projection_<code>.json`，預設 `test21` |

這個工具**一定要有既有的 `G_projection_<code>.json`** 才能運作——它直接借用該檔案已經校正好的 `undistort`（K/D）與 `homography`（H）去把頭/腳像素轉成 sat 平面座標，沒有這些既有參數，方向 7 的公式無法求解。如果 `--g-proj`／`--location-code`（含預設值）都解析不到實際存在的檔案，工具會直接印出錯誤訊息並結束，不會開視窗。

## 使用步驟

### 1.「新增參考點」tab

1. 「開啟影片」載入 mp4；「G_projection.json」欄位若透過 `--g-proj`/`--location-code` 啟動時已自動載入，也可以隨時用「瀏覽」換一份。
2. 用幀數欄位／滑桿選一幀，畫面會自動重新載入（不用另外按按鈕；「載入這一幀」按鈕保留給需要手動強制重新讀取的情況，例如換了影片路徑但幀數值沒變）。畫面顯示的是**原始未去畸變**的影片幀（效能考量，不對每一幀整張去畸變）——只有在按下「完成」鎖定座標後，程式才會把頭/腳兩個點的像素座標做去畸變＋homography 轉換。如果讀取這一幀時發生非預期錯誤，會跳出詳細錯誤訊息（含完整 traceback），不會靜默無反應。
3. 按「新增參考點」：畫面上會出現一組可拖曳的細「叉叉」（紅色＝頭、青色＝腳，不是圓形），預設位置在畫面中央附近。紅色頭叉叉下方會拖出一條紅色垂直虛線，跟著頭叉叉一起移動，方便拖腳叉叉時對齊垂直方向。
4. 用滑鼠**左鍵拖曳**兩個叉叉到參考物件實際的頭部/腳部位置（這個 tab 的畫面拖曳平移功能關閉了，改用捲軸平移，才能讓左鍵直接拖動叉叉；滾輪縮放不受影響）。
5. 按「完成」鎖定位置，輸入這個參考物件的真實高度（公尺，預設 1.6m），按「確認加入列表」加入。
6. 同一幀可以重複步驟 3–5 加入多個參考物件；右下角「這一幀已加入的參考點」列表只顯示目前這一幀的項目，選取後按「刪除選取的參考點」可以個別刪除。
7. 換到別的幀繼續加入更多參考點；跨幀累積的全部參考點會在下一個 tab 看到。

> 部分影片用 `CAP_PROP_POS_FRAMES` 直接跳幀會失敗，工具會自動改成從頭循序讀取到目標幀，狀態列會顯示「此影片 seek 不可靠，已改用循序讀取，可能較慢」——這是正常行為，不是壞掉（跟 `run_vp_calibration_tool.py` 的行為一致）。

### 2.「確認」tab

- 左側列出目前為止加入的**所有**參考點（跨所有幀），格式「Frame N · #id · 高度」。
- 點選一筆：右側會載入該筆所在的那一幀，畫出它的頭/腳細叉叉與連接線（唯讀，不能拖動），方便核對有沒有點錯位置。
- 按「計算」：至少需要 2 個參考點才能求解。計算成功會自動跳到「結果」tab；失敗（例如求解不收斂）會跳出訊息說明原因，不會硬顯示一個不可信的數字。

### 3.「結果」tab

上方文字面板顯示（比照現有校正精靈的呈現方式）：

- `cam_sat_xy`：相機在 sat 平面上的位置
- `z_cam_meters`：相機高度
- `RMSE`：所有參考點殘差的均方根，數值越小代表參考點之間互相印證得越一致
- **跟既有 G_projection 的數字比較**：載入的 `G_projection_<code>.json` 裡原本的 `cam_sat_xy`/`z_cam_meters`，以及兩者的差異（cam_sat 位移 px、z_cam 差異 m 與百分比）——方便判斷這次多參考物解算跟現有兩點手動標定差多少

下方是衛星圖：把這個 location 的 `sat_<code>.png` 疊上黃色叉叉（這次計算出的相機位置）與洋紅色叉叉（既有 G_projection 的相機位置），方便直接用眼睛核對兩者位置差多少。如果 `G_projection.json` 的 `inputs.sat_path` 指向的檔案不存在，文字面板會註明「找不到衛星影像」，並跳過繪圖。

兩個儲存選項：

- **「另存新檔（獨立 JSON）」**：存到 `output/reference_point_calibration/<location_code>/<時間戳記>.json`，內容包含所有參考點的原始座標與高度、計算結果——不會動 `G_projection_<code>.json`。
- **「套用到 G_projection」**：跳確認對話框後，直接覆寫載入的 `G_projection_<code>.json` 裡 `parallax.x_cam_coords_sat`/`y_cam_coords_sat`/`z_cam_meters` 三個欄位（其餘欄位不動）。此動作無法復原，請先確認 RMSE／殘差合理再按。

### 4.「驗證」tab

用真實的關鍵點推論結果（而不是手動標的參考點）對照**兩個** G_projection——「原版」與「新版」——校正結果差多少：底層直接重用 `trafficlab/projection/parallax_reprojection.py`（跟 `scripts/check_parallax_correction.py` 的 `--recompute` 開／關是同一套邏輯），畫出每個關鍵點「校正前（h=0 表觀位置）→校正後」的位移線疊在衛星圖上。**灰點（校正前）一律即時計算**；彩色點（校正後）依欄位而定——「新版」用 `kp_cctv` 原始像素座標＋車型關鍵點高度樣板，套用選的 G_projection 即時重算；「原版」讀 replay JSON 裡當初推論存的舊 `kp_sat`，不重算。所以可以拿來測試「換一個新的 G_projection，這些既有的偵測結果校正起來會不會比較準」，不用重新跑一次關鍵點推論。

版面是左圖右側邊欄：所有操作欄位都在右側側邊欄，圖片顯示在左側。輸出的兩張圖固定用同一個縮放倍率與座標範圍，並疊在同一個畫面裡——原版在下層、新版在上層，側邊欄有一個「上層（新版）透明度」滑桿可以即時調整上層圖片的透明度（純調整顯示，不用重新產生圖片），方便用眼睛比較兩者差異。

側邊欄由上到下：

1. 「G_projection」群組，**兩個獨立欄位**：
   - 「新版（即時重算）」：即前一版唯一的那個 G_projection 欄位。
   - 「原版（讀存好的 kp_sat）」：新增的欄位，用來當比較基準——通常放這份 JSON 當初推論時實際用的那個 G_projection。
   工具啟動時如果有帶 `--g-proj`/`--location-code`，這兩欄會**同時**自動帶入同一個路徑並載入（最常見情境是先確認兩邊一致沒有差異，再換掉其中一邊做比較）；之後可以隨時各自用「瀏覽…」換成別的，兩邊互不影響，也跟「新增參考點」tab 的 G_projection 欄位無關。
2. 「Replay JSON」群組：選一份**已經跑過關鍵點推論**的 replay JSON（`.json`/`.json.gz`，要帶 `kp_cctv` 欄位，例如 `scripts/run_keypoints_openpifpaf.py` 的輸出——跟這個工具自己存的參考點 JSON 是不同東西，不能互換）。
3. 「輸出設定」群組：輸入要輸出的**幀數字**（必填，只會畫這一幀，不會像 CLI 版本那樣自動挑「關鍵點最多的那一幀」）；選填 `tracked_id` 篩選（逗號分隔，例如 `101,205`；留空代表這一幀的所有物件都畫）；透明度滑桿（見上方版面說明）。
4. 按「確認並輸出對照圖」：分別用「原版」與「新版」G_projection 算出兩組校正前/後的點，取兩組點聯集算出共用的縮放範圍，畫成兩張座標範圍與像素尺寸完全一致的圖，各自存一份 PNG 在輸入 JSON 旁邊（檔名 `<原檔名>.parallax_frame<N>.baseline.png`／`<原檔名>.parallax_frame<N>.recomputed.png`），並疊在下方畫面裡顯示。

常見錯誤訊息：「找不到這一幀」（這份 JSON 裡沒有這個 frame_index）、「沒有可用的關鍵點」（新版或原版其中一邊在這一幀沒有有效的 `kp_cctv`/`kp_sat` 關鍵點，或篩選的 `tracked_id` 在這一幀不存在——訊息會註明是哪一邊）、「沒有新版/原版 G_projection」（回步驟 1 先選好對應的欄位）。

## 常見問題

**「計算未成功」訊息**
通常是參考點太少（少於 2 個）或幾何退化（所有參考點幾乎共線、或頭腳點幾乎重疊）。加入更多橫向分散、深度分散的參考物件通常能解決。

**RMSE 偏大**
回到「確認」tab 逐筆點選參考點檢查疊圖：常見原因是某幀的頭/腳叉叉沒拖到準確位置，或高度輸入用了不適合該物件的假設值（例如把車輛當成 1.6m 的行人）。

**跟現有 `pars_stage` 兩點手動標定的結果差多少才算合理**
這個工具沒有內建的自我驗證/合成資料測試（跟 `check_vanishing_point_calibration.py` 不同），需要使用者自行拿算出來的 `z_cam_meters`/`cam_sat_xy` 跟該 location 既有 `G_projection` 裡的數值做量級比對，若差異很大，先檢查是不是有選錯 G_projection 檔案、或參考點的頭腳位置/高度輸入有誤。

## 相關檔案

| 檔案 | 說明 |
|------|------|
| `trafficlab/projection/reference_point_calibration.py` | 核心數學（`ReferencePoint`、`calibrate`），純 numpy/scipy，不依賴 Qt |
| `trafficlab/gui/tools/reference_point_calibration_tool.py` | GUI 本體 |
| `scripts/reference_point_calibration_tool.py` | 啟動腳本 |
| [docs/height-correction-algorithm-survey.md](height-correction-algorithm-survey.md) | 方向 7 的背景、公式推導、跟其他高度校正方案的比較 |
