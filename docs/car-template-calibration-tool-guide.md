# 車輛模板 GCP 校正工具使用說明

`scripts/car_template_calibration_tool.py` 是一個獨立的 GUI 工具，把方向 4「車輛模板作為 PnP GCP」跟方向 7「多參考物最小二乘法」接在一起：不用像方向 7 的 [`reference_point_calibration_tool.py`](reference-point-calibration-tool-guide.md) 那樣，每個參考物件都手動拖 2 個頭/腳十字，而是在 CCTV 畫面上點 2 個輪胎接地點（並標記是哪兩顆輪子），工具就會用既有 ground homography 把這兩點轉到 sat 平面，解一個「旋轉＋縮放＋平移」的相似變換，把整台車的俯視 CAD 模板（24 個關鍵點，各點都有已知真實高度）精確套上正確的 sat 位置、朝向與大小。再把套好的模板每個關鍵點的真實地面位置，跟該幀 replay JSON 裡 OpenPifPaf 已經偵測到的原始像素位置（`kp_cctv`）配對，一次就能產生多組「頭點像素＋腳點真實位置＋真實高度」觀測值（實測每次放置約 1–12 組，遠比方向 7 一次 1 組快）。可以跨多幀多台車重複這個流程累積「放置」列表，最後全部丟進**跟方向 7 完全相同**的 `trafficlab.projection.reference_point_calibration.calibrate()` 做聯合最小二乘，解出 `cam_sat_xy`／`z_cam_meters`。

背景與公式見 [docs/height-correction-algorithm-survey.md](height-correction-algorithm-survey.md) 方向 4。**跟方向 4 原始構想的差異**：原始構想是用 `cv2.solvePnP` 對多點 GCP 解完整相機外參；這個工具改用既有已校正好的 ground homography，把「幫模板定位」簡化成一個 2 點剛體＋縮放的相似變換（不需要重新解相機姿態），是簡化版而非逐字實作。

這個工具**完全獨立於現有的 4 階段校正精靈（CalibrationTab）與方向 7 的 `reference_point_calibration_tool.py`**，預設不會動到 `G_projection_<code>.json`，只會另外存一份自己的結果 JSON；確認結果可信後，才需要額外按「套用到 G_projection」覆寫。

## 啟動

不帶任何參數也能執行，預設會載入 `output/wheel_pair/test21/test21-4.json.gz`（若存在）與 `location/test21/G_projection_test21.json`：

```bash
source /opt/anaconda3/bin/activate trafficlab
python scripts/car_template_calibration_tool.py
```

要換成別的 replay JSON／location，用 `--replay-json`／`--g-proj`（或 `--location-code`）覆寫預設值：

```bash
python scripts/car_template_calibration_tool.py \
  --replay-json output/haware/taipei-cm/taipei-cm.json.gz \
  --g-proj location/taipei-cm/G_projection_taipei-cm.json
```

| Flag | 說明 |
|------|------|
| `--replay-json` | 主要輸入：一份已經跑過 `run_keypoints_openpifpaf.py`、帶 `kp_cctv` 欄位的 replay JSON。`location_code`／對應影片路徑會從裡面自動推斷並載入，不用另外指定影片。 |
| `--g-proj` | 提供既有的 `G_projection_<code>.json`（undistort + homography + parallax），若給了會優先於 `--location-code` |
| `--location-code` | `--g-proj` 的替代寫法，會自動找 `location/<code>/G_projection_<code>.json` |

這個工具**一定要有既有的 `G_projection_<code>.json`** 才能運作——沒有它，方向 4/7 的公式都無法求解。如果 `--g-proj`／`--location-code`（含預設值）都解析不到實際存在的檔案，工具會直接印出錯誤訊息並結束，不會開視窗。

## 使用步驟

一共 5 個分頁：1. 標記 → 2. 微調 → 3. 確認 → 4. 結果 → 5. 驗證。前 3 個是「累積放置」的迴圈，第 4 個是「計算並套用」，第 5 個是選用的視覺化驗證，不影響前面的計算結果。

### 1.「標記」tab

1. 「載入 replay JSON」：載入後會自動讀出 `location_code`、對應的 `mp4_path` 並開啟影片。
2. 「CAD 模板 JSON」：預設 `cad_models/nissan_juke_nismo/keypoint_template_nissan_juke_nismo.json`，可換成別的車型模板（`scripts/build_cad_keypoint_template.py` 產出的同格式 JSON）。
3. 「G_projection.json」：提供地面 homography；透過 `--g-proj`/`--location-code` 啟動時已自動載入，也可以隨時用「瀏覽」換一份。
4. 工具會自動交叉檢查：replay JSON 的 `location_code` 是否跟載入的 `G_projection` 一致、模板的關鍵點順序是否跟目前版本的 Apollo-24 `KP_NAMES` 一致——任一不一致都會鎖住後續操作並顯示警告，避免默默產生看起來合理但其實錯位的資料。
5. 用幀數欄位／滑桿選一幀。畫面預設顯示這一幀**所有**帶 `kp_cctv` 物件的偵測框（藍色；下拉選單目前指到的那台車會用黃色高亮），方便先看一眼這一幀有哪些車可選，再決定要標哪一台。
6. 下方「這一幀的車輛」下拉框列出所有候選物件（用物件本身的 `id` 選取，不是 `tracked_id`——沒有跑過追蹤/`tracked_id` 為空的偵測也一樣選得到，下拉框會標「未追蹤」；附上這一幀有幾個高信心度關鍵點，方便挑選畫面清楚的車輛）。選好候選車輛後，按**「確定選擇」**——這時偵測框才會全部消失，換成這台車的 24 個關鍵點（依部位上色，跟 CCTV+SAT 合成輸出 `--kp-color-mode part` 同一套配色），方便先看清楚輪胎在哪裡再標記。換幀或換選單裡的車都要重新按一次「確定選擇」才能繼續下一步。
7. 選好「點 1」「點 2」要標記的輪子名稱（4 選 2，不限同側，任意兩顆都可以，只要標籤跟畫面上實際位置對得上），按「標記兩個輪胎點」：畫面上出現一組可拖曳的青色叉叉，每個叉叉後方會拖出一條貫穿整個畫面的垂直虛線，方便判斷點擊位置有沒有對準輪胎的垂直中心線。
8. 用滑鼠**左鍵拖曳**兩個叉叉到畫面上這兩顆輪子的**實際輪胎接地點**（不是輪轂中心——接地點是輪胎碰到地面的最低點；模板裡輪轂中心跟接地點的水平座標其實相同，只有高度不同，工具會自動處理這個換算，你只需要點準接地點在畫面上的位置）。
9. 按「完成（自動擬合）」：工具會解一個 2 點精確的「旋轉＋縮放＋平移」相似變換（無殘差——兩個模板輪點會精確落在你點的兩個位置上），然後自動切換到「2. 微調」分頁。這個畫面本身不會再疊出模板的預測/偵測比對點（那個比對圖現在只在「3. 確認」tab 審閱已存放置時才畫）。

### 2.「微調」tab

由左到右三塊：CCTV 參考面板、sat 俯視圖、（變窄的）側邊欄。

- **左：CCTV 參考面板**（唯讀，不可拖曳）——目前這幀畫面，疊上選中車輛的關鍵點，以及剛剛鎖定的兩個輪胎標記叉叉。用來跟右邊 sat 俯視圖對照，確認輪胎位置跟原始畫面吻合。
- **中：sat 俯視圖**——疊出剛剛擬合好的模板，**是車輛直接往正下方看的樣子**（白色車身外框＋紅色線標出車頭方向＋4 顆輪子的小圓點，標記用的那 2 顆輪子是青色，其餘是綠色）。另外還疊了兩個**固定不動**的青色十字，標記的是「當初點擊的兩個輪胎點在 sat 平面上的原始位置」——自動擬合完成當下，模板輪子會精確落在這兩個十字上（0 殘差），但微調（旋轉/平移/翻轉）之後模板會離開這兩個固定點，藉此可以看出目前微調偏離原始點擊多遠。
- **右：側邊欄**（比「標記」tab 窄，因為多了 CCTV 面板要分空間，且沒有長檔案路徑欄位）：

1. 「參考點篩選 / 擬合資訊」：
   | 欄位 | 預設值 | 說明 |
   |---|---|---|
   | 關鍵點信心度閾值 | 0.2 | 低於此信心度的 `kp_cctv` 關鍵點不會被當成參考點；OpenPifPaf 沒偵測到的關鍵點會存成 `[0,0,0]` 或 `[0,-4,0]`（PifPaf 自己的慣例，非本專案定義），這兩種也一律排除，不受信心度欄位單獨控制。 |
   | 最低真實高度 (m) | 0.0（不篩選） | 篩掉真實高度低於此值的關鍵點——高度越低，該筆觀測值對解出 `z_cam` 的貢獻越小（幾乎貼地的點幾乎不受視差影響），預設全部納入。 |

   下方會即時顯示「縮放比例」（這次解出的縮放跟這個 location 標定的 `px_per_m` 的比值——1.0 代表這台車跟模板大小吻合；偏離太多會有警告，通常代表輪子標籤選錯、點擊位置不夠準，或這台車跟所選模板車型差異較大）與「這次放置預計產生幾組參考點」（機率性的，取決於 OpenPifPaf 在那一幀那台車上實際偵測到幾個高信心度關鍵點，實測中位數只有 1 個、平均約 3 個、最多見過 12 個——畫面清楚、車輛沒被遮擋的幀通常能拿到比較多組）。

2. 「微調（僅旋轉／平移，不含縮放）」——**只能調旋轉與平移，不能調縮放**（縮放已經在上一步精確解出，代表這台車相對模板的真實大小，微調時保持不變，連同高度一起等比例套用）：

   | 動作 | 按鍵 | 按鈕 | 單步 | 加 Shift |
   |---|---|---|---|---|
   | 平移 | 方向鍵 | ←↑↓→ | 0.05 m | 0.5 m |
   | 旋轉 | Q / E | ↺ / ↻ | 0.5° | 5° |
   | 翻轉 180° | F | 翻轉 180° | — | — |
   | 重新自動擬合 | R | 重新自動擬合 | 回到上一步剛解出的姿態 | — |

   鍵盤微調前要先點一下中間的 sat 圖讓它取得鍵盤焦點。「翻轉 180°」是為了修正最常見的錯誤——把模板 `+x`（車輛左側）跟畫面左側搞混，標成同軸左右輪時方向就會整個轉反，地圖上會很明顯看起來像轉了 180 度，按這顆按鈕直接修正，不用重標。

3. 「確認加入列表」／「取消此次放置」（在微調區塊下面）：確認無誤後按「確認加入列表」——這一次的「放置」（連同它的 2 個標記點、姿態、當時的信心度／高度篩選設定）會被記錄下來，不是只記錄一組參考點，並自動切回「1. 標記」分頁準備下一次（重新回到偵測框模式，需要重新選車、按「確定選擇」）；「取消此次放置」放棄這次擬合，同樣切回「1. 標記」。

換到別的幀或別台車，重複「標記」→「微調」流程，繼續累積更多放置。

### 3.「確認」tab

- 左側列出目前為止加入的**所有放置**——格式「#放置編號 · frame N · id 車輛編號 · X 組參考點 · scale_ratio」，點選可預覽該次放置在原幀上的疊圖（唯讀，不能再拖動）。
- 疊圖上三種點的意思：
  | 顏色 | 意思 |
  |---|---|
  | 🟢 綠色 | 這台模板**全部 24 個關鍵點**的地面接觸點（h=0）投影回 CCTV 的位置——不管有沒有被偵測到都畫 |
  | 🔵 藍色 | 只在該關鍵點**有被偵測到**時畫：依目前模板姿態、用該點真實高度做視差校正後，理論上這個關鍵點應該出現的像素位置 |
  | 🟠 橘紅色 | 同樣只在有偵測到時畫：OpenPifPaf **實際偵測到**的像素位置（`kp_cctv`） |

  藍色跟橘紅色之間的黃色虛線是「殘差」——最小二乘法在解相機參數時想要縮小的誤差，兩點越近代表這組參考點品質越好。
- 「刪除選取的放置」可以整次放置一起刪除（不是刪除單一參考點）。
- 按「計算」：所有放置在計算當下才展開成完整的參考點列表（每個放置最多貢獻它當時篩選後的所有可用關鍵點），至少需要展開後 2 組參考點才能求解；求解邏輯**跟方向 7 完全共用同一支** `calibrate()`。成功會自動跳到「結果」tab；失敗會跳出訊息說明原因。

### 4.「結果」tab

跟方向 7 的結果頁類似的呈現方式：`cam_sat_xy`、`z_cam_meters`、跟既有 `G_projection` 的差異比較、sat 平面相機位置示意圖，以及兩個儲存選項：

- 「另存新檔」：把目前載入的 `G_projection.json` **完整複製一份**，只改動其中的 `parallax` 區塊（`x_cam_coords_sat`／`y_cam_coords_sat`／`z_cam_meters`），其餘欄位（homography、去畸變參數、`sat_path`……）原封不動，另存到 `output/car_template_calibration/<location_code>/G_projection_<location_code>_calibrated_<時間戳記>.json`。存出來的檔案本身就是一份可直接使用的 `G_projection` JSON（不是自訂的 meta/inputs/results 紀錄格式），可以拿去「5. 驗證」tab 當「修正後」檔案比較，或供其他工具直接讀取。
- 「套用到 G_projection」：邏輯跟「另存新檔」共用同一段 parallax 覆寫，差別只在直接**覆寫**目前載入的那份 `G_projection.json`（跳確認對話框，動作無法復原，除非該檔案本身有版本控制）。

顯示的 `RMSE` 已經換算成公尺（除以 `px_per_m`）——`calibrate()` 內部回傳的原始 `rmse_m` 其實是 sat 平面像素單位，這是既有程式的既有問題（不在這個工具的改動範圍內），這裡刻意換算並誠實標示單位，不要在別處看到 `rmse_m` 就直接當公尺用。

### 5.「驗證」tab

視覺化比較「修正前」跟「修正後」兩份 `G_projection` 在同一幀、同一批 `kp_cctv` 偵測下的 parallax correction 疊圖，沿用既有的 `trafficlab.projection.parallax_reprojection`（`compute_frame_records`／`compute_view_extent`／`plot_frame_pre_post_keypoints`），這個工具沒有修改那支模組本身。不影響「確認」／「結果」tab 已經算出的結果，純粹是輔助判斷用的視覺化。

| 欄位 | 預設值 | 說明 |
|---|---|---|
| 修正前 G_projection JSON | 載入 tab 1 的那份 | 每次在「標記」tab（重新）載入 G_projection 都會自動同步過來，也可以自行瀏覽換成別的檔案 |
| 修正後 G_projection JSON | 空白 | 留空＝套用「確認」tab 本次「計算」出的結果（等同「結果」tab 的「另存新檔」內容，但不落地檔案）；也可以直接指定一個既有檔案（例如「結果」tab 存出來的複本），純粹比較兩個已經存在的 `G_projection` 檔案，不需要本次 session 算過東西 |
| 比較幀 | 0 | 要拿來比較的 `frame_index`；「自動挑選（參考點最多）」按鈕會依目前的 tracked_id 篩選條件，找出參考點數量最多的那一幀並自動填入 |
| 篩選 tracked_id | `5` | 只比較這個 `tracked_id` 的車輛；留空＝不篩選，比較這一幀所有車輛 |

按「產生比較圖」會輸出兩張 PNG 到 `output/car_template_calibration/<location_code>/verify_<時間戳記>_before.png`／`_after.png`，並直接顯示在畫面左右兩側。兩張圖共用同一個縮放/範圍（`compute_view_extent`），方便並排比對同一組點在修正前後跑到哪裡去了。

## 設計上的幾個重要細節

- **不會用 `sat_to_cctv` 把算好的 sat 腳點反投影回 CCTV 像素**：`GProjection.cctv_to_sat` 內部用疊代法去畸變，`sat_to_cctv` 用精確閉式解加畸變，兩者不是精確的反函數，來回轉一次在畸變較大的 location 可能誤差到公尺甚至更多，尤其在畫面邊緣／地平線附近。所以 `ReferencePoint` 多了一個 `foot_sat`（sat 平面座標）欄位，這個工具產生的參考點一律走這條路徑，不受這個誤差影響；手動標記的方向 7 參考點仍照舊用 `foot_px`。
- **自動擬合是精確解，不是殘差最小化**：2 個點對應剛好精確決定「旋轉＋縮放＋平移」這 4 個未知數，所以自動擬合完成當下兩個模板輪點一定精確落在你點的兩個位置上。「縮放比例」警告指的是解出來的縮放跟這個 location 標定值差多少，不是兩個錨點沒對齊。
- **body frame 跟 sat 像素座標手性相反，擬合前會先做一次反射**：CAD 模板的 body frame 是 `x=+左、z=+後`，跟 sat 影像座標 `x=+右、y=+下` 手性剛好相反（同樣的問題也記在 `trafficlab/motion/keypoints_openpifpaf.py` 的 `localize_wheel_pair` 裡）。純 proper rotation（不含反射）沒辦法把一個手性映到另一個手性，硬套的話整台模板會被鏡射放上去——標 `wheel_left`，結果卻套到畫面上車輛的右側輪子。`car_template_placement.py` 的 `_body_xz_to_sat_frame()` 在做相似變換之前就先做這次反射，`fit_two_point_pose` 跟 `apply_pose_to_points`（因此包含俯視圖、CCTV 疊圖、`foot_sat` 全部下游計算）都會經過它，不用另外處理。
- **不會跨幀去重或加權**：同一台車在不同幀是不同的視差方向，是真正獨立的約束，全部保留；但同一次放置內的多組參考點共用同一組點擊誤差，彼此相關，目前沒有做任何加權處理——如果某次放置貢獻了 12 組、另一次只貢獻 1 組，最小二乘會不成比例地偏向那次 12 組的放置。「確認」tab 的放置列表會直接告訴你每次放置貢獻幾組，做決定時列入考量。

## 常見問題

**輪胎標籤選對了，疊圖卻整個轉了 180 度**
模板 `+x` = 車輛左側，很容易跟畫面左側搞混，尤其面對鏡頭的車輛左右邊很直覺地會看反。按「翻轉 180°」直接修正，不用重新標記。

**「計算未成功」或「參考點不足」**
放置數量太少、或每次放置的信心度閾值設太高導致展開後參考點不到 2 組。回「標記」tab 用信心度較低的門檻試試，或換一幀畫面更清楚的車輛。

**縮放比例警告一直出現**
先確認 2 個輪子標籤有沒有選對、有沒有點準接地點（不是輪拱、不是輪胎側面反光）。如果標籤與點擊都確認沒問題，可能代表這個路口實際出現的車輛比預設的模板車型（Nissan Juke Nismo）明顯大或小，考慮換一個更接近的模板。

**「5. 驗證」tab 按「產生比較圖」說沒有可比較的資料**
通常是 tracked_id 篩選條件在這一幀沒有對應的車輛，或信心度閾值篩掉了這一幀所有關鍵點。先按「自動挑選（參考點最多）」讓工具重新找一幀，或把篩選 tracked_id 欄位清空改成比較全部車輛。

## 相關檔案

| 檔案 | 說明 |
|------|------|
| `trafficlab/projection/car_template_placement.py` | 核心數學（`CarTemplate`、`TemplatePose`、`fit_two_point_pose`、`apply_pose_to_points`/`apply_pose_to_template`、`scaled_heights_m`、`anchor_fit_report`、`build_reference_points`），純 numpy，不依賴 Qt，也不依賴 `trafficlab.motion`；`if __name__ == "__main__"` 有一個不依賴 replay JSON/GProjection 的 self-test，含 body↔sat 手性的斷言 |
| `trafficlab/projection/reference_point_calibration.py` | 方向 7 的核心數學（`ReferencePoint`、`calibrate`）——這個工具新增了 `foot_sat` 欄位，其餘完全共用不變 |
| `trafficlab/projection/parallax_reprojection.py` | 「5. 驗證」tab 疊圖比較機制的來源，這個工具沒有修改它 |
| `trafficlab/gui/tools/car_template_calibration_tool.py` | GUI 本體 |
| `scripts/car_template_calibration_tool.py` | 啟動腳本 |
| [docs/height-correction-algorithm-survey.md](height-correction-algorithm-survey.md) | 方向 4 的背景、跟原始 PnP 構想的差異、跟其他高度校正方案的比較 |
| [docs/reference-point-calibration-tool-guide.md](reference-point-calibration-tool-guide.md) | 方向 7 原本的手動標記工具，計算邏輯與這個工具共用 |
