# 高度校正（parallax correction）演算法調查

**問題背景**：目前 `GProjection`（`trafficlab/projection/g_projection.py`）用單點相機模型做高度校正——把相機視為 sat 平面上的一個點（`cam_sat_xy`, `z_cam`），任意高度 h 的關鍵點沿著「相機→該點」的徑向直線，以 `factor = (z_cam - h) / z_cam` 做線性縮放校正（`parallax_correct_ground_to_real`）。這組參數目前是用 GUI（`pars_stage.py`）手動標兩個已知身高的參考物件（頭/腳點）、解交會點與距離比例求出的。

本文件記錄討論過的幾個可能更精準的替代/延伸方向，**均為概念調查，尚未在本專案實作或驗證**，供之後評估用。

---

## 狀態總覽

| 方向 | 類型 | 跟現有架構的相容度 | 狀態 |
|------|------|------|------|
| Vanishing-point 單視角標定 | 幾何標定 | 高（輸出可直接餵給現有 `GProjection` 參數） | ⚠️ 已實作，test21 實測受阻（見內文） |
| PnP + Bundle Adjustment | 攝影測量標定 | 中（需重新設計校正 UI） | 📋 概念調查，未實作 |
| Plane+Parallax（原始文獻查證，2026-08-12 更正） | structure-from-motion 文獻 | 低（前提是相機移動，本專案相機固定，不適用） | 📋 已查證文獻公式，確認不適用（見內文；可行構想改記錄於方向 7） |
| 車輛模板作為 PnP GCP（方法 2+3 混合） | 攝影測量標定 + 統計先驗 | 中（可用現有 keypoint 輸出，但誤差來源多、需交叉驗證） | 📋 概念調查，未實作，風險較高 |
| 只信任地面接觸點 | 幾何迴避策略 | 已存在 | ✅ 已實作（`--localizer wheel_pair`） |
| 學習式單目 3D 偵測 | 深度學習 | 低（架構差異大，需訓練資料） | 🔍 已分析，不建議短期投入 |
| 多參考物最小二乘法（承接自方向 3 更正後的構想） | 幾何 + 統計先驗 | 高（沿用既有 parallax 模型，只需一支回歸腳本） | ✅ 已實作（獨立 GUI 工具，手動多幀多物件標定） |

---

## 1. Vanishing-point-based single-view calibration（單視角消失點標定）

**概念來源**：攝影測量學經典技術（Criminisi/Reid/Zisserman 的 Single View Metrology 系列），交通監控領域有大量延伸做法（用車道線、路緣線、行進中車輛軌跡自動求正交消失點）。

**原理**：不靠人工點兩個已知身高物件去解單點相機模型，而是從畫面本身既有的幾何線索（車道線平行邊、斑馬線、路緣）解出完整相機姿態（pitch/roll/焦距關係），精度通常高於「兩個人工標記點」，且不需要額外現場量測。

**場景幾何線索需求（Scene Geometry）**

場景中需要能提取出至少兩組相互正交（垂直）的平行線：

- **縱向平行線**（用於求解消失點 $VP_{\text{long}}$，代表車流方向）：
  - 數量：至少 2 條（越多越精確）。
  - 常見來源：車道線、分隔島邊緣、路緣石（Curb）、車邊線。
- **橫向平行線**（用於求解消失點 $VP_{\text{lat}}$，與縱向正交）：
  - 數量：至少 2 條。
  - 常見來源：停止線（Stop line）、斑馬線邊框/條紋、路口網狀線、橫向行人穿越道。
- **（可選）垂直平行線**（用於求解 $VP_{\text{vert}}$）：如路燈桿、號誌桿、建築物立面。若畫面中缺乏垂直線，演算法通常可以直接帶入「相機 Roll 角趨近於 0（相機未傾斜）」的物理約束。

**物理尺度先驗（Metric Scale Prior）**

消失點幾何可以完全解出相機的姿態角（Pitch, Roll）與焦距（$f$），但純幾何無名數（Unitless）。若要算出具體公尺單位的相機高度 $z_{\text{cam}}$，只需提供 1 個法規標準尺寸作為尺度基準（無需現場量測）：

- 標準車道寬度（例如：台灣市區道路標準車道寬 $3.0\text{ m} \sim 3.5\text{ m}$）。
- 斑馬線標準尺寸（例如：白色標線寬 $40\text{ cm}$、間距 $40\text{ cm}$）。
- 停止線標準寬度（例如：法規標線寬 $30\text{ cm} \sim 40\text{ cm}$）。

**相機內參假設（Camera Assumptions）**

- 光軸中心（Principal Point）：預設為影像中心點 $(W/2, H/2)$。
- 像素為正方形（Square Pixels）：假設 $f_x = f_y$（現今絕大多數數位感光元件均滿足此條件）。

**可參考實作**：[`kocurvik/deep_vp`](https://github.com/kocurvik/deep_vp)

**跟本專案的接點**：輸出可以直接取代 `pars_stage.py` 現在的頭腳點交會法，產生同樣的 `z_cam`/`cam_sat`（或更完整的外參），下游 `GProjection` 的公式不用換。

**已實作**：`trafficlab/projection/vanishing_point_calibration.py`（核心數學，用合成資料驗證過，f/z_cam/cam_sat 在無雜訊時對到機器精度）、`trafficlab/gui/tools/vp_calibration_tool.py` + `scripts/run_vp_calibration_tool.py`（獨立點線工具，不影響現有校正精靈）、`trafficlab/diagnostics/vanishing_point_calibration_check.py` + `scripts/check_vanishing_point_calibration.py`（自我驗證與跟既有校正的數字比對）。使用說明見 [docs/vp-calibration-tool-guide.md](vp-calibration-tool-guide.md)。

**test21-4.mp4 實測遇到的問題**：焦距公式 `f² = -(VP_long-pp)·(VP_lat-pp)` 需要至少兩組互相垂直方向的消失點都夠明確（線在畫面裡要有看得出來的收斂）。實際在這支影片上點線時：
- 縱向（車道方向）線收斂清楚，沒問題。
- **橫向線幾乎平行**——沒有明顯收斂，消失點位置對點擊誤差極度敏感，甚至直接落在無窮遠。
- 換點**垂直線**（電線杆等）想代替橫向湊一組正交方向，結果一樣看不出收斂。

**判斷**：這不是點線技巧問題，是這台相機的視角/焦距組合對這個路口來說，先天可能只有縱向方向有足夠的透視收斂可用——橫向（跨路寬）實際跨距通常只有十幾公尺，相對於相機視角來說透視效果太弱；垂直方向若相機安裝角度使電線杆等特徵在畫面裡呈現的傾角很小，同樣收斂不明顯。這代表消失點法在「至少兩組正交消失點都夠明確」這個前提上，對 test21 這個 location 目前不成立，需要換一個不依賴線收斂的方法（見下方方向 2、3、5 的討論）。**焦距公式目前寫死用縱向+橫向這一組**，沒有做成「自動挑選最穩的兩組方向」——如果之後想繼續嘗試消失點法，這是一個可以做的延伸，但沒有解決「橫向和垂直都收斂不出來」這個根本的場景幾何限制。

---

## 2. 完整外參標定 + Bundle Adjustment（PnP + 非線性聯合最佳化）

**概念來源**：業界攝影測量標準做法。

**原理**：不要分階段各自獨立求解（先 undistort、再 homography H、再 parallax），而是拿多個已知世界座標的地面控制點（GCP），用 `cv2.solvePnP`／DLT 一次解出真正的相機外參（R, T），必要時再對 K/D/H/parallax 全部參數一起做非線性最小二乘聯合優化。

**優點**：各階段誤差不會互相累積、互相掩蓋。
**缺點**：需要重新設計校正 UI（多點 GCP，而非兩個物件），改動量最大。

---

## 3. Plane+Parallax（平面＋視差）— 文獻查證後更正

**更正說明（2026-08-12）**：本節先前寫成 Plane+Parallax 可以「用車輛自身跨幀的運動自動回歸視差係數」、「拿大量真實移動車輛的軌跡自我標定」，這是錯誤記錄。查證原始論文後，Irani & Anandan 的 Plane+Parallax 前提是**相機在移動、場景靜止**，靠跨 frame 的相機平移（epipole）產生的視差訊號回歸場景結構；TrafficLab 的場景相反——**相機固定、車輛在動**，沒有相機平移可用，論文的求解公式無法套用在這個場景上。原先「用車輛軌跡回歸相機參數」這個構想本身沒錯，但它不是 Plane+Parallax，已重新獨立推導、記錄在**方向 7（多參考物最小二乘法）**。

**概念來源**：Irani & Anandan 的 Plane+Parallax 系列工作（structure-from-motion 經典技術）：

- Irani, M. & Anandan, P., *Parallax Geometry of Pairs of Points for 3D Scene Analysis*, ECCV 1996。[PDF](https://www.weizmann.ac.il/math/irani/sites/math.irani/files/publications/parallax-geometry.pdf)
- Irani, M., Anandan, P., Cohen, M., *Direct Recovery of Planar-Parallax from Multiple Frames*, PAMI 2002, Vol. 24, No. 11, pp. 1528–1534（短版見 ICCV Workshop on Vision Algorithms, 1999）。[PDF](https://www.weizmann.ac.il/math/irani/sites/math.irani/files/publications/direct_recovery.pdf)
- 兩幀奠基之作：Kumar, Anandan, Hanna, *Direct Recovery of Shape from Multiple Views: A Parallax Based Approach*, ICPR 1994；Sawhney, *3D Geometry from Planar Parallax*, CVPR 1994。

**論文的實際公式**：核心關係式（PAMI 2002 Eq. 1）：

$$\vec\mu = \vec p_w - \vec p = -\gamma(t_3\vec p_w - \vec t)$$

$\gamma = H/Z$ 是該點相對參考平面的高度 $H$ 除以相對相機的深度 $Z$（論文所謂的「結構參數」），$\vec t=(t_1,t_2,t_3)$ 是該 frame 相對參考 frame 的 epipole（相機平移在射影座標下的投影）。多幀版本用兩階段交替最小二乘求解（$n$ 個像素、$l$ 個 frame，共 $3l+n$ 個未知數、$ln$ 條約束）：

- **Local Phase**（Eq. 7）：固定所有 epipole，對每個像素的 $\gamma$ 在該像素跨所有 frame 的亮度約束上做最小二乘。
- **Global Phase**（Eq. 8）：固定 $\gamma$，對每個 frame 的 epipole $\vec t^j$ 在所有像素的約束上做最小二乘。

**為什麼不能直接套用在本專案**：$\vec t^j$ 是相機在第 $j$ 幀相對參考幀的平移量投影，這個量存在的前提是相機真的動了。TrafficLab 的攝影機是固定式監視器，$\vec t^j \equiv 0$，Eq. 1 直接退化成 $\vec\mu = 0$——固定相機不會因為「車輛移動」而產生 Plane+Parallax 定義下的視差訊號，車輛的位移不是相機的 epipolar 平移，兩者不能互換代入公式。

**跟本專案現有公式唯一站得住腳的關聯**：$\gamma$ 的 bilinear 結構（視差 = 結構參數 × 與相機平移相關的項），在數學形式上跟本專案 `factor = (z_cam - h) / z_cam` 的單點徑向縮放公式同源，這是這兩者最初被聯想在一起的原因。但論文本身的求解流程（Eq. 7/8）建立在相機移動上，無法照搬；「用多輛車統計高度先驗回歸相機參數」這個可行構想，已改用本專案既有的徑向縮放模型重新推導，見**方向 7**。

---

## 4. 車輛模板作為 PnP GCP（方法 2 + 方向 7 的混合自舉法）

**概念來源**：使用者提出的想法，延伸自方法 2（PnP）與方向 7（車輛統計先驗，原誤記為方法 3 的 Plane+Parallax，見方向 3 的更正說明）。

**原理**：GCP 不用現場量測的控制點，改用不同車輛的各部位 keypoint（輪、頂、鏡等）：Z 值用車型模板高度（body-type 統計先驗，邏輯同方向 7），X/Y 則套用既有 homography 把車輛地面接觸點換算成世界座標來錨定。用這組 GCP 解 `cv2.solvePnP` 得到相機姿態/`z_cam`，再拿去餵給既有的 `--localizer reprojection` 做逐車定位。

**跟本專案的接點**：可以直接沿用 `--method geometric`/`--method segmentation` 等流程已經在抓的車輛 keypoint（`kp_cctv`）與 `--spec-csv`/`prior_dimensions.json` 車型高度先驗，組出 GCP 對應點；解出來的參數可直接替換 `GProjection` 現有的 parallax 參數，下游 `--localizer reprojection` 不用改架構。

**風險與誤差來源**

1. **車型模板 vs 實際車輛落差**：即使限定車型（如 Sedan），車高等尺寸仍有 $\pm 0.05\text{m}$ 量級的個體變異。單一車輛的落差不會被平均掉，會直接混進解出來的相機姿態；若落差是系統性的（例如樣本混入非典型車款），則不會隨樣本數增加而消失。
2. **單一車輛的幾何跨距太小**：一台車的 keypoints 全部落在約 4.5m × 1.5m × 1.5m 的小範圍內，對距離較遠的攝影機而言 conditioning 差，2D keypoint 偵測雜訊容易被放大成大幅姿態誤差，尤其是 pitch 與深度方向。
3. **錨定用的 homography 準確度本身是風險，而且是誤差地板**：
   - homography 精度在畫面各處不均勻，離原始標定點越遠、尤其靠近消失線的地方，同樣像素誤差對應到的世界座標誤差會被放大。
   - homography 假設地面完全平坦，路面坡度/路拱會造成额外系統性誤差。
   - 因此車輛 GCP 選在畫面遠處時，其「已知」X/Y 座標其實可能誤差很大，只是看起來像確定值。
4. **自我一致性掩蓋問題**：校正（用車輛模板+homography 解相機參數）與後續定位（`--localizer reprojection` 同樣用車輛模板去反算車輛姿態）共用同一套模板假設。最佳化過程可能把模板誤差或 homography 誤差吸收進相機高度參數裡，讓殘差變小、看起來一致，但不代表絕對精度正確——而且由於誤差來源（模板、homography、相機參數）三者糾纏在同一組殘差裡，很難單從殘差判斷問題出在哪。對跟校正樣本相似的車輛，這種自我一致性可能讓相對誤差變小；對不像的車輛（校正時漏掉的車型），誤差不會被掩蓋，反而可能疊加。
5. **K 矩陣不可靠，是 `solvePnP` 的共通限制**：`cv2.solvePnP` 求解 R、T 需要先給定準確的相機內參矩陣 K；本專案目前的 K 是用 `fx = image_width` 這種未實測的粗略假設，誤差可達 30–100%（見 `docs/localization-methods.md` 方法 D 否決 PnP 的理由），而 PnP 反推深度跟 fx 成正比，K 的誤差會直接等比例傳導成姿態/深度誤差。下方文獻調查的 AutoCalib 同樣呼叫 `solvePnP`，但論文明講是**假設 K 已知（來自攝影機廠牌/型號規格），不解 K**——代表 AutoCalib 報出來的 8.98% RMS 誤差是建立在「K 準」這個前提上；若直接沿用本專案現有的 K，這個前提不成立，方法 4 會直接繼承 `docs/localization-methods.md` 否決 PnP 時列出的同一個失效模式。要讓方法 4 可行，必須先解決 K 的來源問題（實測棋盤格標定、或借用方法 1 的消失點幾何法反推焦距）。

**選點策略（降低車輛相關誤差，但無法突破 homography 地板）**

1. **空間分佈要跨深度、跨車道**：只用近處點無法解耦 pitch 與 depth；但遠處點 homography 誤差較大，建議依「離原始標定點的距離」加權，近處權重高、遠處權重低，不要等權重丟進同一個最小二乘法。
2. **限定單一車型類別**：只用 `car`/Sedan class，排除 SUV/貨車/公車混雜，把個體車高變異壓到 $\pm 0.05\text{m}$ 量級。
3. **同一空間區塊要有多台車，不能只用一台**：每個深度/車道區塊需要足夠獨立樣本，才能靠大數法則平均掉個體車高偏差。
4. **keypoint 挑幾何穩定、偵測穩定的點——但要注意跟下方文獻調查的衝突**：直覺上應優先用輪胎接觸點/輪轂中心，避免後照鏡（會折疊）、車頂圓角邊緣（偵測器不易抓準）等雜訊大的點；但 AutoCalib（見下方文獻調查）的實證結果指出，後照鏡等側向點是少數能提供深度（Y）軸跨距的 keypoint，拿掉它們會讓 PnP 在深度方向失去解析力。兩者是不同層次的取捨（單點雜訊 vs. 點集合對 PnP 的 conditioning），不能只憑「雜訊大」就排除，需要兩者兼顧評估。
5. **保留至少一個完全獨立於車輛模板與 homography 的驗證點**：自我一致性掩蓋問題只能靠外部獨立量測（已知車道寬、現場量測距離）抓出來，光靠選點數量無法解決。

**誤差量級的大致推估**

- 最終誤差不會小於 homography 本身的精度——這是這個方法的地板，選點策略無法突破。
- 車輛模板的隨機變異項，限定車型 + 足夠樣本數（每區塊數十台以上）後，理論上可壓到接近或小於 homography 誤差量級，不會是主導誤差源。
- 系統性偏差（模板跟實際車隊的整體落差、homography 的平面假設誤差）不會隨樣本數增加而縮小，只能靠獨立驗證點抓出來。
- 若沒有針對 homography 精度與模板系統性偏差做過獨立驗證，不建議把這個方法解出來的結果當作比現有兩點手動標定更準；需要先用已知真實座標的獨立驗證點實測，才能知道實際誤差量級。

**文獻調查：GCP 數量與空間分布如何影響精度**（依使用者要求查證，2026-08-12 補充）

以下三組文獻／實作直接回答「方法 4 要多少個 GCP、怎麼分布，才能有一定精度」：

1. **AutoCalib**（Bhardwaj et al., *AutoCalib: Automatic Traffic Camera Calibration at Scale*, ACM BuildSys 2017 / ACM TOSN 2018；社群復現：[corenel/auto-traffic-camera-calib](https://github.com/corenel/auto-traffic-camera-calib)，已封存）——跟方法 4 幾乎是同一個做法：用車輛 keypoint（車頭燈、車牌、側照鏡等 6 點）+ `SolvePnP` 逐台車獨立解一次相機外參，再統計過濾聚合，不是把所有車的 keypoint 一次丟進同一個最小二乘法。
   - **keypoint 空間跨距比雜訊更關鍵**：論文實測發現車頭燈（LL/RL）、車牌（LIC）、車輛中線（CL）等點幾乎共面，只有兩側照鏡（SWL/SWR）在深度（Y）軸有足夠變異；拿掉 SWL/SWR 後，80% 的單車校正結果誤差超過 50%。這是選點策略第 4 點特別標注衝突的依據。
   - **聚合演算法**：`OrientationFilter(75%)` → `DisplacementFilter(中間 50%)` → `OrientationFilter(75%)` 三階段統計過濾，剔除方向與位移離群的單車校正結果，最後用**中位數**（而非平均，因 `SolvePnP` 是非線性解）聚合旋轉與平移。
   - **樣本規模**：10 支鏡頭、350+ 小時影片的大規模聚合，屬於「大量獨立單車校正 + 統計過濾」量級，不是幾十台的小樣本。
   - **精度基準**：人工標定的靜態 GCP 基準（每支鏡頭 10 個以上，如樹木、電線桿、人行穿越道，用 Google Earth 對應世界座標）本身 RMS 誤差 4.62%（GCP 標定/量測誤差的地板）；AutoCalib（車輛模板法）平均 RMS 誤差 8.98%，約為人工 GCP 基準的兩倍——可作為方法 4 精度上限的參考錨點。

2. **用真實車輛 GNSS 位置當 GCP**（Ojala et al., *Infrastructure camera calibration with GNSS for vehicle localisation*, IET Intelligent Transport Systems 2023；程式碼：[ojalar/gnss-camera-calibration](https://github.com/ojalar/gnss-camera-calibration)）——跟方法 4 的差異是用 RTK GNSS 量到的真實車輛位置取代車型模板（等同方向 7 思路但用實測取代統計先驗），且只解 2D homography（非 3D PnP，因此不受 K 矩陣問題影響），但其**空間分布策略可直接借用**：
   - **具體數量**：三支鏡頭分別只用 43、38、40 組 GNSS↔影像對應點就完成校正，論文明確指出刻意保持樣本數不多，用以證明方法不需要大量資料。
   - **分布做法**：不是隨機撒點，是**駕車覆蓋每一條車道各至少一次**——四車道路口每車道各開一次；雙車道路段雙向來回共 5 趟；六車道路段（雙向各 3 車道）逐車道覆蓋。用意是強制 GCP 同時跨越深度（遠近）與橫向（車道）兩個維度，不能只集中單一車道或單一距離。
   - **精度結果**：10-fold cross-validation，約 140m 範圍內平均誤差 1.5–2.0m（相對誤差約 1–2%），且誤差隨距離增加而放大、在接近消失點的遠處樣本誤差可達 3–6m——印證了本文件「選點策略」第 1 點對遠點誤差放大的推估。

3. **通用攝影測量文獻**（UAV/正射影像領域，非交通攝影機專屬，但可作為選點數量/分布的量化參考）：
   - 實務底線約 5 個 GCP，分布在場域四角+中心；10–15 個以上、且均勻涵蓋周邊與中心（不只邊緣）時 RMSE 隨數量增加而下降，超過此量級後邊際效益迅速趨緩。
   - **分布均勻度通常比單純增加數量更重要**：GCP 集中一處時，即使數量夠多，RMSE 仍然偏高；均勻分布配置持續優於角點集中式配置。若 GCP 本身量測可靠度高（如 RTK 等級），數量的重要性才會超過分布。

**對方法 4 選點策略的具體結論**：不建議沿用方向 7那種不管空間分布、只靠增加樣本數量（數百輛車級別）讓統計先驗誤差自我平均掉的大數法則量級；方法 4 若要落地，應參考上述兩個已驗證量級之一——(a) AutoCalib 式「大量車輛、逐台獨立 PnP + 統計過濾聚合」（幾十到上百台以上），或 (b) Ojala 式「每個空間區塊（車道 × 深度）至少要有代表性樣本，30–40 組對應點即可達到 1–2% 量級相對誤差」——並優先確保空間分布跨車道、跨深度，而非單純堆數量。

---

## 5. 只信任地面接觸點（已實作）

業界交通監控最常見、最穩健的做法：只用物體的地面足印（輪胎接觸點、bbox 底邊中點）定位，因為只有 h=0 的點完全不受高度校正誤差影響。這正是本專案 `--localizer wheel_pair` 在做的事（見 `docs/keypoints-openpifpaf-wheel-pair-localizer.md`），等於直接把「高度校正準不準」這個變數從 pipeline 中的關鍵路徑上拿掉。

---

## 6. 學習式單目 3D 物件偵測（重量級選項）

**概念來源**：自駕車領域現行主流，例如 Mousavian et al. 的幾何約束 2D→3D 方法，或更新的 BEV 系 monocular 3D 偵測網路（可參考 `docs/method-survey.md` 中已調查過的 BEVHeight / BEVHeight++ / HeightFormer / CoBEV 等，那份文件是針對車輛朝向估算調查的，但架構同樣涵蓋高度/深度估計）。

**原理**：完全繞開手動相機標定＋視差公式，直接從單張影像學習回歸物體的 3D box / 離地高度。

**評估**：精度可以很高，但需要訓練資料，且跟本專案現有「幾何模板 + procrustes/reprojection 擬合」架構差異很大，屬於要重新投資的方向。**不建議作為短期選項**；若要投入，應先參考 `docs/method-survey.md` 裡已經對同類 BEV 偵測模型做過的可行性分析（GPU 需求、domain gap 等）。

---

## 7. 多參考物最小二乘法（N-point least-squares reference calibration）

**概念來源**：使用者提出，是對現有單點相機視差模型的直接推廣。

**核心公式**

把相機視為位在衛星平面座標 $O = \text{cam\_sat\_xy}$ 正上方、高度 $z_{\text{cam}}$ 的一個點。對場景中一個真實高度 $h$ 的參考物件，腳點（高度 0）的世界座標是 $F$；頭點的影像像素直接套用「假設高度為 0」的既有地面 homography 換算出來的「表觀地面位置」是 $G$。兩者跟 $O$、$z_{\text{cam}}$、$h$ 的關係是：

$$F - O = (G - O)\cdot\frac{z_{\text{cam}} - h}{z_{\text{cam}}}$$

這正是現有 `parallax_correct_ground_to_real` 用的 `factor = (z_cam - h) / z_cam` 徑向縮放公式，只是這裡反過來用：不是拿已知的 `z_cam` 去校正觀測點，而是拿多組**已知 $F$、$G$、$h$** 的參考物件，反推未知的 $O$（`cam_sat_xy`）與 $z_{\text{cam}}$。

**用多個參考物件建立超定方程組**

每個參考物件 $i$ 提供一組已知量 $(F_i, G_i, h_i)$：

- $F_i$：參考物件腳點套用既有地面 homography 換算出的世界座標（視為地面真值）。
- $G_i$：參考物件頭點套用同一個 homography（假設高度為 0）換算出的「表觀」世界座標。
- $h_i$：這個參考物件的真實高度（此處採用假設平均車高，而非逐一實測）。

未知量只有 3 個：$O_x, O_y, z_{\text{cam}}$。每個參考物件貢獻 2 條方程式（$x$、$y$ 各一條），$N \ge 2$ 就過決定（overdetermined）。方程式裡有 $O$ 與 $1/z_{\text{cam}}$ 的乘積項，對這兩組未知數是雙線性、非線性的，但只有 3 個未知數、結構單純，用非線性最小二乘（例如 `scipy.optimize.least_squares`）取多組 $(F_i,G_i,h_i)$ 聯合求解，應該能穩定收斂。

**用假設平均高度取代逐一實測的影響**

因為 $h_i$ 不是逐一量測、而是用假設的車輛平均高度代入，每個參考物件的 $h_i$ 都帶有跟真實高度之間的落差：

- 若落差是**隨機、零均值**的個體差異（例如同車型內的高度變異），只要參考物件數量夠多、彼此獨立，這個誤差在最小二乘解裡會被平均掉——物件數量越多，解的標準差越小。
- 若落差是**系統性**的（例如這批參考物件的車型分佈整體偏離假設值），則不會隨數量增加而縮小，最小二乘法只會把這個系統性落差鎖進解出來的 $z_{\text{cam}}$/`cam_sat` 裡，變成一個看起來收斂良好、但實際上有偏移的解。
- 要壓低隨機項的影響，建議限定參考物件的類別（例如只用同一車型類別），把個體高度變異壓到最小。

**跟本專案的接點**：$F_i$、$G_i$ 可以直接用現有 YOLO/keypoint 偵測產生的 bbox 底部中點與頂部中點，套用既有 `GProjection` 的地面 homography 換算得到；解出來的 $z_{\text{cam}}$/`cam_sat` 可以直接替換 `GProjection` 現有的 parallax 參數，不需要新增校正 UI，只需要一支批次回歸腳本。

**建議下一步**：寫一支原型腳本，蒐集某個 location 足夠數量、跨深度/跨車道分佈的車輛 bbox 頂/底點，限定車型類別後跑非線性最小二乘，看解出來的 $z_{\text{cam}}$ 跟現有兩點手動標定差多少、殘差分佈長什麼樣子。

**已實作（2026-08-12）**：以獨立 GUI 工具的形式實作，讓使用者手動在影片的多個幀上標記多個參考物件的頭/腳點與真實高度（而非方向 4 提議的自動車輛 keypoint 批次偵測），求解流程與上方公式一致。

- `trafficlab/projection/reference_point_calibration.py` — 核心數學（`ReferencePoint`、`calibrate`，`scipy.optimize.least_squares`），純 numpy/scipy，不依賴 Qt。
- `trafficlab/gui/tools/reference_point_calibration_tool.py` + `scripts/reference_point_calibration_tool.py` — 獨立 PyQt5 工具（不影響現有校正精靈），三個 tab：新增參考點（可拖曳頭/腳細叉叉標記、逐幀可刪除）、確認（跨幀列表 + 逐筆預覽）、結果（顯示 `cam_sat_xy`/`z_cam_meters`/RMSE，可另存獨立 JSON 或選擇性套用到 `G_projection_<code>.json`）。

使用說明見 [docs/reference-point-calibration-tool-guide.md](reference-point-calibration-tool-guide.md)。

---

## 相關檔案索引

| 檔案 | 說明 |
|------|------|
| `trafficlab/projection/g_projection.py` | 現有高度校正實作，`_init_parallax`/`parallax_correct_ground_to_real` |
| `trafficlab/gui/tabs/calibration_stage/pars_stage.py` | 現有的兩物件頭腳點手動標定 UI |
| `trafficlab/projection/parallax_reprojection.py` | 校正前後（pre/post）視差位移診斷工具 |
| `docs/keypoints-openpifpaf-wheel-pair-localizer.md` | 方向 5 的已實作版本 |
| `docs/method-survey.md` | 方向 6 提及的 BEV 3D 偵測模型調查（車輛朝向估算脈絡） |
