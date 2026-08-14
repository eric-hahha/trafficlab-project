# 只用同側前後輪的定位方法（`--localizer wheel_pair`）

## 背景

現有的 `--localizer procrustes`（`OpenPifPafKeypointsLocalizer.localize`）和 `--localizer reprojection`（`localize_reprojection`）都會盡量利用畫面中每一個高信心度的關鍵點：procrustes 對所有可信點做一次 2D Procrustes 擬合，reprojection 依可見的關鍵點 pair 組合走 Method 1/2/3，能用到的點越多，擬合的自由度越高。

`--localizer wheel_pair` 反其道而行：**永遠只看同側（左或右）的前輪、後輪這兩個關鍵點**，即使畫面裡其他關鍵點也很可信也不用。這不是資料不足時的 fallback，是刻意的限制——用來當作「只靠輪胎位置能做到多準」的 baseline，跟另外兩個方法的結果對照，觀察其他關鍵點（車身、車燈、後照鏡等，這些多半是估計出來的高度、不是實測值）到底貢獻了多少準確度。

涉及檔案：
- `trafficlab/motion/keypoints_openpifpaf.py` — `OpenPifPafKeypointsLocalizer.localize_wheel_pair`，以及供它重用的 module-level `_rotation_from_vectors`/`_heading_from_forward`
- `scripts/run_keypoints_openpifpaf.py` — `--localizer wheel_pair` 選項、輸出路徑分支

## 選點與資格判斷

`_FR_WHEEL_PAIRS = [(front_wheel_left, rear_wheel_left), (front_wheel_right, rear_wheel_right)]`——同一側的前後輪組成一組。一組要「合格」，前後兩個關鍵點的信心度都必須 `>= kp_conf`；只要有一邊沒過門檻，這一組就不能用。

兩側都不合格 → 直接回傳 `status='failed_insufficient_kp'`（`sat_coords=None`、`heading=None`、`n_keypoints=0`、`p_sat={}`），不會退而求其次去用別的關鍵點湊。兩側都合格時，任選一組即可（目前固定取 `_FR_WHEEL_PAIRS` 順序中第一個，也就是左側優先——不是真的隨機，跟現有 `localize_reprojection` 的 `lr_pair`/`fr_pair` 選點慣例一致）。

## 位置公式

模板（`build_car_template(dims)`）裡，同側前後輪的 `x`（左右座標）完全相同，都是半輪距 `htw`，只有 `z`（前後座標）不同。兩點的中點因此已經落在正確的前後中心線上（z 對了），但左右方向還偏移了 `htw`——要移到車輛真正的對稱軸（x=0），必須沿著「垂直於前後輪連線」的方向移動，移動量等於半輪距。

這個位移量再乘上一個縮放比例做per-vehicle校正：

```
scale_ratio = 實際前後輪距離（公尺，sat 空間投影後換算） / 模板前後輪距離（公尺，模板原始單位）
位移量 = |該側輪的模板 x 座標| * scale_ratio
```

用這台車自己觀測到的軸距相對模板的比例，去縮放模板的半輪距，而不是不分青紅皂白套用模板的固定值——车體比模板大或小，位移量會跟著等比例調整。

方向與正負號**直接沿用 `localize_reprojection` Method 2（`fr_pair`-only 分支）已經驗證過的手法**，不是重新推導：算出一個把模板前後輪向量（body frame）旋轉到觀測到的前後輪向量（world/sat 空間）的旋轉矩陣 `R`，位移方向就是 `R @ (1,0)`（對應 body frame 的 +x／側向軸）。因為位移量用的是**帶符號**的模板 x 座標（左側 pair 是正的 `htw`，右側 pair 是負的 `htw`），同一條公式 `mid_world - template_x * scale_ratio * s * shift_world` 對兩側都會自動移到正確的方向，不需要另外寫 if-else 判斷左右——這正是 Method 2 原本用來處理這個問題的做法，只是那裡是套用固定模板值、這裡多乘了 `scale_ratio`。

## 朝向（heading）公式

`heading = _heading_from_forward(front_sat - rear_sat)`，即 `atan2(dy, dx) % 360`（無負號）。這跟 `localize_reprojection` 的 `fr_pair`-only 分支、`localize()` 的 heading 慣例完全一樣——**刻意重用同一條公式**，不是自己重新推導。原因是這個 codebase 裡曾經記錄過一次方向號搞錯的真實問題（`wheel_localization.py` 文件裡的 `atan2(-dy, dx)` 慣例其實跟系統其他地方，如 `kinematics.py`、`sat_renderer.py` 畫箭頭的方式對不上），front/rear 是有名字的關鍵點、方向不會有歧義，因此直接用觀測到的前後輪向量，不需要 `_resolve_lr_forward` 那種投票校正。

## 為什麼只用兩個關鍵點（而不是像 Method 2 那樣多用一點）

`localize_reprojection` Method 2 在只有 `fr_pair` 可見時，除了用這組輪子定出前後中心線的位置，還會把畫面中其他每一個可信關鍵點（依其模板座標）投影校正、平均起來，定出第二條中心線——本質上是「有多少可信點就多用多少」，多一點資訊就多一分準度。

`wheel_pair` 刻意不這麼做：不管還有多少其他關鍵點可信，一律只用這兩個輪子算位置和朝向。這樣才能單獨衡量「只靠輪胎信號」這件事本身的準確度上限，作為跟 `procrustes`/`reprojection`（會用到其他多半是估計出來的高度值的關鍵點）比較的基準線。

## 目前預設值

```
--localizer  wheel_pair
--kp-conf    0.2（沿用全域門檻，不另外設定）
confidence   固定 0.4（flat placeholder，量級對齊 localize_reprojection Method 2）
```

輸出路徑：`output/wheel_pair/<location_code>/<video_stem>.json.gz`（跟 `procrustes`/`reprojection` 共用的 `output/haware/...` 分開，避免同一支影片、沒指定 `--out` 時互相覆蓋）。
