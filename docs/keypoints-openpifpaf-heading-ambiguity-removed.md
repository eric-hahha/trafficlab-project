# H-aware：移除 `ambiguous_heading` 翻轉檢驗

**改動檔案**：`trafficlab/motion/haware_localization.py`（`HawareLocalizer.localize()`）
**狀態**：已改動，未 commit

---

## 改了什麼

`localize()` 原本在算出 Procrustes 解之後，會多做一次「前後翻轉」檢驗：

```python
# 原本的做法（已移除）
Q_flip = Q.copy()
Q_flip[:, 1] *= -1                              # 把模板縱向軸整個翻面
qb_flip = Q_flip.mean(0)
Hc_flip = (Q_flip - qb_flip).T @ (P - pb)
U_f, _, Vt_f = np.linalg.svd(Hc_flip)
det_sign_f = float(np.sign(np.linalg.det(Vt_f.T @ U_f.T)))
R_flip = Vt_f.T @ np.diag([1.0, det_sign_f]) @ U_f.T
P_pred_flip = (Q_flip - qb_flip) @ R_flip.T + pb
rms_flip = float(np.sqrt(np.mean(np.sum((P - P_pred_flip) ** 2, axis=1))))

ambiguous = rms >= self._AMBIGUITY_RMS_RATIO * rms_flip   # 0.7
status = 'ambiguous_heading' if ambiguous else 'ok'
if ambiguous:
    heading = None
```

邏輯是：拿同一組偵測到的關鍵點，分別假設「這是車頭朝這邊」跟「其實前後相反」去各自配準一次，如果原本朝向的擬合誤差沒有明顯小於翻轉後的誤差（`rms >= 0.7 * rms_flip`），就代表這組點無法可靠分辨車頭車尾，判定 `ambiguous_heading`，把 `heading` 設成 `None`。

現在整段拿掉，`localize()` 只要 `n >= 2`（關鍵點數量門檻）就直接輸出 Procrustes 解出的 heading，`status` 一律是 `'ok'`：

```python
# 現在的做法
P_pred = (Q - qb) @ R.T + pb
rms    = float(np.sqrt(np.mean(np.sum((P - P_pred) ** 2, axis=1))))
conf   = min(1.0, n / 8.0) * max(0.0, 1.0 - rms / (5.0 * s))

return HawareResult(
    sat_coords=tuple(T_sat), heading=heading, confidence=conf,
    n_keypoints=n, status='ok', p_sat=p_sat,
)
```

同時移除了已無用的 `_AMBIGUITY_RMS_RATIO = 0.7` 常數，以及 docstring / `HawareResult.status` 註解裡對 `ambiguous_heading` 的說明。

## 為什麼改

使用者明確要求：只要能配準，就要輸出 heading，不要因為「翻轉後幾乎一樣好擬合」而把 heading 壓成 `None`。

## 改動範圍

只動 `HawareLocalizer.localize()`（`eval_haware_replay.py --localizer procrustes` 用的路徑）。**沒有動**：

- `HawareLocalizer.localize_reprojection()`（`--localizer reprojection`）——這個方法本來就從沒回傳過 `ambiguous_heading`，不受影響
- `trafficlab/motion/carfusion_localization.py`——CarFusion 用的是另一套獨立的 ambiguous 判斷（檢查所有有效關鍵點是否都在同一縱向半邊），跟這次移除的 RMS 翻轉比對是不同機制，沒有改動
- 下游腳本（`eval_haware_replay.py`、`eval_haware_yolo_crop.py`、`scripts/archive/eval_haware.py`）裡檢查 `status == 'ambiguous_heading'` 的統計/顯示分支——這些分支還在，只是以後從 `localize()` 拿到的結果不會再觸發，變成永遠不會執行到的分支，不影響功能

## 實測結果對照（`test21-6.mp4`，`--method geometric --yolo models/best.pt`）

| | 改動前 | 改動後 |
|---|---|---|
| `ok` | 7 | 975 |
| `ambiguous` | 968 | 0 |
| `failed_insufficient_kp` | 711 | 711（不變，門檻邏輯沒動）|
| 總偵測數 | 1686 | 1686 |

## ⚠️ 需要注意的取捨

這個改動**沒有讓 heading 判斷變準，只是不再把「判斷不出來」標記出來**。原本被判定 `ambiguous_heading` 的 968 個偵測，其幾何意義是「這組關鍵點在翻轉前後幾乎一樣擬合得好」——這代表 Procrustes 解出的旋轉方向，在這些案例裡本來就沒有堅實的幾何證據支持。移除檢驗後，這 968 個案例一樣會輸出一個 heading 數值，但那個數值的可信度跟改動前是一樣低的，只是不再被標記為 `None`／`ambiguous`。

換句話說：下游如果要用 `status` 欄位篩掉不可靠的朝向，這個管道從今以後失效了（`status` 只會是 `ok` 或 `failed_insufficient_kp`）。如果之後發現大量 heading 反著畫（180° 誤判），根源多半就是這批原本會被攔下來的模糊案例。

## 已知需要一起檢查的地方

`docs/haware-intro.md` 目前第 75、81 行仍描述舊版「同側關鍵點」判斷法（更早一版的 ambiguous 邏輯，比這次移除的 RMS 翻轉版本還舊），跟目前程式碼已經有兩層不同步，這份文件本身已經過時，需要另外更新（不在這次改動範圍內）。
