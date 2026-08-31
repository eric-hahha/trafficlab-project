# export_trajectory_3d.py

把 inference pipeline 產出的 `output/.../{footage}.json.gz` 轉成一份**自足資料夾**，
交給下一位同學用 **Three.js 做 3D 行車軌跡重建**（車速以單一平均速度呈現）。

## 為什麼需要這支腳本

原始 `.json.gz` 的座標是**衛星影像像素**（`sat_coords`，原點左上、Y 向下），而且：

- 換算成公尺需要的 `px_per_m` 在 `location/{loc}/G_projection_{loc}.json`，**不在** `.json.gz`。
- 車輛尺寸（w/l/h）在 `prior_dimensions.json`，**不在** `.json.gz`。
- **平均速度不是存起來的欄位**，是 viewer 端即時算的。

所以直接丟 `.json.gz` 對方接不起來。本腳本把這些補齊、換算成公尺、預先算好平均速度，
打包成一個資料夾。

## 用法

```bash
# 最常見：只給輸入檔，其餘用預設
python scripts/export_trajectory_3d.py \
    output/model-yolov8s-seg_tracker-bytetrack/seg_default/Hsinchu/Hsinchu.json.gz

# 指定輸出目錄、關閉過濾、路徑降採樣
python scripts/export_trajectory_3d.py path/to/foo.json.gz \
    --out-dir export_3d/foo --min-move 0 --stride 2
```

預設輸出到 `export_3d/{location}_{stem}/`，內含：

| 檔案 | 內容 |
|---|---|
| `trajectory.json` | 每台車：公尺座標路徑 + `avg_speed_kmh` + `dims_m` + 每點 `heading`/時間 |
| `sat_{loc}.png` | 衛星底圖（3D 場景地面貼圖） |
| `README.md` | 座標系與 schema 說明（隨資料夾一起給對方） |

## 參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `input_path` | — | 輸入 `.json` / `.json.gz`（positional） |
| `--out-dir` | `export_3d/{loc}_{stem}` | 輸出資料夾 |
| `--g-projection` | 自動由 `location_code` 定位 | 覆寫 G_projection 路徑 |
| `--prior-dimensions` | `prior_dimensions.json` | 車輛尺寸表 |
| `--prior-set` | 合併所有 set | 指定 `measurements_visdrone` / `_v2` |
| `--min-frames` | `5` | 路徑點數少於此值的 track 丟棄 |
| `--min-move` | `2.0` | 總移動距離（m）低於此值丟棄，濾停車/誤判 |
| `--stride` | `1` | 路徑點降採樣（不影響平均速度計算） |
| `--no-copy-image` | off | 不複製底圖 |

## 設計重點

- **重用既有函式**：I/O 與 location 推斷用 `trafficlab/trajectory/io.py`；
  `resolve_g_projection_path` / `load_prior_map` 沿用 `scripts/filter_and_enrich_output.py` 的做法；
  平均速度公式鏡像自 `trafficlab/gui/tabs/tab_visualization.py` 的 `_compute_avg_speed_map`，
  因此匯出的 `avg_speed_kmh` 會與 GUI 勾「Avg Speed」顯示的值一致。
- **不靜默 fallback**：`G_projection` 缺失或 `px_per_meter` 無效時**直接報錯**，
  絕不用 `px_per_m = 1.0` 產生看似正常但完全錯誤的公尺座標／速度。
- `--stride` 只降採樣輸出的路徑點，**平均速度仍以完整取樣計算**，故縮檔不影響速度正確性。

## 座標系（給 Three.js 端）

- 單位公尺，`x = sat_x/px_per_m`、`z = sat_y/px_per_m`，原點 = 衛星圖左上角。
- 軌跡貼地平面 `Y = 0`；高度用 `dims_m.height`。
- `sat_y` 螢幕向下增加，Three.js 為 Y-up 右手系——翻軸／貼圖對位由 Three.js 端決定。
- 詳見每次輸出資料夾內自動產生的 `README.md`。
