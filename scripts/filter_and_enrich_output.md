# `filter_and_enrich_output.py`

這支腳本會讀取 TrafficLab 的 `output.json` 或 `output.json.gz`，保留指定的多個 `tracked_id`，並補上給下游模擬或 Blender 較好使用的欄位。

## 功能

- 依 `tracked_id` 篩選車輛
- 在輸出 JSON 的 `meta` 中補上 `px_per_meter`
- 在每筆物件上補上 `dimensions_m`
- 在每筆物件上補上 `position_m`
- 在每筆物件上補上 `velocity_mps`
- 統計每個選定 `tracked_id` 缺少多少 `heading` 與 `speed`

## 使用方式

```bash
python scripts/filter_and_enrich_output.py output.json filtered_output.json --ids 7 373
```

如果第二個參數只給檔名，腳本會自動輸出到 `scripts/` 目錄，也就是這個例子實際會寫到：

```text
scripts/filtered_output.json
```

如果需要指定校正檔或尺寸先驗檔，也可以額外傳入：

```bash
python scripts/filter_and_enrich_output.py \
  output.json \
  filtered_output.json \
  --ids 7 373 \
  --g-projection location/test1/G_projection_test1.json \
  --prior-dimensions prior_dimensions.json \
  --prior-set measurements_visdrone
```

## 欄位來源與計算方式

### `px_per_meter`

來源：`G_projection_{location}.json` 的 `parallax.px_per_meter`

意思是：

```text
1 meter = px_per_meter SAT pixels
```

腳本會把它補到：

```json
{
  "meta": {
    "px_per_meter": 34.41027430558973
  }
}
```

### `dimensions_m`

來源：`prior_dimensions.json`

做法是把每筆物件的 `class` 轉成小寫後，依車種查表。例如：

- `Car` -> `car`
- `Truck` -> `truck`

再補成：

```json
"dimensions_m": {
  "width": 1.8,
  "length": 3.8,
  "height": 1.55
}
```

### `position_m`

來源：`sat_coords`

計算公式：

```text
position_m.x = sat_coords.x / px_per_meter
position_m.y = sat_coords.y / px_per_meter
```

也就是把 SAT 平面上的像素座標換算成公尺座標。

### `velocity_mps`

來源：相同 `tracked_id` 的相鄰兩筆 `position_m`

計算公式：

```text
dt = (current_frame_index - previous_frame_index) / fps
vx = (current_x_m - previous_x_m) / dt
vy = (current_y_m - previous_y_m) / dt
```

輸出格式：

```json
"velocity_mps": [vx, vy]
```

注意事項：

- 某台車第一次出現時，沒有前一筆位置，因此 `velocity_mps` 會是 `null`
- 如果兩次觀測之間有掉幀，腳本會用 `frame_index` 差值計算實際 `dt`

## 缺失統計規則

### `missing_heading_count`

以下情況會算缺少 `heading`：

- 沒有 `heading` 欄位
- `heading` 值為 `null`

### `missing_speed_count`

以下情況會算缺少 `speed`：

- 沒有 `speed_kmh` 欄位
- `speed_kmh` 值為 `null`
- `have_heading == false`

最後一條是沿用目前資料語意：當沒有可用朝向時，`speed_kmh` 即使為 `0.0`，也視為不可可靠使用。

## 輸出欄位範例

```json
{
  "tracked_id": 7,
  "class": "Car",
  "sat_coords": [681.235, 916.751],
  "position_m": [19.797, 26.642],
  "velocity_mps": [0.228, -1.804],
  "dimensions_m": {
    "width": 1.8,
    "length": 3.8,
    "height": 1.55
  }
}
```
