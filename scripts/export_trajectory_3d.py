"""Export a TrafficLab replay .json.gz into a self-contained folder for 3D (Three.js) reconstruction.

用途：把 inference pipeline 產出的 output/.../{footage}.json.gz 轉成一份「可直接交給
下一位同學做 Three.js 3D 重建」的自足資料夾，內含：
  - trajectory.json  每台車一筆：公尺座標路徑 + 預算好的平均速度 + 車輛尺寸 + heading
  - sat_{loc}.png    衛星底圖（當地面貼圖用）
  - README.md        座標系與 schema 說明

原始 .json.gz 不夠自足，因為換算成公尺需要的 px_per_m（在 G_projection_{loc}.json）與
車輛尺寸（在 prior_dimensions.json）都不在裡面，平均速度也只是 viewer 端即時算的、沒存檔。

範例：
  python scripts/export_trajectory_3d.py \
      output/model-yolov8s-seg_tracker-bytetrack/seg_default/Hsinchu/Hsinchu.json.gz

  # 指定輸出目錄、放寬過濾、路徑降採樣
  python scripts/export_trajectory_3d.py path/to/foo.json.gz \
      --out-dir export_3d/foo --min-move 0 --stride 2
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Reuse the shared trajectory I/O helpers rather than re-rolling gzip/json handling.
from trafficlab.trajectory.io import (  # noqa: E402
    frames_from_data,
    infer_location_code,
    load_json,
    resolve_satellite_image_path,
    write_json,
)
from trafficlab.io.prior_dimensions import (  # noqa: E402
    load_location_overrides,
    resolve_dims_layered,
)


# ---------------------------------------------------------------------------
# Small helpers mirrored from scripts/filter_and_enrich_output.py
# (that file is a plain script, not an importable module, so we copy the tiny
#  functions here and keep the behaviour identical.)
# ---------------------------------------------------------------------------
def resolve_g_projection_path(data, explicit_path=None) -> Path:
    """Locate location/{loc}/G_projection_{loc}.json from the replay's location_code."""
    if explicit_path:
        return Path(explicit_path)

    location_code = data.get("location_code") if isinstance(data, dict) else None
    if not location_code:
        raise ValueError(
            "無法定位 G_projection：檔案內缺少 location_code，請用 --g-projection 明確指定。"
        )
    return REPO_ROOT / "location" / location_code / f"G_projection_{location_code}.json"


def load_prior_map(prior_dimensions_path, prior_set=None) -> dict:
    """Return {class_name.lower(): {width,length,height}}.

    未指定 prior_set 時合併所有 set（first-wins），化解 measurements_visdrone 與
    measurements_visdrone_v2 的歧義（兩組共有的類別尺寸相同，motor/two_wheeler 亦相同）。
    """
    prior_data = load_json(prior_dimensions_path)

    if prior_set:
        if prior_set not in prior_data:
            raise ValueError(
                f"未知的 prior set '{prior_set}'。可用：{', '.join(sorted(prior_data))}"
            )
        return {k.lower(): v for k, v in prior_data[prior_set].items()}

    merged: dict = {}
    for class_map in prior_data.values():
        for class_name, dims in class_map.items():
            merged.setdefault(class_name.lower(), dims)
    return merged


# ---------------------------------------------------------------------------
# Trajectory extraction + average speed
# ---------------------------------------------------------------------------
def build_track_history(data) -> dict:
    """Collect per-track ordered samples. Mirrors VisualizationTab.load_file (tab_visualization.py:512-523).

    Returns {tid: {"class": str, "samples": [(frame_index, [sx, sy], heading), ...]}}.
    """
    history: dict = {}
    for frame in frames_from_data(data):
        fi = frame.get("frame_index")
        for obj in frame.get("objects", []):
            tid = obj.get("tracked_id")
            coord = obj.get("sat_coords")
            if tid is None or coord is None:
                continue
            entry = history.setdefault(
                tid, {"class": obj.get("class", "?"), "samples": []}
            )
            entry["samples"].append((fi, coord, obj.get("heading")))
    return history


def path_length_px(coords) -> float:
    return sum(
        math.hypot(coords[k][0] - coords[k - 1][0], coords[k][1] - coords[k - 1][1])
        for k in range(1, len(coords))
    )


def average_speed_kmh(samples, px_per_m, fps) -> float:
    """Average speed over the whole track. Mirrors VisualizationTab._compute_avg_speed_map
    (tab_visualization.py:577-601): total path length / total elapsed time.
    """
    coords = [s[1] for s in samples]
    frame_indices = [s[0] for s in samples]
    total_seconds = (frame_indices[-1] - frame_indices[0]) / fps
    if total_seconds <= 0:
        return 0.0
    path_m = path_length_px(coords) / px_per_m
    return (path_m / total_seconds) * 3.6


# ---------------------------------------------------------------------------
# Core export
# ---------------------------------------------------------------------------
def build_export(data, *, px_per_m, fps, prior_map, min_frames, min_move, stride,
                 dim_by_track=None, dim_by_class=None):
    history = build_track_history(data)

    tracks = []
    filtered_out = 0
    dims_by_class: dict = {}

    for tid in sorted(history, key=lambda t: (t is None, t)):
        entry = history[tid]
        samples = entry["samples"]
        if len(samples) < 2:
            filtered_out += 1
            continue

        # avg speed uses the FULL sample set (before stride downsampling) so it
        # stays identical to the GUI value regardless of --stride.
        coords_full = [s[1] for s in samples]
        path_m = path_length_px(coords_full) / px_per_m
        avg_kmh = average_speed_kmh(samples, px_per_m, fps)

        kept = samples[::stride] if stride > 1 else samples
        # always keep the last sample so the path endpoint is exact
        if stride > 1 and kept[-1][0] != samples[-1][0]:
            kept = list(kept) + [samples[-1]]

        if len(kept) < min_frames or path_m < min_move:
            filtered_out += 1
            continue

        cls = str(entry["class"])
        # 尺寸必須和 pipeline 當初算 sat_coords 時用的一致：尺寸會參與錨點修正
        # h(u)，換一組尺寸連座標都會不同。只讀全域 prior 會讓 dims_m 與 path 互相矛盾。
        dims = resolve_dims_layered(prior_map, cls, tid, dim_by_track, dim_by_class)
        dims_by_class.setdefault(cls.strip().lower(), dims)

        path = []
        for fi, coord, heading in kept:
            path.append(
                {
                    "frame": fi,
                    "t": round(fi / fps, 4),
                    "x": round(coord[0] / px_per_m, 4),
                    "z": round(coord[1] / px_per_m, 4),
                    "heading": (round(heading, 2) if heading is not None else None),
                }
            )

        tracks.append(
            {
                "id": tid,
                "class": cls,
                "dims_m": dims,
                "avg_speed_kmh": round(avg_kmh, 2),
                "path_length_m": round(path_m, 2),
                "path": path,
            }
        )

    dims_by_class = {c: dims_by_class[c] for c in sorted(dims_by_class)}
    return tracks, dims_by_class, filtered_out


def resolve_sat_image(g_proj_path, g_proj_data, location_code) -> Path | None:
    """Resolve the satellite image: inputs.sat_path relative to the G_projection dir,
    falling back to the conventional location/{loc}/sat_{loc}.png search."""
    inputs = g_proj_data.get("inputs", {}) if isinstance(g_proj_data, dict) else {}
    sat_rel = inputs.get("sat_path")
    if sat_rel:
        candidate = Path(g_proj_path).parent / sat_rel
        if candidate.exists():
            return candidate
    return resolve_satellite_image_path(location_code, project_root=REPO_ROOT)


README_TEMPLATE = """# 3D 重建軌跡資料 — {location_code}

由 `scripts/export_trajectory_3d.py` 從 `{source_file}` 匯出。

## 檔案
- `trajectory.json` — 軌跡與平均速度（見下方 schema）
- `{sat_image}` — 衛星底圖，當 3D 場景地面貼圖用
- `README.md` — 本說明

## 座標系（重要）
- 單位：**公尺**。`x = sat_x / px_per_m`、`z = sat_y / px_per_m`。
- 原點：**衛星影像左上角**。地面平面大小 = `sat_size_px / px_per_m`（約 {ground_w:.1f} m × {ground_h:.1f} m）。
- `sat_y` 於螢幕向下增加；Three.js 為 Y-up 右手系，貼圖對位／翻軸（例如 z 或 texture flipY）請在 Three.js 端決定。
- 軌跡貼地平面（**Y = 0**）；高度未編碼，車體高度請用 `dims_m.height`。
- `heading`：度，`atan2(dy, dx)` 於 sat 像素空間（y-down），0° = +x（右），順時針增加。

## trajectory.json schema
```
meta: {{ location_code, source_file, px_per_m, fps, units, sat_image, sat_size_px:[W,H],
         axes_note, track_count, filtered_out }}
dims_by_class: {{ <class>: {{ width, length, height }} }}   # 公尺
tracks: [ {{
  id, class, dims_m:{{width,length,height}}|null,
  avg_speed_kmh,           # 整段平均速度（總路徑長 / 總時間）
  path_length_m,
  path: [ {{ frame, t, x, z, heading }}, ... ]   # t 秒, x/z 公尺
}} ]
```

## 給 Three.js 的最小用法
1. 用 `sat_size_px / px_per_m` 建一個 PlaneGeometry，貼上 `{sat_image}`。
2. 每台 track 用 `dims_m` 建一個 Box（或車模），沿 `path` 的 `(x, z)` 移動，`t` 當時間軸。
3. 車速標籤直接顯示 `avg_speed_kmh`（單一數值，不隨時間變）。
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把 TrafficLab replay .json.gz 匯出成 3D 重建用的自足軌跡資料夾。"
    )
    parser.add_argument("input_path", help="輸入的 .json 或 .json.gz")
    parser.add_argument("--out-dir", help="輸出資料夾（預設 export_3d/{location}_{stem}）")
    parser.add_argument(
        "--g-projection", dest="g_projection", help="明確指定 G_projection_{loc}.json 路徑"
    )
    parser.add_argument(
        "--prior-dimensions",
        default=str(REPO_ROOT / "prior_dimensions.json"),
        help="prior_dimensions.json 路徑",
    )
    parser.add_argument("--prior-set", help="指定 measurement set；不給則合併所有 set")
    parser.add_argument(
        "--min-frames", type=int, default=5, help="路徑點數少於此值的 track 丟棄（預設 5）"
    )
    parser.add_argument(
        "--min-move",
        type=float,
        default=2.0,
        help="總移動距離（公尺）低於此值的 track 丟棄，濾停車/誤判（預設 2.0）",
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="路徑點降採樣間隔（高 fps 縮檔用，預設 1）"
    )
    parser.add_argument(
        "--no-copy-image", action="store_true", help="不複製衛星底圖到輸出資料夾"
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        parser.error(f"找不到輸入檔：{input_path}")
    if args.stride < 1:
        parser.error("--stride 必須 >= 1")

    data = load_json(input_path)
    location_code = infer_location_code(data, input_path) or "unknown"

    g_proj_path = resolve_g_projection_path(data, args.g_projection)
    if not Path(g_proj_path).exists():
        parser.error(
            f"找不到 G_projection：{g_proj_path}\n"
            "  （沒有它就無法把像素換算成公尺、也算不出真實速度；請用 --g-projection 指定。）"
        )
    g_proj_data = load_json(g_proj_path)

    px_per_m = g_proj_data.get("parallax", {}).get("px_per_meter")
    if px_per_m is None or float(px_per_m) <= 0.001:
        parser.error(
            f"G_projection 的 parallax.px_per_meter 無效（{px_per_m!r}）。"
            "拒絕用 1.0 fallback 產生假的公尺座標／速度。"
        )
    px_per_m = float(px_per_m)

    fps = (data.get("meta", {}) or {}).get("fps", 0)
    if not fps or fps <= 0:
        print(f"[warn] meta.fps 無效（{fps!r}），改用 30.0", file=sys.stderr)
        fps = 30.0
    fps = float(fps)

    prior_map = load_prior_map(args.prior_dimensions, args.prior_set)
    ovr_track, ovr_class = load_location_overrides(str(g_proj_path))
    if ovr_track or ovr_class:
        print(f"[info] 套用場地尺寸覆寫：{len(ovr_track)} by track, {len(ovr_class)} by class")

    tracks, dims_by_class, filtered_out = build_export(
        data,
        px_per_m=px_per_m,
        fps=fps,
        prior_map=prior_map,
        min_frames=args.min_frames,
        min_move=args.min_move,
        stride=args.stride,
        dim_by_track=ovr_track,
        dim_by_class=ovr_class,
    )

    # Output folder
    stem = input_path.name
    for suffix in (".json.gz", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "export_3d" / f"{location_code}_{stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Satellite image + size
    sat_image_name = None
    sat_size_px = None
    sat_src = resolve_sat_image(g_proj_path, g_proj_data, location_code)
    if sat_src is not None and Path(sat_src).exists():
        img = cv2.imread(str(sat_src))
        if img is not None:
            sat_size_px = [int(img.shape[1]), int(img.shape[0])]
        sat_image_name = Path(sat_src).name
        if not args.no_copy_image:
            shutil.copy2(sat_src, out_dir / sat_image_name)
    else:
        print("[warn] 找不到衛星底圖，trajectory.json 仍會產生但 sat_image 為 null", file=sys.stderr)

    export = {
        "meta": {
            "location_code": location_code,
            "source_file": str(input_path),
            "px_per_m": round(px_per_m, 6),
            "fps": fps,
            "units": "meters",
            "sat_image": sat_image_name,
            "sat_size_px": sat_size_px,
            "axes_note": (
                "world ground plane. x = sat_x/px_per_m, z = sat_y/px_per_m; "
                "origin = sat image top-left; sat_y 向下(螢幕). Three.js 端自行決定翻軸/翻手性與貼圖對位. "
                "height(Y) 未編碼, 用 dims_m.height. "
                "heading=度, atan2(dy,dx) 於 sat 像素空間(y-down), 0°=+x, 順時針增."
            ),
            "track_count": len(tracks),
            "filtered_out": filtered_out,
            "dims_source": (
                "location_override" if (ovr_track or ovr_class) else "global_prior"
            ),
            "dims_override_counts": {"by_track": len(ovr_track), "by_class": len(ovr_class)},
        },
        "dims_by_class": dims_by_class,
        "tracks": tracks,
    }
    write_json(out_dir / "trajectory.json", export)

    # README
    ground_w = (sat_size_px[0] / px_per_m) if sat_size_px else 0.0
    ground_h = (sat_size_px[1] / px_per_m) if sat_size_px else 0.0
    readme = README_TEMPLATE.format(
        location_code=location_code,
        source_file=input_path.name,
        sat_image=sat_image_name or "(none)",
        ground_w=ground_w,
        ground_h=ground_h,
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    # Summary
    print(f"匯出完成 → {out_dir}")
    print(f"  location={location_code}  px_per_m={px_per_m:.4f}  fps={fps}")
    print(f"  保留 {len(tracks)} 條軌跡，過濾掉 {filtered_out} 條")
    if sat_size_px:
        print(f"  底圖 {sat_image_name} {sat_size_px[0]}x{sat_size_px[1]} → 地面 {ground_w:.1f}m x {ground_h:.1f}m")
    for tr in tracks[:15]:
        print(f"    id={tr['id']:<4} {tr['class']:<12} avg={tr['avg_speed_kmh']:>6.1f} km/h  len={tr['path_length_m']:>6.1f} m  pts={len(tr['path'])}")
    if len(tracks) > 15:
        print(f"    ... 其餘 {len(tracks) - 15} 條略")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
