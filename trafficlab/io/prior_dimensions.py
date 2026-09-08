"""Resolve detector class names against prior_dimensions.json keys.

不同模型吐出的 class 名稱不一致（COCO 的 person/motorcycle、VisDrone 的
pedestrian/motor、v2 的 two_wheeler），但 prior_dimensions.json 每組 set 只存一種鍵。
名稱對不上時 dims 會是 None，該物件就沒有 have_measurements、沒有 sat_floor_box、
也沒有 bbox_3d — 例如 yolov8s-seg 的 "motorcycle" 對不到 measurements_visdrone 的
"motor"，機車就只剩一條軌跡線。這裡集中處理同義名，避免各處各寫一份對照表。
"""

import json
import os

# value 為候選鍵清單，取 prior_map 中第一個存在者（motor 與 two_wheeler 尺寸相同）。
CLASS_ALIASES = {
    "person": ["pedestrian"],
    "people": ["pedestrian"],
    "pedestrian": ["pedestrian"],
    "motorcycle": ["two_wheeler", "motor"],
    "motorbike": ["two_wheeler", "motor"],
    "motor": ["two_wheeler", "motor"],
    "two_wheeler": ["two_wheeler", "motor"],
    "bike": ["bicycle", "two_wheeler", "motor"],
    "bicycle": ["bicycle", "two_wheeler", "motor"],
}


def resolve_dims(prior_map, class_name):
    """Look up dims by class name, then fall back to a synonym alias map.

    prior_map 的鍵須已 lower-case（見 normalize_prior_map）。
    """
    key = str(class_name).strip().lower()
    dims = prior_map.get(key)
    if dims is not None:
        return dims
    for candidate in CLASS_ALIASES.get(key, []):
        if candidate in prior_map:
            return prior_map[candidate]
    return None


def normalize_prior_map(prior_dims):
    """Lower-case the keys of one prior_dimensions.json set."""
    return {str(k).strip().lower(): v for k, v in prior_dims.items()}


# ---------------------------------------------------------------------------
# 場地層級的尺寸覆寫
# ---------------------------------------------------------------------------
# prior_dimensions.json 是全域的類別平均值（car = 1.8×3.8 m 的通用小型車），改它會影響
# 所有場地。但事故分析要的是「這個案子裡這台車」的實際尺寸，而尺寸同時決定 floor box 大小
# 與錨點修正 h(u)，填錯會讓重建位置偏移。所以改成每個場地一份覆寫檔，放在
# location/<loc>/dimensions_<loc>.json，缺檔就維持原本行為。
#
# 格式（兩層都可省略）：
#   {
#     "by_class": { "car": {"width": 1.78, "length": 4.62, "height": 1.47} },
#     "by_track": { "1":   {"width": 1.78, "length": 4.62, "height": 1.47} }
#   }
# 優先序：by_track > by_class > prior_dimensions.json

def load_location_overrides(g_proj_path):
    """讀 location/<loc>/dimensions_<loc>.json。回傳 (by_track, by_class)，缺檔則為空 dict。"""
    d = os.path.dirname(g_proj_path)
    base = os.path.basename(g_proj_path)
    loc = base[len('G_projection_'):-len('.json')] if base.startswith('G_projection_') else ''
    path = os.path.join(d, f'dimensions_{loc}.json')
    if not os.path.isfile(path):
        return {}, {}
    with open(path, 'r') as f:
        data = json.load(f)
    by_track = {int(k): _strip_notes(v)
                for k, v in (data.get('by_track') or {}).items()
                if str(k).lstrip('-').isdigit()}
    by_class = {k: _strip_notes(v)
                for k, v in normalize_prior_map(data.get('by_class') or {}).items()}
    return by_track, by_class


def _strip_notes(dims):
    """濾掉 _note 這類給人看的欄位 —— 尺寸會被原樣複製進匯出檔，註解不該流到下游。"""
    return {k: v for k, v in dims.items() if not str(k).startswith('_')}


def resolve_dims_layered(prior_map, class_name, track_id=None, by_track=None, by_class=None):
    """依 by_track > by_class > prior_map 的順序查尺寸。"""
    if by_track and track_id is not None and track_id in by_track:
        return by_track[track_id]
    if by_class:
        dims = resolve_dims(by_class, class_name)
        if dims is not None:
            return dims
    return resolve_dims(prior_map, class_name)
