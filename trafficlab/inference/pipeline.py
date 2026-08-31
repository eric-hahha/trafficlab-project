import os
import json
import yaml
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from trafficlab.projection.g_projection import GProjection
from trafficlab.motion.kinematics import TrackSmoother, KalmanTrackSmoother, MotorcycleLateralCorrector
from trafficlab.io.prior_dimensions import (
    load_location_overrides,
    normalize_prior_map,
    resolve_dims_layered,
)
import logging

# 設置日誌記錄
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('motorcycle_correction.log'),
        logging.StreamHandler()
    ]
)
from trafficlab.io.replay_writer import ReplayWriter


def _get_mask_bottom_center(mask_xy, tolerance_px=3):
    """從 mask 輪廓點取得底部中心點（最大 y 值附近的平均 x）。"""
    if mask_xy is None or len(mask_xy) == 0:
        return None
    pts = np.array(mask_xy)
    max_y = pts[:, 1].max()
    bottom_pts = pts[pts[:, 1] >= max_y - tolerance_px]
    return (float(bottom_pts[:, 0].mean()), float(max_y))


class InferencePipeline:

    def __init__(
        self,
        location_code,
        footage_path,
        config_path,
        output_root,
        g_proj_path,
        config_name=None,
        log_fn=None,
        progress_fn=None,
        stop_flag_fn=None
    ):
        self.loc_code = location_code
        self.footage_path = footage_path
        self.config_path = config_path
        self.config_name = config_name
        self.output_root = output_root
        self.g_proj_path = g_proj_path
        self.log_fn = log_fn or (lambda msg: None)
        self.progress_fn = progress_fn or (lambda pct: None)
        self.stop_flag_fn = stop_flag_fn or (lambda: False)

    def _build_tracker_config(self, tracking_cfg, output_dir):
        tracker_type = (tracking_cfg or {}).get('tracker_type', 'bytetrack')
        tracker_yaml = {
            "tracker_type": tracker_type,
            "track_high_thresh": tracking_cfg.get('track_high_thresh', 0.25),
            "track_low_thresh": tracking_cfg.get('track_low_thresh', 0.1),
            "new_track_thresh": tracking_cfg.get('new_track_thresh', 0.25),
            "match_thresh": tracking_cfg.get('match_thresh', 0.8),
            "track_buffer": tracking_cfg.get('track_buffer', 30),
            "fuse_score": tracking_cfg.get('fuse_score', True),
        }

        if tracker_type == "botsort":
            tracker_yaml.update({
                "gmc_method": tracking_cfg.get("gmc_method", "sparseOptFlow"),
                "proximity_thresh": tracking_cfg.get("proximity_thresh", 0.5),
                "appearance_thresh": tracking_cfg.get("appearance_thresh", 0.8),
                "with_reid": tracking_cfg.get("with_reid", False),
                "model": tracking_cfg.get("model", "auto"),
            })
        else:
            # Keep any extra tracker params if a config provides them.
            for key in ("gmc_method", "proximity_thresh", "appearance_thresh", "with_reid", "model"):
                if key in tracking_cfg:
                    tracker_yaml[key] = tracking_cfg[key]

        tracker_path = os.path.join(output_dir, f"{tracker_type}_tracker.yaml")
        with open(tracker_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(tracker_yaml, f, sort_keys=False)
        return tracker_path

    def run(self):
        # 1. Load Configs (support new YAML that contains 'configs')
        with open(self.config_path, 'r') as f:
            raw_cfg = yaml.safe_load(f)

        # If YAML contains a 'configs' mapping, pick the selected one
        if isinstance(raw_cfg, dict) and 'configs' in raw_cfg and isinstance(raw_cfg['configs'], dict):
            configs_map = raw_cfg['configs']
            if self.config_name and self.config_name in configs_map:
                full_config = configs_map[self.config_name]
                config_name = self.config_name
            else:
                # fallback to first config key
                first_key = next(iter(configs_map.keys()))
                full_config = configs_map[first_key]
                config_name = first_key
        else:
            full_config = raw_cfg or {}
            config_name = full_config.get('config_name', 'default')

        with open(self.g_proj_path, 'r') as f: g_data = json.load(f)

        g_proj_dir = os.path.dirname(self.g_proj_path)
        g_engine = GProjection(g_data, base_dir=g_proj_dir)

        # Load Priors (normalize keys for case-insensitive lookups)
        with open("prior_dimensions.json", 'r') as f: all_priors = json.load(f)
        measure_set = full_config.get('prior_dimensions', 'measurements_visdrone')
        prior_dims = all_priors.get(measure_set, {})
        prior_dims_norm = normalize_prior_map(prior_dims)

        # 場地層級的尺寸覆寫（location/<loc>/dimensions_<loc>.json），缺檔則為空
        dim_by_track, dim_by_class = load_location_overrides(self.g_proj_path)
        if dim_by_track or dim_by_class:
            self.log_fn(f"Dimension overrides: {len(dim_by_track)} by track, {len(dim_by_class)} by class")

        # Output Setup
        footage_name = os.path.basename(self.footage_path)
        # use chosen config_name for folder naming (already set above)
        model_name = Path(full_config['model']['weights']).stem
        tracking_cfg = full_config.get('tracking', {})
        tracker_type = tracking_cfg.get('tracker_type', 'default')

        config_dir = os.path.join(self.output_root, f"model-{model_name}_tracker-{tracker_type}", config_name)
        os.makedirs(config_dir, exist_ok=True)
        out_subdir = os.path.join(config_dir, self.loc_code)
        os.makedirs(out_subdir, exist_ok=True)

        # Video Init
        cap = cv2.VideoCapture(self.footage_path)
        real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ROI Logic
        use_roi = g_data.get('use_roi', False)
        roi_mask = None
        if use_roi:
            roi_rel = g_data['inputs'].get('roi_path')
            if roi_rel:
                roi_p = os.path.join(g_proj_dir, roi_rel)
                if os.path.exists(roi_p):
                    img = cv2.imread(roi_p, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        if img.shape[:2] != (real_h, real_w):
                            img = cv2.resize(img, (real_w, real_h), interpolation=cv2.INTER_NEAREST)
                        if img.ndim == 3 and img.shape[2] == 4:
                            roi_mask = (cv2.bitwise_not(img[:,:,3]) > 128)
                        elif img.ndim == 3:
                            roi_mask = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 10)

        roi_method = g_data.get('roi_method', 'partial')

        # Model Init
        self.log_fn(f"Loading Model: {full_config['model']['weights']}")
        model = YOLO(full_config['model']['weights'])
        use_explicit_tracker_config = tracking_cfg.get('use_explicit_tracker_config', True)
        tracker_cfg_path = None
        if use_explicit_tracker_config:
            tracker_cfg_path = self._build_tracker_config(tracking_cfg, config_dir)
            self.log_fn(f"Using tracker config: {tracker_cfg_path}")
        else:
            self.log_fn("Using Ultralytics default tracker behavior (no explicit tracker config).")

        # Tracking State
        track_smoothers = {} # tid -> Smoother
        last_seen_frame = {} # tid -> int

        out_data = {
            "mp4_path": self.footage_path,
            "meta": {"resolution": [real_w, real_h], "fps": real_fps},
            "location_code": self.loc_code,
            "mp4_frame_count": total_frames,
            "frames": []
        }

        use_svg = g_data.get('use_svg', False)
        max_frame = full_config.get('frames', {}).get('max_frame', -1)
        frames_to_process = min(total_frames, max_frame) if max_frame > 0 else total_frames

        # Run Loop
        track_kwargs = {
            "source": self.footage_path,
            "device": full_config['model']['device'],
            "persist": tracking_cfg.get('persist', True),
            "verbose": full_config['model'].get('verbose', False),
            "stream": True,
            "conf": full_config['model']['conf'],
            "iou": full_config['model']['iou'],
            "imgsz": full_config['model']['imgsz'],
            "max_det": full_config['model'].get('max_det', 300),
            "agnostic_nms": full_config['model'].get('agnostic_nms', False),
            "half": full_config['model'].get('half', False),
        }
        if tracker_cfg_path is not None:
            track_kwargs["tracker"] = tracker_cfg_path

        results = model.track(**track_kwargs)

        for i, r in enumerate(results):
            if self.stop_flag_fn() or (max_frame > 0 and i >= max_frame): break

            frame_objects = []
            boxes = r.boxes.xyxy.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None]*len(boxes)
            masks_xy = r.masks.xy if r.masks is not None else None

            for j, box in enumerate(boxes):
                # 1. ROI Check
                if roi_mask is not None:
                    x1, y1, x2, y2 = map(int, box)
                    x1 = max(0, min(real_w-1, x1)); y1 = max(0, min(real_h-1, y1))
                    x2 = max(0, min(real_w-1, x2)); y2 = max(0, min(real_h-1, y2))
                    if x1 >= x2 or y1 >= y2: continue

                    if roi_method == 'in':
                        corns = [(x1,y1), (x2,y1), (x1,y2), (x2,y2)]
                        if not all([roi_mask[cy, cx] for cx, cy in corns]): continue
                    else:
                        if np.count_nonzero(roi_mask[y1:y2, x1:x2]) == 0: continue

                # 2. Projection
                cls_name = r.names[int(cls_ids[j])]
                tid = int(track_ids[j]) if track_ids[j] is not None else None
                dims = resolve_dims_layered(prior_dims_norm, cls_name, tid, dim_by_track, dim_by_class)
                have_measurements = (dims is not None)
                h_real = float(dims.get('height', 0.0)) if have_measurements else 0.0

                bx1, by1, bx2, by2 = box
                mask_ref = None
                mask_contour = None
                if masks_xy is not None and j < len(masks_xy) and len(masks_xy[j]) > 0:
                    mask_ref = _get_mask_bottom_center(masks_xy[j])
                    mask_contour = masks_xy[j].tolist()
                    if mask_ref is not None and mask_ref[1] >= real_h - 5:
                        mask_ref = None  # mask 延伸至畫面底部邊緣，投影不可靠，fallback 至 bbox
                # mask 底部中心已是地面接觸點 (h=0)，不需再套視差修正 (down_h/down_h_2)，
                # 否則會把落地點重複拉向相機而偏移。僅在退回 bbox 參考點時沿用 config 的 proj_method
                # (該參考點約在半車高，才需要修正)。
                proj_method = 'none' if mask_ref is not None else g_data.get('proj_method', 'down_h')
                proj_res = g_engine.get_ground_contact_from_box(
                    (bx1, by1, bx2-bx1, by2-by1), h_real,
                    ref_method=g_data.get('ref_method', 'center_bottom_side'),
                    proj_method=proj_method,
                    ref_point=mask_ref
                )
                sat_coords = proj_res['sat_coords']

                # 3. Kinematics (tid 已在尺寸查詢時取得)
                heading = None
                speed = 0.0
                is_def = False

                if tid is not None:
                    if tid not in track_smoothers:
                        vehicle_class = cls_name.strip().lower()
                        kinematics_config = full_config['kinematics']
                        use_kalman = kinematics_config.get('use_kalman', False)  # default off; enable in inference_config.yaml
                        lateral_config = kinematics_config.get('lateral_correction', {})

                        if use_kalman:
                            track_smoothers[tid] = KalmanTrackSmoother(kinematics_config)
                            logging.info(f"創建KalmanTrackSmoother for 車輛類別: {vehicle_class}, track_id: {tid}")
                        elif (lateral_config.get('enabled', False) and
                              vehicle_class in lateral_config.get('vehicle_classes', ['motor', 'two_wheeler'])):
                            track_smoothers[tid] = MotorcycleLateralCorrector(kinematics_config, vehicle_class)
                            logging.info(f"創建MotorcycleLateralCorrector for 車輛類別: {vehicle_class}, track_id: {tid}")
                        else:
                            track_smoothers[tid] = TrackSmoother(kinematics_config)
                            logging.info(f"創建標準TrackSmoother for 車輛類別: {vehicle_class}, track_id: {tid}")
                        
                        last_seen_frame[tid] = i - 1

                    svg_h = g_engine.get_svg_heading(sat_coords) if use_svg else None

                    prev_f = last_seen_frame.get(tid, i-1)
                    dt = (i - prev_f) / real_fps
                    if dt <= 0: dt = 1.0/real_fps

                    # Pass px_per_m to smoother
                    k_res = track_smoothers[tid].update(sat_coords, dt, g_engine.px_per_m, svg_heading=svg_h)
                    last_seen_frame[tid] = i

                    speed = k_res['speed_kmh']
                    heading = k_res['heading']
                    is_def = k_res['default_heading']

                    corrected_sat_coords = k_res.get('corrected_position')
                    if corrected_sat_coords is not None:
                        sat_coords = corrected_sat_coords

                have_heading = (heading is not None)
                if not have_heading: speed = 0.0

                # 3b. Footprint anchor -> geometric centre
                # mask 底部中心落在「面向相機那一側」的接地輪廓上，不是 footprint 中心；
                # 拿它當 floor box 中心會讓整個框往相機方向偏（Hsinchu1 的碰撞因此從
                # 側撞變成前撞）。只在 mask 路徑修正 —— bbox 路徑的 down_h/down_h_2
                # 已經沿視線做過另一種位移，疊加會過度修正。
                # 放在 smoother 之後：這個偏移對同一台車緩慢變化，不會污染速度與 heading。
                # 不以 have_heading 為條件 —— 否則 heading 首次出現的那一幀會一次補上
                # 整段位移（汽車約 1.7 m 的橫移）；heading 為 None 時走等向 fallback。
                if mask_ref is not None and have_measurements:
                    centered = g_engine.footprint_anchor_to_center(
                        sat_coords, heading, dims['width'], dims['length']
                    )
                    if centered is not None:
                        sat_coords = centered

                # 4. 3D Lifting
                sat_floor_box = None
                bbox_3d = None

                if have_heading and have_measurements:
                    w_m, l_m = dims['width'], dims['length']
                    px_m = g_engine.px_per_m

                    ang = np.radians(heading)
                    c, s = np.cos(ang), np.sin(ang)
                    dx, dy = (l_m * px_m)/2, (w_m * px_m)/2
                    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
                    R = np.array([[c, -s], [s, c]])

                    sat_floor_box = (corners @ R.T + sat_coords).tolist()
                    bbox_3d = g_engine.sat_floor_to_cctv_3d(sat_floor_box, h_real)

                obj_data = {
                    "id": j,
                    "tracked_id": tid,
                    "class": cls_name,
                    "confidence": float(confs[j]),
                    "bbox_2d": [float(x) for x in box],
                    "mask_contour": mask_contour,
                    "reference_point": proj_res['cctv_ref_point'],
                    "sat_coords": sat_coords,
                    "have_heading": have_heading,
                    "have_measurements": have_measurements,
                    "default_heading": is_def,
                    "heading": heading,
                    "speed_kmh": speed,
                    "sat_floor_box": sat_floor_box,
                    "bbox_3d": bbox_3d
                }
                frame_objects.append(obj_data)

            out_data['frames'].append({"frame_index": i, "objects": frame_objects})
            self.progress_fn(int((i / frames_to_process) * 100))

        cap.release()

        out_data["animation_frame_count"] = i
        out_filename = f"{os.path.splitext(footage_name)[0]}.json.gz"
        out_path = os.path.join(out_subdir, out_filename)

        # Write gzip-compressed JSON so viewer can load .json.gz
        ReplayWriter.write(out_path, out_data)
