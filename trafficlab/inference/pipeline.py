import os
import json
import logging
import yaml
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from trafficlab.projection.g_projection import GProjection
from trafficlab.motion.kinematics import TrackSmoother, MotorcycleLateralCorrector
from trafficlab.motion.segmentation_car import mask_to_polygon, polygon_tight_bbox
from trafficlab.io.replay_writer import ReplayWriter


class InferencePipeline:

    SEG_OUTPUT_ROOT = "output/yolobox-seg"

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

    @staticmethod
    def config_output_dir(output_root, weights, tracker_type, config_name, is_seg=False):
        model_name = Path(weights).stem
        root = InferencePipeline.SEG_OUTPUT_ROOT if is_seg else output_root
        return os.path.join(root, f"model-{model_name}_{tracker_type}", config_name)

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
        prior_dims_norm = {k.strip().lower(): v for k, v in prior_dims.items()}

        # Segmentation-based tight-box localization (model.type: seg)
        seg_class_map = full_config['model'].get('classes')
        is_seg = full_config['model'].get('type') == 'seg'
        if is_seg and not seg_class_map:
            raise ValueError(
                "model.type is 'seg' but model.classes (COCO name -> internal class name map) is missing in config"
            )

        # Ground-projection reference point + parallax method.
        # Resolved from the inference config first, then the G_projection file,
        # then the historical defaults. Lets one G_projection serve configs that
        # need different projection behaviour (e.g. seg tight-box wanting
        # proj_method: match instead of the bbox-oriented down_h).
        proj_cfg = full_config.get('projection', {}) or {}
        ref_method = proj_cfg.get('ref_method', g_data.get('ref_method', 'center_bottom_side'))
        proj_method = proj_cfg.get('proj_method', g_data.get('proj_method', 'down_h'))
        ref_src = 'config' if 'ref_method' in proj_cfg else ('G_projection' if 'ref_method' in g_data else 'default')
        proj_src = 'config' if 'proj_method' in proj_cfg else ('G_projection' if 'proj_method' in g_data else 'default')
        self.log_fn(
            f"Projection: ref_method={ref_method} (from {ref_src}), "
            f"proj_method={proj_method} (from {proj_src})"
        )

        # Output Setup
        footage_name = os.path.basename(self.footage_path)
        # use chosen config_name for folder naming (already set above)
        model_name = Path(full_config['model']['weights']).stem
        tracking_cfg = full_config.get('tracking', {})
        tracker_type = tracking_cfg.get('tracker_type', 'default')

        config_dir = InferencePipeline.config_output_dir(
            self.output_root, full_config['model']['weights'], tracker_type, config_name, is_seg=is_seg
        )
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
        tracker_cfg_path = None
        if tracker_type != 'default':
            tracker_cfg_path = self._build_tracker_config(tracking_cfg, config_dir)
            self.log_fn(f"Using tracker config: {tracker_cfg_path}")
        else:
            self.log_fn("Using Ultralytics default tracker behavior (tracker_type: default).")

        # Tracking State
        track_smoothers = {} # tid -> Smoother
        last_seen_frame = {} # tid -> int

        out_data = {
            "mp4_path": self.footage_path,
            "meta": {"resolution": [real_w, real_h], "fps": real_fps, "config_name": config_name},
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
        if is_seg:
            # Aligns r.masks.data to orig_shape (letterbox-unaware resize otherwise
            # misaligns mask pixels against the box/ROI pipeline's frame coordinates).
            track_kwargs["retina_masks"] = True

        # Optional diagnostic pass: log all raw detections at low conf for comparison.
        # Enable via `debug: {conf_log: true}` in the inference config. Off by default.
        enable_conf_log = full_config.get('debug', {}).get('conf_log', False)
        process_conf = full_config['model']['conf']
        _conf_tracked = None
        _h_tracked = None
        if enable_conf_log:
            _all_log_path = os.path.join(out_subdir, f"{Path(footage_name).stem}_conf_all.log")
            _conf_all = logging.getLogger(f"conf_all_{id(self)}")
            _conf_all.setLevel(logging.INFO)
            _conf_all.propagate = False
            _h_all = logging.FileHandler(_all_log_path, mode="w", encoding="utf-8")
            _h_all.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            _conf_all.addHandler(_h_all)

            _predict_kwargs = {
                "source": self.footage_path,
                "device": full_config['model']['device'],
                "verbose": False,
                "stream": True,
                "conf": 0.02,
                "iou": full_config['model']['iou'],
                "imgsz": full_config['model']['imgsz'],
                "max_det": full_config['model'].get('max_det', 300),
                "agnostic_nms": full_config['model'].get('agnostic_nms', False),
                "half": full_config['model'].get('half', False),
            }
            self.log_fn("Debug: running detection pass for conf logging...")
            for _i, _r in enumerate(model.predict(**_predict_kwargs)):
                if max_frame > 0 and _i >= max_frame:
                    break
                _confs = _r.boxes.conf.cpu().numpy()
                _cls_ids = _r.boxes.cls.cpu().numpy()
                for _j in range(len(_r.boxes)):
                    _cls = _r.names[int(_cls_ids[_j])]
                    _conf = float(_confs[_j])
                    _tag = " [BELOW_THRESHOLD]" if _conf < process_conf else ""
                    _conf_all.info(f"frame={_i} tid=None cls={_cls} conf={_conf:.4f}{_tag}")
            _conf_all.removeHandler(_h_all)
            _h_all.close()

            _tracked_log_path = os.path.join(out_subdir, f"{Path(footage_name).stem}_conf_tracked.log")
            _conf_tracked = logging.getLogger(f"conf_tracked_{id(self)}")
            _conf_tracked.setLevel(logging.INFO)
            _conf_tracked.propagate = False
            _h_tracked = logging.FileHandler(_tracked_log_path, mode="w", encoding="utf-8")
            _h_tracked.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            _conf_tracked.addHandler(_h_tracked)

        results = model.track(**track_kwargs)

        for i, r in enumerate(results):
            if self.stop_flag_fn() or (max_frame > 0 and i >= max_frame): break

            frame_objects = []
            boxes = r.boxes.xyxy.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None]*len(boxes)
            masks_data = r.masks.data.cpu().numpy() if (is_seg and r.masks is not None) else None

            for j, box in enumerate(boxes):
                cls_name = r.names[int(cls_ids[j])]

                mask_contour = None  # image-space [[x,y],...] polygon, seg mode only
                if is_seg:
                    if masks_data is not None and j < len(masks_data):
                        mask_contour = mask_to_polygon(masks_data[j].astype(bool))
                        tight = polygon_tight_bbox(mask_contour)
                        if tight is not None:
                            box = np.array(tight, dtype=box.dtype)
                        else:
                            self.log_fn(f"frame={i}: empty seg mask for detection {j} (class={cls_name}), falling back to detector box")
                    else:
                        self.log_fn(f"frame={i}: no seg mask returned for detection {j} (class={cls_name}), falling back to detector box")

                    internal_name = seg_class_map.get(cls_name)
                    if internal_name is None:
                        continue
                    cls_name = internal_name

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
                if _conf_tracked is not None:
                    _conf_tracked.info(f"frame={i} tid={int(track_ids[j]) if track_ids[j] is not None else None} cls={cls_name} conf={float(confs[j]):.4f}")
                dims = prior_dims_norm.get(cls_name.strip().lower())
                have_measurements = (dims is not None)
                h_real = float(dims.get('height', 0.0)) if have_measurements else 0.0

                bx1, by1, bx2, by2 = box
                proj_res = g_engine.get_ground_contact_from_box(
                    (bx1, by1, bx2-bx1, by2-by1), h_real,
                    ref_method=ref_method,
                    proj_method=proj_method
                )
                sat_coords = proj_res['sat_coords']

                # 3. Kinematics
                tid = int(track_ids[j]) if track_ids[j] is not None else None
                heading = None
                speed = 0.0
                is_def = False

                if tid is not None:
                    if tid not in track_smoothers:
                        # 根據車輛類別選擇合適的軌跡平滑器
                        vehicle_class = cls_name.strip().lower()
                        kinematics_config = full_config['kinematics']
                        
                        # 檢查是否啟用橫向修正且為機車類別
                        lateral_config = kinematics_config.get('lateral_correction', {})
                        if (lateral_config.get('enabled', False) and
                            vehicle_class in lateral_config.get('vehicle_classes', ['motor', 'two_wheeler'])):
                            track_smoothers[tid] = MotorcycleLateralCorrector(kinematics_config, vehicle_class)
                        else:
                            track_smoothers[tid] = TrackSmoother(kinematics_config)
                        
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

                # 3b. Footprint anchor -> geometric centre (seg tight-box only)
                # mask 底部中心落在「面向相機那一側」的接地輪廓上，不是 footprint 中心；
                # 拿它當 floor box 中心會讓整個框往相機方向偏。放在 smoother 之後：
                # 這個偏移對同一台車緩慢變化，不會污染速度與 heading。不以 have_heading
                # 為條件——否則 heading 首次出現的那一幀會一次補上整段位移；heading 為
                # None 時 footprint_anchor_to_center 內建等向 fallback。
                if is_seg and have_measurements:
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
        if _conf_tracked is not None and _h_tracked is not None:
            _conf_tracked.removeHandler(_h_tracked)
            _h_tracked.close()

        out_data["animation_frame_count"] = i
        out_filename = f"{os.path.splitext(footage_name)[0]}.json.gz"
        out_path = os.path.join(out_subdir, out_filename)

        # Write gzip-compressed JSON so viewer can load .json.gz
        ReplayWriter.write(out_path, out_data)
