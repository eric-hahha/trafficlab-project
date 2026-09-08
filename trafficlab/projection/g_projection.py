import numpy as np
import cv2
import math
import os

from trafficlab.projection.svg_parser import SVGParser

class GProjection:
    def __init__(self, config_data: dict, base_dir: str = "."):
        self.config = config_data
        self.base_dir = base_dir
        self._init_matrices()
        self._init_parallax()
        self._init_svg()
        
    def _init_matrices(self):
        und = self.config.get('undistort', {})
        self.K = np.array(und.get('K'), dtype=np.float64)
        self.D = np.array(und.get('D'), dtype=np.float64)
        self.P = self.K.copy()
        
        hom = self.config.get('homography', {})
        self.H = np.array(hom.get('H'), dtype=np.float64)
        self.H_inv = np.linalg.inv(self.H)
        
    def _init_parallax(self):
        par = self.config.get('parallax', {})
        self.z_cam = par.get('z_cam_meters', 10.0)
        x_sat = par.get('x_cam_coords_sat', 0.0)
        y_sat = par.get('y_cam_coords_sat', 0.0)
        self.cam_sat = np.array([x_sat, y_sat], dtype=np.float64)
        self.cam_sat_valid = bool(np.linalg.norm(self.cam_sat) > 1.0)
        self.px_per_m = par.get('px_per_meter', 1.0)
        if self.px_per_m <= 0.001: self.px_per_m = 1.0

    def _init_svg(self):
        self.svg_parser = None
        if self.config.get('use_svg', False):
            rel_path = self.config.get('inputs', {}).get('layout_path')
            if rel_path:
                full_path = os.path.join(self.base_dir, rel_path)
                align_A = self.config.get('layout_svg', {}).get('A')
                self.svg_parser = SVGParser(full_path, align_A)

    def get_svg_heading(self, sat_pt):
        if self.svg_parser and self.svg_parser.valid:
            return self.svg_parser.get_nearest_heading(sat_pt)
        return None

    # --- TRANSFORMATIONS ---
    def cctv_to_undistorted(self, u, v):
        src = np.array([[[u, v]]], dtype=np.float64)
        dst = cv2.undistortPoints(src, self.K, self.D, P=self.P)
        return tuple(dst[0,0])

    def undistorted_to_flat_sat(self, u_u, v_u):
        src = np.array([[[u_u, v_u]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, self.H)
        return tuple(dst[0,0])

    def flat_sat_to_undistorted(self, x, y):
        src = np.array([[[x, y]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, self.H_inv)
        return tuple(dst[0,0])
    
    def undistorted_to_cctv(self, u_u, v_u):
        fx, fy = self.K[0,0], self.K[1,1]
        cx, cy = self.K[0,2], self.K[1,2]
        x_norm = (u_u - cx) / fx
        y_norm = (v_u - cy) / fy
        obj_pts = np.array([[[x_norm, y_norm, 1.0]]], dtype=np.float64)
        img_pts, _ = cv2.projectPoints(obj_pts, (0,0,0), (0,0,0), self.K, self.D)
        return tuple(img_pts[0,0])

    # --- PARALLAX ---
    def parallax_correct_ground_to_real(self, apparent_sat_pt, h):
        if self.z_cam == 0: return apparent_sat_pt
        A = np.array(apparent_sat_pt); C = self.cam_sat
        factor = (self.z_cam - h) / self.z_cam
        real_pt = C + (A - C) * factor
        return tuple(real_pt)

    def parallax_project_real_to_ground(self, real_sat_pt, h):
        if abs(self.z_cam - h) < 0.01: return real_sat_pt 
        R = np.array(real_sat_pt); C = self.cam_sat
        factor = self.z_cam / (self.z_cam - h)
        apparent_pt = C + (R - C) * factor
        return tuple(apparent_pt)

    def cctv_to_sat(self, u, v, h=0.0):
        u_u, v_u = self.cctv_to_undistorted(u, v)
        apparent_pt = self.undistorted_to_flat_sat(u_u, v_u)
        if h != 0: return self.parallax_correct_ground_to_real(apparent_pt, h)
        return apparent_pt

    def sat_to_cctv(self, x, y, h=0.0):
        if h != 0: flat_pt = self.parallax_project_real_to_ground((x, y), h)
        else: flat_pt = (x, y)
        u_u, v_u = self.flat_sat_to_undistorted(flat_pt[0], flat_pt[1])
        return self.undistorted_to_cctv(u_u, v_u)

    def get_ground_contact_from_box(self, rect, h_meters, ref_method="center_bottom_side", proj_method="down_h"):
        if hasattr(rect, 'x'): rx, ry, rw, rh = rect.x(), rect.y(), rect.width(), rect.height()
        else: rx, ry, rw, rh = rect
        cx = rx + rw/2
        if ref_method == "center_bottom_side": cy = ry + rh
        else: cy = ry + rh/2
            
        apparent_sat = self.cctv_to_sat(cx, cy, h=0)
        if proj_method == "down_h": final_sat = self.parallax_correct_ground_to_real(apparent_sat, h_meters)
        elif proj_method == "down_h_2": final_sat = self.parallax_correct_ground_to_real(apparent_sat, h_meters / 2.0)
        elif proj_method == "match": final_sat = apparent_sat  # no parallax correction
        else: final_sat = apparent_sat  # unrecognised value: same as "match"

        gc_cctv = self.sat_to_cctv(final_sat[0], final_sat[1], h=0)
        return { "sat_coords": final_sat, "cctv_ref_point": (cx, cy), "cctv_ground_point": gc_cctv }

    def footprint_anchor_to_center(self, sat_pt, heading_deg, width_m, length_m):
        """把「面向相機那一側的接地輪廓點」還原成 footprint 幾何中心。

        mask 底部中心取的是影像最低點；在掠射視角（本專案的路口監視器仰角約 5°~23°）
        下它等同於 footprint 上離相機最近的邊，而不是幾何中心。直接拿它當 floor box
        的中心會讓整個框往相機方向偏，偏移量正好是矩形在視線方向上的支撐距離
        h(u) = |u·前|·L/2 + |u·右|·W/2，所以沿視線往內推回 h(u) 即可還原。

        heading_deg 為 None（尚未取得朝向）時改用朝向均勻分布下的期望支撐距離
        (L+W)/π。方向 u 只由相機幾何決定、不需要朝向，所以這個 fallback 讓修正從
        第一幀就生效；等 heading 出現時 h(u) 只會微調（汽車約 0.05 m），不會像原本
        那樣在 heading 首次出現的那一幀一次補上整段位移。

        回傳 None 表示條件不足（相機座標未校正、或物件幾乎貼在相機上導致方向不可靠），
        呼叫端應維持原本的座標。
        """
        if not self.cam_sat_valid:
            return None

        p = np.array(sat_pt, dtype=np.float64)
        u = self.cam_sat - p
        dist = np.linalg.norm(u)
        if dist < self.px_per_m:   # 不到 1 公尺，視線方向已無意義
            return None
        u = u / dist

        if heading_deg is None:
            support_m = (length_m + width_m) / math.pi
        else:
            rad = math.radians(heading_deg)
            fwd = np.array([math.cos(rad), math.sin(rad)], dtype=np.float64)
            right = np.array([-math.sin(rad), math.cos(rad)], dtype=np.float64)
            support_m = abs(float(u @ fwd)) * length_m / 2.0 + abs(float(u @ right)) * width_m / 2.0

        return (p - u * support_m * self.px_per_m).tolist()

    def sat_floor_to_cctv_3d(self, sat_poly, obj_height_m):
        """
        Lifts a SAT floor polygon (4 points) to a 3D box (8 points) in CCTV pixel space.
        Matches legacy 'produce_stage4.py' logic.
        """
        # 1. Project Floor points (Z=0) from SAT -> CCTV Undistorted
        # Logic: SAT -> Flat -> Undistorted
        undistorted_ground_pts = []
        for p in sat_poly:
            flat_pt = p # On ground, flat = real
            u_u, v_u = self.flat_sat_to_undistorted(flat_pt[0], flat_pt[1])
            undistorted_ground_pts.append([u_u, v_u])
        
        # 2. Project Ceiling points (Z=h)
        # Logic: Parallax shift on SAT plane, then map back
        undistorted_top_pts = []
        if abs(self.z_cam - obj_height_m) < 0.01: factor = 1.0
        else: factor = self.z_cam / (self.z_cam - obj_height_m)
        
        c_sat = self.cam_sat
        
        for p in sat_poly:
            p_arr = np.array(p)
            # Find where the 'head' would appear on the ground plane (apparent position)
            # Formula reversed from correction: Apparent = C + (Real - C) * factor
            apparent_pt = c_sat + (p_arr - c_sat) * factor
            
            u_u, v_u = self.flat_sat_to_undistorted(apparent_pt[0], apparent_pt[1])
            undistorted_top_pts.append([u_u, v_u])
            
        # Combine: 4 Bottom, 4 Top
        all_undist = np.array(undistorted_ground_pts + undistorted_top_pts, dtype=np.float64)
        
        # 3. Apply Distortion (CCTV Intrinsics)
        # Convert to Normalized Ray: (u-cx)/fx
        fx, fy = self.K[0,0], self.K[1,1]
        cx, cy = self.K[0,2], self.K[1,2]
        
        obj_pts = []
        for pt in all_undist:
            x_n = (pt[0] - cx) / fx
            y_n = (pt[1] - cy) / fy
            obj_pts.append([x_n, y_n, 1.0])
            
        obj_pts = np.array(obj_pts, dtype=np.float64)
        rvec = np.zeros(3); tvec = np.zeros(3)
        
        distorted, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.K, self.D)
        return distorted.reshape(-1, 2).tolist()
        return distorted.reshape(-1, 2).tolist()
