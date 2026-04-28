import logging
import math
from collections import deque

import cv2
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('motorcycle_correction.log'),
        logging.StreamHandler()
    ]
)


class TrackSmoother:
    def __init__(self, config: dict):
        self.cfg = config

        # State
        self.speed_state = None
        self.heading_vec_state = None
        self.velocity_vec_state = None

        # Params
        h_ema = self.cfg.get('heading_ema', {})
        self.h_alpha_min = h_ema.get('alpha_min', 0.05)
        self.h_alpha_max = h_ema.get('alpha_max', 0.6)
        self.h_speed_ref = h_ema.get('speed_ref', 5.0)

        # History
        reg_win = 8
        self.pos_history = deque(maxlen=reg_win)
        jitter_frames = self.cfg.get('heading_sat_coords_jitter_frames', 8)
        self.jitter_hist = deque(maxlen=jitter_frames)

        self.cosine_reject_counter = 0
        self.max_physics_speed = 200.0

    def update(self, current_sat_pos: list, dt: float, px_per_m: float, svg_heading: float = None) -> dict:
        curr_pt = np.array(current_sat_pos, dtype=float)

        if dt > 0 and len(self.pos_history) > 0:
            prev_pt = np.array(self.pos_history[-1], dtype=float)
            dist_px = np.linalg.norm(curr_pt - prev_pt)
            dist_m = dist_px / px_per_m
            inst_speed = (dist_m / dt) * 3.6

            if inst_speed > self.max_physics_speed:
                return {
                    "speed_kmh": self.speed_state if self.speed_state else 0.0,
                    "heading": self._vec_to_deg(self.heading_vec_state) if self.heading_vec_state is not None else None,
                    "default_heading": False
                }

        self.pos_history.append(curr_pt.tolist())
        self.jitter_hist.append(curr_pt.tolist())

        is_jittering = self._check_jitter(px_per_m)

        raw_speed_kmh = 0.0
        raw_heading_deg = None

        if dt > 0 and len(self.pos_history) >= 2:
            prev_pt = np.array(self.pos_history[-2], dtype=float)
            dist_px = np.linalg.norm(curr_pt - prev_pt)
            dist_m = dist_px / px_per_m
            raw_speed_kmh = (dist_m / dt) * 3.6

            reg_heading = self._compute_regression_heading()
            if reg_heading is not None:
                raw_heading_deg = reg_heading
            elif dist_px > 0.5:
                dx, dy = curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1]
                raw_heading_deg = (math.degrees(math.atan2(dy, dx)) + 360) % 360

        self.speed_state = self._smooth_speed(raw_speed_kmh)

        min_speed = self.cfg.get('heading_min_speed_for_update', 0.1)
        final_heading = None
        is_default = False

        if raw_speed_kmh > min_speed and not is_jittering and raw_heading_deg is not None:
            final_heading = self._smooth_heading(raw_heading_deg, raw_speed_kmh)

            if svg_heading is not None:
                final_heading = self._apply_snapping(final_heading, svg_heading)
        else:
            if self.heading_vec_state is not None:
                final_heading = self._vec_to_deg(self.heading_vec_state)
            elif svg_heading is not None:
                final_heading = svg_heading
                is_default = True

        return {
            "speed_kmh": self.speed_state if self.speed_state else 0.0,
            "heading": final_heading,
            "default_heading": is_default
        }

    def _check_jitter(self, px_per_m):
        if len(self.jitter_hist) < 2:
            return False
        rad_m = self.cfg.get('heading_sat_coords_jitter_radius', 0.6)
        rad_px = rad_m * px_per_m
        pts = np.array(self.jitter_hist, dtype=float)
        return np.linalg.norm(np.max(pts, axis=0) - np.min(pts, axis=0)) < rad_px

    def _smooth_speed(self, raw):
        alpha = self.cfg.get('speed_ema_alpha', 0.4)
        if self.speed_state is None:
            return raw
        return alpha * raw + (1 - alpha) * self.speed_state

    def _compute_regression_heading(self):
        if len(self.pos_history) < 3:
            return None
        pts = np.array(self.pos_history, dtype=np.float32)
        if np.linalg.norm(pts[-1] - pts[0]) < 2.0:
            return None
        [vx, vy, _, _] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        disp = pts[-1] - pts[0]
        if np.dot(disp, [vx[0], vy[0]]) < 0:
            vx, vy = -vx, -vy
        return self._vec_to_deg([vx[0], vy[0]])

    def _smooth_heading(self, raw_deg, speed_kmh):
        rad = math.radians(raw_deg)
        curr_vec = np.array([math.cos(rad), math.sin(rad)], dtype=float)

        vel_alpha = 0.25
        if self.velocity_vec_state is None:
            self.velocity_vec_state = curr_vec
        else:
            self.velocity_vec_state = vel_alpha * curr_vec + (1 - vel_alpha) * self.velocity_vec_state
            self.velocity_vec_state /= max(np.linalg.norm(self.velocity_vec_state), 1e-6)
        curr_vec = self.velocity_vec_state

        if self.heading_vec_state is None:
            self.heading_vec_state = curr_vec
            return self._vec_to_deg(curr_vec)

        speed_ms = speed_kmh / 3.6
        ratio = min(1.0, speed_ms / self.h_speed_ref)
        alpha = self.h_alpha_min + (self.h_alpha_max - self.h_alpha_min) * ratio

        new_vec = alpha * curr_vec + (1 - alpha) * self.heading_vec_state
        new_vec /= max(np.linalg.norm(new_vec), 1e-6)

        target_deg = self._vec_to_deg(new_vec)
        prev_deg = self._vec_to_deg(self.heading_vec_state)
        delta = (target_deg - prev_deg + 180) % 360 - 180

        max_jump = self.cfg.get('heading_max_jump', 5)
        if abs(delta) > max_jump:
            delta = np.clip(delta, -max_jump, max_jump)
            target_deg = (prev_deg + delta + 360) % 360

        r_final = math.radians(target_deg)
        self.heading_vec_state = np.array([math.cos(r_final), math.sin(r_final)], dtype=float)
        return target_deg

    def _apply_snapping(self, current_heading, svg_heading):
        diff = (current_heading - svg_heading + 180) % 360 - 180
        if abs(diff) < 15:
            return (current_heading - diff * 0.3 + 360) % 360
        return current_heading

    def _vec_to_deg(self, vec):
        return (math.degrees(math.atan2(vec[1], vec[0])) + 360) % 360


class TrajectoryAnalyzer:
    def calculate_displacement_components(self, pos_history: deque, current_pos: np.ndarray) -> tuple:
        if len(pos_history) < 2:
            return 0.0, 0.0, np.array([1.0, 0.0], dtype=float)

        prev_pos = np.array(pos_history[-1], dtype=float)
        movement_vec = current_pos - prev_pos

        if len(pos_history) >= 3:
            start_pos = np.array(pos_history[-3], dtype=float)
            heading_vec = prev_pos - start_pos
            if np.linalg.norm(heading_vec) > 0:
                heading_vec = heading_vec / np.linalg.norm(heading_vec)
            else:
                heading_vec = np.array([1.0, 0.0], dtype=float)
        else:
            heading_vec = movement_vec / max(np.linalg.norm(movement_vec), 1e-6)

        longitudinal_disp = np.dot(movement_vec, heading_vec)
        lateral_vec = movement_vec - longitudinal_disp * heading_vec
        lateral_disp = np.linalg.norm(lateral_vec)
        return lateral_disp, abs(longitudinal_disp), heading_vec

    def compute_expected_path(self, positions: deque) -> np.ndarray:
        if len(positions) < 2:
            return np.array(positions[-1], dtype=float) if positions else np.array([0.0, 0.0], dtype=float)

        pts = np.array(list(positions), dtype=float)
        if len(pts) >= 4:
            recent_pts = pts[-4:]
            weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
            movements = []
            for i in range(1, len(recent_pts)):
                movements.append((recent_pts[i] - recent_pts[i - 1]) * weights[i - 1])
            avg_movement = np.sum(movements, axis=0) / np.sum(weights[:-1])
            return pts[-1] + avg_movement

        if len(pts) >= 3:
            recent_pts = pts[-3:]
            movements = [recent_pts[i] - recent_pts[i - 1] for i in range(1, len(recent_pts))]
            if len(movements) > 1:
                avg_movement = np.average(movements, axis=0, weights=np.array([0.3, 0.7]))
            else:
                avg_movement = np.mean(movements, axis=0)
            return pts[-1] + avg_movement

        direction = pts[-1] - pts[-2]
        return pts[-1] + direction

    def calculate_smoothness_metric(self, trajectory: list) -> float:
        if len(trajectory) < 3:
            return 1.0
        trajectory = np.array(trajectory, dtype=float)
        second_derivatives = []
        for i in range(1, len(trajectory) - 1):
            d2 = trajectory[i + 1] - 2 * trajectory[i] + trajectory[i - 1]
            second_derivatives.append(np.linalg.norm(d2))
        if not second_derivatives:
            return 1.0
        avg_curvature = np.mean(second_derivatives)
        return 1.0 / (1.0 + avg_curvature)


class QualityValidator:
    def __init__(self, config: dict):
        self.config = config.get('quality_validation', {})
        self.enabled = self.config.get('enabled', True)
        self.smoothness_threshold = self.config.get('smoothness_threshold', 0.8)
        self.log_corrections = self.config.get('log_corrections', True)

    def validate_correction(self, before_trajectory: list, after_trajectory: list) -> dict:
        if not self.enabled:
            return {'validation_enabled': False}

        analyzer = TrajectoryAnalyzer()
        smoothness_before = analyzer.calculate_smoothness_metric(before_trajectory)
        smoothness_after = analyzer.calculate_smoothness_metric(after_trajectory)
        improvement_ratio = (smoothness_after - smoothness_before) / (smoothness_before + 1e-6)
        result = {
            'smoothness_before': smoothness_before,
            'smoothness_after': smoothness_after,
            'improvement_ratio': improvement_ratio,
            'meets_threshold': smoothness_after >= self.smoothness_threshold,
            'validation_passed': smoothness_after > smoothness_before
        }

        if not result['meets_threshold'] and self.log_corrections:
            print(f"警告: 修正後軌跡平滑度 {smoothness_after:.3f} 低於閾值 {self.smoothness_threshold}")
        return result

    def log_correction_metrics(self, track_id: int, correction_data: dict):
        if not self.log_corrections:
            return
        log_entry = {
            'track_id': track_id,
            'lateral_ratio': correction_data.get('lateral_ratio', 0),
            'correction_strength': correction_data.get('correction_strength', 0),
            'improvement': correction_data.get('quality_metrics', {}).get('improvement_ratio', 0)
        }
        print(f"橫向軌跡修正記錄: {log_entry}")


class MotorcycleMotionFilter:
    """Lightweight alpha-beta style filter tuned for motorcycle jitter."""

    def __init__(self, config: dict):
        self.enabled = config.get('enabled', True)
        self.max_blend_offset_px = config.get('max_blend_offset_px', 12.0)
        self.prediction_blend = config.get('prediction_blend', 0.85)
        self.longitudinal_blend = config.get('longitudinal_blend', 0.65)
        self.lateral_blend = config.get('lateral_blend', 0.2)
        self.velocity_blend = config.get('velocity_blend', 0.12)
        self.heading_lock_enabled = config.get('heading_lock_enabled', True)
        self.heading_lock_window = config.get('heading_lock_window', 10)
        self.heading_lock_angle_deg = config.get('heading_lock_angle_deg', 12.0)
        self.heading_lock_speed_kmh = config.get('heading_lock_speed_kmh', 10.0)
        self.heading_lock_min_lateral_px = config.get('heading_lock_min_lateral_px', 1.0)
        self.heading_lock_alpha = config.get('heading_lock_alpha', 0.12)
        self.heading_lock_lateral_blend = config.get('heading_lock_lateral_blend', 0.0)

        self.position = None
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.lock_heading_vec = None

    def _estimate_heading(self, pos_history: deque, window: int = 3) -> np.ndarray:
        window = max(2, int(window))
        if len(pos_history) >= window:
            recent = np.array(list(pos_history)[-window:], dtype=float)
            heading = recent[-1] - recent[0]
        elif len(pos_history) >= 1:
            recent = np.array(list(pos_history), dtype=float)
            heading = recent[-1] - recent[0]
        else:
            heading = np.array([1.0, 0.0], dtype=float)

        norm = np.linalg.norm(heading)
        if norm < 1e-6:
            return np.array([1.0, 0.0], dtype=float)
        return heading / norm

    def _update_lock_heading(self, pos_history: deque) -> np.ndarray:
        path_heading = self._estimate_heading(pos_history, self.heading_lock_window)
        if self.lock_heading_vec is None:
            self.lock_heading_vec = path_heading
            return self.lock_heading_vec

        candidate = path_heading
        if np.dot(candidate, self.lock_heading_vec) < 0:
            candidate = -candidate

        updated = (1 - self.heading_lock_alpha) * self.lock_heading_vec + self.heading_lock_alpha * candidate
        norm = np.linalg.norm(updated)
        if norm > 1e-6:
            self.lock_heading_vec = updated / norm
        return self.lock_heading_vec

    def _apply_heading_lock(self, innovation: np.ndarray, heading_unit: np.ndarray, dt: float, px_per_m: float, pos_history: deque) -> tuple[np.ndarray, bool, np.ndarray]:
        lock_heading = heading_unit
        perpendicular = np.array([-heading_unit[1], heading_unit[0]], dtype=float)
        longitudinal = np.dot(innovation, heading_unit) * heading_unit
        lateral = np.dot(innovation, perpendicular) * perpendicular

        if not self.heading_lock_enabled or len(pos_history) < max(3, self.heading_lock_window):
            return longitudinal + lateral, False, heading_unit

        movement_vec = innovation
        move_norm = np.linalg.norm(movement_vec)
        lock_perpendicular = np.array([-lock_heading[1], lock_heading[0]], dtype=float)
        lock_longitudinal = np.dot(innovation, lock_heading) * lock_heading
        lock_lateral = np.dot(innovation, lock_perpendicular) * lock_perpendicular
        lateral_norm = np.linalg.norm(lock_lateral)
        speed_kmh = (move_norm / max(px_per_m, 1e-6) / max(dt, 1e-3)) * 3.6
        if move_norm < 1e-6 or speed_kmh < self.heading_lock_speed_kmh or lateral_norm < self.heading_lock_min_lateral_px:
            return longitudinal + lateral, False, heading_unit

        movement_unit = movement_vec / move_norm
        cos_angle = np.clip(np.dot(movement_unit, lock_heading), -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_angle))
        if angle_deg <= self.heading_lock_angle_deg:
            return longitudinal + lateral, False, heading_unit

        locked = lock_longitudinal + self.heading_lock_lateral_blend * lock_lateral
        return locked, True, lock_heading

    def filter(self, measurement: np.ndarray, dt: float, px_per_m: float, pos_history: deque) -> np.ndarray:
        measurement = np.array(measurement, dtype=float)
        if not self.enabled:
            return measurement

        if self.position is None:
            self.position = measurement.copy()
            self.velocity = np.array([0.0, 0.0], dtype=float)
            return measurement

        dt = max(float(dt), 1e-3)
        predicted_pos = self.position + self.velocity * dt
        heading_unit = self._update_lock_heading(pos_history)

        innovation = measurement - predicted_pos
        locked_innovation, heading_locked, heading_unit = self._apply_heading_lock(innovation, heading_unit, dt, px_per_m, pos_history)
        perpendicular = np.array([-heading_unit[1], heading_unit[0]], dtype=float)
        longitudinal = np.dot(locked_innovation, heading_unit) * heading_unit
        lateral = np.dot(locked_innovation, perpendicular) * perpendicular

        lateral_blend = self.heading_lock_lateral_blend if heading_locked else self.lateral_blend
        filtered_pos = predicted_pos + self.longitudinal_blend * longitudinal + lateral_blend * lateral
        filtered_velocity = self.velocity + self.velocity_blend * (filtered_pos - self.position) / dt
        if heading_locked:
            filtered_velocity = np.dot(filtered_velocity, heading_unit) * heading_unit

        offset = filtered_pos - measurement
        offset_norm = np.linalg.norm(offset)
        if offset_norm > self.max_blend_offset_px:
            blend = min(1.0, self.max_blend_offset_px / offset_norm)
            filtered_pos = measurement + offset * blend
            filtered_velocity = self.velocity + self.velocity_blend * (filtered_pos - self.position) / dt

        blended_pos = self.prediction_blend * filtered_pos + (1 - self.prediction_blend) * measurement
        self.position = blended_pos
        self.velocity = filtered_velocity
        return blended_pos


class MotorcycleLateralCorrector(TrackSmoother):
    def __init__(self, config: dict, vehicle_class: str = None):
        super().__init__(config)
        self.vehicle_class = vehicle_class
        self.lateral_config = config.get('lateral_correction', {})
        self.filter_config = config.get('motorcycle_filter', {})

        logging.info(f"MotorcycleLateralCorrector初始化: vehicle_class={vehicle_class}")
        logging.info(f"lateral_correction配置: {self.lateral_config}")
        logging.info(f"motorcycle_filter配置: {self.filter_config}")

        self.lateral_enabled = self.lateral_config.get('enabled', False)
        self.vehicle_classes = self.lateral_config.get('vehicle_classes', ['motor', 'two_wheeler'])
        self.lateral_threshold = self.lateral_config.get('threshold', 1.5)
        self.correction_strength = self.lateral_config.get('strength', 0.5)
        self.smoothing_window = self.lateral_config.get('window_size', 7)
        self.max_correction_distance = self.lateral_config.get('max_distance', 2.5)
        self.min_jump_lateral_px = self.lateral_config.get('min_jump_lateral_px', 0.6)
        self.prediction_weight = self.lateral_config.get('prediction_weight', 0.3)
        self.shock_ratio_threshold = self.lateral_config.get('shock_ratio_threshold', 6.0)
        self.shock_lateral_px = self.lateral_config.get('shock_lateral_px', 4.0)
        self.shock_longitudinal_factor = self.lateral_config.get('shock_longitudinal_factor', 0.12)
        self.shock_lateral_factor = self.lateral_config.get('shock_lateral_factor', 0.02)

        adaptive_config = self.lateral_config.get('adaptive_strength', {})
        self.adaptive_enabled = adaptive_config.get('enabled', True)
        self.min_strength = adaptive_config.get('min_strength', 0.45)
        self.max_strength = adaptive_config.get('max_strength', 1.0)
        self.consecutive_threshold = adaptive_config.get('consecutive_threshold', 1)

        self.correction_history = deque(maxlen=self.smoothing_window)
        self.consecutive_corrections = 0
        self.trajectory_analyzer = TrajectoryAnalyzer()
        self.quality_validator = QualityValidator(config)
        self.motion_filter = MotorcycleMotionFilter(self.filter_config)

    def is_motorcycle(self, vehicle_class: str = None) -> bool:
        check_class = vehicle_class or self.vehicle_class
        return check_class in self.vehicle_classes if check_class else False

    def update(self, current_sat_pos: list, dt: float, px_per_m: float,
              svg_heading: float = None, vehicle_class: str = None) -> dict:
        if vehicle_class:
            self.vehicle_class = vehicle_class

        raw_pos = np.array(current_sat_pos, dtype=float)
        filtered_pos = raw_pos
        correction_applied = False

        if self.is_motorcycle():
            filtered_pos = self.motion_filter.filter(raw_pos, dt, px_per_m, self.pos_history)

        corrected_pos = filtered_pos
        if self.lateral_enabled and self.is_motorcycle():
            corrected_pos = self.apply_lateral_correction(raw_pos, filtered_pos, px_per_m)
            correction_applied = not np.allclose(corrected_pos, filtered_pos, atol=1e-3)

        result = super().update(corrected_pos.tolist(), dt, px_per_m, svg_heading)
        result.update({
            'lateral_correction_applied': correction_applied,
            'original_position': raw_pos.tolist(),
            'filtered_position': filtered_pos.tolist(),
            'corrected_position': corrected_pos.tolist(),
            'vehicle_class': self.vehicle_class
        })
        return result

    def detect_lateral_jump(self, measurement_pos: np.ndarray, reference_pos: np.ndarray) -> tuple:
        if len(self.pos_history) < 2:
            return False, 0.0, np.array([0.0, 0.0], dtype=float)

        lateral_disp, longitudinal_disp, heading_vec = self.trajectory_analyzer.calculate_displacement_components(
            self.pos_history, measurement_pos
        )
        residual_vec = measurement_pos - reference_pos

        if longitudinal_disp < 0.1:
            lateral_ratio = float('inf') if lateral_disp > self.min_jump_lateral_px else 0.0
        else:
            lateral_ratio = lateral_disp / longitudinal_disp

        heading_norm = max(np.linalg.norm(heading_vec), 1e-6)
        heading_unit = heading_vec / heading_norm
        perpendicular = np.array([-heading_unit[1], heading_unit[0]], dtype=float)
        residual_lateral = abs(np.dot(residual_vec, perpendicular))
        is_jump = (
            lateral_ratio > self.lateral_threshold or
            lateral_disp > self.min_jump_lateral_px * 1.5 or
            residual_lateral > self.min_jump_lateral_px * 1.5
        )
        return is_jump, lateral_ratio, heading_vec

    def calculate_correction_strength(self) -> float:
        if not self.adaptive_enabled:
            return self.correction_strength
        if self.consecutive_corrections >= self.consecutive_threshold:
            adaptive_factor = min(1.0, self.consecutive_corrections / (self.consecutive_threshold * 2))
            strength = self.correction_strength + adaptive_factor * (self.max_strength - self.correction_strength)
        else:
            strength = max(self.min_strength, self.correction_strength)
        return min(self.max_strength, strength)

    def apply_lateral_correction(self, measurement_pos: np.ndarray, filtered_pos: np.ndarray, px_per_m: float) -> np.ndarray:
        is_jump, lateral_ratio, heading_vec = self.detect_lateral_jump(measurement_pos, filtered_pos)
        if not is_jump:
            self.consecutive_corrections = 0
            return filtered_pos

        heading_norm = max(np.linalg.norm(heading_vec), 1e-6)
        heading_unit = heading_vec / heading_norm
        perpendicular = np.array([-heading_unit[1], heading_unit[0]], dtype=float)
        residual = measurement_pos - filtered_pos

        longitudinal_component = np.dot(residual, heading_unit) * heading_unit
        lateral_component = np.dot(residual, perpendicular) * perpendicular
        residual_lateral = abs(np.dot(residual, perpendicular))

        is_shock = (
            (np.isfinite(lateral_ratio) and lateral_ratio >= self.shock_ratio_threshold) or
            residual_lateral >= self.shock_lateral_px
        )

        if not is_shock:
            self.consecutive_corrections = 0
            return filtered_pos

        corrected_pos = (
            filtered_pos +
            self.shock_longitudinal_factor * longitudinal_component +
            self.shock_lateral_factor * lateral_component
        )

        max_correction_px = self.max_correction_distance * px_per_m
        offset = corrected_pos - filtered_pos
        offset_norm = np.linalg.norm(offset)
        if offset_norm > max_correction_px:
            corrected_pos = filtered_pos + offset * (max_correction_px / offset_norm)

        self.consecutive_corrections += 1
        correction_data = {
            'original_pos': measurement_pos.tolist(),
            'filtered_pos': filtered_pos.tolist(),
            'corrected_pos': corrected_pos.tolist(),
            'lateral_ratio': lateral_ratio,
            'correction_strength': 1.0
        }
        self.correction_history.append(correction_data)
        logging.info(
            "機車濾波修正: ratio=%.2f shock=%s original=%s filtered=%s corrected=%s",
            lateral_ratio,
            is_shock,
            measurement_pos,
            filtered_pos,
            corrected_pos
        )
        return corrected_pos

    def apply_additional_smoothing(self, corrected_pos: np.ndarray, reference_pos: np.ndarray) -> np.ndarray:
        if len(self.pos_history) < 2:
            return corrected_pos

        recent_positions = list(self.pos_history)[-3:] if len(self.pos_history) >= 3 else list(self.pos_history)
        if len(recent_positions) < 2:
            return corrected_pos

        movements = []
        for i in range(1, len(recent_positions)):
            movement = np.array(recent_positions[i], dtype=float) - np.array(recent_positions[i - 1], dtype=float)
            movements.append(movement)

        avg_movement = np.mean(movements, axis=0)
        predicted_pos = np.array(recent_positions[-1], dtype=float) + avg_movement
        return (1 - self.prediction_weight) * corrected_pos + self.prediction_weight * predicted_pos
