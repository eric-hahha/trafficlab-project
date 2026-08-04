import math


_SIDE_ON_MIN_ASPECT = 1.8
_FRONT_BACK_MAX_ASPECT = 1.3
_MIN_BBOX_AREA = 400  # pixels²; ignore tiny boxes


def _angular_distance(a: float, b: float) -> float:
    """Smallest angle between two headings in degrees (0-180)."""
    diff = abs((a - b + 180) % 360 - 180)
    return diff


def _closest_candidate(candidates: list[float], road_heading: float) -> float:
    return min(candidates, key=lambda c: _angular_distance(c, road_heading))


def estimate_heading_from_bbox(
    bbox_2d: list[float],
    sat_coords: tuple[float, float],
    cam_pos_sat: tuple[float, float],
    road_heading: float | None = None,
) -> tuple[float, float] | None:
    """Estimate vehicle heading from YOLO bbox shape + camera geometry.

    Uses bbox aspect ratio to determine if the vehicle is showing its side
    (high aspect) or front/back (low aspect), then maps to a world heading
    using the camera-to-vehicle line-of-sight direction.

    Args:
        bbox_2d: [x1, y1, x2, y2] in CCTV pixels.
        sat_coords: Vehicle ground position in satellite space (x, y).
        cam_pos_sat: Camera position in satellite space (x_cam, y_cam).
            Available from GProjection.x_cam_coords_sat / y_cam_coords_sat.
        road_heading: SVG road heading hint (degrees, 0-360) used to
            disambiguate the two candidate directions. If None, returns the
            candidate that points "away" from the camera (forward-biased).

    Returns:
        (heading_deg, confidence) where heading_deg is 0-360 (0=East,
        90=North) and confidence is 0-1, or None if bbox is too small.
    """
    x1, y1, x2, y2 = bbox_2d
    w = x2 - x1
    h = y2 - y1

    if w * h < _MIN_BBOX_AREA:
        return None

    aspect = w / max(h, 1.0)

    # Line-of-sight angle from camera to vehicle (East=0, North=90).
    # Sat y increases downward, so negate dy for standard math convention.
    dx = sat_coords[0] - cam_pos_sat[0]
    dy = sat_coords[1] - cam_pos_sat[1]
    los_angle = math.degrees(math.atan2(-dy, dx)) % 360

    if aspect >= _SIDE_ON_MIN_ASPECT:
        # Vehicle is showing its side → travelling perpendicular to LOS.
        candidates = [(los_angle + 90) % 360, (los_angle - 90) % 360]
        confidence = min(1.0, (aspect - _SIDE_ON_MIN_ASPECT) / 1.2)
    elif aspect <= _FRONT_BACK_MAX_ASPECT:
        # Vehicle is showing its front or rear → travelling along LOS.
        candidates = [los_angle % 360, (los_angle + 180) % 360]
        confidence = min(1.0, (_FRONT_BACK_MAX_ASPECT - aspect) / 0.5)
    else:
        # Ambiguous aspect ratio — still produce a guess, low confidence.
        candidates = [
            (los_angle + 90) % 360,
            (los_angle - 90) % 360,
            los_angle % 360,
            (los_angle + 180) % 360,
        ]
        confidence = 0.1

    if road_heading is not None:
        heading = _closest_candidate(candidates, road_heading)
    else:
        # No road hint: prefer the direction that points away from the camera
        # (vehicles more often travel away from the camera vantage point).
        away_from_cam = (los_angle + 180) % 360
        heading = _closest_candidate(candidates, away_from_cam)

    return heading, confidence
