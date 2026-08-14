"""Segmentation-based fragment merging for PifPaf instance splitting.

PifPaf's CIF/CAF decoder occasionally splits one physical vehicle into two
(or more) separate annotations when the association field between, e.g.,
front and rear keypoints is weak (see docs/keypoints-openpifpaf-intro.md,
"PifPaf Instance 分裂問題"). car-segmenter (vendored under
trafficlab.vendor.car_segmenter) gives one instance mask per real vehicle —
grouping PifPaf keypoints by mask membership merges those fragments using
the vehicle's actual pixel extent as ground truth, instead of guessing from
a PifPaf fragment's own (possibly degenerate) bbox.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from trafficlab.vendor.car_segmenter import CarSegmenter, CarSegmenterConfig


class CarMaskSource:
    """Wraps CarSegmenter for single-frame instance-mask lookup."""

    def __init__(self, model_path: str = 'models/yolo11m-seg.pt', confidence: float = 0.3,
                 device: str | None = None):
        self._segmenter = CarSegmenter(CarSegmenterConfig(
            model_path=model_path, classes=('car',), confidence=confidence,
            track=True, device=device,
        ))

    def detect(self, frame_bgr: np.ndarray):
        """Run instance segmentation on a single BGR frame. Returns sv.Detections
        with `.mask` ((N,H,W) bool), `.xyxy` (mask-derived bbox), `.tracker_id`
        (persistent per-vehicle id from car-segmenter's built-in ByteTrack)."""
        return self._segmenter.predict(frame_bgr)

    def detect_records(self, frame_bgr: np.ndarray) -> list:
        """detect() + detections_to_records() in one call, for callers that
        only want the serializable record form (e.g. record_car_masks.py)."""
        return detections_to_records(self.detect(frame_bgr))


def assign_predictions_to_masks(predictions, masks: np.ndarray, kp_conf: float) -> list:
    """Assign each PifPaf annotation to the segmentation mask its confident
    keypoints mostly fall inside.

    predictions: list of openpifpaf Annotation (each `.data` is (24,3) [x,y,conf]).
    masks: (M, H, W) bool array, one plane per segmentation instance (M may be 0).
    Returns a list[int | None] of length len(predictions): the winning mask
    index per annotation, by majority vote among its confident keypoints, or
    None if none of its keypoints fall inside any mask (left unmerged rather
    than guessed at — a fragment segmentation missed shouldn't be forced into
    the wrong group).
    """
    n_masks = masks.shape[0] if masks is not None else 0
    if n_masks == 0:
        return [None] * len(predictions)
    H, W = masks.shape[1], masks.shape[2]

    group_of: list = []
    for ann in predictions:
        kp = ann.data
        votes: dict = {}
        for i in range(kp.shape[0]):
            x, y, conf = float(kp[i, 0]), float(kp[i, 1]), float(kp[i, 2])
            if conf < kp_conf or (x == 0.0 and y == 0.0):
                continue
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < W and 0 <= yi < H):
                continue
            for m in range(n_masks):
                if masks[m, yi, xi]:
                    votes[m] = votes.get(m, 0) + 1
        group_of.append(max(votes, key=votes.get) if votes else None)
    return group_of


def mask_to_polygon(mask: np.ndarray) -> list | None:
    """Simplify a (H,W) bool instance mask to a flat [[x,y], ...] polygon —
    same findContours + approxPolyDP recipe already used for the FOV polygon
    in trafficlab/gui/tabs/calibration_stage/homf_stage.py:240-258. A polygon
    (tens of points) is orders of magnitude smaller to serialize than the
    dense mask it came from, which is what makes recording a whole video's
    worth of per-frame instances to a file (record_car_masks.py) practical.

    Returns None if the mask has no contour (shouldn't happen for a real
    detection, but a detection's mask is never assumed non-empty here).
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.001 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    return [[float(x), float(y)] for x, y in contour[:, 0, :]]


def polygon_contains_point(polygon, x: float, y: float) -> bool:
    """Point-in-polygon test — the file-loaded-record equivalent of indexing
    a dense mask array directly."""
    if not polygon:
        return False
    contour = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


def assign_predictions_to_polygons(predictions, polygons: Sequence, kp_conf: float) -> list:
    """Polygon-based counterpart of assign_predictions_to_masks, for car
    instances loaded from a record_car_masks.py file (where only the
    simplified polygon was persisted, not the dense mask). Kept as a
    separate implementation rather than sharing one with the live dense-mask
    path — that path is already exercised and verified; this one exists
    purely to serve the --seg-masks-json record-loading path.

    predictions: list of openpifpaf Annotation (each `.data` is (24,3) [x,y,conf]).
    polygons: list of [[x,y], ...] point lists, one per segmentation instance
    (or None for an instance whose mask had no contour). Same majority-vote
    assignment and None-if-unmatched semantics as assign_predictions_to_masks.
    """
    n_polys = len(polygons)
    if n_polys == 0:
        return [None] * len(predictions)

    group_of: list = []
    for ann in predictions:
        kp = ann.data
        votes: dict = {}
        for i in range(kp.shape[0]):
            x, y, conf = float(kp[i, 0]), float(kp[i, 1]), float(kp[i, 2])
            if conf < kp_conf or (x == 0.0 and y == 0.0):
                continue
            for m in range(n_polys):
                if polygon_contains_point(polygons[m], x, y):
                    votes[m] = votes.get(m, 0) + 1
        group_of.append(max(votes, key=votes.get) if votes else None)
    return group_of


def detections_to_records(detections) -> list:
    """Convert an sv.Detections instance-segmentation result into a plain,
    JSON-serializable record list: [{'tracker_id', 'bbox_xyxy', 'confidence',
    'polygon'}, ...]. Shared shape between the live path (CarMaskSource.
    detect_records) and record_car_masks.py's on-disk format, so a
    --seg-masks-json record and a live detection look identical downstream.

    tracker_id follows car-segmenter's own convention (segmenter.py's
    annotate()): a tracker_id < 0 means the ByteTrack warm-up period hasn't
    produced a stable id yet, treated the same as no id (None) here.
    """
    n = len(detections)
    tracker_ids = detections.tracker_id if detections.tracker_id is not None else [None] * n
    records = []
    for i in range(n):
        tid = tracker_ids[i]
        records.append({
            'tracker_id': int(tid) if tid is not None and tid >= 0 else None,
            'bbox_xyxy': [float(v) for v in detections.xyxy[i]],
            'confidence': float(detections.confidence[i]),
            'polygon': mask_to_polygon(detections.mask[i]) if detections.mask is not None else None,
        })
    return records


def merge_keypoints_by_group(predictions, group_of: Sequence, kp_conf: float):
    """Merge annotations sharing the same non-None group id into one (24,3)
    keypoint array — per keypoint slot, keep whichever contributing
    annotation has the higher confidence there. This is what actually fixes
    instance splitting: two fragments assigned to the same mask become a
    single, more complete detection instead of two partial ones.

    Returns (merged, leftover):
      merged:   {group_id: kp_24} for every group with >=1 member.
      leftover: indices into `predictions` whose group_of entry was None —
                left untouched (not dropped) so nothing disappears silently.
    """
    merged: dict = {}
    leftover: list = []
    for j, g in enumerate(group_of):
        if g is None:
            leftover.append(j)
            continue
        kp = predictions[j].data
        if g not in merged:
            merged[g] = kp.copy()
            continue
        cur = merged[g]
        better = kp[:, 2] > cur[:, 2]
        cur[better] = kp[better]
    return merged, leftover
