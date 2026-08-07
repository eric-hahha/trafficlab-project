"""Constants shared across the package."""

from __future__ import annotations

# COCO class ids for vehicle categories, as produced by every Ultralytics
# *-seg.pt checkpoint (yolov8*-seg.pt, yolo11*-seg.pt, ...).
COCO_VEHICLE_CLASS_IDS: dict[str, int] = {
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "train": 6,
    "truck": 7,
}

DEFAULT_MODEL = "yolo11n-seg.pt"
DEFAULT_CLASSES = ("car",)
DEFAULT_CONFIDENCE = 0.3
DEFAULT_IOU = 0.5
