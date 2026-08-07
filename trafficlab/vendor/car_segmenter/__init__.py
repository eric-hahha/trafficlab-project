from .config import COCO_VEHICLE_CLASS_IDS
from .segmenter import CarSegmenter, CarSegmenterConfig

__all__ = [
    "CarSegmenter",
    "CarSegmenterConfig",
    "COCO_VEHICLE_CLASS_IDS",
]

__version__ = "0.1.0"
