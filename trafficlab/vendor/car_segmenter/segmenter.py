"""Core video car detection + instance segmentation pipeline.

Built on top of:
- ultralytics (YOLO*-seg checkpoints) for the actual segmentation inference
- supervision (roboflow/supervision) for detection filtering and drawing
  masks/boxes/labels onto frames, plus video read/write
- trackers (roboflow/trackers) for persistent per-vehicle tracker IDs.
  `supervision.ByteTrack` is deprecated (removed as of supervision 0.30) in
  favor of this package, so it's used directly here instead.

The public surface is intentionally small so this package can be dropped
into any other project: construct a `CarSegmenter`, call `process_frame`
per-frame or `process_video` for a full file, done.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import supervision as sv
from tqdm import tqdm
from trackers import ByteTrackTracker
from ultralytics import YOLO

from .config import (
    COCO_VEHICLE_CLASS_IDS,
    DEFAULT_CLASSES,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEFAULT_MODEL,
)


@dataclass
class CarSegmenterConfig:
    """All tunable knobs in one place.

    Args:
        model_path: Any Ultralytics *-seg checkpoint or path to a custom
            trained one, e.g. "yolo11n-seg.pt", "yolov8s-seg.pt".
        classes: Vehicle categories to keep, drawn from
            `car_segmenter.config.COCO_VEHICLE_CLASS_IDS`.
        confidence: Minimum detection confidence.
        iou: NMS IoU threshold.
        track: Assign persistent tracker IDs across frames (ByteTrack).
        device: "cuda", "mps", "cpu", or None to let ultralytics pick.
        codec: FOURCC code used by `process_video` to write the output file.
            "mp4v" (default) is the most widely supported for encoding, but
            produces files that some players (QuickTime, Safari, iOS) refuse
            to play. Use "avc1" for H.264 output if your OpenCV build has it
            available, for better compatibility on Apple platforms.
    """

    model_path: str = DEFAULT_MODEL
    classes: Sequence[str] = field(default_factory=lambda: DEFAULT_CLASSES)
    confidence: float = DEFAULT_CONFIDENCE
    iou: float = DEFAULT_IOU
    track: bool = True
    device: str | None = None
    codec: str = "mp4v"

    def resolve_class_ids(self) -> list[int]:
        try:
            return [COCO_VEHICLE_CLASS_IDS[name] for name in self.classes]
        except KeyError as exc:
            valid = ", ".join(sorted(COCO_VEHICLE_CLASS_IDS))
            raise ValueError(
                f"Unknown class {exc}. Valid options: {valid}"
            ) from exc


class CarSegmenter:
    """Detect + segment vehicles in images or video frames."""

    def __init__(self, config: CarSegmenterConfig | None = None) -> None:
        self.config = config or CarSegmenterConfig()
        self.model = YOLO(self.config.model_path)
        self._class_ids = self.config.resolve_class_ids()
        self.tracker = ByteTrackTracker() if self.config.track else None

        self.mask_annotator = sv.MaskAnnotator()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def predict(self, frame: np.ndarray) -> sv.Detections:
        """Run segmentation on a single BGR frame and keep only target classes."""
        result = self.model(
            frame,
            verbose=False,
            conf=self.config.confidence,
            iou=self.config.iou,
            device=self.config.device,
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[np.isin(detections.class_id, self._class_ids)]
        if self.tracker is not None:
            detections = self.tracker.update(detections)
        return detections

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Draw segmentation masks, boxes and labels for `detections` onto a copy of `frame`."""
        # ByteTrackTracker reports tracker_id == -1 for a couple of frames
        # while a track is still warming up (see `minimum_consecutive_frames`).
        labels = [
            f"#{tracker_id} {self.model.names[class_id]} {confidence:.2f}"
            if tracker_id is not None and tracker_id >= 0
            else f"{self.model.names[class_id]} {confidence:.2f}"
            for class_id, confidence, tracker_id in zip(
                detections.class_id,
                detections.confidence,
                detections.tracker_id
                if detections.tracker_id is not None
                else [None] * len(detections),
            )
        ]

        annotated = frame.copy()
        annotated = self.mask_annotator.annotate(annotated, detections)
        annotated = self.box_annotator.annotate(annotated, detections)
        annotated = self.label_annotator.annotate(annotated, detections, labels=labels)
        return annotated

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, sv.Detections]:
        """Run detection + annotation on a single frame. Returns (annotated_frame, detections)."""
        detections = self.predict(frame)
        annotated = self.annotate(frame, detections)
        return annotated, detections

    def iter_video(
        self,
        source_path: str | Path,
        stride: int = 1,
        on_frame: Callable[[int, np.ndarray, sv.Detections], None] | None = None,
    ) -> Iterator[tuple[np.ndarray, sv.Detections]]:
        """Stream (annotated_frame, detections) pairs for every frame of a video.

        Useful when the caller wants to do their own thing with each frame
        (display, push to a websocket, custom writer, ...) instead of getting
        a finished output file. See `process_video` for the file-in/file-out
        convenience wrapper.
        """
        frame_generator = sv.get_video_frames_generator(
            source_path=str(source_path), stride=stride
        )
        for index, frame in enumerate(frame_generator):
            annotated, detections = self.process_frame(frame)
            if on_frame is not None:
                on_frame(index, annotated, detections)
            yield annotated, detections

    def process_video(
        self,
        source_path: str | Path,
        target_path: str | Path,
        show_progress: bool = True,
    ) -> None:
        """Read a video file, run detection+segmentation on every frame, write annotated output."""
        video_info = sv.VideoInfo.from_video_path(video_path=str(source_path))
        frame_iterator = self.iter_video(source_path)
        if show_progress:
            frame_iterator = tqdm(
                frame_iterator, total=video_info.total_frames, desc="car-segmenter"
            )

        with sv.VideoSink(
            target_path=str(target_path), video_info=video_info, codec=self.config.codec
        ) as sink:
            for annotated_frame, _ in frame_iterator:
                sink.write_frame(frame=annotated_frame)
