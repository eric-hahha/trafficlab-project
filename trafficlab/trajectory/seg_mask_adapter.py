"""Reuse a seg-mode inference replay's mask contours as a car-segmenter record.

scripts/run_inference.py with model.type: "seg" already runs the segmentation
model over every frame and keeps each detection's mask polygon in the replay's
`mask_contour` field. scripts/run_keypoints_openpifpaf.py --method
segmentation needs the same per-frame car instances, and would otherwise run
the segmentation model a *second* time over the same video.

This module converts the former into the scripts/record_car_masks.py on-disk
format the latter's --seg-masks-json already loads, so the segmentation model
runs exactly once for a "seg tight-box localization + wheel_pair correction"
workflow. A side effect worth having: both replays then carry the same
tracker ids, which is exactly the correspondence
scripts/correct_trajectory_wheel_pair.py's --track-id-map wants.

Field mapping (replay object -> record_car_masks instance):
    tracked_id    -> tracker_id
    bbox_2d       -> bbox_xyxy
    confidence    -> confidence
    mask_contour  -> polygon
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

# Which replay `class` values become car-segmenter instances. Matches
# CarMaskSource's hardcoded classes=('car',): the PifPaf checkpoint these
# masks group keypoints for is a *car* keypoint model, so letting a
# two_wheeler mask compete for those keypoints would only mis-assign them.
DEFAULT_CAR_CLASSES = ("car",)


@dataclass
class AdaptStats:
    """What the conversion actually kept and dropped, for the caller to report."""

    n_frames: int = 0
    n_instances: int = 0
    n_skipped_other_class: int = 0
    n_skipped_no_contour: int = 0

    def summary(self) -> str:
        parts = [f"{self.n_instances} instances across {self.n_frames} frames"]
        if self.n_skipped_other_class:
            parts.append(f"{self.n_skipped_other_class} skipped (class not in car classes)")
        if self.n_skipped_no_contour:
            parts.append(f"{self.n_skipped_no_contour} skipped (no mask_contour)")
        return "; ".join(parts)


def replay_to_seg_mask_records(
    data: dict[str, Any],
    *,
    car_classes: Iterable[str] = DEFAULT_CAR_CLASSES,
) -> tuple[dict[str, Any], AdaptStats]:
    """Build a record_car_masks.py-shaped dict from a seg-mode replay.

    Raises ValueError if the replay carries no mask_contour at all — that
    means it was not produced by a model.type: "seg" config, and silently
    handing run_keypoints_openpifpaf.py an empty instance record would look
    like "the segmenter found no cars" rather than "wrong input file".
    """
    class_filter = {str(c) for c in car_classes}
    stats = AdaptStats()
    out_frames: list[dict[str, Any]] = []
    saw_any_contour_field = False

    for frame in data.get("frames") or []:
        frame_index = frame.get("frame_index")
        if frame_index is None:
            continue

        instances = []
        for obj in frame.get("objects") or []:
            if str(obj.get("class")) not in class_filter:
                stats.n_skipped_other_class += 1
                continue

            polygon = obj.get("mask_contour")
            if polygon:
                saw_any_contour_field = True
            bbox = obj.get("bbox_2d")
            if not polygon or not bbox:
                # A seg-mode detection whose mask came back empty falls back
                # to the detector box (pipeline.py logs it). Without a polygon
                # there is nothing for assign_predictions_to_polygons to test
                # against, so drop it here and let the count say so rather
                # than emitting an instance that can never match.
                stats.n_skipped_no_contour += 1
                continue

            tracked_id = obj.get("tracked_id")
            confidence = obj.get("confidence")
            instances.append(
                {
                    "tracker_id": int(tracked_id) if tracked_id is not None else None,
                    "bbox_xyxy": [float(v) for v in bbox],
                    "confidence": float(confidence) if confidence is not None else 0.0,
                    "polygon": [[float(p[0]), float(p[1])] for p in polygon],
                }
            )

        stats.n_instances += len(instances)
        out_frames.append({"frame_index": int(frame_index), "instances": instances})

    stats.n_frames = len(out_frames)

    if not saw_any_contour_field:
        raise ValueError(
            "This replay has no mask_contour on any object, so it cannot stand in "
            "for a car-segmenter recording. Only a model.type: \"seg\" inference "
            "run produces mask_contour."
        )

    meta = data.get("meta") or {}
    out_data = {
        "video_path": data.get("mp4_path"),
        # run_keypoints_openpifpaf.py's _load_seg_masks_json copies these three
        # into its run_config as the provenance of the instances it used. Name
        # the real origin instead of a model path we did not run ourselves.
        "seg_model": f"replay:{meta.get('config_name')}",
        "seg_conf": None,
        "seg_device": None,
        "meta": {
            "resolution": meta.get("resolution"),
            "fps": meta.get("fps"),
            "source": "seg_mask_adapter",
            "source_config_name": meta.get("config_name"),
        },
        "frame_count": data.get("mp4_frame_count"),
        "frames": out_frames,
    }
    return out_data, stats


def default_seg_masks_output_path(target_path, location_code: str | None = None):
    """output/car_masks/<location>/seg-mask_<stem>.from-replay.json.gz

    Same directory record_car_masks.py writes to, with a .from-replay marker
    so an adapted file never silently overwrites a real recording.
    """
    from pathlib import Path

    target_path = Path(target_path)
    name = target_path.name
    for suffix in (".json.gz", ".json"):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            break
    else:
        stem = target_path.stem

    root = Path("output") / "car_masks"
    if location_code:
        root = root / location_code
    return root / f"seg-mask_{stem}.from-replay.json.gz"
