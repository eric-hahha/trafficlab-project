"""PyTorch dataset adapter for SKoPe3D vehicle keypoint fine-tuning."""

from __future__ import annotations

import csv
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


PIRAZH_TO_SKOPE3D = {
    0: 16,  # left-front wheel
    1: 17,  # left-back wheel
    2: 18,  # right-front wheel
    3: 19,  # right-back wheel
    8: 32,  # front auto logo
    12: 0,  # right-front roof corner
    13: 1,  # left-front roof corner
    14: 3,  # left-back roof corner
    15: 2,  # right-back roof corner
}

DEFAULT_MEAN_STD = Path(__file__).with_name("veri_mean_std.pth.tar")


@dataclass(frozen=True)
class SKoPe3DSample:
    scene: int
    frame: int
    vehicle: int
    keypoint_name: str
    metadata_name: str | None


def _read_csv_rows(zf: zipfile.ZipFile, name: str) -> list[list[str]]:
    text = zf.read(name).decode("utf-8").strip()
    return list(csv.reader(text.splitlines()))


def _float_rows(rows: list[list[str]]) -> list[list[float | str]]:
    parsed: list[list[float | str]] = []
    for row in rows:
        values: list[float | str] = []
        for value in row:
            try:
                values.append(float(value))
            except ValueError:
                values.append(value)
        parsed.append(values)
    return parsed


def _parse_scene_number(path: Path) -> int:
    match = re.search(r"scene_(\d+)\.zip$", path.name)
    if not match:
        raise ValueError(f"Cannot parse scene number from {path}")
    return int(match.group(1))


def _parse_member_ids(name: str) -> tuple[int, int] | None:
    match = re.search(r"scene_\d+/(\d+)/keypoint_(\d+)\.csv$", name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _gaussian_heatmap(size: int, x: float, y: float, sigma: float) -> np.ndarray:
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    heatmap = np.exp(-((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2.0 * sigma ** 2))
    return heatmap.astype(np.float32)


def _orientation_from_roof(kp_by_skope_idx: np.ndarray) -> int:
    """Quantize the rear-to-front roof direction into Pirazh's 8 bins.

    The label set is from ``trafficlab.keypoint.inference.ORIENTATION_LABELS``:
    front, rear, left, left-front, left-rear, right, right-front, right-rear.
    Image coordinates use x to the right and y downward.
    """
    rf = kp_by_skope_idx[0, :2]
    lf = kp_by_skope_idx[1, :2]
    rb = kp_by_skope_idx[2, :2]
    lb = kp_by_skope_idx[3, :2]
    front = (rf + lf) / 2.0
    rear = (rb + lb) / 2.0
    dx, dy = front - rear

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0

    # Dominant vertical image motion corresponds to front/rear view.
    if abs(dx) < abs(dy) * 0.5:
        return 0 if dy > 0 else 1
    if abs(dy) < abs(dx) * 0.5:
        return 5 if dx > 0 else 2
    if dx > 0 and dy > 0:
        return 6
    if dx < 0 and dy > 0:
        return 3
    if dx > 0 and dy < 0:
        return 7
    return 4


class _VideoFrameCache:
    def __init__(self, extracted_video: Path):
        self.extracted_video = extracted_video
        self._cap: cv2.VideoCapture | None = None
        self._last_frame: int | None = None
        self._last_image: np.ndarray | None = None

    def read(self, frame_index: int) -> np.ndarray:
        if self._last_frame == frame_index and self._last_image is not None:
            return self._last_image.copy()
        if self._cap is None:
            self._cap = cv2.VideoCapture(str(self.extracted_video))
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.extracted_video}")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame_index} from {self.extracted_video}")
        self._last_frame = frame_index
        self._last_image = frame
        return frame.copy()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self) -> None:
        self.close()


class SKoPe3DDataset(Dataset):
    """Read SKoPe3D scene zips and return Pirazh fine-tuning tensors.

    Each item contains:
        img224: float tensor (3, 224, 224)
        img56: float tensor (3, 56, 56)
        heatmap_gt: float tensor (20, 56, 56)
        heatmap_mask: bool tensor (20,)
        orient_gt: int64 tensor scalar
    """

    def __init__(
        self,
        data_root: str | Path,
        scenes: Iterable[int] | None = None,
        cache_dir: str | Path = "/tmp/trafficlab_skope3d_cache",
        mean_std_path: str | Path = DEFAULT_MEAN_STD,
        heatmap_size: int = 56,
        image_size: int = 224,
        sigma: float = 1.0,
        min_box_size: int = 8,
        max_samples: int | None = None,
        extract_video: bool = True,
    ):
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir)
        self.heatmap_size = heatmap_size
        self.image_size = image_size
        self.sigma = sigma
        self.min_box_size = min_box_size
        self.extract_video = extract_video

        mean_std = torch.load(str(mean_std_path), map_location="cpu", weights_only=False)
        self.mean = np.asarray(mean_std["mean"], dtype=np.float32)
        self.std = np.asarray(mean_std["std"], dtype=np.float32)

        requested = set(scenes) if scenes is not None else None
        self.scene_zips: dict[int, Path] = {}
        for zip_path in sorted(self.data_root.glob("scene_*.zip")):
            scene = _parse_scene_number(zip_path)
            if requested is None or scene in requested:
                self.scene_zips[scene] = zip_path
        if not self.scene_zips:
            raise FileNotFoundError(f"No SKoPe3D scene zips found in {self.data_root}")

        self.samples = self._index_samples(max_samples=max_samples)
        if not self.samples:
            raise RuntimeError("No SKoPe3D keypoint samples found")

        self._video_caches: dict[int, _VideoFrameCache] = {}

    def _index_samples(self, max_samples: int | None) -> list[SKoPe3DSample]:
        samples: list[SKoPe3DSample] = []
        for scene, zip_path in self.scene_zips.items():
            with zipfile.ZipFile(zip_path) as zf:
                names = set(zf.namelist())
                keypoint_names = sorted(
                    (name for name in names if re.search(r"/keypoint_\d+\.csv$", name)),
                    key=lambda n: tuple(int(x) for x in re.findall(r"\d+", n)[-2:]),
                )
                for keypoint_name in keypoint_names:
                    ids = _parse_member_ids(keypoint_name)
                    if ids is None:
                        continue
                    if not self._is_usable_sample(zf, keypoint_name):
                        continue
                    frame, vehicle = ids
                    metadata_name = f"scene_{scene}/{frame}/metadata_{vehicle}.csv"
                    samples.append(
                        SKoPe3DSample(
                            scene=scene,
                            frame=frame,
                            vehicle=vehicle,
                            keypoint_name=keypoint_name,
                            metadata_name=metadata_name if metadata_name in names else None,
                        )
                    )
                    if max_samples is not None and len(samples) >= max_samples:
                        return samples
        return samples

    def _is_usable_sample(self, zf: zipfile.ZipFile, keypoint_name: str) -> bool:
        rows = _float_rows(_read_csv_rows(zf, keypoint_name))
        if len(rows) < 36:
            return False
        x1, y1 = float(rows[1][0]), float(rows[1][1])
        x2, y2 = float(rows[2][0]), float(rows[2][1])
        if x2 - x1 < self.min_box_size or y2 - y1 < self.min_box_size:
            return False
        skope_kps = np.asarray(
            [[float(row[0]), float(row[1]), float(row[2])] for row in rows[3:36]],
            dtype=np.float32,
        )
        for skope_idx in PIRAZH_TO_SKOPE3D.values():
            if skope_kps[skope_idx, 2] == 1:
                return True
        return False

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_video_path(self, scene: int) -> Path:
        video_path = self.cache_dir / f"scene_{scene}" / "original.avi"
        if video_path.exists():
            return video_path
        if not self.extract_video:
            raise FileNotFoundError(f"Video has not been extracted: {video_path}")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path = self.scene_zips[scene]
        member_name = f"scene_{scene}/original.avi"
        tmp_path = video_path.with_suffix(".tmp")
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(member_name) as src, open(tmp_path, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
        tmp_path.replace(video_path)
        return video_path

    def _read_frame(self, scene: int, frame: int) -> np.ndarray:
        cache = self._video_caches.get(scene)
        if cache is None:
            cache = _VideoFrameCache(self._extract_video_path(scene))
            self._video_caches[scene] = cache
        return cache.read(frame)

    def _normalize_crop(self, crop_bgr: np.ndarray, size: int) -> torch.Tensor:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        interpolation = cv2.INTER_AREA if crop_bgr.shape[0] > size or crop_bgr.shape[1] > size else cv2.INTER_LINEAR
        resized = cv2.resize(rgb, (size, size), interpolation=interpolation)
        resized = (resized - self.mean.reshape(1, 1, 3)) / self.std.reshape(1, 1, 3)
        return torch.from_numpy(resized.transpose(2, 0, 1)).float()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        zip_path = self.scene_zips[sample.scene]

        with zipfile.ZipFile(zip_path) as zf:
            rows = _float_rows(_read_csv_rows(zf, sample.keypoint_name))

        if len(rows) < 36:
            raise RuntimeError(f"Unexpected keypoint CSV length in {sample.keypoint_name}: {len(rows)}")

        bbox_tl = rows[1]
        bbox_br = rows[2]
        x1, y1 = int(round(float(bbox_tl[0]))), int(round(float(bbox_tl[1])))
        x2, y2 = int(round(float(bbox_br[0]))), int(round(float(bbox_br[1])))

        frame = self._read_frame(sample.scene, sample.frame)
        height, width = frame.shape[:2]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        crop = frame[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape[:2]

        skope_kps = np.asarray(
            [[float(row[0]), float(row[1]), float(row[2])] for row in rows[3:36]],
            dtype=np.float32,
        )

        heatmap_gt = np.zeros((20, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        heatmap_mask = np.zeros((20,), dtype=bool)
        scale_x = self.heatmap_size / float(crop_w)
        scale_y = self.heatmap_size / float(crop_h)

        for pirazh_idx, skope_idx in PIRAZH_TO_SKOPE3D.items():
            x, y, visible = skope_kps[skope_idx]
            if visible != 1:
                continue
            if x < 0 or y < 0 or x >= crop_w or y >= crop_h:
                continue
            hx = min(self.heatmap_size - 1, max(0.0, x * scale_x))
            hy = min(self.heatmap_size - 1, max(0.0, y * scale_y))
            heatmap_gt[pirazh_idx] = _gaussian_heatmap(self.heatmap_size, hx, hy, self.sigma)
            heatmap_mask[pirazh_idx] = True

        roof_visible = np.all(skope_kps[[0, 1, 2, 3], 2] == 1)
        orient_gt = _orientation_from_roof(skope_kps) if roof_visible else 0

        return {
            "img224": self._normalize_crop(crop, self.image_size),
            "img56": self._normalize_crop(crop, self.heatmap_size),
            "heatmap_gt": torch.from_numpy(heatmap_gt),
            "heatmap_mask": torch.from_numpy(heatmap_mask),
            "orient_gt": torch.tensor(orient_gt, dtype=torch.long),
            "scene": torch.tensor(sample.scene, dtype=torch.long),
            "frame": torch.tensor(sample.frame, dtype=torch.long),
            "vehicle": torch.tensor(sample.vehicle, dtype=torch.long),
        }


def parse_scene_spec(spec: str | None) -> list[int] | None:
    """Parse strings like ``0-3,8,10`` into a sorted scene list."""
    if spec is None or spec.strip() == "":
        return None
    scenes: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            scenes.update(range(int(start), int(end) + 1))
        else:
            scenes.add(int(part))
    return sorted(scenes)


__all__ = ["PIRAZH_TO_SKOPE3D", "SKoPe3DDataset", "parse_scene_spec"]
