from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _resample_filter(name: str) -> int:
    if hasattr(Image, "Resampling"):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


class GastricSegmentationDataset(Dataset):
    """WL/NBI gastric lesion segmentation dataset.

    The source dataset is strictly read-only. This class only reads images and
    masks from dataset/<modal>/<split>/{images,masks}.
    """

    def __init__(
        self,
        data_root: str | Path,
        modal: str,
        split: str,
        img_size: int = 224,
        augment: bool = False,
        mask_threshold: int = 0,
        limit_samples: int | None = None,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.modal = modal
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.mask_threshold = mask_threshold

        if modal not in {"WL", "NBI"}:
            raise ValueError(f"modal must be WL or NBI, got {modal!r}")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train, val, or test, got {split!r}")

        image_dir = self.data_root / modal / split / "images"
        mask_dir = self.data_root / modal / split / "masks"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing mask directory: {mask_dir}")

        image_paths = sorted(image_dir.glob("*.jpg"))
        samples = []
        for image_path in image_paths:
            mask_path = mask_dir / f"{image_path.stem}.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")
            samples.append((image_path, mask_path))

        if limit_samples is not None:
            samples = samples[:limit_samples]
        if not samples:
            raise RuntimeError(f"No samples found for {modal}/{split} in {self.data_root}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> tuple[np.ndarray, tuple[int, int]]:
        with Image.open(path) as img:
            original_size = img.size
            img = img.convert("RGB").resize(
                (self.img_size, self.img_size), _resample_filter("BILINEAR")
            )
            arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr, original_size

    def _load_mask(self, path: Path) -> np.ndarray:
        with Image.open(path) as mask:
            mask = mask.convert("L").resize(
                (self.img_size, self.img_size), _resample_filter("NEAREST")
            )
            arr = np.asarray(mask, dtype=np.uint8)
        # Use non-zero foreground so both 0/1 and 0/255 masks are handled.
        return (arr != 0).astype(np.int64)

    def foreground_ratio_summary(self, max_items: int = 32) -> dict[str, float | int]:
        ratios = []
        for _, mask_path in self.samples[:max_items]:
            ratios.append(float(self._load_mask(mask_path).mean()))
        if not ratios:
            return {"count": 0, "min": 0.0, "mean": 0.0, "max": 0.0}
        ratios_arr = np.asarray(ratios, dtype=np.float64)
        return {
            "count": int(ratios_arr.size),
            "min": float(ratios_arr.min()),
            "mean": float(ratios_arr.mean()),
            "max": float(ratios_arr.max()),
        }

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        k = np.random.randint(0, 4)
        if k:
            image = np.rot90(image, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        return image, mask

    def __getitem__(self, idx: int) -> dict[str, object]:
        image_path, mask_path = self.samples[idx]
        image, original_size = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        if self.augment:
            image, mask = self._augment(image, mask)

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).long()
        return {
            "image": image_tensor,
            "label": mask_tensor,
            "case_name": image_path.stem,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "original_size": original_size,
            "modal": self.modal,
            "split": self.split,
        }
