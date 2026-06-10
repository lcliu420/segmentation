from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    MODALS = {"ALL", "WL", "NBI", "DELETE"}
    SPLITS = {"train", "val", "test"}
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, data_root, modal="ALL", split="train", img_size=224, augment=False):
        modal = modal.upper()
        split = split.lower()
        if modal not in self.MODALS:
            raise ValueError(f"modal must be one of {sorted(self.MODALS)}, got {modal}")
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {sorted(self.SPLITS)}, got {split}")

        self.data_root = Path(data_root)
        self.modal = modal
        self.split = split
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.image_dir = self.data_root / modal / split / "images"
        self.mask_dir = self.data_root / modal / split / "masks"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No paired samples found in {self.image_dir} and {self.mask_dir}")

    def _collect_samples(self):
        image_paths = sorted(self.image_dir.glob("*.jpg"))
        mask_by_stem = {path.stem: path for path in self.mask_dir.glob("*.png")}
        samples = []
        missing_masks = []

        for image_path in image_paths:
            mask_path = mask_by_stem.get(image_path.stem)
            if mask_path is None:
                missing_masks.append(image_path.name)
                continue
            samples.append((image_path, mask_path))

        extra_masks = sorted(set(mask_by_stem) - {path.stem for path in image_paths})
        if missing_masks or extra_masks:
            details = []
            if missing_masks:
                details.append(f"missing masks for images: {missing_masks[:5]}")
            if extra_masks:
                details.append(f"masks without images: {extra_masks[:5]}")
            raise RuntimeError("; ".join(details))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        original_w, original_h = image.size

        if self.augment:
            image, mask = self._augment(image, mask)

        size = (self.img_size, self.img_size)
        image = image.resize(size, Image.BILINEAR)
        mask = mask.resize(size, Image.NEAREST)

        image = np.asarray(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = (image - self.MEAN) / self.STD

        mask = np.asarray(mask, dtype=np.uint8)
        mask = torch.from_numpy((mask > 127).astype(np.int64))

        return {
            "image": image,
            "label": mask,
            "name": image_path.stem,
            "original_size": torch.tensor([original_h, original_w], dtype=torch.long),
        }

    @staticmethod
    def _augment(image, mask):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            angle = random.uniform(-20.0, 20.0)
            image = image.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle, resample=Image.NEAREST)
        return image, mask
