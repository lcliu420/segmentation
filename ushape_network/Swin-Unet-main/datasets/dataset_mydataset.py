import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0, 1))
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, axes=(0, 1), order=3, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def remap_mask(mask):
    unique_values = sorted(np.unique(mask).tolist())
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for new_value, old_value in enumerate(unique_values):
        remapped[mask == old_value] = new_value
    return remapped


def resize_image(image, output_size):
    height, width = image.shape[:2]
    target_h, target_w = output_size
    if height == target_h and width == target_w:
        return image
    if image.ndim == 3:
        factors = (target_h / height, target_w / width, 1)
    else:
        factors = (target_h / height, target_w / width)
    return zoom(image, factors, order=3 if image.ndim == 3 else 0)


def resize_label(label, output_size):
    height, width = label.shape
    target_h, target_w = output_size
    if height == target_h and width == target_w:
        return label
    return zoom(label, (target_h / height, target_w / width), order=0)


def image_to_tensor(image):
    if image.ndim == 2:
        image = image[..., None]
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image.transpose(2, 0, 1))
    return image


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        image = resize_image(image, self.output_size)
        label = resize_label(label, self.output_size)

        sample["image"] = image_to_tensor(image)
        sample["label"] = torch.from_numpy(label.astype(np.int64))
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = resize_image(image, self.output_size)
        label = resize_label(label, self.output_size)

        sample["image"] = image_to_tensor(image)
        sample["label"] = torch.from_numpy(label.astype(np.int64))
        return sample


class MyDataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.base_dir = Path(base_dir)
        self.split = split
        self.transform = transform
        self.image_dir = self.base_dir / split / "images"
        self.mask_dir = self.base_dir / split / "masks"
        self.samples = self._collect_samples()

    def _collect_samples(self):
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        samples = []
        for image_path in sorted(self.image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMG_EXTENSIONS:
                continue

            mask_path = None
            for suffix in IMG_EXTENSIONS:
                candidate = self.mask_dir / f"{image_path.stem}{suffix}"
                if candidate.exists():
                    mask_path = candidate
                    break

            if mask_path is None:
                raise FileNotFoundError(f"Mask not found for image: {image_path.name}")

            samples.append((image_path, mask_path, image_path.stem))

        if not samples:
            raise RuntimeError(f"No image/mask pairs found under {self.base_dir / self.split}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path, case_name = self.samples[idx]

        image = np.array(Image.open(image_path).convert("RGB"))
        label = np.array(Image.open(mask_path))
        label = remap_mask(label)

        sample = {"image": image, "label": label, "case_name": case_name}
        if self.transform is not None:
            sample = self.transform(sample)
            sample["case_name"] = case_name
        return sample
