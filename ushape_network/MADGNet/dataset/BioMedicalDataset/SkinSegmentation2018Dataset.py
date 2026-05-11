import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class MedicalSegmentationDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        split,
        image_size=352,
        augment=False,
        boundary_width=3,
        image_dir_name="images",
        mask_dir_name="masks",
    ):
        super(MedicalSegmentationDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.boundary_width = boundary_width
        self.image_dir_name = image_dir_name
        self.mask_dir_name = mask_dir_name
        self.samples = self.build_samples()

        print("{} split size: {}".format(split, len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label_path, sample_name = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(label_path).convert("L")

        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        if self.augment:
            image, mask = self.apply_augmentation(image, mask)

        image_tensor = TF.to_tensor(image)

        # Treat any non-zero grayscale value as foreground so masks like {0, 2}
        # are binarized correctly instead of collapsing to all background.
        mask_np = (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)
        region = torch.from_numpy(mask_np).unsqueeze(0).float()
        distance_np = self.build_distance_map(mask_np)
        boundary_np = self.build_boundary_map(mask_np)

        distance = torch.from_numpy(distance_np).unsqueeze(0).float()
        boundary = torch.from_numpy(boundary_np).unsqueeze(0).float()

        return {
            "image": image_tensor,
            "region": region,
            "distance": distance,
            "boundary": boundary,
            "name": sample_name,
        }

    def build_samples(self):
        frame_path = os.path.join(self.dataset_dir, "{}_frame.csv".format(self.split))
        if os.path.isfile(frame_path):
            return self.build_samples_from_csv(frame_path)
        return self.build_samples_from_split_dirs()

    def build_samples_from_csv(self, frame_path):
        image_dir = os.path.join(self.dataset_dir, self.image_dir_name)
        label_dir = os.path.join(self.dataset_dir, self.mask_dir_name)
        self.validate_directory(image_dir, "image")
        self.validate_directory(label_dir, "mask")

        frame = pd.read_csv(frame_path)
        mask_index = self.build_mask_index(label_dir)
        samples = []

        for _, row in frame.iterrows():
            image_name = self.get_first_valid_value(row, ["image_path", "image", "image_name"])
            if image_name is None:
                raise KeyError("Cannot find image column in {}".format(frame_path))

            mask_name = self.get_first_valid_value(row, ["new_mask_path", "mask_path", "label_path", "mask", "mask_name"])
            if mask_name is None:
                mask_name = self.match_mask_name(image_name, mask_index, label_dir)

            samples.append(
                (
                    os.path.join(image_dir, image_name),
                    os.path.join(label_dir, mask_name),
                    os.path.splitext(os.path.basename(image_name))[0],
                )
            )

        return samples

    def build_samples_from_split_dirs(self):
        image_dir = os.path.join(self.dataset_dir, self.split, self.image_dir_name)
        label_dir = os.path.join(self.dataset_dir, self.split, self.mask_dir_name)
        self.validate_directory(image_dir, "image")
        self.validate_directory(label_dir, "mask")

        image_names = sorted(
            file_name for file_name in os.listdir(image_dir) if self.is_image_file(file_name)
        )
        mask_index = self.build_mask_index(label_dir)
        samples = []

        for image_name in image_names:
            mask_name = self.match_mask_name(image_name, mask_index, label_dir)
            samples.append(
                (
                    os.path.join(image_dir, image_name),
                    os.path.join(label_dir, mask_name),
                    os.path.splitext(image_name)[0],
                )
            )

        return samples

    def get_first_valid_value(self, row, column_names):
        for column_name in column_names:
            if column_name in row and isinstance(row[column_name], str) and row[column_name].strip():
                return row[column_name].strip()
        return None

    def validate_directory(self, directory_path, directory_type):
        if not os.path.isdir(directory_path):
            raise FileNotFoundError("Cannot find {} directory: {}".format(directory_type, directory_path))

    def build_mask_index(self, label_dir):
        mask_index = {}
        for file_name in os.listdir(label_dir):
            if not self.is_image_file(file_name):
                continue
            stem = os.path.splitext(file_name)[0].lower()
            mask_index[stem] = file_name
        return mask_index

    def match_mask_name(self, image_name, mask_index, label_dir):
        image_stem = os.path.splitext(image_name)[0].lower()
        for candidate in [image_stem, "{}_segmentation".format(image_stem)]:
            if candidate in mask_index:
                return mask_index[candidate]
        raise FileNotFoundError(
            "Cannot find mask for image '{}' in '{}'".format(image_name, label_dir)
        )

    def is_image_file(self, file_name):
        return os.path.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS

    def apply_augmentation(self, image, mask):
        if random.random() < 0.8:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
            image = TF.adjust_saturation(image, random.uniform(0.85, 1.15))
            image = TF.adjust_hue(image, random.uniform(-0.03, 0.03))

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.3:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        width, height = image.size
        angle = random.uniform(-15.0, 15.0)
        translate = (
            int(random.uniform(-0.05, 0.05) * width),
            int(random.uniform(-0.05, 0.05) * height),
        )
        scale = random.uniform(0.9, 1.1)
        shear = random.uniform(-8.0, 8.0)
        image = TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        mask = TF.affine(
            mask,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )

        if random.random() < 0.2:
            image = TF.gaussian_blur(image, kernel_size=5, sigma=random.uniform(0.1, 1.5))

        return image, mask

    def build_boundary_map(self, mask):
        dilated = ndimage.binary_dilation(mask, iterations=self.boundary_width)
        eroded = ndimage.binary_erosion(mask, iterations=self.boundary_width)
        boundary = np.logical_xor(dilated, eroded).astype(np.float32)
        return boundary

    def build_distance_map(self, mask):
        if mask.max() == 0:
            return np.zeros_like(mask, dtype=np.float32)

        distance = ndimage.distance_transform_edt(mask)
        distance = distance.astype(np.float32)
        max_value = float(distance.max())
        if max_value > 0:
            distance /= max_value
        return distance


ISIC2018Dataset = MedicalSegmentationDataset
