from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

from PIL import Image, ImageEnhance
from torch import Tensor


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def _ensure_hw_tuple(size: Union[int, Sequence[int]]) -> Tuple[int, int]:
    """将输入尺寸统一解析为 `(height, width)`。"""

    if isinstance(size, int):
        return size, size

    if len(size) != 2:
        raise ValueError("输入尺寸必须是 int 或长度为 2 的序列。")

    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError("输入尺寸必须为正整数。")

    return height, width


class PairCompose:
    """顺序执行多个图像与 mask 的联合变换。"""

    def __init__(
        self,
        transforms: Sequence[
            Callable[[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]
        ],
    ) -> None:
        self.transforms = list(transforms)

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


class ResizePair:
    """
    同步调整图像和 mask 的尺寸。

    - 图像使用双线性插值，保证视觉平滑
    - mask 使用最近邻插值，避免标签值被插值污染
    """

    def __init__(self, size: Union[int, Sequence[int]]) -> None:
        self.height, self.width = _ensure_hw_tuple(size)

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        target_size = (self.width, self.height)
        image = image.resize(target_size, resample=Image.BILINEAR)
        mask = mask.resize(target_size, resample=Image.NEAREST)
        return image, mask


class RandomHorizontalFlipPair:
    """以指定概率对图像和 mask 同时做水平翻转。"""

    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.probability:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask


class RandomVerticalFlipPair:
    """以指定概率对图像和 mask 同时做垂直翻转。"""

    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.probability:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask


class RandomRotatePair:
    """
    以指定概率对图像和 mask 同时做随机旋转。

    图像使用双线性插值，mask 使用最近邻插值，以避免标签边界被平滑。
    """

    def __init__(self, degrees: float = 15.0, probability: float = 0.5) -> None:
        if degrees < 0:
            raise ValueError("旋转角度范围必须大于等于 0。")
        self.degrees = degrees
        self.probability = probability

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() >= self.probability:
            return image, mask

        angle = random.uniform(-self.degrees, self.degrees)
        image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)
        return image, mask


class RandomColorJitterPair:
    """
    仅对图像做颜色扰动，不修改 mask。

    当前保留最基础的亮度、对比度和饱和度扰动，
    适合作为 baseline 阶段的轻量增强。
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
        probability: float = 0.5,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.probability = probability

    @staticmethod
    def _sample_factor(strength: float) -> float:
        if strength <= 0:
            return 1.0
        low = max(0.0, 1.0 - strength)
        high = 1.0 + strength
        return random.uniform(low, high)

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() >= self.probability:
            return image, mask

        image = ImageEnhance.Brightness(image).enhance(
            self._sample_factor(self.brightness)
        )
        image = ImageEnhance.Contrast(image).enhance(
            self._sample_factor(self.contrast)
        )
        image = ImageEnhance.Color(image).enhance(
            self._sample_factor(self.saturation)
        )
        return image, mask


@dataclass(frozen=True)
class NormalizeTensor:
    """对图像 tensor 做逐通道归一化。"""

    mean: Tuple[float, float, float] = DEFAULT_MEAN
    std: Tuple[float, float, float] = DEFAULT_STD

    def __call__(self, image: Tensor) -> Tensor:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("输入图像 tensor 形状必须为 `3 x H x W`。")

        for channel, (mean, std) in enumerate(zip(self.mean, self.std)):
            image[channel] = (image[channel] - mean) / std
        return image


def build_transforms(
    split: str,
    image_size: Union[int, Sequence[int]] = (512, 512),
    enable_augmentation: bool = True,
    normalize: bool = True,
) -> Tuple[
    Optional[Callable[[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]],
    Optional[Callable[[Tensor], Tensor]],
    Optional[Callable[[Tensor], Tensor]],
]:
    """
    构建当前实验使用的预处理与增强流程。

    返回值依次为：
    - joint_transform: 图像和 mask 共享的空间变换
    - image_transform: 仅作用于图像 tensor 的变换
    - mask_transform: 仅作用于 mask tensor 的变换，当前阶段暂不需要
    """

    transforms: List[
        Callable[[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]
    ] = [ResizePair(image_size)]

    if split == "train" and enable_augmentation:
        transforms.extend(
            [
                RandomHorizontalFlipPair(probability=0.5),
                RandomVerticalFlipPair(probability=0.2),
                RandomRotatePair(degrees=15.0, probability=0.5),
                RandomColorJitterPair(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    probability=0.5,
                ),
            ]
        )

    joint_transform = PairCompose(transforms)
    image_transform = NormalizeTensor() if normalize else None
    mask_transform = None
    return joint_transform, image_transform, mask_transform
