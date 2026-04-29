from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


SUPPORTED_MODALITIES = ("WL", "NBI")
SUPPORTED_SPLITS = ("train", "val", "test")
IMAGE_SUFFIX = ".jpg"
MASK_SUFFIX = ".png"


@dataclass(frozen=True)
class SampleInfo:
    """保存单个样本的图像路径和 mask 路径。"""

    image_path: Path
    mask_path: Path


class GastricSegmentationDataset(Dataset):
    """
    胃部医学图像分割数据集。

    设计目标：
    - 从 `sorted_075/<modality>/<split>/` 读取数据
    - 按文件名自动配对图像和 mask
    - 将原始 mask 的 `0/2` 编码动态映射为训练使用的 `0/1`
    - 不改动 `sorted_075` 下的任何原始文件

    参数说明：
    - data_root: 数据集根目录，默认对应 `sorted_075`
    - modality: 模态，当前支持 `WL` 和 `NBI`
    - split: 数据划分，当前支持 `train`、`val`、`test`
    - joint_transform: 对图像和 mask 同时生效的变换，后续可用于 resize 或空间增强
    - image_transform: 仅作用于图像的变换
    - mask_transform: 仅作用于 mask 的变换
    - return_paths: 是否在返回结果中附带原始文件路径，方便调试和可视化
    - validate_pairs: 初始化时是否校验图像与 mask 是否完全配对
    """

    def __init__(
        self,
        data_root: Union[str, Path] = "sorted_075",
        modality: str = "WL",
        split: str = "train",
        joint_transform: Optional[
            Callable[[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]
        ] = None,
        image_transform: Optional[Callable[[Tensor], Tensor]] = None,
        mask_transform: Optional[Callable[[Tensor], Tensor]] = None,
        return_paths: bool = False,
        validate_pairs: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.modality = modality
        self.split = split
        self.joint_transform = joint_transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.return_paths = return_paths

        self._validate_inputs()
        self.samples = self._collect_samples(validate_pairs=validate_pairs)

        if not self.samples:
            raise RuntimeError(
                f"在 {self.data_root / self.modality / self.split} 下没有找到可用样本。"
            )

    def _validate_inputs(self) -> None:
        if self.modality not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"不支持的模态: {self.modality}，当前仅支持 {SUPPORTED_MODALITIES}。"
            )

        if self.split not in SUPPORTED_SPLITS:
            raise ValueError(
                f"不支持的 split: {self.split}，当前仅支持 {SUPPORTED_SPLITS}。"
            )

        if not self.data_root.exists():
            raise FileNotFoundError(f"数据根目录不存在: {self.data_root}")

    def _collect_samples(self, validate_pairs: bool) -> List[SampleInfo]:
        split_root = self.data_root / self.modality / self.split
        image_dir = split_root / "images"
        mask_dir = split_root / "masks"

        if not image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {image_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask 目录不存在: {mask_dir}")

        image_map = {path.stem: path for path in sorted(image_dir.glob(f"*{IMAGE_SUFFIX}"))}
        mask_map = {path.stem: path for path in sorted(mask_dir.glob(f"*{MASK_SUFFIX}"))}

        if validate_pairs:
            image_keys = set(image_map)
            mask_keys = set(mask_map)

            only_images = sorted(image_keys - mask_keys)
            only_masks = sorted(mask_keys - image_keys)

            if only_images or only_masks:
                message = [
                    f"{self.modality}/{self.split} 中图像与 mask 配对不完整。"
                ]
                if only_images:
                    message.append(f"仅存在图像的样本数: {len(only_images)}")
                if only_masks:
                    message.append(f"仅存在 mask 的样本数: {len(only_masks)}")
                raise RuntimeError(" ".join(message))

        common_keys = sorted(set(image_map) & set(mask_map))
        return [
            SampleInfo(image_path=image_map[key], mask_path=mask_map[key])
            for key in common_keys
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str]]:
        sample = self.samples[index]

        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path).convert("L")

        if image.size != mask.size:
            raise ValueError(
                f"图像与 mask 尺寸不一致: {sample.image_path.name} -> "
                f"{image.size} vs {mask.size}"
            )

        if self.joint_transform is not None:
            image, mask = self.joint_transform(image, mask)

        image_tensor = self._image_to_tensor(image)
        mask_tensor = self._mask_to_tensor(mask)

        if self.image_transform is not None:
            image_tensor = self.image_transform(image_tensor)
        if self.mask_transform is not None:
            mask_tensor = self.mask_transform(mask_tensor)

        result: Dict[str, Union[Tensor, str]] = {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_name": sample.image_path.name,
        }

        if self.return_paths:
            result["image_path"] = str(sample.image_path)
            result["mask_path"] = str(sample.mask_path)

        return result

    @staticmethod
    def _image_to_tensor(image: Image.Image) -> Tensor:
        """
        将 RGB 图像转换为 `C x H x W` 的 float tensor。

        输出范围为 `[0, 1]`，暂不做归一化，归一化将在后续预处理阶段补充。
        """

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))
        return torch.from_numpy(image_array)

    @staticmethod
    def _mask_to_tensor(mask: Image.Image) -> Tensor:
        """
        将原始 mask 转换为单通道二值 tensor。

        当前数据集中 mask 的像素值实际为 `0/2`，这里统一映射为：
        - 背景: 0.0
        - 前景: 1.0
        """

        mask_array = np.asarray(mask, dtype=np.uint8)
        binary_mask = (mask_array > 0).astype(np.float32)
        binary_mask = np.expand_dims(binary_mask, axis=0)
        return torch.from_numpy(binary_mask)


def build_dataset(
    data_root: Union[str, Path],
    modality: str,
    split: str,
    joint_transform: Optional[
        Callable[[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]
    ] = None,
    image_transform: Optional[Callable[[Tensor], Tensor]] = None,
    mask_transform: Optional[Callable[[Tensor], Tensor]] = None,
    return_paths: bool = False,
) -> GastricSegmentationDataset:
    """构建一个数据集实例，供训练和测试脚本复用。"""

    return GastricSegmentationDataset(
        data_root=data_root,
        modality=modality,
        split=split,
        joint_transform=joint_transform,
        image_transform=image_transform,
        mask_transform=mask_transform,
        return_paths=return_paths,
    )
