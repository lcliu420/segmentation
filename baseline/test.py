"""
胃部医学图像分割 baseline 的测试入口。

当前版本已经具备：
- 加载训练好的 checkpoint
- 在 test 集上做推理
- 统计 Dice / IoU / Precision / Recall
- 可选保存预测 mask
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import GastricSegmentationDataset, build_transforms
from models import build_unet
from utils import compute_segmentation_metrics, ensure_dir, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="测试 WL 或 NBI 模态的分割 baseline。"
    )
    parser.add_argument(
        "--modality",
        choices=["WL", "NBI"],
        required=True,
        help="选择需要测试的模态。",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("sorted_075"),
        help="数据集根目录。",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="训练完成后的模型权重路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="用于保存评估指标和预测结果的目录。",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="测试结果保存目录名；如果不传，则默认使用 `<modality>_unet`。",
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=[512, 512],
        help="测试时统一使用的输入尺寸。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="测试 batch size。",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader 的 worker 数量；Windows 下默认建议设为 0。",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=64,
        help="U-Net 的基础通道数，需要与训练时保持一致。",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="测试时使用的二值化阈值。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="测试设备选择。",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="在 CUDA 环境下启用自动混合精度推理。",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="保存测试集预测 mask。",
    )
    return parser


def resolve_device(device_arg: str) -> torch.device:
    """根据命令行参数解析测试设备。"""

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境不可用 CUDA，请改用 `--device cpu` 或 `--device auto`。")

    return torch.device(device_arg)


def build_test_dataloader(args: argparse.Namespace) -> DataLoader:
    """构建测试集 DataLoader。"""

    joint_transform, image_transform, mask_transform = build_transforms(
        split="test",
        image_size=tuple(args.image_size),
        enable_augmentation=False,
        normalize=True,
    )
    dataset = GastricSegmentationDataset(
        data_root=args.data_root,
        modality=args.modality,
        split="test",
        joint_transform=joint_transform,
        image_transform=image_transform,
        mask_transform=mask_transform,
        return_paths=True,
    )

    common_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device != "cpu" or torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        common_kwargs["persistent_workers"] = True

    return DataLoader(dataset, shuffle=False, **common_kwargs)


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """构建模型并加载训练好的权重。"""

    model = build_unet(
        in_channels=3,
        num_classes=1,
        base_channels=args.base_channels,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def save_prediction_mask(mask: np.ndarray, output_path: Path) -> None:
    """将二值预测结果保存为 0/255 的 PNG 图像。"""

    output_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    output_image.save(output_path)


@torch.no_grad()
def run_test(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float,
    prediction_dir: Optional[Path] = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    """在测试集上执行推理并统计指标。"""

    total_samples = 0
    metric_sums = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    progress = tqdm(dataloader, desc="测试中", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        image_names = batch["image_name"]
        batch_size = images.size(0)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)

        batch_metrics = compute_segmentation_metrics(logits, masks, threshold)
        for name, value in batch_metrics.items():
            metric_sums[name] += value * batch_size
        total_samples += batch_size

        if prediction_dir is not None:
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float().cpu().numpy()
            for index, image_name in enumerate(image_names):
                pred_mask = preds[index, 0]
                output_name = Path(image_name).with_suffix(".png").name
                save_prediction_mask(pred_mask, prediction_dir / output_name)

        progress.set_postfix(dice=f"{batch_metrics['dice']:.4f}")

    if total_samples == 0:
        raise RuntimeError("测试集没有样本，无法计算指标。")

    return {
        "dice": metric_sums["dice"] / total_samples,
        "iou": metric_sums["iou"] / total_samples,
        "precision": metric_sums["precision"] / total_samples,
        "recall": metric_sums["recall"] / total_samples,
    }


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    experiment_name = args.experiment_name or f"{args.modality.lower()}_unet"
    experiment_dir = ensure_dir(args.output_dir / experiment_name)
    prediction_dir = (
        ensure_dir(experiment_dir / "test_predictions") if args.save_predictions else None
    )

    dataloader = build_test_dataloader(args)
    model = load_model(args, device)

    print(f"测试设备：{device}")
    print(f"混合精度：{use_amp}")
    print(f"测试集样本数：{len(dataloader.dataset)}")
    print(f"使用权重：{args.checkpoint}")

    metrics = run_test(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=args.threshold,
        prediction_dir=prediction_dir,
        use_amp=use_amp,
    )

    result = {
        "modality": args.modality,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "amp_enabled": use_amp,
        "image_size": list(args.image_size),
        "threshold": args.threshold,
        "metrics": metrics,
    }
    save_json(result, experiment_dir / "test_metrics.json")

    print(
        f"测试完成：dice={metrics['dice']:.4f}, "
        f"iou={metrics['iou']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}"
    )
    print(f"测试结果已保存到：{experiment_dir / 'test_metrics.json'}")
    if prediction_dir is not None:
        print(f"预测 mask 已保存到：{prediction_dir}")


if __name__ == "__main__":
    main()
