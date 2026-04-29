"""
胃部医学图像分割 baseline 的训练入口。

当前版本已经具备：
- 单模态训练：`WL` 或 `NBI`
- 训练集与验证集加载
- 标准 U-Net 构建
- BCE + Dice 联合损失
- 验证集 Dice / IoU / Precision / Recall 统计
- 最优模型与最新模型保存
- 训练日志导出
- GPU 训练支持
- 可选混合精度训练
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import GastricSegmentationDataset, build_transforms
from models import build_unet
from utils import (
    BCEDiceLoss,
    build_experiment_name,
    compute_segmentation_metrics,
    ensure_dir,
    save_history_csv,
    save_json,
    set_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="训练 WL 或 NBI 模态的分割 baseline。"
    )
    parser.add_argument(
        "--modality",
        choices=["WL", "NBI"],
        required=True,
        help="选择需要训练的模态。",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("sorted_075"),
        help="数据集根目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="用于保存日志、模型权重和训练配置的目录。",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="实验名称；如果不传，则默认使用 `<modality>_unet`。",
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=[512, 512],
        help="训练和验证时统一使用的输入尺寸。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="训练和验证的 batch size。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="总训练轮数。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="初始学习率。",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="权重衰减。",
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
        help="U-Net 的基础通道数。",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="验证指标计算时使用的二值化阈值。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="训练设备选择。",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "cosine"],
        help="学习率调度器类型。",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="在 CUDA 环境下启用自动混合精度训练。",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="关闭训练集数据增强。",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="关闭图像归一化。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅跑一个训练 batch 和一个验证 batch，用于检查流程是否通畅。",
    )
    return parser


def resolve_device(device_arg: str) -> torch.device:
    """根据命令行参数解析训练设备。"""

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境不可用 CUDA，请改用 `--device cpu` 或 `--device auto`。")

    return torch.device(device_arg)


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """构建训练集与验证集 DataLoader。"""

    image_size = tuple(args.image_size)

    train_joint_transform, train_image_transform, train_mask_transform = build_transforms(
        split="train",
        image_size=image_size,
        enable_augmentation=not args.no_augmentation,
        normalize=not args.no_normalize,
    )
    val_joint_transform, val_image_transform, val_mask_transform = build_transforms(
        split="val",
        image_size=image_size,
        enable_augmentation=False,
        normalize=not args.no_normalize,
    )

    train_dataset = GastricSegmentationDataset(
        data_root=args.data_root,
        modality=args.modality,
        split="train",
        joint_transform=train_joint_transform,
        image_transform=train_image_transform,
        mask_transform=train_mask_transform,
    )
    val_dataset = GastricSegmentationDataset(
        data_root=args.data_root,
        modality=args.modality,
        split="val",
        joint_transform=val_joint_transform,
        image_transform=val_image_transform,
        mask_transform=val_mask_transform,
    )

    common_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device != "cpu" or torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        common_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    return train_loader, val_loader


def create_experiment_dirs(
    output_dir: Path, experiment_name: str
) -> Tuple[Path, Path, Path]:
    """创建实验目录、checkpoint 目录和日志目录。"""

    experiment_dir = ensure_dir(output_dir / experiment_name)
    checkpoint_dir = ensure_dir(experiment_dir / "checkpoints")
    log_dir = ensure_dir(experiment_dir / "logs")
    return experiment_dir, checkpoint_dir, log_dir


def move_batch_to_device(
    batch: Dict[str, Union[Tensor, str]], device: torch.device
) -> Tuple[Tensor, Tensor]:
    """将一个 batch 中的图像和 mask 移动到目标设备。"""

    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    return images, masks


def aggregate_metric_sums(
    metric_sums: Dict[str, float],
    batch_metrics: Dict[str, float],
    batch_size: int,
) -> None:
    """按 batch size 累加指标，便于最后求 epoch 平均值。"""

    for name, value in batch_metrics.items():
        metric_sums[name] += value * batch_size


def finalize_epoch_metrics(
    metric_sums: Dict[str, float],
    total_loss: float,
    total_samples: int,
) -> Dict[str, float]:
    """将累计值转换为当前 epoch 的平均指标。"""

    if total_samples == 0:
        raise RuntimeError("当前 epoch 没有处理任何样本。")

    return {
        "loss": total_loss / total_samples,
        "dice": metric_sums["dice"] / total_samples,
        "iou": metric_sums["iou"] / total_samples,
        "precision": metric_sums["precision"] / total_samples,
        "recall": metric_sums["recall"] / total_samples,
    }


def run_train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Adam,
    scaler: Optional[GradScaler],
    device: torch.device,
    threshold: float,
    epoch_index: int,
    total_epochs: int,
    use_amp: bool = False,
    dry_run: bool = False,
) -> Dict[str, float]:
    """执行一个训练 epoch。"""

    model.train()
    total_loss = 0.0
    total_samples = 0
    metric_sums = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    progress = tqdm(
        dataloader,
        desc=f"训练 Epoch [{epoch_index}/{total_epochs}]",
        leave=False,
    )

    for batch in progress:
        images, masks = move_batch_to_device(batch, device)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if use_amp:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_metrics = compute_segmentation_metrics(logits.detach(), masks, threshold)

        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        aggregate_metric_sums(metric_sums, batch_metrics, batch_size)

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{batch_metrics['dice']:.4f}",
        )

        if dry_run:
            break

    return finalize_epoch_metrics(metric_sums, total_loss, total_samples)


@torch.no_grad()
def run_validation_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    epoch_index: int,
    total_epochs: int,
    use_amp: bool = False,
    dry_run: bool = False,
) -> Dict[str, float]:
    """执行一个验证 epoch。"""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_sums = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    progress = tqdm(
        dataloader,
        desc=f"验证 Epoch [{epoch_index}/{total_epochs}]",
        leave=False,
    )

    for batch in progress:
        images, masks = move_batch_to_device(batch, device)
        batch_size = images.size(0)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        batch_metrics = compute_segmentation_metrics(logits, masks, threshold)

        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        aggregate_metric_sums(metric_sums, batch_metrics, batch_size)

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{batch_metrics['dice']:.4f}",
        )

        if dry_run:
            break

    return finalize_epoch_metrics(metric_sums, total_loss, total_samples)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Adam,
    scheduler: Optional[CosineAnnealingLR],
    scaler: Optional[GradScaler],
    epoch: int,
    best_val_dice: float,
    args: argparse.Namespace,
) -> None:
    """保存模型权重和训练状态。"""

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_dice": best_val_dice,
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def maybe_build_scheduler(
    optimizer: Adam,
    scheduler_name: str,
    epochs: int,
) -> Optional[CosineAnnealingLR]:
    """按配置构建学习率调度器。"""

    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"不支持的 scheduler: {scheduler_name}")


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    experiment_name = args.experiment_name or build_experiment_name(
        modality=args.modality,
        model_name="unet",
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        scheduler=args.scheduler,
        use_amp=use_amp,
    )
    experiment_dir, checkpoint_dir, log_dir = create_experiment_dirs(
        args.output_dir, experiment_name
    )

    train_loader, val_loader = build_dataloaders(args)

    model = build_unet(
        in_channels=3,
        num_classes=1,
        base_channels=args.base_channels,
    ).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = maybe_build_scheduler(optimizer, args.scheduler, args.epochs)
    scaler = GradScaler(enabled=use_amp)

    config = {
        "modality": args.modality,
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "experiment_name": experiment_name,
        "image_size": list(args.image_size),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "base_channels": args.base_channels,
        "threshold": args.threshold,
        "seed": args.seed,
        "device": str(device),
        "scheduler": args.scheduler,
        "amp_enabled": use_amp,
        "augmentation_enabled": not args.no_augmentation,
        "normalize_enabled": not args.no_normalize,
        "dry_run": args.dry_run,
        "loss": "BCE + Dice",
        "best_model_rule": "验证集 Dice 最大",
    }
    save_json(config, experiment_dir / "config.json")

    print(f"实验名称：{experiment_name}")
    print(f"训练设备：{device}")
    print(f"混合精度：{use_amp}")
    print(f"训练集样本数：{len(train_loader.dataset)}")
    print(f"验证集样本数：{len(val_loader.dataset)}")
    print(f"日志目录：{log_dir}")
    print(f"权重目录：{checkpoint_dir}")

    best_val_dice = -1.0
    history: List[Dict[str, Union[float, int]]] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            threshold=args.threshold,
            epoch_index=epoch,
            total_epochs=args.epochs,
            use_amp=use_amp,
            dry_run=args.dry_run,
        )
        val_metrics = run_validation_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
            epoch_index=epoch,
            total_epochs=args.epochs,
            use_amp=use_amp,
            dry_run=args.dry_run,
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_log: Dict[str, Union[float, int]] = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        history.append(epoch_log)
        save_history_csv(history, log_dir / "history.csv")

        save_checkpoint(
            path=checkpoint_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if use_amp else None,
            epoch=epoch,
            best_val_dice=max(best_val_dice, val_metrics["dice"]),
            args=args,
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if use_amp else None,
                epoch=epoch,
                best_val_dice=best_val_dice,
                args=args,
            )
            save_json(
                {
                    "best_epoch": epoch,
                    "best_val_dice": best_val_dice,
                    "best_val_iou": val_metrics["iou"],
                    "best_val_precision": val_metrics["precision"],
                    "best_val_recall": val_metrics["recall"],
                },
                experiment_dir / "best_metrics.json",
            )

        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"lr={current_lr:.6f}"
        )

        if args.dry_run:
            print("dry-run 已完成，提前结束训练。")
            break

    total_minutes = (time.time() - start_time) / 60.0
    save_json(
        {
            "experiment_name": experiment_name,
            "best_val_dice": best_val_dice,
            "epochs_completed": len(history),
            "total_minutes": total_minutes,
            "device": str(device),
            "amp_enabled": use_amp,
        },
        experiment_dir / "summary.json",
    )

    print(f"训练结束，总耗时约 {total_minutes:.2f} 分钟。")
    print(f"最优验证集 Dice：{best_val_dice:.4f}")
    print(f"实验结果已保存到：{experiment_dir}")


if __name__ == "__main__":
    main()
