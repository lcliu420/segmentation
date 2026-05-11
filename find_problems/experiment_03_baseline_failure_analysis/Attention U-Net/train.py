from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.gastric_dataset import GastricSegmentationDataset
from models.attention_unet import AttentionUNet
from utils import DiceLoss, dice_iou_from_logits


ATTENTION_UNET_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ATTENTION_UNET_ROOT.parent
REPO_ROOT = EXPERIMENT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Attention U-Net for WL/NBI gastric segmentation.")
    parser.add_argument("--data_root", type=str, default=str(REPO_ROOT / "dataset"))
    parser.add_argument("--modal", type=str, choices=["WL", "NBI"], required=True)
    parser.add_argument("--model_name", type=str, default="attention_unet")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--limit_samples", type=int, default=None, help="Optional smoke-test limit per split.")
    return parser.parse_args()


def seed_everything(seed: int, deterministic: bool) -> None:
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(snapshot_path: str) -> None:
    log_path = os.path.join(snapshot_path, "log.txt")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def make_loader(args: argparse.Namespace, split: str, shuffle: bool, augment: bool) -> DataLoader:
    dataset = GastricSegmentationDataset(
        data_root=args.data_root,
        modal=args.modal,
        split=split,
        img_size=args.img_size,
        augment=augment,
        limit_samples=args.limit_samples,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def log_foreground_sanity(loader: DataLoader, split: str, max_items: int = 32) -> None:
    summary = loader.dataset.foreground_ratio_summary(max_items=max_items)
    logging.info(
        "%s foreground ratio sanity over first %d samples: min=%.6f mean=%.6f max=%.6f",
        split,
        summary["count"],
        summary["min"],
        summary["mean"],
        summary["max"],
    )
    if summary["count"] == 0 or summary["max"] <= 0.0:
        raise RuntimeError(
            f"{split} masks appear to be all background in the foreground sanity check. "
            "Check mask values and data_root before training."
        )


def write_history_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr", "is_best"])


def append_history(path: Path, row: dict[str, object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr", "is_best"],
        )
        writer.writerow(row)


def train(args: argparse.Namespace, model: nn.Module, snapshot_path: str, device: torch.device) -> None:
    Path(snapshot_path).mkdir(parents=True, exist_ok=True)
    setup_logging(snapshot_path)
    logging.info(str(args))

    train_loader = make_loader(args, "train", shuffle=True, augment=True)
    val_loader = make_loader(args, "val", shuffle=False, augment=False)
    logging.info("Train samples: %d", len(train_loader.dataset))
    logging.info("Val samples: %d", len(val_loader.dataset))
    log_foreground_sanity(train_loader, "train")
    log_foreground_sanity(val_loader, "val")

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.max_epochs, 1))

    history_path = Path(snapshot_path) / "history.csv"
    write_history_header(history_path)

    best_val_dice = -1.0
    logging.info("#################### Start Training ####################")
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        train_bar = tqdm(train_loader, desc=f"Train {epoch:03d}/{args.max_epochs:03d}", leave=False)
        for batch in train_bar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            batch_size = images.size(0)
            outputs = model(images)
            loss_ce = ce_loss(outputs, labels)
            loss_dice = dice_loss(outputs, labels, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_size
            train_sample_count += batch_size
            train_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        train_loss = train_loss_sum / max(train_sample_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_dice_sum = 0.0
        val_iou_sum = 0.0
        val_sample_count = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Val {epoch:03d}/{args.max_epochs:03d}", leave=False)
            for batch in val_bar:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                batch_size = images.size(0)
                outputs = model(images)
                loss_ce = ce_loss(outputs, labels)
                loss_dice = dice_loss(outputs, labels, softmax=True)
                loss = 0.4 * loss_ce + 0.6 * loss_dice
                dice, iou = dice_iou_from_logits(outputs, labels)
                val_loss_sum += loss.item() * batch_size
                val_dice_sum += dice * batch_size
                val_iou_sum += iou * batch_size
                val_sample_count += batch_size

        val_loss = val_loss_sum / max(val_sample_count, 1)
        val_dice = val_dice_sum / max(val_sample_count, 1)
        val_iou = val_iou_sum / max(val_sample_count, 1)
        lr = optimizer.param_groups[0]["lr"]
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(snapshot_path, "best_model.pth"))
        torch.save(model.state_dict(), os.path.join(snapshot_path, "last_model.pth"))

        append_history(
            history_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "lr": lr,
                "is_best": int(is_best),
            },
        )
        logging.info(
            "epoch=%03d train_loss=%.6f val_loss=%.6f val_dice=%.6f val_iou=%.6f lr=%.6e best=%s",
            epoch,
            train_loss,
            val_loss,
            val_dice,
            val_iou,
            lr,
            is_best,
        )
        scheduler.step()

    logging.info("Training finished. Best val Dice: %.6f", best_val_dice)


def main() -> None:
    args = parse_args()
    args.num_classes = 2
    if args.run_name is None:
        args.run_name = f"{args.model_name}_{args.modal}"
    if args.output_dir is None:
        args.output_dir = str(EXPERIMENT_ROOT / "outputs" / "runs" / args.run_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed, bool(args.deterministic))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = AttentionUNet(in_channels=3, num_classes=args.num_classes).to(device)
    train(args, model, args.output_dir, device)


if __name__ == "__main__":
    main()
