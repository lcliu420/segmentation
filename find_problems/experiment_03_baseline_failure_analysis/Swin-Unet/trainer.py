from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.gastric_dataset import GastricSegmentationDataset
from utils import DiceLoss, dice_iou_from_logits


def _setup_logging(snapshot_path: str) -> None:
    log_path = os.path.join(snapshot_path, "log.txt")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def _make_loader(args, split: str, shuffle: bool, augment: bool) -> DataLoader:
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


def _write_history_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr", "is_best"])


def _append_history(path: Path, row: dict[str, object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr", "is_best"],
        )
        writer.writerow(row)


def _log_foreground_sanity(loader: DataLoader, split: str, max_items: int = 32) -> None:
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


def trainer_gastric(args, model: nn.Module, snapshot_path: str, device: torch.device) -> str:
    Path(snapshot_path).mkdir(parents=True, exist_ok=True)
    _setup_logging(snapshot_path)
    logging.info(str(args))

    train_loader = _make_loader(args, "train", shuffle=True, augment=True)
    val_loader = _make_loader(args, "val", shuffle=False, augment=False)
    logging.info("Train samples: %d", len(train_loader.dataset))
    logging.info("Val samples: %d", len(val_loader.dataset))
    _log_foreground_sanity(train_loader, "train")
    _log_foreground_sanity(val_loader, "val")

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.max_epochs, 1))

    history_path = Path(snapshot_path) / "history.csv"
    _write_history_header(history_path)

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

        _append_history(
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
    return "Training Finished!"
