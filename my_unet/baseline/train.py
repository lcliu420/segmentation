import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from datasets.dataset import SegmentationDataset
from metrics import average_metrics, compute_metrics
from networks.vision_transformer import CSwinUnet


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, inputs, target, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target_one_hot = torch.zeros_like(inputs)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        smooth = 1e-5
        loss = 0.0
        for cls_idx in range(self.n_classes):
            score = inputs[:, cls_idx]
            target_cls = target_one_hot[:, cls_idx]
            intersect = torch.sum(score * target_cls)
            y_sum = torch.sum(target_cls * target_cls)
            z_sum = torch.sum(score * score)
            loss += 1.0 - (2.0 * intersect + smooth) / (z_sum + y_sum + smooth)
        return loss / self.n_classes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../dataset")
    parser.add_argument("--modal", type=str, default="ALL", choices=["ALL", "WL", "NBI"])
    parser.add_argument("--cfg", type=str, default="configs/cswin_tiny_224_lite.yaml", metavar="FILE")
    parser.add_argument("--output_dir", type=str, default="../outputs/B")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--ce_weight", type=float, default=0.4)
    parser.add_argument("--dice_weight", type=float, default=0.6)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)

    # Kept for compatibility with config.py from the original project.
    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache-mode", type=str, default="part", choices=["no", "full", "part"])
    parser.add_argument("--resume")
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--amp-opt-level", type=str, default="O1", choices=["O0", "O1", "O2"])
    parser.add_argument("--tag")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    return parser.parse_args()


def format_param(value):
    return str(value).replace("+", "").replace("-", "m").replace(".", "p")


def build_run_name(args):
    if args.run_name:
        return args.run_name
    return (
        f"bs{args.batch_size}"
        f"_lr{format_param(args.base_lr)}"
        f"_ep{args.max_epochs}"
        f"_img{args.img_size}"
        f"_seed{args.seed}"
        f"_ce{format_param(args.ce_weight)}"
        f"_dice{format_param(args.dice_weight)}"
    )


def set_seed(seed, deterministic=True):
    cudnn.benchmark = not deterministic
    cudnn.deterministic = bool(deterministic)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def prepare_config(args):
    config = get_config(args)
    config.defrost()
    config.DATA.IMG_SIZE = args.img_size
    config.MODEL.NUM_CLASSES = args.num_classes
    config.freeze()
    return config


def build_loaders(args):
    train_set = SegmentationDataset(args.data_root, args.modal, "train", args.img_size, augment=True)
    val_set = SegmentationDataset(args.data_root, args.modal, "val", args.img_size, augment=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def tensor_metrics(pred, target):
    rows = []
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    for idx in range(pred_np.shape[0]):
        rows.append(compute_metrics(pred_np[idx], target_np[idx]))
    return rows


def run_epoch(model, loader, criterion_ce, criterion_dice, optimizer, device, args, epoch, train=True):
    model.train(train)
    total_loss = 0.0
    total_seen = 0
    metric_rows = []
    desc = f"{'Train' if train else 'Val'} {epoch:03d}/{args.max_epochs:03d}"
    progress = tqdm(loader, ncols=100, desc=desc)

    max_batches = args.max_train_batches if train else args.max_val_batches
    for batch_idx, batch in enumerate(progress):
        if max_batches and batch_idx >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).long()

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss_ce = criterion_ce(outputs, labels)
            loss_dice = criterion_dice(outputs, labels, softmax=True)
            loss = args.ce_weight * loss_ce + args.dice_weight * loss_dice

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_seen += batch_size
        preds = torch.argmax(torch.softmax(outputs.detach(), dim=1), dim=1)
        metric_rows.extend(tensor_metrics(preds, labels))
        progress.set_postfix(loss=f"{loss.item():.4f}")

    metrics = average_metrics(metric_rows)
    metrics["loss"] = total_loss / max(total_seen, 1)
    return metrics


def save_history_row(path, row, fieldnames):
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def to_builtin(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def save_best_result(run_dir, args, run_name, row, train_metrics, val_metrics, checkpoint_path):
    record = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "selection_metric": "val_dice",
        "best_epoch": row["epoch"],
        "checkpoint": str(checkpoint_path),
        "args": vars(args).copy(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "flat_metrics": row,
    }
    with (run_dir / "best_result.json").open("w", encoding="utf-8") as f:
        json.dump(to_builtin(record), f, indent=2)

    csv_row = {
        "run_name": run_name,
        "modal": args.modal,
        "best_epoch": row["epoch"],
        "base_lr": args.base_lr,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "img_size": args.img_size,
        "seed": args.seed,
        "ce_weight": args.ce_weight,
        "dice_weight": args.dice_weight,
        "train_loss": row["train_loss"],
        "train_dice": row["train_dice"],
        "train_iou": row["train_iou"],
        "train_boundary_iou": row["train_boundary_iou"],
        "train_hd95_medpy": row["train_hd95_medpy"],
        "train_mae": row["train_mae"],
        "val_loss": row["val_loss"],
        "val_dice": row["val_dice"],
        "val_iou": row["val_iou"],
        "val_boundary_iou": row["val_boundary_iou"],
        "val_hd95_medpy": row["val_hd95_medpy"],
        "val_mae": row["val_mae"],
        "checkpoint": str(checkpoint_path),
    }
    with (run_dir / "best_result.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        writer.writeheader()
        writer.writerow(to_builtin(csv_row))


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    config = prepare_config(args)

    if args.num_classes != 2:
        raise ValueError("This dataset is binary segmentation; please keep --num_classes 2.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = build_run_name(args)
    run_dir = Path(args.output_dir) / args.modal / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.csv"

    config_path = Path(config.MODEL.PRETRAIN_CKPT)
    if not config_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {config_path}")

    train_loader, val_loader = build_loaders(args)
    print(f"Using device: {device}")
    print(f"Dataset: {args.modal} | train={len(train_loader.dataset)} | val={len(val_loader.dataset)}")

    model = CSwinUnet(config, img_size=args.img_size, num_classes=args.num_classes).to(device)
    model.load_from(config)

    criterion_ce = CrossEntropyLoss()
    criterion_dice = DiceLoss(args.num_classes)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    config_record = vars(args).copy()
    config_record["run_name"] = run_name
    config_record["run_dir"] = str(run_dir)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    fields = [
        "epoch", "lr", "train_loss", "train_dice", "train_iou", "train_boundary_iou",
        "train_hd95_medpy", "train_mae", "val_loss", "val_dice", "val_iou",
        "val_boundary_iou", "val_hd95_medpy", "val_mae", "best_dice", "is_best",
    ]
    best_dice = -1.0

    print("#################### Start Training ####################")
    for epoch in range(1, args.max_epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion_ce, criterion_dice, optimizer, device, args, epoch, train=True
        )
        val_metrics = run_epoch(
            model, val_loader, criterion_ce, criterion_dice, optimizer, device, args, epoch, train=False
        )

        val_dice = val_metrics["dice"]
        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
            best_ckpt_path = ckpt_dir / "best.pth"
            torch.save(model.state_dict(), best_ckpt_path)
        latest_ckpt_path = ckpt_dir / "latest.pth"
        torch.save(model.state_dict(), latest_ckpt_path)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "train_boundary_iou": train_metrics["boundary_iou"],
            "train_hd95_medpy": train_metrics["hd95_medpy"],
            "train_mae": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_boundary_iou": val_metrics["boundary_iou"],
            "val_hd95_medpy": val_metrics["hd95_medpy"],
            "val_mae": val_metrics["mae"],
            "best_dice": best_dice,
            "is_best": int(is_best),
        }
        save_history_row(history_path, row, fields)
        if is_best:
            save_best_result(run_dir, args, run_name, row, train_metrics, val_metrics, best_ckpt_path)

        print(
            f"TRAIN | epoch={epoch:03d}/{args.max_epochs:03d} | dataset={args.modal:<3} "
            f"| dice={train_metrics['dice']:.4f} | iou={train_metrics['iou']:.4f} "
            f"| b_iou={train_metrics['boundary_iou']:.4f} | hd95_medpy={train_metrics['hd95_medpy']:.4f} "
            f"| mae={train_metrics['mae']:.4f} | loss={train_metrics['loss']:.4f}"
        )
        print(
            f"VAL   | epoch={epoch:03d}/{args.max_epochs:03d} | dataset={args.modal:<3} "
            f"| dice={val_metrics['dice']:.4f} | iou={val_metrics['iou']:.4f} "
            f"| b_iou={val_metrics['boundary_iou']:.4f} | hd95_medpy={val_metrics['hd95_medpy']:.4f} "
            f"| mae={val_metrics['mae']:.4f} | loss={val_metrics['loss']:.4f}"
        )
        if is_best:
            print(f"BEST  | epoch={epoch:03d}/{args.max_epochs:03d} best_dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
