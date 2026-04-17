from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


CLASS_NAMES = ["NBI", "WL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ResNet18 model for NBI / WL classification."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("prepared_data"),
        help="Dataset root containing train/ and val/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "resnet18",
        help="Directory for checkpoints and reports.",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    freeze_group = parser.add_mutually_exclusive_group()
    freeze_group.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_true",
        help="Freeze ResNet18 backbone and train only the final classifier.",
    )
    freeze_group.add_argument(
        "--full-finetune",
        dest="freeze_backbone",
        action="store_false",
        help="Fine-tune the full ResNet18 model.",
    )
    parser.set_defaults(freeze_backbone=None)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, for example cuda or cpu.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_freeze_backbone(
    freeze_backbone_arg: bool | None, device: torch.device
) -> bool:
    if freeze_backbone_arg is not None:
        return freeze_backbone_arg
    return device.type == "cpu"


def build_transforms(image_size: int) -> dict[str, transforms.Compose]:
    weights = ResNet18_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return {"train": train_transform, "val": val_transform}


def create_dataloaders(
    data_dir: Path, image_size: int, batch_size: int, num_workers: int
) -> tuple[dict[str, DataLoader], dict[str, int]]:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Expected train/ and val/ under {data_dir}")

    data_transforms = build_transforms(image_size)
    datasets_map = {
        split: datasets.ImageFolder(data_dir / split, transform=data_transforms[split])
        for split in ("train", "val")
    }
    class_names = datasets_map["train"].classes
    if class_names != CLASS_NAMES:
        raise RuntimeError(
            f"Unexpected class order {class_names}, expected {CLASS_NAMES}."
        )

    loaders = {
        "train": DataLoader(
            datasets_map["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            datasets_map["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    sizes = {split: len(dataset) for split, dataset in datasets_map.items()}
    return loaders, sizes


def build_model(freeze_backbone: bool) -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(CLASS_NAMES))
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += (preds == labels).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_labels: list[int] = []
    all_preds: list[int] = []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += (preds == labels).sum().item()
        total_samples += batch_size
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc, all_labels, all_preds


def save_history(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def save_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix.tolist()):
            writer.writerow([class_name, *row])


def save_training_summary(summary: dict, path: Path) -> None:
    def to_jsonable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: to_jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [to_jsonable(item) for item in value]
        if isinstance(value, tuple):
            return [to_jsonable(item) for item in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary), handle, ensure_ascii=False, indent=2)


def print_run_header(
    device: torch.device,
    dataset_sizes: dict[str, int],
    args: argparse.Namespace,
    freeze_backbone: bool,
) -> None:
    print("=" * 72, flush=True)
    print("ResNet18 training started", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Training samples: {dataset_sizes['train']}", flush=True)
    print(f"Validation samples: {dataset_sizes['val']}", flush=True)
    print(f"Classes: {CLASS_NAMES}", flush=True)
    print(
        "Config: "
        f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, "
        f"weight_decay={args.weight_decay}, image_size={args.image_size}, "
        f"freeze_backbone={freeze_backbone}",
        flush=True,
    )
    if args.freeze_backbone is None and device.type == "cpu":
        print(
            "CPU mode detected: defaulting to frozen backbone for faster training.",
            flush=True,
        )
    print("=" * 72, flush=True)


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    best_val_acc: float,
    elapsed: float,
) -> None:
    print(
        f"[Epoch {epoch:02d}/{total_epochs:02d}] "
        f"train_loss={train_loss:.4f} "
        f"train_acc={train_acc:.4f} "
        f"val_loss={val_loss:.4f} "
        f"val_acc={val_acc:.4f} "
        f"best_val_acc={best_val_acc:.4f} "
        f"time={elapsed:.1f}s",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    report_dir = output_dir / "reports"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    freeze_backbone = resolve_freeze_backbone(args.freeze_backbone, device)
    loaders, dataset_sizes = create_dataloaders(
        data_dir=data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(freeze_backbone=freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    run_args = vars(args).copy()
    run_args["effective_freeze_backbone"] = freeze_backbone

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_state_dict = None
    best_epoch = -1
    best_labels: list[int] = []
    best_preds: list[int] = []

    print_run_header(
        device=device,
        dataset_sizes=dataset_sizes,
        args=args,
        freeze_backbone=freeze_backbone,
    )

    for epoch in range(1, args.epochs + 1):
        print(f"Starting epoch {epoch:02d}/{args.epochs:02d}...", flush=True)
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc, labels, preds = evaluate(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
        )
        elapsed = time.time() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "elapsed_seconds": round(elapsed, 2),
            }
        )

        current_best_val_acc = max(best_val_acc, val_acc)
        print_epoch_summary(
            epoch=epoch,
            total_epochs=args.epochs,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            best_val_acc=current_best_val_acc,
            elapsed=elapsed,
        )

        latest_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "class_names": CLASS_NAMES,
            "args": run_args,
        }
        torch.save(latest_payload, checkpoint_dir / "last_model.pt")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            best_labels = labels
            best_preds = preds
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "class_names": CLASS_NAMES,
                    "args": run_args,
                },
                checkpoint_dir / "best_model.pt",
            )
            print(
                f"New best model saved at epoch {epoch:02d} "
                f"with val_acc={best_val_acc:.4f}",
                flush=True,
            )

    if best_state_dict is None:
        raise RuntimeError("Training finished without saving a best checkpoint.")

    model.load_state_dict(best_state_dict)
    matrix = confusion_matrix(best_labels, best_preds, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(
        best_labels,
        best_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    save_history(history, report_dir / "history.csv")
    save_confusion_matrix(matrix, report_dir / "confusion_matrix.csv")
    save_training_summary(
        {
            "model": "resnet18",
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "class_names": CLASS_NAMES,
            "dataset_sizes": dataset_sizes,
            "freeze_backbone": freeze_backbone,
            "device": str(device),
            "metrics": report,
            "args": run_args,
        },
        report_dir / "summary.json",
    )

    print("=" * 72, flush=True)
    print(
        f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}",
        flush=True,
    )
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}", flush=True)
    print(f"Training report saved to: {report_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
