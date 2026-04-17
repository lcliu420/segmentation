from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class PendingImageDataset(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.root = root
        self.transform = transform
        self.files = sorted(
            path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.files:
            raise RuntimeError(f"No supported images found in: {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.files[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict pending images with a trained ResNet18 model."
    )
    parser.add_argument(
        "--pending-dir",
        type=Path,
        default=Path("pending"),
        help="Directory containing mixed pending images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts") / "resnet18" / "checkpoints" / "best_model.pt",
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to receive copied prediction results. Defaults to an auto-generated threshold-named directory.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory to save inference reports. Defaults to an auto-generated threshold-named directory.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Confidence threshold below which images go to uncertain.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, for example cuda or cpu.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transform(image_size: int) -> transforms.Compose:
    weights = ResNet18_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_model(class_count: int, checkpoint: dict, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_threshold_tag(threshold: float) -> str:
    scaled = int(round(threshold * 100))
    return f"t{scaled:03d}"


def choose_unique_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    index = 2
    while True:
        candidate = base_dir.parent / f"{base_dir.name}_{index}"
        if not candidate.exists():
            return candidate
        index += 1


def resolve_output_dir(output_dir_arg: Path | None, threshold: float) -> Path:
    if output_dir_arg is not None:
        return output_dir_arg.resolve()
    threshold_tag = format_threshold_tag(threshold)
    return choose_unique_dir((Path.cwd() / f"sorted_output_{threshold_tag}").resolve())


def resolve_report_dir(report_dir_arg: Path | None, threshold: float) -> Path:
    if report_dir_arg is not None:
        return report_dir_arg.resolve()
    threshold_tag = format_threshold_tag(threshold)
    base_dir = (Path.cwd() / "artifacts" / "resnet18" / f"inference_{threshold_tag}").resolve()
    return choose_unique_dir(base_dir)


def copy_prediction(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    pending_dir = args.pending_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = resolve_output_dir(args.output_dir, args.threshold)
    report_dir = resolve_report_dir(args.report_dir, args.threshold)
    ensure_dir(report_dir)

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names: list[str] = checkpoint.get("class_names", ["NBI", "WL"])
    checkpoint_args = checkpoint.get("args", {})
    image_size = int(checkpoint_args.get("image_size", 224))

    dataset = PendingImageDataset(pending_dir, build_transform(image_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = build_model(len(class_names), checkpoint, device)

    for class_name in class_names:
        ensure_dir(output_dir / class_name)
    ensure_dir(output_dir / "uncertain")

    print("=" * 72, flush=True)
    print("Pending inference started", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Pending images: {len(dataset)}", flush=True)
    print(f"Confidence threshold: {args.threshold}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print("=" * 72, flush=True)

    start_time = time.time()
    softmax = nn.Softmax(dim=1)
    rows: list[dict[str, str | float]] = []
    counts = {class_name: 0 for class_name in class_names}
    counts["uncertain"] = 0
    processed = 0

    with torch.no_grad():
        for inputs, names in loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            probabilities = softmax(logits).cpu()

            for name, probs in zip(names, probabilities):
                confidence, pred_index = torch.max(probs, dim=0)
                predicted_label = class_names[int(pred_index)]
                confidence_value = float(confidence.item())
                final_label = predicted_label if confidence_value >= args.threshold else "uncertain"

                source_path = pending_dir / name
                target_path = output_dir / final_label / name
                copy_prediction(source_path, target_path)

                rows.append(
                    {
                        "filename": name,
                        "predicted_label": predicted_label,
                        "confidence": round(confidence_value, 6),
                        "final_output_label": final_label,
                        "source_path": str(source_path),
                        "target_path": str(target_path),
                    }
                )
                counts[final_label] += 1
                processed += 1

            print(
                f"Processed {processed}/{len(dataset)} images "
                f"(NBI={counts.get('NBI', 0)}, WL={counts.get('WL', 0)}, uncertain={counts['uncertain']})",
                flush=True,
            )

    report_path = report_dir / "predictions.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "predicted_label",
                "confidence",
                "final_output_label",
                "source_path",
                "target_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = report_dir / "inference_summary.txt"
    elapsed = time.time() - start_time
    summary_lines = [
        f"device={device}",
        f"pending_images={len(dataset)}",
        f"confidence_threshold={args.threshold}",
        f"NBI={counts.get('NBI', 0)}",
        f"WL={counts.get('WL', 0)}",
        f"uncertain={counts['uncertain']}",
        f"elapsed_seconds={elapsed:.2f}",
        f"checkpoint={checkpoint_path}",
        f"predictions_csv={report_path}",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("=" * 72, flush=True)
    print(f"Inference finished in {elapsed:.1f}s", flush=True)
    print(f"NBI: {counts.get('NBI', 0)}", flush=True)
    print(f"WL: {counts.get('WL', 0)}", flush=True)
    print(f"uncertain: {counts['uncertain']}", flush=True)
    print(f"Prediction report saved to: {report_path}", flush=True)
    print(f"Summary saved to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
