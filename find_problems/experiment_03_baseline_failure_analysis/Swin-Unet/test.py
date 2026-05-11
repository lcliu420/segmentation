from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from datasets.gastric_dataset import GastricSegmentationDataset
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import segmentation_metrics


SWIN_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SWIN_ROOT.parent
REPO_ROOT = EXPERIMENT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Swin-Unet for WL/NBI gastric segmentation.")
    parser.add_argument("--data_root", type=str, default=str(REPO_ROOT / "dataset"))
    parser.add_argument("--modal", type=str, choices=["WL", "NBI"], required=True)
    parser.add_argument("--model_name", type=str, default="swin_unet")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--cfg", type=str, default=str(SWIN_ROOT / "configs" / "swin_tiny_patch4_window7_224_lite.yaml"))
    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache-mode", type=str, default="part", choices=["no", "full", "part"])
    parser.add_argument("--resume")
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--amp-opt-level", type=str, default="O0", choices=["O0", "O1", "O2"])
    parser.add_argument("--tag")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--boundary_width", type=int, default=3)
    parser.add_argument("--limit_samples", type=int, default=None, help="Optional smoke-test limit for test split.")
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


def save_prediction_png(pred: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((pred.astype(np.uint8) * 255)).save(path)


def main() -> None:
    args = parse_args()
    args.num_classes = 2
    if args.run_name is None:
        args.run_name = f"{args.model_name}_{args.modal}"
    if args.output_dir is None:
        args.output_dir = str(EXPERIMENT_ROOT / "outputs" / "runs" / args.run_name)
    if args.checkpoint is None:
        args.checkpoint = str(Path(args.output_dir) / "best_model.pth")

    seed_everything(args.seed, bool(args.deterministic))
    config = get_config(args)
    config.defrost()
    pretrain_path = Path(config.MODEL.PRETRAIN_CKPT)
    if not pretrain_path.is_absolute():
        config.MODEL.PRETRAIN_CKPT = str((SWIN_ROOT / pretrain_path).resolve())
    config.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).to(device)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dataset = GastricSegmentationDataset(
        data_root=args.data_root,
        modal=args.modal,
        split="test",
        img_size=args.img_size,
        augment=False,
        limit_samples=args.limit_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    pred_dir = EXPERIMENT_ROOT / "predictions" / args.model_name / args.modal / "test"
    metrics_dir = EXPERIMENT_ROOT / "outputs" / "prediction_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{args.model_name}_{args.modal}_prediction_metrics.csv"

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Test {args.modal}"):
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            outputs = model(images)
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().numpy()
            case_names = batch["case_name"]
            for pred, gt, case_name in zip(preds, labels, case_names):
                pred_bool = pred > 0
                gt_bool = gt > 0
                save_prediction_png(pred_bool, pred_dir / f"{case_name}.png")
                row = {
                    "image": case_name,
                    "modal": args.modal,
                    "split": "test",
                    "model": args.model_name,
                }
                row.update(segmentation_metrics(pred_bool, gt_bool, boundary_width=args.boundary_width))
                rows.append(row)

    fieldnames = [
        "image",
        "modal",
        "split",
        "model",
        "Dice",
        "IoU",
        "Boundary_IoU",
        "HD95",
        "MAE",
        "pred_fg_ratio",
        "gt_fg_ratio",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved predictions to: {pred_dir}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
