import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from datasets.dataset import SegmentationDataset
from metrics import average_metrics, compute_metrics
from networks.vision_transformer import CSwinUnet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../dataset")
    parser.add_argument("--modal", type=str, default="ALL", choices=["ALL", "WL", "NBI", "DELETE"])
    parser.add_argument("--cfg", type=str, default="configs/cswin_tiny_224_lite.yaml", metavar="FILE")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/CSWinUNet")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_test_batches", type=int, default=0)

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


def resolve_run_dir(args, checkpoint):
    if args.run_name:
        return Path(args.output_dir) / args.modal / args.run_name
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return Path(args.output_dir) / args.modal


def prepare_config(args):
    config = get_config(args)
    config.defrost()
    config.DATA.IMG_SIZE = args.img_size
    config.MODEL.NUM_CLASSES = args.num_classes
    config.freeze()
    return config


def save_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.num_classes != 2:
        raise ValueError("This dataset is binary segmentation; please keep --num_classes 2.")

    torch.manual_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    config = prepare_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SegmentationDataset(args.data_root, args.modal, "test", args.img_size, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CSwinUnet(config, img_size=args.img_size, num_classes=args.num_classes).to(device)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    run_dir = resolve_run_dir(args, checkpoint)
    pred_dir = run_dir / "predictions" / "test"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Test {args.modal}", ncols=100)):
            if args.max_test_batches and batch_idx >= args.max_test_batches:
                break
            images = batch["image"].to(device)
            labels = batch["label"].numpy()
            outputs = model(images)
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().numpy()
            names = batch["name"]

            for idx, name in enumerate(names):
                pred = preds[idx].astype(np.uint8)
                target = labels[idx].astype(np.uint8)
                metrics = compute_metrics(pred, target)
                row = {"name": name}
                row.update(metrics)
                rows.append(row)
                Image.fromarray(pred * 255).save(pred_dir / f"{name}.png")

    summary = average_metrics(rows)
    per_image_fields = ["name", "dice", "iou", "boundary_iou", "hd95_medpy", "mae"]
    save_csv(run_dir / "metrics_test_per_image.csv", rows, per_image_fields)
    save_csv(run_dir / "metrics_test.csv", [summary], ["dice", "iou", "boundary_iou", "hd95_medpy", "mae"])

    print(
        f"TEST | dataset={args.modal:<3} | dice={summary['dice']:.4f} | iou={summary['iou']:.4f} "
        f"| b_iou={summary['boundary_iou']:.4f} | hd95_medpy={summary['hd95_medpy']:.4f} "
        f"| mae={summary['mae']:.4f}"
    )
    print(f"Predictions saved to: {pred_dir}")


if __name__ == "__main__":
    main()
