from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_gastric


SWIN_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SWIN_ROOT.parent
REPO_ROOT = EXPERIMENT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Swin-Unet for WL/NBI gastric segmentation.")
    parser.add_argument("--data_root", type=str, default=str(REPO_ROOT / "dataset"))
    parser.add_argument("--modal", type=str, choices=["WL", "NBI"], required=True)
    parser.add_argument("--model_name", type=str, default="swin_unet")
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
    parser.add_argument("--cfg", type=str, default=str(SWIN_ROOT / "configs" / "swin_tiny_patch4_window7_224_lite.yaml"))
    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache-mode", type=str, default="part", choices=["no", "full", "part"])
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--amp-opt-level", type=str, default="O0", choices=["O0", "O1", "O2"])
    parser.add_argument("--tag")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
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


def main() -> None:
    args = parse_args()
    args.num_classes = 2
    if args.run_name is None:
        args.run_name = f"{args.model_name}_{args.modal}"
    if args.output_dir is None:
        args.output_dir = str(EXPERIMENT_ROOT / "outputs" / "runs" / args.run_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed, bool(args.deterministic))
    config = get_config(args)
    config.defrost()
    pretrain_path = Path(config.MODEL.PRETRAIN_CKPT)
    if not pretrain_path.is_absolute():
        config.MODEL.PRETRAIN_CKPT = str((SWIN_ROOT / pretrain_path).resolve())
    config.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).to(device)
    net.load_from(config)

    trainer_gastric(args, net, args.output_dir, device)


if __name__ == "__main__":
    main()
