import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tabulate import tabulate

from utils.dataloader import test_dataset
from utils.metrics import average_metrics, segmentation_metrics


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('true', '1', 'yes', 'y')


def resolve_split(data_root, modal, split):
    split_path = Path(data_root) / modal / split
    image_root = split_path / 'images'
    mask_root = split_path / 'masks'
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError(f'Missing {modal}/{split}/images or {modal}/{split}/masks under {data_root}')
    return split_path, image_root, mask_root


def load_model_weights(model, pth_path, device):
    checkpoint = torch.load(pth_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key.replace('module.', '', 1)] = value
    model.load_state_dict(cleaned)
    return checkpoint


def build_model(model_name, pretrained_path):
    if model_name == 'unet_v2':
        from unet_v2.UNet_v2 import UNetV2
        return UNetV2(n_classes=1, deep_supervision=False, pretrained_path=pretrained_path)
    if model_name == 'unet_sdi':
        from models import UNetSDI
        return UNetSDI(n_classes=1)
    raise ValueError(f'Unsupported model: {model_name}')


def infer_checkpoint_run_dir(pth_path):
    pth_path = Path(pth_path)
    if pth_path.parent.name == 'checkpoints':
        return pth_path.parent.parent
    return pth_path.parent


@torch.no_grad()
def evaluate(model, split_path, testsize, device, save_dir=None, threshold=0.5):
    image_root = split_path / 'images'
    mask_root = split_path / 'masks'
    loader = test_dataset(str(image_root), str(mask_root), testsize)
    if hasattr(model, 'deep_supervision'):
        model.deep_supervision = False
    model.eval()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _ in range(loader.size):
        image, gt, name = loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt = gt / (gt.max() + 1e-8)
        image = image.to(device, non_blocking=True)

        logits = model(image)
        logits = F.interpolate(logits, size=gt.shape, mode='bilinear', align_corners=False)
        prob = torch.sigmoid(logits).detach().cpu().numpy().squeeze()
        pred = prob >= threshold

        row = {'name': name}
        row.update(segmentation_metrics(pred, gt > 0.5))
        rows.append(row)

        if save_dir is not None:
            Image.fromarray((pred.astype(np.uint8) * 255)).save(save_dir / name)

    return average_metrics(rows), rows


def write_summary(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_per_image(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['name', 'dice', 'iou', 'boundary_iou', 'hd95', 'mae']
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='dataset', help='dataset root containing WL/NBI')
    parser.add_argument('--modal', type=str, required=True, choices=['WL', 'NBI'], help='dataset modal to test')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--testsize', type=int, default=352, help='testing image size')
    parser.add_argument('--pth_path', type=str, required=True, help='trained checkpoint path')
    parser.add_argument('--model', type=str, default='unet_v2', choices=['unet_v2', 'unet_sdi'], help='model to test')
    parser.add_argument('--pretrained_path', type=str, default='pvt_pth/pvt_v2_b2.pth', help='pretrained PVT path')
    parser.add_argument('--output_root', type=str, default=None, help='optional output root; defaults to checkpoint run folder')
    parser.add_argument('--save_pred', type=str2bool, default=True, help='save binary prediction masks')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    return parser.parse_args()


def main():
    opt = parse_args()

    if opt.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(opt.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested, but torch.cuda.is_available() is False.')
    print(f'Using device: {device}')

    split_path, _, _ = resolve_split(opt.data_root, opt.modal, opt.split)
    model = build_model(opt.model, opt.pretrained_path).to(device)
    model_display_name = model.__class__.__name__
    checkpoint_run_dir = infer_checkpoint_run_dir(opt.pth_path)
    if opt.output_root is None:
        output_dir = checkpoint_run_dir
    else:
        output_dir = Path(opt.output_root) / model_display_name / checkpoint_run_dir.name
    pred_dir = output_dir / 'predictions' / opt.split if opt.save_pred else None

    checkpoint = load_model_weights(model, opt.pth_path, device)
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    metrics, rows = evaluate(model, split_path, opt.testsize, device, pred_dir, opt.threshold)
    summary_row = {'modal': opt.modal, 'split': opt.split, 'checkpoint': opt.pth_path}
    summary_row.update(metrics)
    write_summary(output_dir / f'metrics_{opt.split}.csv', summary_row)
    write_per_image(output_dir / f'metrics_{opt.split}_per_image.csv', rows)

    print(tabulate(
        [[opt.modal, opt.split, metrics['dice'], metrics['iou'], metrics['boundary_iou'], metrics['hd95'], metrics['mae']]],
        headers=['modal', 'split', 'dice', 'iou', 'boundary_iou', 'hd95', 'mae'],
        floatfmt='.4f',
    ))
    if pred_dir is not None:
        print(f'Saved predictions to: {pred_dir}')


if __name__ == '__main__':
    main()
