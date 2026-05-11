import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tabulate import tabulate
from tqdm import tqdm

from utils.dataloader import get_loader, test_dataset
from utils.metrics import average_light_metrics, light_segmentation_metrics


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('true', '1', 'yes', 'y')


def structure_loss(pred, mask):
    if mask.shape[-2:] != pred.shape[-2:]:
        pred = F.interpolate(pred, size=mask.shape[-2:], mode='bilinear', align_corners=False)

    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def resolve_split(data_root, modal, split):
    split_path = Path(data_root) / modal / split
    image_root = split_path / 'images'
    mask_root = split_path / 'masks'
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError(f'Missing {modal}/{split}/images or {modal}/{split}/masks under {data_root}')
    return split_path, image_root, mask_root


def check_dataset(data_root, modal):
    rows = []
    for split in ('train', 'val', 'test'):
        _, image_root, mask_root = resolve_split(data_root, modal, split)
        images = sorted([p for p in image_root.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        masks = sorted([p for p in mask_root.iterdir() if p.suffix.lower() == '.png'])
        image_stems = {p.stem for p in images}
        mask_stems = {p.stem for p in masks}
        bad_values = 0
        bad_size = 0
        for mask_path in masks:
            image_path = next((p for p in images if p.stem == mask_path.stem), None)
            if image_path is None:
                continue
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    bad_size += 1
                values = set(np.unique(np.asarray(mask)).tolist())
                if not values.issubset({0, 255}):
                    bad_values += 1
        rows.append([
            modal,
            split,
            len(images),
            len(masks),
            len(image_stems - mask_stems),
            len(mask_stems - image_stems),
            bad_size,
            bad_values,
        ])

    print(tabulate(
        rows,
        headers=['modal', 'split', 'images', 'masks', 'missing_masks', 'missing_images', 'bad_size', 'bad_values'],
        tablefmt='github',
    ))


def set_lr(optimizer, init_lr, epoch, decay_rate, decay_epoch):
    lr = init_lr * (decay_rate ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def current_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def build_run_name(opt):
    if opt.run_name:
        return opt.run_name
    aug_name = 'aug' if str2bool(opt.augmentation) else 'noaug'
    scale_name = 'ms' if str2bool(opt.multi_scale) else 'single'
    model_name = opt.model.replace('_', '')
    return (
        f'{opt.modal.lower()}_{model_name}_e{opt.epoch}_bs{opt.batchsize}_'
        f'{opt.trainsize}_{opt.optimizer.lower()}_{aug_name}_{scale_name}'
    )


def save_config(config_path, opt, run_name, model_name, device):
    config = vars(opt).copy()
    config.update({
        'run_name': run_name,
        'model_name': model_name,
        'device_resolved': str(device),
        'training_metrics': ['Dice', 'IoU'],
        'test_metrics': ['Dice', 'IoU', 'Boundary IoU', 'HD95', 'MAE'],
    })
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open('w') as f:
        json.dump(config, f, indent=2)


def build_model(model_name, pretrained_path):
    if model_name == 'unet_v2':
        from unet_v2.UNet_v2 import UNetV2
        return UNetV2(n_classes=1, deep_supervision=True, pretrained_path=pretrained_path)
    if model_name == 'unet_sdi':
        from models import UNetSDI
        return UNetSDI(n_classes=1)
    raise ValueError(f'Unsupported model: {model_name}')


def batch_metrics_from_logits(logits, masks):
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
    probs = torch.sigmoid(logits.detach()).cpu().numpy()
    targets = masks.detach().cpu().numpy()
    metrics = []
    for pred, target in zip(probs, targets):
        pred = np.squeeze(pred)
        target = np.squeeze(target)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        metrics.append(light_segmentation_metrics(pred >= 0.5, target >= 0.5))
    return metrics


def summarize_metric_rows(rows):
    keys = ['dice', 'iou']
    if not rows:
        return {key: float('nan') for key in keys}
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def format_epoch_metrics(split, epoch, total_epoch, dataset, metrics, loss=None, lr=None):
    line = (
        f'{split:<5} | epoch={epoch:03d}/{total_epoch:03d} | dataset={dataset:<8} '
        f'| dice={metrics["dice"]:7.4f} | iou={metrics["iou"]:7.4f}'
    )
    if loss is not None and lr is not None:
        line += f' | lr={lr:9.2e} | loss={loss:8.4f}'
    return line


def train_one_epoch(train_loader, model, optimizer, epoch, opt, device):
    model.train()
    losses = []
    train_metric_rows = []
    size_rates = [0.75, 1.0, 1.25] if str2bool(opt.multi_scale) else [1.0]
    progress = tqdm(
        enumerate(train_loader, start=1),
        total=len(train_loader),
        desc=f'Epoch {epoch:03d}/{opt.epoch:03d}',
        ncols=120,
        leave=True,
    )

    for _, (images, gts) in progress:
        images = images.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True)

        for rate in size_rates:
            optimizer.zero_grad()
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            batch_images = images
            batch_gts = gts
            if rate != 1.0:
                batch_images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                batch_gts = F.interpolate(gts, size=(trainsize, trainsize), mode='nearest')

            outputs = model(batch_images)
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
            loss = sum(structure_loss(output, batch_gts) for output in outputs)
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if rate == 1.0:
                losses.append(float(loss.detach().cpu().item()))
                train_metric_rows.extend(batch_metrics_from_logits(outputs[0], batch_gts))

        progress.set_postfix(lr=f'{current_lr(optimizer):.2e}', loss=f'{np.mean(losses):.4f}')

    train_loss = float(np.mean(losses)) if losses else 0.0
    return train_loss, summarize_metric_rows(train_metric_rows)


@torch.no_grad()
def evaluate(model, split_path, testsize, device, save_dir=None, threshold=0.5, desc='VAL'):
    image_root = split_path / 'images'
    mask_root = split_path / 'masks'
    loader = test_dataset(str(image_root), str(mask_root), testsize)
    model_was_deep_supervision = getattr(model, 'deep_supervision', False)
    model.deep_supervision = False
    model.eval()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    losses = []
    for _ in tqdm(range(loader.size), desc=desc, ncols=120, leave=False):
        image, gt, name = loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt = gt / (gt.max() + 1e-8)
        image = image.to(device, non_blocking=True)
        gt_tensor = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0).to(device)

        logits = model(image)
        logits = F.interpolate(logits, size=gt.shape, mode='bilinear', align_corners=False)
        losses.append(float(structure_loss(logits, gt_tensor).detach().cpu().item()))
        prob = torch.sigmoid(logits).detach().cpu().numpy().squeeze()
        pred = prob >= threshold

        row = {'name': name}
        row.update(light_segmentation_metrics(pred, gt > 0.5))
        rows.append(row)

        if save_dir is not None:
            Image.fromarray((pred.astype(np.uint8) * 255)).save(save_dir / name)

    model.deep_supervision = model_was_deep_supervision
    val_loss = float(np.mean(losses)) if losses else 0.0
    return average_light_metrics(rows), rows, val_loss


def append_history(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    fieldnames = [
        'epoch', 'train_loss', 'train_dice', 'train_iou',
        'val_loss', 'val_dice', 'val_iou', 'lr',
    ]
    with csv_path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_val_metrics(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'dice', 'iou'])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, model, optimizer, epoch, best_dice, opt):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'best_dice': best_dice,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(opt),
    }, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'SGD'])
    parser.add_argument('--augmentation', type=str2bool, default=False, help='use random flip and rotation')
    parser.add_argument('--multi_scale', type=str2bool, default=False, help='use multi-scale training at 0.75x, 1.0x and 1.25x')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training image size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='decay learning rate every n epochs')
    parser.add_argument('--data_root', type=str, default='dataset', help='dataset root containing WL/NBI')
    parser.add_argument('--modal', type=str, required=True, choices=['WL', 'NBI'], help='dataset modal to train')
    parser.add_argument('--model', type=str, default='unet_v2', choices=['unet_v2', 'unet_sdi'], help='model to train')
    parser.add_argument('--pretrained_path', type=str, default='pvt_pth/pvt_v2_b2.pth', help='pretrained PVT path')
    parser.add_argument('--train_save', type=str, default='outputs', help='output root')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--run_name', type=str, default=None, help='optional experiment name saved in config')
    parser.add_argument('--check_data', action='store_true', help='check dataset and exit')
    return parser.parse_args()


def main():
    opt = parse_args()
    check_dataset(opt.data_root, opt.modal)
    if opt.check_data:
        return

    if opt.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(opt.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested, but torch.cuda.is_available() is False.')
    print(f'Using device: {device}')

    train_split, train_image_root, train_mask_root = resolve_split(opt.data_root, opt.modal, 'train')
    val_split, _, _ = resolve_split(opt.data_root, opt.modal, 'val')

    model = build_model(opt.model, opt.pretrained_path).to(device)
    model_display_name = model.__class__.__name__

    run_name = build_run_name(opt)
    output_dir = Path(opt.train_save) / model_display_name / run_name
    checkpoint_dir = output_dir / 'checkpoints'
    metrics_csv = output_dir / 'metrics_val.csv'
    history_csv = output_dir / 'history.csv'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(output_dir / 'config.json', opt, run_name, model_display_name, device)

    print(model.__class__.__name__.center(50, '='))
    print(f'experiment: {run_name}')
    print(f'outputs: {output_dir}')

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, weight_decay=1e-4, momentum=0.9)
    print(optimizer)

    train_loader = get_loader(
        str(train_image_root),
        str(train_mask_root),
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        augmentation=opt.augmentation,
        num_workers=opt.num_workers,
        pin_memory=device.type == 'cuda',
    )

    print('#' * 20, 'Start Training', '#' * 20)
    best_dice = -1.0
    for epoch in range(1, opt.epoch + 1):
        lr = set_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train_loss, train_metrics = train_one_epoch(train_loader, model, optimizer, epoch, opt, device)
        print(format_epoch_metrics('TRAIN', epoch, opt.epoch, opt.modal, train_metrics, loss=train_loss, lr=lr))
        val_metrics, _, val_loss = evaluate(model, val_split, opt.trainsize, device, desc=f'Val {opt.modal}')
        print(format_epoch_metrics('VAL', epoch, opt.epoch, opt.modal, val_metrics, loss=val_loss, lr=lr))

        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
        row.update(val_metrics)
        write_val_metrics(metrics_csv, row)

        is_best = val_metrics['dice'] > best_dice
        updated_best_dice = max(best_dice, val_metrics['dice'])
        history_row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_dice': train_metrics['dice'],
            'train_iou': train_metrics['iou'],
            'val_loss': val_loss,
            'val_dice': val_metrics['dice'],
            'val_iou': val_metrics['iou'],
            'lr': lr,
        }
        append_history(history_csv, history_row)
        if is_best:
            best_dice = updated_best_dice
            save_checkpoint(checkpoint_dir / 'best.pth', model, optimizer, epoch, best_dice, opt)
            print(f'BEST  | epoch={epoch:03d}/{opt.epoch:03d} best_dice={best_dice:.4f}')
        save_checkpoint(checkpoint_dir / 'latest.pth', model, optimizer, epoch, best_dice, opt)


if __name__ == '__main__':
    main()
