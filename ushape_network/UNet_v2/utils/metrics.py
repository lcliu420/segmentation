import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt


def _as_bool(mask):
    return np.asarray(mask) > 0.5


def _surface(mask):
    mask = _as_bool(mask)
    if not mask.any():
        return mask
    return mask ^ binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)


def dice_iou_mae(pred, gt):
    pred = _as_bool(pred)
    gt = _as_bool(gt)
    intersection = np.logical_and(pred, gt).sum(dtype=np.float64)
    pred_sum = pred.sum(dtype=np.float64)
    gt_sum = gt.sum(dtype=np.float64)
    smooth = 1e-5
    dice = (2.0 * intersection + smooth) / (pred_sum + gt_sum + smooth)
    iou = (intersection + smooth) / (pred_sum + gt_sum - intersection + smooth)
    mae = np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32)))
    return dice, iou, mae


def light_segmentation_metrics(pred, gt):
    dice, iou, _ = dice_iou_mae(pred, gt)
    return {
        'dice': float(dice),
        'iou': float(iou),
    }


def boundary_iou(pred, gt, dilation_ratio=0.02):
    pred = _as_bool(pred)
    gt = _as_bool(gt)
    h, w = gt.shape[-2:]
    iterations = max(1, int(round(dilation_ratio * np.sqrt(h * h + w * w))))

    pred_boundary = _surface(pred)
    gt_boundary = _surface(gt)
    pred_band = binary_dilation(pred_boundary, iterations=iterations)
    gt_band = binary_dilation(gt_boundary, iterations=iterations)

    union = np.logical_or(pred_band, gt_band).sum(dtype=np.float64)
    if union == 0:
        return 1.0
    intersection = np.logical_and(pred_band, gt_band).sum(dtype=np.float64)
    return intersection / union


def hd95(pred, gt):
    pred = _as_bool(pred)
    gt = _as_bool(gt)
    pred_surface = _surface(pred)
    gt_surface = _surface(gt)

    if not pred_surface.any() and not gt_surface.any():
        return 0.0
    if not pred_surface.any() or not gt_surface.any():
        h, w = gt.shape[-2:]
        return float(np.sqrt(h * h + w * w))

    pred_to_gt = distance_transform_edt(~gt_surface)[pred_surface]
    gt_to_pred = distance_transform_edt(~pred_surface)[gt_surface]
    distances = np.concatenate([pred_to_gt, gt_to_pred])
    return float(np.percentile(distances, 95))


def segmentation_metrics(pred, gt):
    dice, iou, mae = dice_iou_mae(pred, gt)
    return {
        'dice': float(dice),
        'iou': float(iou),
        'boundary_iou': float(boundary_iou(pred, gt)),
        'hd95': float(hd95(pred, gt)),
        'mae': float(mae),
    }


def average_metrics(rows):
    keys = ['dice', 'iou', 'boundary_iou', 'hd95', 'mae']
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def average_light_metrics(rows):
    keys = ['dice', 'iou']
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
