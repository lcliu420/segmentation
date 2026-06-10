import math

import numpy as np
from scipy import ndimage


EPS = 1e-7


def to_binary(mask):
    arr = np.asarray(mask)
    return arr.astype(bool)


def dice_score(pred, target):
    pred = to_binary(pred)
    target = to_binary(target)
    inter = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * inter + EPS) / (denom + EPS))


def iou_score(pred, target):
    pred = to_binary(pred)
    target = to_binary(target)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return float((inter + EPS) / (union + EPS))


def mae_score(pred, target):
    pred = to_binary(pred).astype(np.float32)
    target = to_binary(target).astype(np.float32)
    return float(np.mean(np.abs(pred - target)))


def mask_boundary(mask, dilation_ratio=0.02, min_width=2):
    mask = to_binary(mask)
    h, w = mask.shape
    img_diag = math.sqrt(h * h + w * w)
    dilation = max(min_width, int(round(dilation_ratio * img_diag)))
    struct = np.ones((3, 3), dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=struct, iterations=dilation, border_value=0)
    return np.logical_xor(mask, eroded)


def boundary_iou_score(pred, target, dilation_ratio=0.02):
    pred_boundary = mask_boundary(pred, dilation_ratio=dilation_ratio)
    target_boundary = mask_boundary(target, dilation_ratio=dilation_ratio)
    inter = np.logical_and(pred_boundary, target_boundary).sum()
    union = np.logical_or(pred_boundary, target_boundary).sum()
    if union == 0:
        return 1.0
    return float((inter + EPS) / (union + EPS))


def hd95_score(pred, target):
    pred = to_binary(pred)
    target = to_binary(target)
    if not pred.any() and not target.any():
        return 0.0

    h, w = target.shape
    penalty = float(math.sqrt(h * h + w * w))
    if pred.any() != target.any():
        return penalty

    pred_border = mask_boundary(pred, dilation_ratio=0.0, min_width=1)
    target_border = mask_boundary(target, dilation_ratio=0.0, min_width=1)
    if not pred_border.any() or not target_border.any():
        return penalty

    dt_target = ndimage.distance_transform_edt(~target_border)
    dt_pred = ndimage.distance_transform_edt(~pred_border)
    pred_to_target = dt_target[pred_border]
    target_to_pred = dt_pred[target_border]
    distances = np.concatenate([pred_to_target, target_to_pred])
    if distances.size == 0:
        return penalty
    return float(np.percentile(distances, 95))


def surface_distances_medpy(result, reference, voxelspacing=None, connectivity=1):
    result = to_binary(result)
    reference = to_binary(reference)
    if not result.any():
        raise RuntimeError("The first supplied array does not contain any binary object.")
    if not reference.any():
        raise RuntimeError("The second supplied array does not contain any binary object.")

    footprint = ndimage.generate_binary_structure(result.ndim, connectivity)
    result_border = np.logical_xor(
        result,
        ndimage.binary_erosion(result, structure=footprint, iterations=1),
    )
    reference_border = np.logical_xor(
        reference,
        ndimage.binary_erosion(reference, structure=footprint, iterations=1),
    )
    dt = ndimage.distance_transform_edt(~reference_border, sampling=voxelspacing)
    return dt[result_border]


def hd95_medpy_score(pred, target, voxelspacing=None, connectivity=1):
    pred = to_binary(pred)
    target = to_binary(target)
    if not pred.any() and not target.any():
        return 0.0

    penalty = float(math.sqrt(sum(dim * dim for dim in target.shape)))
    if pred.any() != target.any():
        return penalty

    pred_to_target = surface_distances_medpy(
        pred,
        target,
        voxelspacing=voxelspacing,
        connectivity=connectivity,
    )
    target_to_pred = surface_distances_medpy(
        target,
        pred,
        voxelspacing=voxelspacing,
        connectivity=connectivity,
    )
    distances = np.concatenate([pred_to_target, target_to_pred])
    if distances.size == 0:
        return penalty
    return float(np.percentile(distances, 95))


def compute_metrics(pred, target):
    return {
        "dice": dice_score(pred, target),
        "iou": iou_score(pred, target),
        "boundary_iou": boundary_iou_score(pred, target),
        "hd95_medpy": hd95_medpy_score(pred, target),
        "mae": mae_score(pred, target),
    }


def average_metrics(rows):
    if not rows:
        return {"dice": 0.0, "iou": 0.0, "boundary_iou": 0.0, "hd95_medpy": 0.0, "mae": 0.0}
    keys = ["dice", "iou", "boundary_iou", "hd95_medpy", "mae"]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
