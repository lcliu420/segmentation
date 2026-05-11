from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist


class DiceLoss(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor: torch.Tensor) -> torch.Tensor:
        tensor_list = []
        for i in range(self.n_classes):
            tensor_list.append((input_tensor == i).unsqueeze(1))
        return torch.cat(tensor_list, dim=1).float()

    def _dice_loss(self, score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        return 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)

    def forward(self, inputs: torch.Tensor, target: torch.Tensor, weight=None, softmax: bool = False) -> torch.Tensor:
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), f"predict {inputs.size()} & target {target.size()} shape do not match"
        loss = 0.0
        for i in range(self.n_classes):
            loss += self._dice_loss(inputs[:, i], target[:, i]) * weight[i]
        return loss / self.n_classes


def dice_iou_from_logits(outputs: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1) > 0
    gt = labels > 0
    intersection = torch.logical_and(pred, gt).sum(dim=(1, 2)).float()
    pred_sum = pred.sum(dim=(1, 2)).float()
    gt_sum = gt.sum(dim=(1, 2)).float()
    union = torch.logical_or(pred, gt).sum(dim=(1, 2)).float()
    dice = (2 * intersection + 1e-6) / (pred_sum + gt_sum + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return float(dice.mean().item()), float(iou.mean().item())


def binary_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * np.logical_and(pred, gt).sum()) / denom)


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def mask_boundary(mask: np.ndarray, width: int = 3) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    structure = ndi.generate_binary_structure(2, 1)
    dilated = ndi.binary_dilation(mask, structure=structure, iterations=width)
    eroded = ndi.binary_erosion(mask, structure=structure, iterations=width, border_value=0)
    return np.logical_xor(dilated, eroded)


def boundary_iou(pred: np.ndarray, gt: np.ndarray, width: int = 3) -> float:
    pred_b = mask_boundary(pred, width)
    gt_b = mask_boundary(gt, width)
    union = np.logical_or(pred_b, gt_b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred_b, gt_b).sum() / union)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    diagonal = float(math.hypot(*gt.shape))
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return diagonal

    pred_surface = mask_boundary(pred, width=1)
    gt_surface = mask_boundary(gt, width=1)
    pred_points = np.argwhere(pred_surface)
    gt_points = np.argwhere(gt_surface)
    if len(pred_points) == 0 or len(gt_points) == 0:
        return diagonal
    distances = cdist(pred_points, gt_points)
    pred_to_gt = distances.min(axis=1)
    gt_to_pred = distances.min(axis=0)
    return float(max(np.percentile(pred_to_gt, 95), np.percentile(gt_to_pred, 95)))


def segmentation_metrics(pred: np.ndarray, gt: np.ndarray, boundary_width: int = 3) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    return {
        "Dice": binary_dice(pred, gt),
        "IoU": binary_iou(pred, gt),
        "Boundary_IoU": boundary_iou(pred, gt, width=boundary_width),
        "HD95": hd95(pred, gt),
        "MAE": float(np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean()),
        "pred_fg_ratio": float(pred.mean()),
        "gt_fg_ratio": float(gt.mean()),
    }
