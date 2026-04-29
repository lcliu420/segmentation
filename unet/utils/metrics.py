from __future__ import annotations

import torch
from torch import Tensor
from typing import Dict


def compute_segmentation_metrics(
    logits: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    计算二分类分割任务的常用指标。

    这里按 batch 内逐样本计算后取平均，更适合训练和验证阶段做日志统计。
    """

    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (targets > 0.5).float()

    dims = (1, 2, 3)
    true_positive = (preds * targets).sum(dim=dims)
    false_positive = (preds * (1.0 - targets)).sum(dim=dims)
    false_negative = ((1.0 - preds) * targets).sum(dim=dims)

    dice = (2.0 * true_positive + eps) / (
        2.0 * true_positive + false_positive + false_negative + eps
    )
    iou = (true_positive + eps) / (
        true_positive + false_positive + false_negative + eps
    )
    precision = (true_positive + eps) / (true_positive + false_positive + eps)
    recall = (true_positive + eps) / (true_positive + false_negative + eps)

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
    }
