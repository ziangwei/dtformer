"""Segmentation evaluation metrics.
语义分割评估指标。

Confusion-matrix based IoU, F1, and pixel accuracy computation.
Supports distributed aggregation via ``torch.distributed.all_reduce``.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.distributed as dist


class Metrics:
    """Streaming confusion-matrix based metrics.

    Args:
        num_classes: Number of segmentation classes.
        ignore_index: Label index to ignore (typically 255).
        device: Torch device for the confusion matrix.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        device: torch.device | str = "cpu",
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.hist = torch.zeros(
            num_classes, num_classes, dtype=torch.long, device=device,
        )

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate predictions into the confusion matrix.

        Args:
            pred: ``(B, C, H, W)`` logits or softmax probabilities.
            target: ``(B, H, W)`` ground truth labels.
        """
        pred_cls = pred.argmax(dim=1).flatten()
        target_flat = target.flatten()

        valid = target_flat != self.ignore_index
        pred_cls = pred_cls[valid]
        target_flat = target_flat[valid]

        self.hist += torch.bincount(
            target_flat * self.num_classes + pred_cls,
            minlength=self.num_classes ** 2,
        ).view(self.num_classes, self.num_classes).to(self.hist.device)

    def reduce(self) -> None:
        """All-reduce the confusion matrix across distributed processes."""
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.hist, op=dist.ReduceOp.SUM)

    def compute_iou(self) -> Tuple[List[float], float]:
        """Compute per-class IoU and mIoU (%).

        Returns:
            (per_class_iou_list, miou)
        """
        h = self.hist.float()
        iou = h.diag() / (h.sum(0) + h.sum(1) - h.diag() + 1e-10)
        iou[iou.isnan()] = 0.0
        miou = iou.mean().item() * 100.0
        per_class = (iou * 100.0).cpu().tolist()
        return [round(x, 2) for x in per_class], round(miou, 2)

    def compute_f1(self) -> Tuple[List[float], float]:
        """Compute per-class F1 and mean F1 (%)."""
        h = self.hist.float()
        f1 = 2 * h.diag() / (h.sum(0) + h.sum(1) + 1e-10)
        f1[f1.isnan()] = 0.0
        mf1 = f1.mean().item() * 100.0
        per_class = (f1 * 100.0).cpu().tolist()
        return [round(x, 2) for x in per_class], round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[List[float], float]:
        """Compute per-class pixel accuracy and mean accuracy (%)."""
        h = self.hist.float()
        acc = h.diag() / (h.sum(1) + 1e-10)
        acc[acc.isnan()] = 0.0
        macc = acc.mean().item() * 100.0
        per_class = (acc * 100.0).cpu().tolist()
        return [round(x, 2) for x in per_class], round(macc, 2)
