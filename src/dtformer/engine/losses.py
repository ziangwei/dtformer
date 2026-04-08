"""Loss functions for semantic segmentation.
语义分割损失函数。

The main segmentation loss (CE with ignore_index) is built into the
DTFormer segmentor.  This module provides the ``accuracy`` utility
used for logging during training.
"""

from __future__ import annotations

import torch


def accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Compute pixel-wise classification accuracy.

    Args:
        pred: ``(B, C, H, W)`` logits.
        target: ``(B, H, W)`` labels.
        ignore_index: Label to ignore.

    Returns:
        Scalar accuracy in [0, 1].
    """
    pred_cls = pred.argmax(dim=1)
    valid = target != ignore_index
    correct = (pred_cls[valid] == target[valid]).sum()
    total = valid.sum()
    if total == 0:
        return torch.tensor(0.0, device=pred.device)
    return correct.float() / total.float()
