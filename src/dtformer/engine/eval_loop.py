"""Evaluation loop.
评估循环。

Supports single-scale, multi-scale + flip, and sliding-window inference.
Reports mIoU and per-class IoU.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .metrics import Metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sliding window inference
# ---------------------------------------------------------------------------
def slide_inference(
    model: nn.Module,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    text_features: Optional[torch.Tensor],
    crop_size: Sequence[int],
    stride_rate: float,
    num_classes: int,
) -> torch.Tensor:
    """Sliding-window inference for large images.

    Args:
        model: Segmentation model (inference mode).
        rgb: ``(1, 3, H, W)`` RGB input.
        depth: ``(1, 3, H, W)`` depth input.
        text_features: ``(1, T, Ct)`` or ``None``.
        crop_size: ``[h_crop, w_crop]``.
        stride_rate: Fraction of crop_size for stride.
        num_classes: Number of output classes.

    Returns:
        ``(1, C, H, W)`` logits.
    """
    h_crop, w_crop = crop_size
    _, _, h_img, w_img = rgb.shape

    # Upscale if image smaller than crop
    if h_img < h_crop or w_img < w_crop:
        rgb = F.interpolate(rgb, size=(h_crop, w_crop), mode="bilinear", align_corners=True)
        depth = F.interpolate(depth, size=(h_crop, w_crop), mode="bilinear", align_corners=True)
        _, _, h_img, w_img = rgb.shape

    h_stride = int(stride_rate * h_crop)
    w_stride = int(stride_rate * w_crop)
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    preds = rgb.new_zeros((1, num_classes, h_img, w_img))
    count = rgb.new_zeros((1, 1, h_img, w_img))

    for hi in range(h_grids):
        for wi in range(w_grids):
            y1 = hi * h_stride
            x1 = wi * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            crop_rgb = rgb[:, :, y1:y2, x1:x2]
            crop_depth = depth[:, :, y1:y2, x1:x2]
            logit = model(crop_rgb, crop_depth, text_features=text_features)
            preds[:, :, y1:y2, x1:x2] += logit
            count[:, :, y1:y2, x1:x2] += 1

    return preds / count


# ---------------------------------------------------------------------------
# Core inference step (single scale, optional flip)
# ---------------------------------------------------------------------------
def _infer_once(
    model: nn.Module,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    text_features: Optional[torch.Tensor],
    crop_size: Optional[Sequence[int]],
    stride_rate: float,
    num_classes: int,
) -> torch.Tensor:
    """Single forward pass, optionally with sliding window."""
    if crop_size is not None:
        return slide_inference(
            model, rgb, depth, text_features,
            crop_size, stride_rate, num_classes,
        )
    return model(rgb, depth, text_features=text_features)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    num_classes: int = 40,
    ignore_index: int = 255,
    device: torch.device | str = "cuda",
    scales: List[float] | None = None,
    flip: bool = True,
    crop_size: Optional[List[int]] = None,
    stride_rate: float = 0.667,
) -> float:
    """Run evaluation and return mIoU (%).

    Supports multi-scale + flip test-time augmentation.

    Args:
        model: DTFormer model.
        dataloader: Validation data loader.
        num_classes: Number of segmentation classes.
        ignore_index: Ignore label index.
        device: Torch device.
        scales: List of scales for MST (e.g. [0.75, 1.0, 1.25]).
        flip: Whether to apply horizontal flip augmentation.
        crop_size: ``[h, w]`` for sliding window (``None`` = direct).
        stride_rate: Stride as fraction of crop size.

    Returns:
        mIoU as a float percentage.
    """
    model.eval()
    scales = scales or [1.0]
    metrics = Metrics(num_classes, ignore_index, device)

    for batch in dataloader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        text_feat = batch.get("text_features")
        if text_feat is not None:
            text_feat = text_feat.to(device, non_blocking=True).float()

        if label.dim() == 2:
            label = label.unsqueeze(0)

        B, H, W = label.shape
        agg_logits = torch.zeros(B, num_classes, H, W, device=device)

        for scale in scales:
            new_H = int(math.ceil(H * scale / 32)) * 32
            new_W = int(math.ceil(W * scale / 32)) * 32

            s_rgb = F.interpolate(rgb, size=(new_H, new_W), mode="bilinear", align_corners=True)
            s_depth = F.interpolate(depth, size=(new_H, new_W), mode="bilinear", align_corners=True)

            logits = _infer_once(
                model, s_rgb, s_depth, text_feat,
                crop_size, stride_rate, num_classes,
            )
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=True)
            agg_logits += logits.softmax(dim=1)

            # Flip augmentation
            if flip:
                s_rgb_f = torch.flip(s_rgb, dims=(3,))
                s_depth_f = torch.flip(s_depth, dims=(3,))
                logits_f = _infer_once(
                    model, s_rgb_f, s_depth_f, text_feat,
                    crop_size, stride_rate, num_classes,
                )
                logits_f = torch.flip(logits_f, dims=(3,))
                logits_f = F.interpolate(logits_f, size=(H, W), mode="bilinear", align_corners=True)
                agg_logits += logits_f.softmax(dim=1)

        metrics.update(agg_logits, label)

    # Distributed aggregation
    metrics.reduce()

    _, miou = metrics.compute_iou()
    return miou
