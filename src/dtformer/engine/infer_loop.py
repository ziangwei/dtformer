"""Inference loop.
推理循环。

Single-image and batch inference with optional visualization output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .eval_loop import _infer_once

logger = logging.getLogger(__name__)


@torch.no_grad()
def infer_single(
    model: nn.Module,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    text_features: Optional[torch.Tensor] = None,
    *,
    num_classes: int = 40,
    crop_size: Optional[Sequence[int]] = None,
    stride_rate: float = 0.667,
    scales: Optional[Sequence[float]] = None,
    flip: bool = False,
) -> torch.Tensor:
    """Run inference on a single image pair and return class predictions.

    Args:
        model: DTFormer model in eval mode.
        rgb: ``(1, 3, H, W)`` RGB tensor (normalized).
        depth: ``(1, 3, H, W)`` depth tensor (3-ch replicated, normalized).
        text_features: ``(1, T, Ct)`` or ``None``.
        num_classes: Number of output classes.
        crop_size: ``[h, w]`` for sliding window inference (None = direct).
        stride_rate: Stride as fraction of crop size.
        scales: Optional multi-scale list (default single-scale 1.0).
        flip: Whether to apply horizontal flip TTA.

    Returns:
        ``(H, W)`` integer tensor of predicted class indices.
    """
    import math

    model.eval()
    device = next(model.parameters()).device
    rgb = rgb.to(device)
    depth = depth.to(device)
    if text_features is not None:
        text_features = text_features.to(device).float()

    _, _, H, W = rgb.shape
    scales = scales or [1.0]
    agg = torch.zeros(1, num_classes, H, W, device=device)

    for scale in scales:
        new_H = int(math.ceil(H * scale / 32)) * 32
        new_W = int(math.ceil(W * scale / 32)) * 32
        s_rgb = F.interpolate(rgb, size=(new_H, new_W), mode="bilinear", align_corners=True)
        s_depth = F.interpolate(depth, size=(new_H, new_W), mode="bilinear", align_corners=True)

        logits = _infer_once(model, s_rgb, s_depth, text_features, crop_size, stride_rate, num_classes)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=True)
        agg += logits.softmax(dim=1)

        if flip:
            logits_f = _infer_once(
                model, torch.flip(s_rgb, (3,)), torch.flip(s_depth, (3,)),
                text_features, crop_size, stride_rate, num_classes,
            )
            logits_f = torch.flip(logits_f, dims=(3,))
            logits_f = F.interpolate(logits_f, size=(H, W), mode="bilinear", align_corners=True)
            agg += logits_f.softmax(dim=1)

    return agg.argmax(dim=1).squeeze(0)  # (H, W)


def save_prediction(
    pred: torch.Tensor,
    save_path: str,
    palette: Optional[np.ndarray] = None,
) -> None:
    """Save a prediction map as a color PNG.

    Args:
        pred: ``(H, W)`` integer class map.
        save_path: Output file path.
        palette: ``(num_classes, 3)`` RGB palette. If ``None``, uses random.
    """
    from PIL import Image

    pred_np = pred.cpu().numpy().astype(np.uint8)
    img = Image.fromarray(pred_np, mode="P")

    if palette is not None:
        flat_pal = palette.flatten().tolist()
    else:
        rng = np.random.RandomState(42)
        flat_pal = rng.randint(0, 256, size=768).tolist()

    img.putpalette(flat_pal)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)
    logger.info(f"Saved prediction → {save_path}")
