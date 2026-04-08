"""Evaluation loop.
评估循环。

Supports single-scale, multi-scale + flip, and sliding-window inference.
Reports mIoU and per-class IoU.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
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
# Prediction visualization
# ---------------------------------------------------------------------------
def save_vis_prediction(
    pred: torch.Tensor,
    label: torch.Tensor,
    save_path: str,
    palette: Optional[np.ndarray] = None,
    ignore_index: int = 255,
) -> None:
    """Save a side-by-side prediction vs. ground truth visualization.

    Args:
        pred: ``(H, W)`` integer class map (predicted).
        label: ``(H, W)`` integer class map (ground truth).
        save_path: Output file path.
        palette: ``(num_classes, 3)`` RGB palette. If ``None``, uses random.
        ignore_index: Label value to treat as ignore (shown as black).
    """
    from PIL import Image

    pred_np = pred.cpu().numpy().astype(np.uint8)
    label_np = label.cpu().numpy().astype(np.uint8)

    if palette is not None:
        flat_pal = palette.flatten().tolist()
    else:
        rng = np.random.RandomState(42)
        flat_pal = rng.randint(0, 256, size=768).tolist()

    # Set ignore pixels to black
    flat_pal_arr = list(flat_pal)
    if ignore_index < 256:
        idx3 = ignore_index * 3
        if idx3 + 2 < len(flat_pal_arr):
            flat_pal_arr[idx3] = 0
            flat_pal_arr[idx3 + 1] = 0
            flat_pal_arr[idx3 + 2] = 0

    pred_img = Image.fromarray(pred_np, mode="P")
    pred_img.putpalette(flat_pal_arr)

    label_img = Image.fromarray(label_np, mode="P")
    label_img.putpalette(flat_pal_arr)

    # Side by side
    pred_rgb = pred_img.convert("RGB")
    label_rgb = label_img.convert("RGB")
    w, h = pred_rgb.size
    combined = Image.new("RGB", (w * 2, h))
    combined.paste(label_rgb, (0, 0))
    combined.paste(pred_rgb, (w, 0))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    combined.save(save_path)


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
    use_amp: bool = False,
    class_names: Optional[Sequence[str]] = None,
    save_vis: bool = False,
    vis_dir: Optional[str] = None,
    vis_max: int = 20,
) -> Dict[str, object]:
    """Run evaluation and return mIoU (%) and per-class results.

    Supports multi-scale + flip test-time augmentation and optional AMP.

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
        use_amp: Whether to use AMP (autocast) during evaluation.
        class_names: Optional list of class name strings for logging.
        save_vis: If True, save prediction visualizations.
        vis_dir: Directory for visualization images (required if save_vis).
        vis_max: Maximum number of visualizations to save.

    Returns:
        Dictionary with keys:

        - ``"miou"``: float, mean IoU percentage.
        - ``"per_class_iou"``: list of per-class IoU percentages.
        - ``"macc"``: float, mean pixel accuracy percentage.
        - ``"per_class_acc"``: list of per-class accuracy percentages.
    """
    model.eval()
    scales = scales or [1.0]
    metrics = Metrics(num_classes, ignore_index, device)

    vis_count = 0

    for batch_idx, batch in enumerate(dataloader):
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

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = _infer_once(
                        model, s_rgb, s_depth, text_feat,
                        crop_size, stride_rate, num_classes,
                    )
            else:
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

                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits_f = _infer_once(
                            model, s_rgb_f, s_depth_f, text_feat,
                            crop_size, stride_rate, num_classes,
                        )
                else:
                    logits_f = _infer_once(
                        model, s_rgb_f, s_depth_f, text_feat,
                        crop_size, stride_rate, num_classes,
                    )
                logits_f = torch.flip(logits_f, dims=(3,))
                logits_f = F.interpolate(logits_f, size=(H, W), mode="bilinear", align_corners=True)
                agg_logits += logits_f.softmax(dim=1)

        metrics.update(agg_logits, label)

        # Save visualization (first vis_max samples)
        if save_vis and vis_dir and vis_count < vis_max:
            pred_cls = agg_logits.argmax(dim=1)  # (B, H, W)
            for b in range(B):
                if vis_count >= vis_max:
                    break
                save_vis_prediction(
                    pred_cls[b], label[b],
                    os.path.join(vis_dir, f"vis_{vis_count:04d}.png"),
                    ignore_index=ignore_index,
                )
                vis_count += 1

    # Distributed aggregation
    metrics.reduce()

    per_class_iou, miou = metrics.compute_iou()
    per_class_acc, macc = metrics.compute_pixel_acc()

    # Log per-class IoU table
    if class_names is not None:
        logger.info("Per-class IoU:")
        for ci, iou_val in enumerate(per_class_iou):
            name = class_names[ci] if ci < len(class_names) else f"class_{ci}"
            logger.info(f"  {name:>24s}: {iou_val:.2f}%")
        logger.info(f"  {'mIoU':>24s}: {miou:.2f}%")

    return {
        "miou": miou,
        "per_class_iou": per_class_iou,
        "macc": macc,
        "per_class_acc": per_class_acc,
    }
