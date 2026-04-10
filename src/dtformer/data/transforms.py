"""Data augmentation and preprocessing transforms.
数据增强与预处理。

Provides RGBD-aware transforms for training and evaluation:
  - Random horizontal flip
  - Multi-scale resize
  - Random crop + pad
  - Normalisation (ImageNet for RGB; depth-specific for depth)

All transforms operate on numpy arrays (H, W, C) and return numpy arrays.
Tensor conversion is handled by the dataset ``__getitem__``.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Normalization constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEPTH_MEAN = (0.48, 0.48, 0.48)
DEPTH_STD = (0.28, 0.28, 0.28)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def normalize(
    img: np.ndarray,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """Scale to [0,1], subtract mean, divide by std."""
    img = img.astype(np.float64) / 255.0
    img = img - mean
    img = img / std
    return img


def pad_to_shape(
    img: np.ndarray,
    target_shape: Tuple[int, int],
    pad_value: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad *img* to at least *target_shape*, centred.

    Returns:
        ``(padded_img, margin)`` where ``margin = [top, bottom, left, right]``.
    """
    th, tw = target_shape
    h, w = img.shape[:2]
    pad_h = max(th - h, 0)
    pad_w = max(tw - w, 0)
    margin = np.array([pad_h // 2, pad_h - pad_h // 2,
                        pad_w // 2, pad_w - pad_w // 2], dtype=np.uint32)
    img = cv2.copyMakeBorder(
        img, int(margin[0]), int(margin[1]), int(margin[2]), int(margin[3]),
        cv2.BORDER_CONSTANT, value=pad_value,
    )
    return img, margin


def random_crop(
    img: np.ndarray,
    crop_pos: Tuple[int, int],
    crop_size: Tuple[int, int],
    pad_value: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop at *crop_pos* then pad to *crop_size* if smaller."""
    sh, sw = crop_pos
    ch, cw = crop_size
    cropped = img[sh : sh + ch, sw : sw + cw, ...]
    padded, margin = pad_to_shape(cropped, crop_size, pad_value)
    return padded, margin


def generate_random_crop_pos(
    img_shape: Tuple[int, int],
    crop_size: Tuple[int, int],
) -> Tuple[int, int]:
    """Generate a random top-left corner for cropping."""
    h, w = img_shape
    ch, cw = crop_size
    pos_h = random.randint(0, max(h - ch, 0))
    pos_w = random.randint(0, max(w - cw, 0))
    return pos_h, pos_w


# ---------------------------------------------------------------------------
# Composed transforms
# ---------------------------------------------------------------------------
class TrainTransform:
    """Training augmentation pipeline: flip → scale → normalize → crop.

    Args:
        crop_size: ``(height, width)`` for random crop.
        scale_array: List of scale factors for multi-scale augmentation.
        rgb_mean / rgb_std: RGB normalisation parameters.
        depth_mean / depth_std: Depth normalisation parameters.
    """

    def __init__(
        self,
        crop_size: Tuple[int, int],
        scale_array: Optional[List[float]] = None,
        rgb_mean: Sequence[float] = IMAGENET_MEAN,
        rgb_std: Sequence[float] = IMAGENET_STD,
        depth_mean: Sequence[float] = DEPTH_MEAN,
        depth_std: Sequence[float] = DEPTH_STD,
    ):
        self.crop_size = crop_size
        self.scale_array = scale_array
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.depth_mean = depth_mean
        self.depth_std = depth_std

    def __call__(
        self,
        rgb: np.ndarray,
        gt: np.ndarray,
        depth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Random horizontal flip
        if random.random() >= 0.5:
            rgb = cv2.flip(rgb, 1)
            gt = cv2.flip(gt, 1)
            depth = cv2.flip(depth, 1)

        # Multi-scale resize
        if self.scale_array is not None:
            scale = random.choice(self.scale_array)
            h, w = rgb.shape[:2]
            sh, sw = int(h * scale), int(w * scale)
            rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, (sw, sh), interpolation=cv2.INTER_LINEAR)

        # Normalize
        rgb = normalize(rgb, self.rgb_mean, self.rgb_std)
        depth = normalize(depth, self.depth_mean, self.depth_std)

        # Random crop
        crop_pos = generate_random_crop_pos(rgb.shape[:2], self.crop_size)
        rgb, _ = random_crop(rgb, crop_pos, self.crop_size, 0)
        gt, _ = random_crop(gt, crop_pos, self.crop_size, 255)
        depth, _ = random_crop(depth, crop_pos, self.crop_size, 0)

        # HWC -> CHW
        rgb = rgb.transpose(2, 0, 1)
        depth = depth.transpose(2, 0, 1)

        return rgb, gt, depth


class ValTransform:
    """Validation transform: normalize only (no augmentation).

    Args:
        rgb_mean / rgb_std: RGB normalisation parameters.
        depth_mean / depth_std: Depth normalisation parameters.
    """

    def __init__(
        self,
        rgb_mean: Sequence[float] = IMAGENET_MEAN,
        rgb_std: Sequence[float] = IMAGENET_STD,
        depth_mean: Sequence[float] = DEPTH_MEAN,
        depth_std: Sequence[float] = DEPTH_STD,
    ):
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.depth_mean = depth_mean
        self.depth_std = depth_std

    def __call__(
        self,
        rgb: np.ndarray,
        gt: np.ndarray,
        depth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb = normalize(rgb, self.rgb_mean, self.rgb_std)
        depth = normalize(depth, self.depth_mean, self.depth_std)
        rgb = rgb.transpose(2, 0, 1)
        depth = depth.transpose(2, 0, 1)
        return rgb, gt, depth
