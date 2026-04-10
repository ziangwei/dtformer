"""NYU Depth v2 dataset.
NYU Depth v2 数据集。

Handles RGB + Depth loading, label mapping, and text prior association
for NYUDepthv2 (40-class semantic segmentation).

Expected directory layout::

    {data_root}/
        RGB/          # 0.jpg, 1.jpg, ...
        Depth/        # 0.png, 1.png, ...
        Label/        # 0.png, 1.png, ...
        train.txt     # one filename stem per line
        test.txt

Ground-truth labels are 1-indexed in the PNG files; we subtract 1 so that
class indices are 0-based (0 .. 39), and the original 0 (unlabelled)
becomes 255 (ignore index).
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.utils.data as data

from ..text_store import TextStore

logger = logging.getLogger(__name__)


class NYUDepthv2(data.Dataset):
    """NYU Depth v2 RGBD segmentation dataset.

    Args:
        data_root: Path to dataset root (contains RGB/, Depth/, Label/).
        split: ``"train"`` or ``"val"``  (val reads ``test.txt``).
        transform: Callable ``(rgb, gt, depth) -> (rgb, gt, depth)``.
        text_store: Optional :class:`TextStore` for text prior features.
        file_length: If set, resample the split to this many items per epoch.
    """

    NUM_CLASSES = 40

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        text_store: Optional[TextStore] = None,
        file_length: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.text_store = text_store
        self._file_length = file_length

        # Read split file
        split_file = "train.txt" if split == "train" else "test.txt"
        split_path = os.path.join(data_root, split_file)
        with open(split_path, "r") as f:
            self._file_names = [line.strip() for line in f if line.strip()]

        logger.info(
            f"[NYUDepthv2] split={split}, images={len(self._file_names)}, "
            f"text_mode={text_store.text_mode if text_store else 'none'}"
        )

    def __len__(self) -> int:
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index: int) -> Dict[str, object]:
        if self._file_length is not None:
            names = self._resample(self._file_length)
            name = names[index]
        else:
            name = self._file_names[index]

        # --- Load images ---
        rgb_path = os.path.join(self.data_root, "RGB", name + ".jpg")
        depth_path = os.path.join(self.data_root, "Depth", name + ".png")
        label_path = os.path.join(self.data_root, "Label", name + ".png")

        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)  # BGR, uint8
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        # Depth: single-channel → 3-channel (backbone expects 3-ch input)
        if depth is not None and depth.ndim == 2:
            depth = cv2.merge([depth, depth, depth])

        # Ground-truth: 1-indexed → 0-indexed (unlabelled 0 → 255)
        label = self._remap_label(label)

        # --- Transform ---
        if self.transform is not None:
            rgb, label, depth = self.transform(rgb, label, depth)

        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        label = torch.from_numpy(np.ascontiguousarray(label)).long()
        depth = torch.from_numpy(np.ascontiguousarray(depth)).float()

        # --- Text features ---
        text_features, text_names = self._get_text(name)

        return {
            "rgb": rgb,
            "depth": depth,
            "label": label,
            "text_features": text_features,
            "text_names": text_names,
            "path": rgb_path,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _remap_label(label: np.ndarray) -> np.ndarray:
        """Convert 1-indexed NYU labels to 0-indexed; map 0 → 255 (ignore)."""
        out = label.astype(np.int64) - 1
        out[out < 0] = 255
        return out.astype(np.uint8)

    def _get_text(
        self,
        name: str,
    ) -> Tuple[torch.Tensor, List[str]]:
        if self.text_store is None:
            return torch.zeros(1, 512, dtype=torch.float32), []
        image_key = f"{name}.jpg"
        feats, names = self.text_store.get_text_features(image_key)
        return feats.float(), names

    def _resample(self, length: int) -> List[str]:
        """Resample the file list to *length* items (with wrap-around)."""
        n = len(self._file_names)
        names = self._file_names * (length // n)
        remainder = length % n
        if remainder > 0:
            indices = torch.randperm(n).tolist()[:remainder]
            names += [self._file_names[i] for i in indices]
        return names
