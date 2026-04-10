"""Batch collation utilities.
批量拼接工具。

Custom collate function that handles text features alongside
fixed-size RGB / Depth / mask tensors.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def rgbd_text_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of sample dicts into a batched dict.

    Expected keys per sample:
        - ``rgb``: ``Tensor[3, H, W]``
        - ``depth``: ``Tensor[3, H, W]``
        - ``label``: ``Tensor[H, W]``  (long)
        - ``text_features``: ``Tensor[T, D]``
        - ``text_names``: ``List[str]``
        - ``path``: ``str``

    Returns:
        ``Dict`` with stacked tensors (``rgb``, ``depth``, ``label``,
        ``text_features``) and lists (``text_names``, ``path``).
    """
    rgb = torch.stack([s["rgb"] for s in batch])
    depth = torch.stack([s["depth"] for s in batch])
    label = torch.stack([s["label"] for s in batch])

    # Text features may vary in token count across modes;
    # within a batch they should be uniform (dataset pads to fixed length).
    text_features = torch.stack([s["text_features"] for s in batch])

    return {
        "rgb": rgb,
        "depth": depth,
        "label": label,
        "text_features": text_features,
        "text_names": [s.get("text_names", []) for s in batch],
        "path": [s.get("path", "") for s in batch],
    }
