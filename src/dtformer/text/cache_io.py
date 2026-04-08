"""Explicit cache file I/O.
显式缓存文件读写。

All text embeddings are pre-computed offline and stored as explicit files.
This module provides read/write helpers for:
  - Fixed vocabulary embedding tables  (``vocab_embeds.pt``)
  - Per-image label lists              (``image_labels.json``)
  - Per-image embedding tables          (``image_embeds.pt``)

Cache paths are always explicit — no hidden runtime directories.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed vocabulary cache
# ---------------------------------------------------------------------------
def save_vocab_embeds(
    path: str | Path,
    class_names: List[str],
    embeds: torch.Tensor,
) -> None:
    """Save a vocabulary embedding table.

    File format::

        {
            "class_names": ["wall", "floor", ...],
            "embeds": Tensor[C, D],
        }
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"class_names": class_names, "embeds": embeds.cpu()}, path)
    logger.info(f"Saved vocab embeds ({len(class_names)} classes) -> {path}")


def load_vocab_embeds(path: str | Path) -> Dict[str, Any]:
    """Load a vocabulary embedding table.

    Returns:
        ``{"class_names": List[str], "embeds": Tensor[C, D]}``
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert "class_names" in payload and "embeds" in payload, (
        f"Invalid vocab cache: {path}"
    )
    return payload


# ---------------------------------------------------------------------------
# Per-image label lists (JSON)
# ---------------------------------------------------------------------------
def save_image_labels(
    path: str | Path,
    labels: Dict[str, List[str]],
) -> None:
    """Save per-image label lists as JSON.

    Args:
        path: Output JSON path.
        labels: ``{image_key: [label1, label2, ...]}``
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved image labels ({len(labels)} images) -> {path}")


def load_image_labels(path: str | Path) -> Dict[str, List[str]]:
    """Load per-image label lists from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Per-image embedding cache
# ---------------------------------------------------------------------------
def save_image_embeds(
    path: str | Path,
    feats: Dict[str, torch.Tensor],
    names: Optional[Dict[str, List[str]]] = None,
    pad_len: int = 0,
) -> None:
    """Save per-image embedding cache.

    File format::

        {
            "feats": {image_key: Tensor[K, D], ...},
            "names": {image_key: [label1, ...], ...},
            "pad_len": int,
        }
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feats": {k: v.cpu() for k, v in feats.items()},
        "names": names or {},
        "pad_len": pad_len,
    }
    torch.save(payload, path)
    logger.info(f"Saved image embeds ({len(feats)} images, pad_len={pad_len}) -> {path}")


def load_image_embeds(path: str | Path) -> Dict[str, Any]:
    """Load per-image embedding cache.

    Returns:
        ``{"feats": {key: Tensor}, "names": {key: List[str]}, "pad_len": int}``
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "feats" not in payload:
        raise ValueError(f"Invalid image embed cache: {path}")
    return payload
