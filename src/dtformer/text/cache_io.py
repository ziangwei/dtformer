"""Explicit cache file I/O.
显式缓存文件读写。

All text embeddings are pre-computed offline and stored as explicit files.
This module provides read/write helpers for:
  - Fixed vocabulary embedding tables  (``vocab_embeds.pt``)
  - Per-image label lists              (``image_labels.json``)

Cache paths are always explicit — no hidden runtime directories.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

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
def load_image_labels(path: str | Path) -> Dict[str, List[str]]:
    """Load per-image label lists from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
