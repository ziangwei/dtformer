"""Text prior store.
文本先验存储。

Runtime interface for loading pre-computed text embeddings and associating
them with individual images.

Supported modes (set via config ``text_mode``):
  - ``"fixed"``:  全量词表 embedding，所有图共享同一张表。
  - ``"image_specific"``:  每张图有自己的 top-K 标签子集，
    从固定词表 embedding 表中查向量。

All embedding files are pre-built offline by ``tools/build_clip_cache.py``.
This module does NOT call CLIP at runtime.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..text.cache_io import load_image_labels, load_vocab_embeds
from ..text.templates import normalize_label

logger = logging.getLogger(__name__)


class TextStore:
    """Manages text embeddings for a dataset split.

    Args:
        text_mode: ``"fixed"`` or ``"image_specific"``.
        vocab_embeds_path: Path to ``vocab_embeds.pt`` (required for both modes).
        image_labels_path: Path to ``image_labels.json``
            (required for ``image_specific``).
        max_labels: Maximum labels per image (``image_specific`` only).
            0 = use natural maximum from the label file.
        text_dim: Expected embedding dimension (for zero-padding).
    """

    def __init__(
        self,
        text_mode: str = "image_specific",
        vocab_embeds_path: Optional[str] = None,
        image_labels_path: Optional[str] = None,
        max_labels: int = 6,
        text_dim: int = 512,
    ):
        self.text_mode = text_mode
        self.text_dim = text_dim
        self.max_labels = max_labels

        # Fixed vocabulary table: Tensor[C, D]
        self._vocab_embeds: Optional[torch.Tensor] = None
        self._vocab_names: List[str] = []
        # Lookup: normalized label -> row index in _vocab_embeds
        self._label_to_idx: Dict[str, int] = {}

        # Per-image data (image_specific only)
        self._image_labels: Dict[str, List[str]] = {}
        self._pad_len: int = 0

        # Load
        if vocab_embeds_path and os.path.isfile(vocab_embeds_path):
            self._load_vocab(vocab_embeds_path)

        if text_mode == "image_specific":
            if image_labels_path and os.path.isfile(image_labels_path):
                self._load_image_labels_and_lookup(image_labels_path)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    def _load_vocab(self, path: str) -> None:
        data = load_vocab_embeds(path)
        self._vocab_names = data["class_names"]
        self._vocab_embeds = data["embeds"].float()  # [C, D]
        for i, name in enumerate(self._vocab_names):
            self._label_to_idx[normalize_label(name)] = i
        logger.info(f"[TextStore] Loaded vocab ({len(self._vocab_names)} classes) from {path}")

    def _load_image_labels_and_lookup(self, path: str) -> None:
        """Load per-image label lists and resolve to indices for vocab lookup."""
        raw = load_image_labels(path)

        # Normalize and truncate
        max_len = 0
        for key, labels in raw.items():
            normed = []
            seen = set()
            for lb in labels:
                n = normalize_label(lb)
                if n and n not in seen:
                    normed.append(n)
                    seen.add(n)
            if self.max_labels > 0:
                normed = normed[: self.max_labels]
            raw[key] = normed
            max_len = max(max_len, len(normed))

        self._image_labels = raw
        self._pad_len = self.max_labels if self.max_labels > 0 else max_len
        logger.info(
            f"[TextStore] Loaded image labels ({len(raw)} images, "
            f"pad_len={self._pad_len}) from {path}"
        )

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------
    def get_text_features(
        self,
        image_key: str,
    ) -> Tuple[torch.Tensor, List[str]]:
        """Return text embeddings and label names for one image.

        Args:
            image_key: Image identifier (e.g. basename ``"0.jpg"`` or path).

        Returns:
            ``(features, names)`` where features is ``Tensor[T, D]`` and
            names is ``List[str]``.
        """
        if self.text_mode == "fixed":
            return self._get_fixed()
        return self._get_image_specific(image_key)

    def _get_fixed(self) -> Tuple[torch.Tensor, List[str]]:
        if self._vocab_embeds is not None:
            return self._vocab_embeds, list(self._vocab_names)
        return torch.zeros(0, self.text_dim), []

    def _get_image_specific(
        self,
        image_key: str,
    ) -> Tuple[torch.Tensor, List[str]]:
        # Vocab lookup from per-image label list
        labels = self._try_lookup_labels(image_key)
        if labels and self._vocab_embeds is not None:
            return self._labels_to_padded_embeds(labels)

        # Empty fallback
        pad = self._pad_len or 1
        return torch.zeros(pad, self.text_dim), []

    # ------------------------------------------------------------------
    # Key matching helpers (handles path format mismatch)
    # ------------------------------------------------------------------
    @staticmethod
    def _canonical_key(k: str) -> str:
        """Normalise an image key to ``rgb/<number>.jpg`` for fuzzy matching."""
        k = (k or "").replace("\\", "/").strip()
        base = os.path.basename(k)
        m = re.search(r"(\d+)", base)
        if m:
            return f"rgb/{m.group(1)}.jpg"
        return base.lower()

    def _try_lookup_labels(self, key: str) -> Optional[List[str]]:
        for k in (key, os.path.basename(key), self._canonical_key(key)):
            if k in self._image_labels:
                return self._image_labels[k]
        return None

    def _labels_to_padded_embeds(
        self,
        labels: List[str],
    ) -> Tuple[torch.Tensor, List[str]]:
        """Look up labels in vocab table, pad/truncate to ``_pad_len``."""
        rows: List[torch.Tensor] = []
        valid_names: List[str] = []
        D = self._vocab_embeds.shape[-1]  # type: ignore[union-attr]
        for lb in labels:
            idx = self._label_to_idx.get(lb)
            if idx is not None:
                rows.append(self._vocab_embeds[idx])  # type: ignore[index]
                valid_names.append(lb)
            else:
                rows.append(torch.zeros(D))
                valid_names.append(lb)

        feats = torch.stack(rows) if rows else torch.zeros(0, D)

        # Pad / truncate
        pad_len = self._pad_len or len(rows)
        if feats.shape[0] < pad_len:
            pad = torch.zeros(pad_len - feats.shape[0], D)
            feats = torch.cat([feats, pad], dim=0)
        elif feats.shape[0] > pad_len:
            feats = feats[:pad_len]
            valid_names = valid_names[:pad_len]

        return feats, valid_names
