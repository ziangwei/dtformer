"""Unified TextStore construction from config.
统一的 TextStore 配置构建。

All entry points (train / eval / infer) should use
``build_text_store_from_config`` to avoid three-way drift.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .text_store import TextStore


def build_text_store_from_config(
    text_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    *,
    vocab_embeds_override: Optional[str] = None,
    image_labels_override: Optional[str] = None,
) -> TextStore:
    """Build a :class:`TextStore` from merged config dicts.

    Args:
        text_cfg: The ``text:`` section of the experiment config.
        dataset_cfg: The ``dataset:`` section of the experiment config.
        vocab_embeds_override: CLI override for vocab_embeds path.
        image_labels_override: CLI override for image_labels_json path.

    Returns:
        A ready-to-use :class:`TextStore`.
    """
    data_root = dataset_cfg.get("data_root", "datasets/NYUDepthv2")

    # Resolve vocab embeds path
    vocab_path = vocab_embeds_override or dataset_cfg.get("vocab_embeds")
    if vocab_path and not os.path.isabs(vocab_path):
        vocab_path = os.path.join(data_root, vocab_path)

    # Resolve image labels path
    labels_path = image_labels_override or dataset_cfg.get("image_labels_json")
    if labels_path and not os.path.isabs(labels_path):
        labels_path = os.path.join(data_root, labels_path)

    return TextStore(
        text_mode=text_cfg.get("mode", "fixed"),
        vocab_embeds_path=vocab_path,
        image_labels_path=labels_path,
        max_labels=text_cfg.get("max_image_labels", 6),
        text_dim=text_cfg.get("text_dim", 512),
    )
