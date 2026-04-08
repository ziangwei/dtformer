#!/usr/bin/env python3
"""Build CLIP embedding caches (offline).
离线构建 CLIP 嵌入缓存。

Usage:
    # Build vocabulary embedding table for NYU
    python tools/build_clip_cache.py \\
        --dataset NYUDepthv2 \\
        --output datasets/NYUDepthv2/cache/vocab_embeds.pt

    # Build per-image embedding cache (image_specific mode)
    python tools/build_clip_cache.py \\
        --dataset NYUDepthv2 \\
        --image-labels datasets/NYUDepthv2/image_labels.json \\
        --max-labels 6 \\
        --output datasets/NYUDepthv2/cache/image_embeds.pt

This script encodes text offline so that training / evaluation never
needs to load CLIP.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.dtformer.text.cache_io import (
    load_image_labels,
    save_image_embeds,
    save_vocab_embeds,
)
from src.dtformer.text.clip_backend import (
    encode_labels,
    encode_vocabulary,
    unload_clip,
)
from src.dtformer.text.templates import normalize_label
from src.dtformer.text.vocabularies import get_vocabulary

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_vocab_cache(
    dataset_name: str,
    output_path: str,
    model_name: str | None = None,
    template_set: str = "clip",
    max_templates: int = 3,
) -> None:
    """Build and save a vocabulary embedding table."""
    class_names = get_vocabulary(dataset_name)
    logger.info(f"Encoding {len(class_names)} classes for {dataset_name}...")

    embeds = encode_vocabulary(
        class_names,
        template_set=template_set,
        max_templates=max_templates,
        model_name=model_name,
    )
    save_vocab_embeds(output_path, class_names, embeds)
    unload_clip()


def build_image_cache(
    dataset_name: str,
    image_labels_path: str,
    output_path: str,
    max_labels: int = 6,
    model_name: str | None = None,
    template_set: str = "clip",
    max_templates: int = 3,
) -> None:
    """Build per-image embedding cache from label JSON + vocab table."""
    # Step 1: batch-encode all unique labels
    raw_labels = load_image_labels(image_labels_path)
    all_labels: list[str] = []
    for labels in raw_labels.values():
        all_labels.extend(labels)

    logger.info(
        f"Batch-encoding {len(set(all_labels))} unique labels from "
        f"{len(raw_labels)} images..."
    )
    label_embeds = encode_labels(
        all_labels,
        template_set=template_set,
        max_templates=max_templates,
        model_name=model_name,
    )
    unload_clip()

    # Step 2: assemble per-image features
    text_dim = next(iter(label_embeds.values())).shape[0] if label_embeds else 512
    feats: dict[str, torch.Tensor] = {}
    names: dict[str, list[str]] = {}

    # Compute pad length
    pad_len = max_labels if max_labels > 0 else max(
        len(v) for v in raw_labels.values()
    )

    for key, label_list in raw_labels.items():
        # Normalize, deduplicate, truncate
        seen: set[str] = set()
        clean: list[str] = []
        for lb in label_list:
            n = normalize_label(lb)
            if n and n not in seen:
                clean.append(n)
                seen.add(n)
        if max_labels > 0:
            clean = clean[:max_labels]

        # Look up embeddings
        rows = []
        for lb in clean:
            if lb in label_embeds:
                rows.append(label_embeds[lb])
            else:
                rows.append(torch.zeros(text_dim))

        if rows:
            img_feats = torch.stack(rows)
        else:
            img_feats = torch.zeros(0, text_dim)

        # Pad / truncate
        if img_feats.shape[0] < pad_len:
            pad = torch.zeros(pad_len - img_feats.shape[0], text_dim)
            img_feats = torch.cat([img_feats, pad], dim=0)
        elif img_feats.shape[0] > pad_len:
            img_feats = img_feats[:pad_len]

        feats[key] = img_feats
        names[key] = clean

    save_image_embeds(output_path, feats, names, pad_len)
    logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CLIP embedding caches.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. NYUDepthv2)")
    parser.add_argument("--output", required=True, help="Output .pt file path")
    parser.add_argument("--image-labels", default=None, help="Per-image label JSON (for image_specific mode)")
    parser.add_argument("--max-labels", type=int, default=6, help="Max labels per image (0 = no limit)")
    parser.add_argument("--model-name", default=None, help="CLIP model name (default: ViT-B-16 openai)")
    parser.add_argument("--template-set", default="clip", help="Template set: clip / none")
    parser.add_argument("--max-templates", type=int, default=3, help="Max templates per label")

    args = parser.parse_args()

    if args.image_labels:
        build_image_cache(
            dataset_name=args.dataset,
            image_labels_path=args.image_labels,
            output_path=args.output,
            max_labels=args.max_labels,
            model_name=args.model_name,
            template_set=args.template_set,
            max_templates=args.max_templates,
        )
    else:
        build_vocab_cache(
            dataset_name=args.dataset,
            output_path=args.output,
            model_name=args.model_name,
            template_set=args.template_set,
            max_templates=args.max_templates,
        )


if __name__ == "__main__":
    main()
