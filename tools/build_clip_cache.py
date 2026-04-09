#!/usr/bin/env python3
"""Build CLIP vocabulary embedding cache (offline).
离线构建 CLIP 词表嵌入缓存。

Usage:
    # Build vocabulary embedding table for NYU
    python tools/build_clip_cache.py \\
        --dataset NYUDepthv2 \\
        --output datasets/NYUDepthv2/cache/vocab_embeds.pt

    # Build vocabulary embedding table for SUNRGBD
    python tools/build_clip_cache.py \\
        --dataset SUNRGBD \\
        --output datasets/SUNRGBD/cache/vocab_embeds.pt

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

from src.dtformer.text.cache_io import save_vocab_embeds
from src.dtformer.text.clip_backend import encode_vocabulary, unload_clip
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CLIP vocabulary embedding cache.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. NYUDepthv2, SUNRGBD)")
    parser.add_argument("--output", required=True, help="Output .pt file path")
    parser.add_argument("--model-name", default=None, help="CLIP model name (default: ViT-B-16 openai)")
    parser.add_argument("--template-set", default="clip", help="Template set: clip / none")
    parser.add_argument("--max-templates", type=int, default=3, help="Max templates per label")

    args = parser.parse_args()

    build_vocab_cache(
        dataset_name=args.dataset,
        output_path=args.output,
        model_name=args.model_name,
        template_set=args.template_set,
        max_templates=args.max_templates,
    )


if __name__ == "__main__":
    main()
