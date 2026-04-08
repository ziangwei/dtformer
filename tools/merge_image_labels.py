#!/usr/bin/env python3
"""Merge / intersect multiple per-image label JSON files.
合并多个 VLM 标签源，取交集。

Usage:
    # Strict intersection of two VLM outputs
    python tools/merge_image_labels.py \\
        --inputs qwen_labels.json internvl_labels.json \\
        --output datasets/NYUDepthv2/image_labels.json

    # Consensus: label must appear in at least 2 of 3 sources
    python tools/merge_image_labels.py \\
        --inputs a.json b.json c.json \\
        --min-count 2 \\
        --output merged.json

Each input JSON has format: ``{"image_key": ["label1", "label2", ...]}``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dtformer.text.templates import normalize_label

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_labels(
    sources: List[Dict[str, List[str]]],
    min_count: int = 0,
    key_mode: str = "common",
) -> Dict[str, List[str]]:
    """Merge multiple label dicts by intersection or consensus.

    Args:
        sources: List of ``{image_key: [labels]}`` dicts.
        min_count: Minimum number of sources a label must appear in.
            0 or len(sources) = strict intersection.
        key_mode: ``"common"`` = only images in all sources;
            ``"union"`` = all images from any source.

    Returns:
        Merged dict ``{image_key: [labels]}``.
    """
    n = len(sources)
    if min_count <= 0 or min_count > n:
        min_count = n  # strict intersection

    # Collect all image keys
    if key_mode == "union":
        all_keys = set()
        for s in sources:
            all_keys.update(s.keys())
    else:
        all_keys = set(sources[0].keys())
        for s in sources[1:]:
            all_keys &= set(s.keys())

    result: Dict[str, List[str]] = {}
    for key in sorted(all_keys):
        label_counter: Counter = Counter()
        first_order: List[str] = []

        for s in sources:
            if key not in s:
                continue
            for lb in s[key]:
                norm = normalize_label(lb)
                if norm not in label_counter:
                    first_order.append(norm)
                label_counter[norm] += 1

        # Keep labels that meet the threshold, preserving first-seen order
        kept = [lb for lb in first_order if label_counter[lb] >= min_count]
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for lb in kept:
            if lb not in seen:
                deduped.append(lb)
                seen.add(lb)
        if deduped:
            result[key] = deduped

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-image label files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSON files")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--min-count", type=int, default=0,
                        help="Min sources per label (0 = strict intersection)")
    parser.add_argument("--key-mode", choices=["common", "union"], default="common",
                        help="Image key selection strategy")

    args = parser.parse_args()

    sources = []
    for p in args.inputs:
        logger.info(f"Loading {p}...")
        sources.append(load_json(p))

    merged = merge_labels(sources, args.min_count, args.key_mode)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    logger.info(f"Merged {len(merged)} images -> {args.output}")


if __name__ == "__main__":
    main()
