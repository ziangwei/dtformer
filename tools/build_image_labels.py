#!/usr/bin/env python3
"""Generate per-image label lists from ground-truth segmentation masks.
从语义分割真值标签提取每张图的类名列表。

Usage:
    python tools/build_image_labels.py \\
        --data-root datasets/NYUDepthv2 \\
        --dataset NYUDepthv2 \\
        --top-k 5 \\
        --output datasets/NYUDepthv2/top5_labels_per_image.json

For each image, reads the label PNG, counts pixel occurrences of each
class, and outputs the top-K most frequent class names (excluding
ignore classes).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dtformer.text.vocabularies import get_vocabulary

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# NYU-specific: IDs to skip (0=unlabelled, 38/39/40="other*" in 1-indexed)
NYU_SKIP_IDS = {0, 38, 39, 40}
# SUNRGBD: only 0=unlabelled to skip
SUNRGBD_SKIP_IDS = {0}


def extract_labels_nyu(
    data_root: str,
    class_names: list[str],
    top_k: int = 5,
) -> dict[str, list[str]]:
    """Extract per-image label lists from NYU label PNGs."""
    label_dir = os.path.join(data_root, "Label")
    result: dict[str, list[str]] = {}

    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith(".png"):
            continue
        path = os.path.join(label_dir, fname)
        label_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if label_img is None:
            continue

        # Count pixel occurrences per class (1-indexed)
        unique, counts = np.unique(label_img, return_counts=True)
        # Filter and sort by frequency
        pairs = []
        for uid, cnt in zip(unique.tolist(), counts.tolist()):
            if uid in NYU_SKIP_IDS:
                continue
            idx = uid - 1  # 1-indexed -> 0-indexed
            if 0 <= idx < len(class_names):
                pairs.append((class_names[idx], cnt))

        # Sort by count descending, take top-K
        pairs.sort(key=lambda x: x[1], reverse=True)
        labels = [name for name, _ in pairs[:top_k]]

        key = f"RGB/{fname.replace('.png', '.jpg')}"
        result[key] = labels

    return result


def extract_labels_sunrgbd(
    data_root: str,
    class_names: list[str],
    top_k: int = 5,
) -> dict[str, list[str]]:
    """Extract per-image label lists from SUNRGBD label PNGs."""
    label_dir = os.path.join(data_root, "labels")  # lowercase for SUNRGBD
    result: dict[str, list[str]] = {}

    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith(".png"):
            continue
        path = os.path.join(label_dir, fname)
        label_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if label_img is None:
            continue

        # Count pixel occurrences per class (1-indexed)
        unique, counts = np.unique(label_img, return_counts=True)
        pairs = []
        for uid, cnt in zip(unique.tolist(), counts.tolist()):
            if uid in SUNRGBD_SKIP_IDS:
                continue
            idx = uid - 1  # 1-indexed -> 0-indexed
            if 0 <= idx < len(class_names):
                pairs.append((class_names[idx], cnt))

        pairs.sort(key=lambda x: x[1], reverse=True)
        labels = [name for name, _ in pairs[:top_k]]

        key = f"RGB/{fname.replace('.png', '.jpg')}"
        result[key] = labels

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-image label lists from segmentation masks."
    )
    parser.add_argument("--data-root", required=True, help="Dataset root directory")
    parser.add_argument("--dataset", default="NYUDepthv2", help="Dataset name")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K labels per image")
    parser.add_argument("--output", required=True, help="Output JSON path")

    args = parser.parse_args()

    class_names = get_vocabulary(args.dataset)
    logger.info(f"Using {len(class_names)} classes from {args.dataset}")

    if "NYU" in args.dataset:
        labels = extract_labels_nyu(args.data_root, class_names, args.top_k)
    elif "SUN" in args.dataset:
        labels = extract_labels_sunrgbd(args.data_root, class_names, args.top_k)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported yet")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(labels)} image label entries -> {args.output}")


if __name__ == "__main__":
    main()
