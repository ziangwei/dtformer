#!/usr/bin/env python3
"""DTFormer single-image inference.
DTFormer 单图推理。

Usage (fixed mode — full vocabulary embeddings):
    python tools/infer.py --config configs/experiments/nyu_dtformer_s.yaml \
                          --checkpoint checkpoints/best.pth \
                          --rgb path/to/rgb.jpg \
                          --depth path/to/depth.png \
                          --output pred.png

Usage (image_specific — per-image labels from JSON):
    python tools/infer.py --config configs/experiments/nyu_dtformer_s.yaml \
                          --checkpoint checkpoints/best.pth \
                          --rgb datasets/NYUDepthv2/RGB/0.jpg \
                          --depth datasets/NYUDepthv2/Depth/0.png \
                          --text-mode image_specific \
                          --output pred.png

Usage (image_specific — labels passed via CLI):
    python tools/infer.py --config configs/experiments/nyu_dtformer_s.yaml \
                          --checkpoint checkpoints/best.pth \
                          --rgb path/to/rgb.jpg \
                          --depth path/to/depth.png \
                          --text-mode image_specific \
                          --labels wall floor table chair \
                          --output pred.png
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DTFormer Inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rgb", required=True, help="Path to RGB image")
    parser.add_argument("--depth", required=True, help="Path to depth image")
    parser.add_argument("--output", default="prediction.png")

    # Text interface (unified with train/eval)
    parser.add_argument("--text-mode", default=None,
                        help="Override text mode: fixed / image_specific (default: from config)")
    parser.add_argument("--image-labels-json", default=None,
                        help="Override path to image_labels.json")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Directly specify labels for this image (image_specific only)")
    parser.add_argument("--max-labels", type=int, default=None,
                        help="Override max labels per image")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    from tools.train import _load_config
    cfg = _load_config(args.config)
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("eval", {})
    text_cfg = cfg.get("text", {})
    dataset_cfg = cfg.get("dataset", {})

    # --- Text store ---
    from src.dtformer.data.text_factory import build_text_store_from_config

    # Apply CLI overrides
    if args.text_mode:
        text_cfg = {**text_cfg, "mode": args.text_mode}
    if args.max_labels is not None:
        text_cfg = {**text_cfg, "max_image_labels": args.max_labels}

    text_store = build_text_store_from_config(
        text_cfg, dataset_cfg,
        image_labels_override=args.image_labels_json,
    )

    # --- Get text features ---
    text_mode = text_cfg.get("mode", "fixed")
    if args.labels:
        # CLI labels: manually build embedding from vocab lookup
        from src.dtformer.text.templates import normalize_label
        normed = [normalize_label(lb) for lb in args.labels]
        text_feat, text_names = text_store._labels_to_padded_embeds(normed)
        text_feat = text_feat.unsqueeze(0)  # (1, K, D)
    else:
        # Standard: use TextStore query API
        image_key = os.path.basename(args.rgb)
        text_feat, text_names = text_store.get_text_features(image_key)
        text_feat = text_feat.unsqueeze(0)  # (1, T, D)

    if text_feat.shape[1] > 0:
        logger.info(f"Text mode: {text_mode}, features shape: {text_feat.shape}")
    else:
        logger.warning("No text features available — running without text prior")
        text_feat = None

    # --- Build model ---
    from src.dtformer.models.segmentors.dtformer import DTFormer
    model = DTFormer(
        backbone=model_cfg.get("backbone", "DTFormer_S"),
        num_classes=dataset_cfg.get("num_classes", 40),
        text_dim=model_cfg.get("text_dim", 512),
        decoder_embed_dim=model_cfg.get("decoder_embed_dim", 512),
        tsae_stages=model_cfg.get("tsae_stages", [1, 2, 3]),
        tsad_stages=model_cfg.get("tsad_stages", [1, 2, 3]),
        decoder_in_index=model_cfg.get("decoder_in_index", [1, 2, 3]),
    ).to(device)

    from src.dtformer.engine.checkpoint_io import load_checkpoint
    load_checkpoint(args.checkpoint, model)

    # --- Load and preprocess images ---
    from src.dtformer.data.transforms import IMAGENET_MEAN, IMAGENET_STD, DEPTH_MEAN, DEPTH_STD

    rgb_img = np.array(Image.open(args.rgb).convert("RGB"), dtype=np.float32) / 255.0
    depth_img = np.array(Image.open(args.depth).convert("L"), dtype=np.float32) / 255.0

    # Normalize
    rgb_img = (rgb_img - IMAGENET_MEAN) / IMAGENET_STD
    depth_3ch = np.stack([depth_img] * 3, axis=-1)
    depth_3ch = (depth_3ch - DEPTH_MEAN) / DEPTH_STD

    # HWC -> CHW -> batch
    rgb_t = torch.from_numpy(rgb_img.transpose(2, 0, 1)).float().unsqueeze(0)
    depth_t = torch.from_numpy(depth_3ch.transpose(2, 0, 1)).float().unsqueeze(0)

    # --- Inference ---
    from src.dtformer.engine.infer_loop import infer_single, save_prediction

    pred = infer_single(
        model, rgb_t, depth_t, text_feat,
        num_classes=dataset_cfg.get("num_classes", 40),
        crop_size=eval_cfg.get("crop_size"),
        stride_rate=eval_cfg.get("stride_rate", 0.667),
    )

    save_prediction(pred, args.output)
    logger.info(f"Done. Prediction saved to {args.output}")


if __name__ == "__main__":
    main()
