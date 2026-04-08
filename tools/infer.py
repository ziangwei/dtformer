#!/usr/bin/env python3
"""DTFormer single-image inference.
DTFormer 单图推理。

Usage:
    python tools/infer.py --config configs/experiments/nyu_dtformer_s.yaml \
                          --checkpoint checkpoints/best.pth \
                          --rgb path/to/rgb.jpg \
                          --depth path/to/depth.png \
                          --output pred.png
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="DTFormer Inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rgb", required=True, help="Path to RGB image")
    parser.add_argument("--depth", required=True, help="Path to depth image")
    parser.add_argument("--output", default="prediction.png")
    parser.add_argument("--vocab-embeds", default=None, help="Path to vocab_embeds.pt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    from tools.train import _load_config
    cfg = _load_config(args.config)
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("eval", {})

    # Build model
    from src.dtformer.models.segmentors.dtformer import DTFormer
    model = DTFormer(
        backbone=model_cfg.get("backbone", "DTFormer_S"),
        num_classes=model_cfg.get("num_classes", 40),
        text_dim=model_cfg.get("text_dim", 512),
        decoder_embed_dim=model_cfg.get("decoder_embed_dim", 512),
        tsae_stages=model_cfg.get("tsae_stages", [1, 2, 3]),
        tsad_stages=model_cfg.get("tsad_stages", [1, 2, 3]),
        decoder_in_index=model_cfg.get("decoder_in_index", [1, 2, 3]),
    ).to(device)

    from src.dtformer.engine.checkpoint_io import load_checkpoint
    load_checkpoint(args.checkpoint, model)

    # Load and preprocess images
    from src.dtformer.data.transforms import IMAGENET_MEAN, IMAGENET_STD, DEPTH_MEAN, DEPTH_STD

    rgb_img = np.array(Image.open(args.rgb).convert("RGB"), dtype=np.float32) / 255.0
    depth_img = np.array(Image.open(args.depth).convert("L"), dtype=np.float32) / 255.0

    # Normalize
    rgb_img = (rgb_img - IMAGENET_MEAN) / IMAGENET_STD
    depth_3ch = np.stack([depth_img] * 3, axis=-1)
    depth_3ch = (depth_3ch - DEPTH_MEAN) / DEPTH_STD

    # HWC → CHW → batch
    rgb_t = torch.from_numpy(rgb_img.transpose(2, 0, 1)).float().unsqueeze(0)
    depth_t = torch.from_numpy(depth_3ch.transpose(2, 0, 1)).float().unsqueeze(0)

    # Text features (optional)
    text_feat = None
    if args.vocab_embeds and os.path.exists(args.vocab_embeds):
        data = torch.load(args.vocab_embeds, map_location="cpu")
        text_feat = data["embeds"].unsqueeze(0)  # (1, C, D)

    # Inference
    from src.dtformer.engine.infer_loop import infer_single, save_prediction

    pred = infer_single(
        model, rgb_t, depth_t, text_feat,
        num_classes=model_cfg.get("num_classes", 40),
        crop_size=eval_cfg.get("crop_size"),
        stride_rate=eval_cfg.get("stride_rate", 0.667),
    )

    save_prediction(pred, args.output)
    logger.info(f"Done. Prediction saved to {args.output}")


if __name__ == "__main__":
    main()
