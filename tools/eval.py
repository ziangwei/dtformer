#!/usr/bin/env python3
"""DTFormer evaluation entry point.
DTFormer 评估入口。

Usage:
    python tools/eval.py --config configs/experiments/nyu_dtformer_s.yaml \
                         --checkpoint checkpoints/best.pth
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="DTFormer Evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--multi-scale", action="store_true", help="Use MST")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load config (same loader as train)
    from tools.train import _load_config
    cfg = _load_config(args.config)
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("eval", {})
    text_cfg = cfg.get("text", {})
    dataset_cfg = cfg.get("dataset", {})

    # Dataset
    from src.dtformer.data.datasets.nyu import NYUDepthv2
    from src.dtformer.data.transforms import ValTransform
    from src.dtformer.data.collate import rgbd_text_collate
    from src.dtformer.data.text_store import TextStore

    text_store = TextStore(
        mode=text_cfg.get("mode", "fixed"),
        vocab_embeds_path=dataset_cfg.get("vocab_embeds_path"),
        image_embeds_path=dataset_cfg.get("image_embeds_path"),
        image_labels_path=dataset_cfg.get("image_labels_path"),
        max_labels=text_cfg.get("max_image_labels", 6),
    )

    val_ds = NYUDepthv2(
        root=dataset_cfg.get("root", "data/NYUDepthv2"),
        split="test",
        transform=ValTransform(),
        text_store=text_store,
    )

    val_sampler = (
        torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
        if distributed else None
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, sampler=val_sampler,
        num_workers=2, pin_memory=True, collate_fn=rgbd_text_collate,
    )

    # Model
    from src.dtformer.models.segmentors.dtformer import DTFormer
    model = DTFormer(
        backbone=model_cfg.get("backbone", "DTFormer_S"),
        num_classes=model_cfg.get("num_classes", 40),
        text_dim=model_cfg.get("text_dim", 512),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.25),
        decoder_embed_dim=model_cfg.get("decoder_embed_dim", 512),
        tsae_stages=model_cfg.get("tsae_stages", [1, 2, 3]),
        tsad_stages=model_cfg.get("tsad_stages", [1, 2, 3]),
        tsad_use_topk=model_cfg.get("tsad_use_topk", False),
        tsad_top_m=model_cfg.get("tsad_top_m", 5),
        decoder_in_index=model_cfg.get("decoder_in_index", [1, 2, 3]),
    ).to(device)

    # Load checkpoint
    from src.dtformer.engine.checkpoint_io import load_checkpoint
    load_checkpoint(args.checkpoint, model)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
        )

    # Evaluate
    from src.dtformer.engine.eval_loop import evaluate

    if args.multi_scale:
        scales = eval_cfg.get("scale_array", [0.75, 1.0, 1.25])
    else:
        scales = [1.0]

    miou = evaluate(
        model, val_loader,
        num_classes=model_cfg.get("num_classes", 40),
        device=device,
        scales=scales,
        flip=eval_cfg.get("flip", True),
        crop_size=eval_cfg.get("crop_size"),
        stride_rate=eval_cfg.get("stride_rate", 0.667),
    )

    if local_rank == 0:
        logging.getLogger(__name__).info(f"mIoU: {miou:.2f}%")


if __name__ == "__main__":
    main()
