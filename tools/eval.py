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
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Dataset factory (mirrors tools/train.py)
# ------------------------------------------------------------------
def _build_dataset(dataset_cfg: dict, split: str, transform, text_store):
    """Build dataset by name (NYUDepthv2 / SUNRGBD)."""
    name = dataset_cfg.get("name", "NYUDepthv2")
    data_root = dataset_cfg.get("data_root", f"datasets/{name}")

    if "NYU" in name:
        from src.dtformer.data.datasets.nyu import NYUDepthv2
        return NYUDepthv2(
            data_root=data_root,
            split=split,
            transform=transform,
            text_store=text_store,
        )
    elif "SUN" in name:
        from src.dtformer.data.datasets.sunrgbd import SUNRGBD
        return SUNRGBD(
            data_root=data_root,
            split=split,
            transform=transform,
            text_store=text_store,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")


def main():
    parser = argparse.ArgumentParser(description="DTFormer Evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--multi-scale", action="store_true", help="Use MST")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use AMP during evaluation")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--save-vis", action="store_true",
                        help="Save prediction visualizations")
    parser.add_argument("--vis-dir", default=None,
                        help="Directory for visualizations (default: log_dir/vis)")
    parser.add_argument("--vis-max", type=int, default=20,
                        help="Max number of visualizations to save")
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
    train_cfg = cfg.get("train", {})
    ckpt_cfg = cfg.get("checkpoint", {})

    # --- Text store ---
    from src.dtformer.data.text_factory import build_text_store_from_config
    text_store = build_text_store_from_config(text_cfg, dataset_cfg)

    # --- Dataset ---
    from src.dtformer.data.transforms import ValTransform
    from src.dtformer.data.collate import rgbd_text_collate

    val_ds = _build_dataset(dataset_cfg, "val", ValTransform(), text_store)

    val_sampler = (
        torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
        if distributed else None
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, sampler=val_sampler,
        num_workers=2, pin_memory=True, collate_fn=rgbd_text_collate,
    )

    # --- Model ---
    from src.dtformer.models.segmentors.dtformer import DTFormer
    model = DTFormer(
        backbone=model_cfg.get("backbone", "DTFormer_S"),
        num_classes=dataset_cfg.get("num_classes", 40),
        text_dim=model_cfg.get("text_dim", 512),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.25),
        decoder_embed_dim=model_cfg.get("decoder_embed_dim", 512),
        tsae_stages=model_cfg.get("tsae_stages", [1, 2, 3]),
        tsae_share_factors=model_cfg.get("tsae_share_factors", None),
        tsad_stages=model_cfg.get("tsad_stages", [1, 2, 3]),
        tsad_use_topk=model_cfg.get("tsad_use_topk", False),
        tsad_top_m=model_cfg.get("tsad_top_m", 5),
        decoder_in_index=model_cfg.get("decoder_in_index", [1, 2, 3]),
    ).to(device)

    # BN eps/momentum customization
    from tools.train import _set_bn_params
    _set_bn_params(
        model,
        eps=train_cfg.get("bn_eps", 1e-3),
        momentum=train_cfg.get("bn_momentum", 0.1),
    )

    # Load checkpoint
    from src.dtformer.engine.checkpoint_io import load_checkpoint
    load_checkpoint(args.checkpoint, model)

    # SyncBatchNorm conversion + DDP
    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
        )

    # Class names
    class_names = dataset_cfg.get("class_names", None)

    # Visualization directory
    vis_dir = args.vis_dir
    if args.save_vis and vis_dir is None:
        vis_dir = os.path.join(
            ckpt_cfg.get("log_dir", "checkpoints/run"), "vis",
        )

    # --- Evaluate ---
    from src.dtformer.engine.eval_loop import evaluate

    if args.multi_scale:
        scales = eval_cfg.get("scale_array", [0.75, 1.0, 1.25])
    else:
        scales = [1.0]

    result = evaluate(
        model, val_loader,
        num_classes=dataset_cfg.get("num_classes", 40),
        device=device,
        scales=scales,
        flip=eval_cfg.get("flip", True),
        crop_size=eval_cfg.get("crop_size"),
        stride_rate=eval_cfg.get("stride_rate", 0.667),
        use_amp=args.amp,
        class_names=class_names,
        save_vis=args.save_vis,
        vis_dir=vis_dir,
        vis_max=args.vis_max,
    )

    if local_rank == 0:
        miou = result["miou"]
        per_class_iou = result["per_class_iou"]
        macc = result["macc"]

        logger.info(f"mIoU: {miou:.2f}%  |  mAcc: {macc:.2f}%")

        # Per-class table
        if per_class_iou is not None:
            for ci, iou_val in enumerate(per_class_iou):
                name = class_names[ci] if class_names and ci < len(class_names) else f"class_{ci}"
                logger.info(f"  {name:>24s}: {iou_val:.2f}%")

        if args.save_vis:
            logger.info(f"Visualizations saved to {vis_dir}")


if __name__ == "__main__":
    main()
