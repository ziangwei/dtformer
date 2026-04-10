#!/usr/bin/env python3
"""DTFormer training entry point.
DTFormer 训练入口。

Usage (single GPU):
    python tools/train.py --config configs/experiments/nyu_dtformer_s.yaml

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=4 tools/train.py \
        --config configs/experiments/nyu_dtformer_s.yaml
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

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------
def _load_config(path: str) -> dict:
    """Load and merge experiment config (dataset + model + experiment)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(path)))

    for key in ("dataset_config", "model_config"):
        ref = cfg.pop(key, None)
        if ref:
            # Try: relative to project root, then as-is
            candidates = [
                os.path.join(project_root, ref),
                ref,
                os.path.join(os.path.dirname(path), "..", ref),
            ]
            ref_path = next((p for p in candidates if os.path.exists(p)), ref)
            with open(ref_path) as f2:
                sub = yaml.safe_load(f2) or {}
            for sk, sv in sub.items():
                if sk not in cfg:
                    cfg[sk] = sv
                elif isinstance(sv, dict) and isinstance(cfg[sk], dict):
                    cfg[sk] = {**sv, **cfg[sk]}
    return cfg


def _set_bn_params(model: nn.Module, eps: float = 1e-3, momentum: float = 0.1) -> None:
    """Set eps and momentum for all BatchNorm layers."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eps = eps
            m.momentum = momentum


# ------------------------------------------------------------------
# Dataset factory
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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DTFormer Training")
    parser.add_argument("--config", required=True, help="Experiment config YAML")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--amp", action="store_true", default=True, help="Use AMP")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-compile", action="store_true", default=False,
                        help="Apply torch.compile() (PyTorch >= 2.0)")
    parser.add_argument("--no-tensorboard", dest="tensorboard", action="store_false",
                        default=True, help="Disable TensorBoard logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Distributed setup ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + local_rank)

    # --- Config ---
    cfg = _load_config(args.config)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    text_cfg = cfg.get("text", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    dataset_cfg = cfg.get("dataset", {})

    # --- Text store ---
    from src.dtformer.data.text_factory import build_text_store_from_config
    text_store = build_text_store_from_config(text_cfg, dataset_cfg)

    # --- Data ---
    from src.dtformer.data.transforms import TrainTransform, ValTransform
    from src.dtformer.data.collate import rgbd_text_collate

    train_ds = _build_dataset(
        dataset_cfg, "train",
        TrainTransform(
            crop_size=tuple(train_cfg.get("crop_size",
                [dataset_cfg.get("image_height", 480), dataset_cfg.get("image_width", 640)])),
            scale_array=train_cfg.get("scale_array", [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]),
        ),
        text_store,
    )
    val_ds = _build_dataset(dataset_cfg, "val", ValTransform(), text_store)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 16) // world_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=rgbd_text_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=rgbd_text_collate,
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
        aux_rate=model_cfg.get("aux_rate", 0.0),
        pretrained=model_cfg.get("pretrained"),
    ).to(device)

    # BN eps/momentum
    _set_bn_params(
        model,
        eps=train_cfg.get("bn_eps", 1e-3),
        momentum=train_cfg.get("bn_momentum", 0.1),
    )

    # SyncBatchNorm + DDP
    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("Converted all BN layers to SyncBatchNorm")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )

    # Class names for per-class IoU logging
    class_names = dataset_cfg.get("class_names", None)

    # --- Train ---
    from src.dtformer.engine.train_loop import train

    train(
        model,
        train_loader,
        val_loader,
        optimizer_name=train_cfg.get("optimizer", "AdamW"),
        lr=train_cfg.get("lr", 6e-5),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        lr_power=train_cfg.get("lr_power", 0.9),
        epochs=train_cfg.get("epochs", 500),
        warmup_epochs=train_cfg.get("warmup_epochs", 10),
        use_amp=args.amp,
        log_dir=ckpt_cfg.get("log_dir", "checkpoints/run"),
        save_start_epoch=ckpt_cfg.get("save_start_epoch", 250),
        save_interval=ckpt_cfg.get("save_interval", 25),
        resume_from=args.resume,
        num_classes=dataset_cfg.get("num_classes", 40),
        eval_scales=eval_cfg.get("scale_array", [1.0]),
        eval_flip=eval_cfg.get("flip", True),
        eval_crop_size=eval_cfg.get("crop_size"),
        eval_stride_rate=eval_cfg.get("stride_rate", 0.667),
        local_rank=local_rank,
        world_size=world_size,
        torch_compile=args.torch_compile,
        use_tensorboard=args.tensorboard,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()
