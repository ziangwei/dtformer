"""Training loop.
训练主循环。

Epoch-based training with DDP, AMP, gradient clipping, and periodic
validation.  Designed to be called from ``tools/train.py``.

Usage via ``torchrun``:
    torchrun --nproc_per_node=4 tools/train.py --config ...
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from .checkpoint_io import CheckpointManager, load_checkpoint
from .eval_loop import evaluate
from .metrics import Metrics
from .optim import build_optimizer
from .schedulers import WarmUpPolyLR

logger = logging.getLogger(__name__)


def _all_reduce_scalar(t: torch.Tensor) -> torch.Tensor:
    """All-reduce a scalar tensor across DDP workers."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return t


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    # Optimizer
    optimizer_name: str = "AdamW",
    lr: float = 6e-5,
    weight_decay: float = 0.01,
    lr_power: float = 0.9,
    # Schedule
    epochs: int = 500,
    warmup_epochs: int = 10,
    # AMP
    use_amp: bool = True,
    # Gradient
    grad_clip_norm: float = 1.0,
    # Checkpointing
    log_dir: str = "checkpoints/run",
    save_start_epoch: int = 250,
    save_interval: int = 25,
    resume_from: Optional[str] = None,
    # Evaluation
    eval_interval: int = 25,
    num_classes: int = 40,
    ignore_index: int = 255,
    eval_scales: Optional[list] = None,
    eval_flip: bool = True,
    eval_crop_size: Optional[list] = None,
    eval_stride_rate: float = 0.667,
    # Misc
    print_interval: int = 50,
    local_rank: int = 0,
    world_size: int = 1,
):
    """Run the full training loop.

    Args:
        model: The DTFormer model (already wrapped in DDP if distributed).
        train_loader: Training data loader.
        val_loader: Validation data loader (``None`` to skip eval).
        ... (see argument list above)
    """
    device = next(model.parameters()).device

    # --- Optimizer & Scheduler ---
    raw_model = model.module if hasattr(model, "module") else model
    optimizer = build_optimizer(raw_model, optimizer_name, lr, weight_decay)

    iters_per_epoch = len(train_loader)
    total_iters = epochs * iters_per_epoch
    warmup_iters = warmup_epochs * iters_per_epoch
    lr_scheduler = WarmUpPolyLR(lr, lr_power, total_iters, warmup_iters)

    scaler = GradScaler(enabled=use_amp)

    ckpt_mgr = CheckpointManager(log_dir, max_keep=5)
    best_miou = 0.0
    start_epoch = 1

    # --- Resume ---
    if resume_from:
        ckpt = load_checkpoint(resume_from, raw_model, optimizer)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou = ckpt.get("metric", 0.0)
        logger.info(f"Resumed from epoch {start_epoch - 1}, best mIoU={best_miou}")

    # --- Training loop ---
    for epoch in range(start_epoch, epochs + 1):
        model.train()

        # Distributed sampler shuffle
        if hasattr(train_loader, "sampler") and hasattr(
            train_loader.sampler, "set_epoch"
        ):
            train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        t0 = time.time()

        for idx, batch in enumerate(train_loader):
            global_iter = (epoch - 1) * iters_per_epoch + idx

            # Update LR
            cur_lr = lr_scheduler.get_lr(global_iter)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            # Move to device
            rgb = batch["rgb"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            text_feat = batch.get("text_features")
            if text_feat is not None:
                text_feat = text_feat.to(device, non_blocking=True).float()

            # Forward + backward
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(rgb, depth, label=label, text_features=text_feat)
            else:
                loss = model(rgb, depth, label=label, text_features=text_feat)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            epoch_loss += loss.item()

            # Logging
            if (idx + 1) % print_interval == 0 and local_rank == 0:
                avg = epoch_loss / (idx + 1)
                elapsed = time.time() - t0
                eta_epoch = elapsed / (idx + 1) * (iters_per_epoch - idx - 1)
                logger.info(
                    f"Epoch [{epoch}/{epochs}] "
                    f"Iter [{idx + 1}/{iters_per_epoch}] "
                    f"Loss {loss.item():.4f} (avg {avg:.4f}) "
                    f"LR {cur_lr:.2e} "
                    f"ETA {eta_epoch / 60:.1f}min"
                )

        # --- Epoch summary ---
        avg_loss = epoch_loss / max(iters_per_epoch, 1)
        if dist.is_available() and dist.is_initialized():
            avg_loss_t = torch.tensor(avg_loss, device=device)
            avg_loss = _all_reduce_scalar(avg_loss_t).item()

        if local_rank == 0:
            logger.info(
                f"Epoch {epoch} done — avg loss {avg_loss:.4f} "
                f"({time.time() - t0:.0f}s)"
            )

        # --- Validation ---
        if (
            val_loader is not None
            and epoch >= save_start_epoch
            and epoch % eval_interval == 0
        ):
            miou = evaluate(
                model,
                val_loader,
                num_classes=num_classes,
                ignore_index=ignore_index,
                device=device,
                scales=eval_scales or [1.0],
                flip=eval_flip,
                crop_size=eval_crop_size,
                stride_rate=eval_stride_rate,
            )

            if local_rank == 0:
                logger.info(f"Epoch {epoch} — mIoU: {miou:.2f}%")

                if miou > best_miou:
                    best_miou = miou
                    logger.info(f"New best mIoU: {best_miou:.2f}%")

                ckpt_mgr.save_if_best(
                    raw_model, optimizer, epoch, epoch * iters_per_epoch, miou,
                )

        # --- Periodic save ---
        elif (
            local_rank == 0
            and epoch >= save_start_epoch
            and epoch % save_interval == 0
        ):
            from .checkpoint_io import save_checkpoint as _save

            _save(
                f"{log_dir}/epoch-{epoch}.pth",
                raw_model, optimizer, epoch,
                epoch * iters_per_epoch,
            )

    if local_rank == 0:
        logger.info(f"Training finished. Best mIoU: {best_miou:.2f}%")
