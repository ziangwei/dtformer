"""Checkpoint save/load utilities.
检查点保存/加载工具。

Handles model state, optimizer state, scheduler state, and epoch tracking.
Keeps top-K checkpoints ranked by metric (mIoU).
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    iteration: int = 0,
    metric: Optional[float] = None,
) -> None:
    """Save a training checkpoint.

    Strips the ``module.`` prefix from DDP-wrapped models automatically.
    """
    state_dict = OrderedDict()
    raw_sd = model.state_dict()
    for k, v in raw_sd.items():
        key = k[7:] if k.startswith("module.") else k
        state_dict[key] = v

    ckpt = {
        "model": state_dict,
        "epoch": epoch,
        "iteration": iteration,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if metric is not None:
        ckpt["metric"] = metric

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(ckpt, path)
    logger.info(f"Saved checkpoint → {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = False,
) -> dict:
    """Load a checkpoint and restore model (+ optionally optimizer) state.

    Returns:
        The raw checkpoint dict (with ``epoch``, ``iteration``, etc.).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Extract model state
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    clean_sd = OrderedDict()
    for k, v in sd.items():
        key = k[7:] if k.startswith("module.") else k
        clean_sd[key] = v

    missing, unexpected = model.load_state_dict(clean_sd, strict=strict)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.info(f"Unexpected keys (ignored): {unexpected}")

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        logger.info("Restored optimizer state")

    logger.info(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt


class CheckpointManager:
    """Keeps the top-K checkpoints ranked by metric (higher = better).

    Args:
        save_dir: Directory to save checkpoints.
        max_keep: Maximum number of checkpoints to keep.
    """

    def __init__(self, save_dir: str, max_keep: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self._records: list[dict] = []  # [{"path": ..., "metric": ...}]

    def save_if_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        iteration: int,
        metric: float,
    ) -> Optional[str]:
        """Save a checkpoint if it ranks in top-K by metric.

        Returns the saved path, or ``None`` if not saved.
        """
        # Always save if we haven't filled the quota yet
        if (
            len(self._records) >= self.max_keep
            and metric <= self._records[-1]["metric"]
        ):
            return None

        fname = f"epoch-{epoch}_miou_{metric:.2f}.pth"
        fpath = str(self.save_dir / fname)

        save_checkpoint(fpath, model, optimizer, epoch, iteration, metric)

        self._records.append({"path": fpath, "metric": metric})
        self._records.sort(key=lambda r: r["metric"], reverse=True)

        # Evict the worst if over quota
        while len(self._records) > self.max_keep:
            worst = self._records.pop()
            if os.path.exists(worst["path"]):
                os.remove(worst["path"])
                logger.info(f"Removed old checkpoint: {worst['path']}")

        return fpath
