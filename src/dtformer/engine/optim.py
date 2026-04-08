"""Optimizer construction.
优化器构建。

AdamW / SGD with per-parameter-group weight decay:
  - Conv/Linear weights → with decay
  - Biases, BN/LN params → no decay
"""

from __future__ import annotations

from typing import List

import torch.nn as nn


def _group_weight(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
) -> List[dict]:
    """Separate parameters into decay / no-decay groups.

    - Conv, Linear weights → WITH weight decay
    - Biases → NO weight decay
    - BatchNorm, LayerNorm, GroupNorm params → NO weight decay
    """
    decay_params, no_decay_params = [], []
    decay_ids, no_decay_ids = set(), set()

    norm_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.SyncBatchNorm, nn.GroupNorm, nn.LayerNorm,
    )

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                          nn.ConvTranspose2d, nn.ConvTranspose3d)):
            decay_params.append(m.weight)
            decay_ids.add(id(m.weight))
            if m.bias is not None:
                no_decay_params.append(m.bias)
                no_decay_ids.add(id(m.bias))
        elif isinstance(m, norm_types):
            if m.weight is not None:
                no_decay_params.append(m.weight)
                no_decay_ids.add(id(m.weight))
            if m.bias is not None:
                no_decay_params.append(m.bias)
                no_decay_ids.add(id(m.bias))

    # Catch remaining params (e.g. learnable scales, temperatures)
    for p in model.parameters():
        pid = id(p)
        if pid not in decay_ids and pid not in no_decay_ids:
            decay_params.append(p)
            decay_ids.add(pid)

    return [
        {"params": decay_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: nn.Module,
    name: str = "AdamW",
    lr: float = 6e-5,
    weight_decay: float = 0.01,
    momentum: float = 0.9,
):
    """Build optimizer with proper weight-decay grouping.

    Args:
        model: The model whose parameters to optimize.
        name: ``"AdamW"`` or ``"SGDM"``.
        lr: Base learning rate.
        weight_decay: Weight decay factor.
        momentum: Momentum (only for SGD).
    """
    import torch

    param_groups = _group_weight(model, lr, weight_decay)

    if name == "AdamW":
        return torch.optim.AdamW(
            param_groups, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay,
        )
    elif name == "SGDM":
        return torch.optim.SGD(
            param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")
