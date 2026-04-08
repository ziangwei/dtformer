"""Learning rate schedulers.
学习率调度器。

Implements WarmUpPolyLR: linear warmup followed by polynomial decay.
"""

from __future__ import annotations


class WarmUpPolyLR:
    """Warm-up + polynomial decay LR schedule.

    - Warmup:  ``lr = base_lr * (iter / warmup_iters)``
    - Decay:   ``lr = base_lr * (1 - iter / total_iters) ^ power``

    Args:
        base_lr: Initial learning rate.
        power: Polynomial decay power (default 0.9).
        total_iters: Total number of training iterations.
        warmup_iters: Number of linear warmup iterations.
    """

    def __init__(
        self,
        base_lr: float,
        power: float,
        total_iters: int,
        warmup_iters: int,
    ):
        self.base_lr = base_lr
        self.power = power
        self.total_iters = float(total_iters)
        self.warmup_iters = warmup_iters

    def get_lr(self, cur_iter: int) -> float:
        if cur_iter < self.warmup_iters:
            return self.base_lr * (cur_iter / max(self.warmup_iters, 1))
        return self.base_lr * (
            (1.0 - cur_iter / self.total_iters) ** self.power
        )
