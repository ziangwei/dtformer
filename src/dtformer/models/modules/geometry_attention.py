"""Geometry-aware attention primitives.
几何感知注意力原语。

GeoPriorGen: generates depth-dependent positional bias using sinusoidal
encoding and learnable decay masks.

Two GSA (Geometry-aware Self-Attention) variants:
  - DecomposedGSA: separable H/W attention (stages 0-2)
  - FullGSA: full 2D attention (stage 3)

Internal tensors use NHWC layout; the caller handles NCHW conversion.

Numerical equivalence with the original implementation is required.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class DWConv2d(nn.Module):
    """Depthwise 2-D convolution (NHWC → NCHW → DWConv → NHWC)."""

    def __init__(self, dim: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


def angle_transform(
    x: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE-like rotary encoding to *x*."""
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    return (x * cos) + (torch.stack([-x2, x1], dim=-1).flatten(-2) * sin)


# ---------------------------------------------------------------------------
# GeoPriorGen
# ---------------------------------------------------------------------------
class GeoPriorGen(nn.Module):
    """Generate geometry-aware positional bias from depth maps.

    Produces sinusoidal position encoding and learnable decay masks that
    incorporate depth-dependent distances.

    Args:
        embed_dim: Channel dimension of the current stage.
        num_heads: Number of attention heads.
        initial_value: Controls the base decay rate.
        heads_range: Range across heads for the decay schedule.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        initial_value: float,
        heads_range: float,
    ):
        super().__init__()
        angle = 1.0 / (
            10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2)
        )
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.weight = nn.Parameter(torch.ones(2, 1, 1, 1), requires_grad=True)
        decay = torch.log(
            1
            - 2
            ** (
                -initial_value
                - heads_range
                * torch.arange(num_heads, dtype=torch.float)
                / num_heads
            )
        )
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    # --- Decay generators ---------------------------------------------------
    def generate_depth_decay(
        self, H: int, W: int, depth_grid: torch.Tensor
    ) -> torch.Tensor:
        B, _, H, W = depth_grid.shape
        grid_d = depth_grid.reshape(B, H * W, 1)
        mask_d = grid_d[:, :, None, :] - grid_d[:, None, :, :]
        mask_d = mask_d.abs().sum(dim=-1)
        mask_d = mask_d.unsqueeze(1) * self.decay[None, :, None, None]
        return mask_d

    def generate_pos_decay(self, H: int, W: int) -> torch.Tensor:
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid(index_h, index_w, indexing="ij")
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = mask.abs().sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_1d_depth_decay(
        self, H: int, W: int, depth_grid: torch.Tensor
    ) -> torch.Tensor:
        mask = depth_grid[:, :, :, :, None] - depth_grid[:, :, :, None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None, None]
        assert mask.shape[2:] == (W, H, H)
        return mask

    def generate_1d_decay(self, length: int) -> torch.Tensor:
        index = torch.arange(length).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    # --- Main ---------------------------------------------------------------
    def forward(
        self,
        hw_tuple: Tuple[int, int],
        depth_map: torch.Tensor,
        split_or_not: bool = False,
    ):
        """Generate geometry prior for the given spatial size.

        Args:
            hw_tuple: ``(H, W)`` of the feature map.
            depth_map: ``(B, 1, H_in, W_in)`` raw depth; will be interpolated.
            split_or_not: ``True`` → decomposed (H/W separate);
                ``False`` → full 2-D.

        Returns:
            ``((sin, cos), (mask_h, mask_w))`` if decomposed, or
            ``((sin, cos), mask)`` if full.
        """
        depth_map = F.interpolate(
            depth_map, size=hw_tuple, mode="bilinear", align_corners=False,
        )

        if split_or_not:
            index = torch.arange(hw_tuple[0] * hw_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(
                hw_tuple[0], hw_tuple[1], -1
            )
            cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(
                hw_tuple[0], hw_tuple[1], -1
            )
            mask_d_h = self.generate_1d_depth_decay(
                hw_tuple[0], hw_tuple[1], depth_map.transpose(-2, -1),
            )
            mask_d_w = self.generate_1d_depth_decay(
                hw_tuple[1], hw_tuple[0], depth_map,
            )
            mask_h = self.generate_1d_decay(hw_tuple[0])
            mask_w = self.generate_1d_decay(hw_tuple[1])
            mask_h = (
                self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2)
                + self.weight[1] * mask_d_h
            )
            mask_w = (
                self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2)
                + self.weight[1] * mask_d_w
            )
            return (sin, cos), (mask_h, mask_w)
        else:
            index = torch.arange(hw_tuple[0] * hw_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(
                hw_tuple[0], hw_tuple[1], -1
            )
            cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(
                hw_tuple[0], hw_tuple[1], -1
            )
            mask = self.generate_pos_decay(hw_tuple[0], hw_tuple[1])
            mask_d = self.generate_depth_decay(
                hw_tuple[0], hw_tuple[1], depth_map,
            )
            mask = self.weight[0] * mask + self.weight[1] * mask_d
            return (sin, cos), mask


# ---------------------------------------------------------------------------
# Decomposed GSA (separable H/W attention)
# ---------------------------------------------------------------------------
class DecomposedGSA(nn.Module):
    """Decomposed Geometry-aware Self-Attention (stages 0-2).

    Performs attention separately along H and W axes with depth-dependent
    positional bias, then composes the results.
    """

    def __init__(self, embed_dim: int, num_heads: int, value_factor: int = 1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * value_factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * value_factor, embed_dim, bias=True)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, rel_pos, split_or_not: bool = False):
        bsz, h, w, _ = x.size()
        (sin, cos), (mask_h, mask_w) = rel_pos
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)
        k = k * self.scaling

        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)

        # W-axis attention
        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)
        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)
        qk_mat_w = qk_mat_w + mask_w.transpose(1, 2)
        qk_mat_w = torch.softmax(qk_mat_w, -1)
        v = torch.matmul(qk_mat_w, v)

        # H-axis attention
        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)
        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
        qk_mat_h = qk_mat_h + mask_h.transpose(1, 2)
        qk_mat_h = torch.softmax(qk_mat_h, -1)
        output = torch.matmul(qk_mat_h, v)

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


# ---------------------------------------------------------------------------
# Full GSA (full 2D attention)
# ---------------------------------------------------------------------------
class FullGSA(nn.Module):
    """Full 2-D Geometry-aware Self-Attention (stage 3).

    Standard multi-head attention over all H*W positions with depth-based
    positional bias.
    """

    def __init__(self, embed_dim: int, num_heads: int, value_factor: int = 1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * value_factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * value_factor, embed_dim, bias=True)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, rel_pos, split_or_not: bool = False):
        bsz, h, w, _ = x.size()
        (sin, cos), mask = rel_pos
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)
        k = k * self.scaling

        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)
        qr = qr.flatten(2, 3)
        kr = kr.flatten(2, 3)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4).flatten(2, 3)

        qk_mat = qr @ kr.transpose(-1, -2)
        qk_mat = qk_mat + mask
        qk_mat = torch.softmax(qk_mat, -1)
        output = torch.matmul(qk_mat, vr).transpose(1, 2).reshape(bsz, h, w, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------
class FeedForwardNetwork(nn.Module):
    """FFN with optional depthwise conv and sub-LayerNorm.

    Args:
        embed_dim: Input/output dimension.
        ffn_dim: Hidden dimension.
        subln: If ``True``, apply LayerNorm inside the FFN.
        subconv: If ``True``, apply depthwise conv inside the FFN.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        activation_fn=F.gelu,
        dropout: float = 0.0,
        activation_dropout: float = 0.0,
        layernorm_eps: float = 1e-6,
        subln: bool = False,
        subconv: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = nn.Dropout(activation_dropout)
        self.dropout_module = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.ffn_layernorm = (
            nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        )
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        residual = x
        if self.dwconv is not None:
            x = self.dwconv(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x
