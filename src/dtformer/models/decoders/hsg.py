"""Hierarchical Semantic-Guided (HSG) decoder.
HSG 解码器（层级语义引导）。

Architecture (per the paper):
  For each decoder level i:
    1. (Optional) TSA-D: text-guided cross-attention (Eq. 7-8)
    2. Upsample to the highest resolution
  Concatenate all levels → 1×1 squeeze → Hamburger (NMF) → 1×1 align → cls_seg

The Hamburger module performs Non-negative Matrix Factorization (NMF) for
global context aggregation, replacing traditional pooling methods.

All mmcv/mmseg dependencies have been removed:
  - ConvModule → ConvBN (simple Conv2d + SyncBN + ReLU wrapper)
  - mmseg.ops.resize → F.interpolate
  - BaseDecodeHead(BaseModule) → nn.Module
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.semantic_alignment import TSAD


# ---------------------------------------------------------------------------
# Lightweight ConvBN block (replaces mmcv.cnn.ConvModule)
# ---------------------------------------------------------------------------
class ConvBN(nn.Module):
    """Conv2d + optional SyncBatchNorm + optional ReLU.

    Replaces ``mmcv.cnn.ConvModule`` with a minimal, dependency-free version.
    All convolutions in HSG are 1×1, so we default kernel_size=1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        norm: bool = True,
        act: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=not norm,
        )
        self.bn = nn.SyncBatchNorm(out_channels) if norm else nn.Identity()
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# NMF matrix decomposition
# ---------------------------------------------------------------------------
class _MatrixDecomposition2DBase(nn.Module):
    """Base class for 2D matrix decomposition (Hamburger internals)."""

    def __init__(self, args: dict | None = None):
        super().__init__()
        args = dict(args) if args else {}
        self.spatial = args.get("SPATIAL", True)
        self.S = args.get("MD_S", 1)
        self.D = args.get("MD_D", 512)
        self.R = args.get("MD_R", 64)
        self.train_steps = args.get("TRAIN_STEPS", 6)
        self.eval_steps = args.get("EVAL_STEPS", 7)
        self.inv_t = args.get("INV_T", 100)
        self.eta = args.get("ETA", 0.9)
        self.rand_init = args.get("RAND_INIT", True)

    def _build_bases(self, B: int, S: int, D: int, R: int, cuda: bool = False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def local_inference(self, x, bases):
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)
        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)
        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x: torch.Tensor, return_bases: bool = False):
        B, C, H, W = x.shape
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=x.is_cuda)
        else:
            if not hasattr(self, "bases"):
                self.register_buffer(
                    "bases",
                    self._build_bases(1, self.S, D, self.R, cuda=x.is_cuda),
                )
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)
        coef = self.compute_coef(x, bases, coef)
        x = torch.bmm(bases, coef.transpose(1, 2))

        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)
        return x


class NMF2D(_MatrixDecomposition2DBase):
    """Non-Negative Matrix Factorization for Hamburger module."""

    def __init__(self, args: dict | None = None):
        super().__init__(args)
        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        bases = torch.rand((B * S, D, R))
        if cuda:
            bases = bases.cuda()
        return F.normalize(bases, dim=1)

    def local_step(self, x, bases, coef):
        # Multiplicative update rules for NMF
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)

        numerator = torch.bmm(x, coef)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)
        return bases, coef

    def compute_coef(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        return coef


class Hamburger(nn.Module):
    """Hamburger module: 1×1 → NMF → 1×1 with residual connection."""

    def __init__(
        self,
        ham_channels: int = 512,
        ham_kwargs: dict | None = None,
    ):
        super().__init__()
        self.ham_in = ConvBN(ham_channels, ham_channels, norm=False, act=False)
        self.ham = NMF2D(ham_kwargs)
        self.ham_out = ConvBN(ham_channels, ham_channels, norm=True, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.ham_in(x), inplace=True)
        y = self.ham(y)
        y = self.ham_out(y)
        return F.relu(x + y, inplace=True)


# ---------------------------------------------------------------------------
# Base decode head (replaces mmseg BaseDecodeHead)
# ---------------------------------------------------------------------------
class BaseDecodeHead(nn.Module, metaclass=ABCMeta):
    """Minimal base class for decode heads (replaces mmseg version).

    Provides:
      - Input channel / index validation
      - ``_transform_inputs()``: selects and optionally resizes multi-scale features
      - ``cls_seg()``: dropout → 1×1 conv → class logits
    """

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        channels: int,
        num_classes: int,
        dropout_ratio: float = 0.1,
        in_index: Union[int, Sequence[int]] = -1,
        input_transform: str | None = None,
        align_corners: bool = False,
    ):
        super().__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None

    def _init_inputs(self, in_channels, in_index, input_transform):
        if input_transform is not None:
            assert input_transform in ("resize_concat", "multiple_select")
        self.input_transform = input_transform
        self.in_index = in_index

        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == "resize_concat":
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = list(in_channels)
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(
        self, inputs: Sequence[torch.Tensor],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Select and optionally resize multi-scale features."""
        if self.input_transform == "resize_concat":
            feats = [inputs[i] for i in self.in_index]
            upsampled = [
                F.interpolate(
                    x, size=feats[0].shape[2:],
                    mode="bilinear", align_corners=self.align_corners,
                )
                for x in feats
            ]
            return torch.cat(upsampled, dim=1)
        elif self.input_transform == "multiple_select":
            return [inputs[i] for i in self.in_index]
        else:
            return inputs[self.in_index]

    def cls_seg(self, feat: torch.Tensor) -> torch.Tensor:
        """Dropout → 1×1 conv → class logits."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        return self.conv_seg(feat)

    @abstractmethod
    def forward(
        self,
        inputs: Sequence[torch.Tensor],
        text_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...


# ---------------------------------------------------------------------------
# HSG Head
# ---------------------------------------------------------------------------
class HSGHead(BaseDecodeHead):
    """Hierarchical Semantic-Guided decoder with per-layer TSA-D.

    Architecture:
      1. For each input level i (from backbone stages selected by ``in_index``):
         - If TSA-D is enabled at that stage: apply cross-attention with text
         - Upsample to the highest input resolution
      2. Concatenate all levels → 1×1 squeeze → Hamburger (NMF) → 1×1 align
      3. ``cls_seg()`` produces final logits

    Args:
        in_channels: Per-stage channel dimensions (e.g. [128, 256, 512]).
        in_index: Which backbone stages to use (e.g. [1, 2, 3]).
        channels: Internal feature dimension (ham_channels). Default 512.
        num_classes: Number of segmentation classes.
        text_dim: Dimension of text embeddings (Ct).
        tsad_stages: Which global stage indices get TSA-D (default [1,2,3]).
        tsad_use_topk: Whether TSA-D uses Top-K token selection.
        tsad_top_m: K value for Top-K selection.
        backbone_num_heads: Per-stage head counts from the backbone.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        in_index: Sequence[int] = (1, 2, 3),
        channels: int = 512,
        num_classes: int = 40,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        # Text / TSA-D config
        text_dim: int = 512,
        tsad_stages: Sequence[int] = (1, 2, 3),
        tsad_use_topk: bool = False,
        tsad_top_m: int = 5,
        backbone_num_heads: Sequence[int] = (4, 4, 8, 16),
        tsad_logit_init: float = 1 / 0.07,
    ):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            in_index=in_index,
            input_transform="multiple_select",
            align_corners=align_corners,
        )

        self.ham_channels = channels
        self.text_dim = text_dim
        self.backbone_num_heads = list(backbone_num_heads)

        # --- Per-layer TSA-D modules ---
        global_enable = set(int(s) for s in tsad_stages)
        self.dec_tsad_enabled: List[bool] = []
        self.dec_tsad_layers = nn.ModuleList()

        for local_i, c_in in enumerate(self.in_channels):
            global_idx = (
                self.in_index[local_i]
                if isinstance(self.in_index, (list, tuple))
                else local_i
            )
            enabled = global_idx in global_enable
            self.dec_tsad_enabled.append(enabled)
            if enabled:
                self.dec_tsad_layers.append(
                    TSAD(
                        query_dim=c_in,
                        text_dim=text_dim,
                        use_topk=tsad_use_topk,
                        top_m=tsad_top_m,
                        num_heads=self.backbone_num_heads[global_idx],
                        alpha_init=0.1,
                        logit_scale_init=tsad_logit_init,
                    )
                )
            else:
                # Placeholder to keep indices aligned
                self.dec_tsad_layers.append(nn.Identity())

        # --- Hamburger path ---
        self.squeeze = ConvBN(sum(self.in_channels), self.ham_channels)
        self.hamburger = Hamburger(self.ham_channels)
        self.align = ConvBN(self.ham_channels, self.channels)

    def _apply_tsad(
        self,
        x_bchw: torch.Tensor,
        tsad_layer: nn.Module,
        text_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply TSA-D with BCHW ↔ NHWC conversion and safety checks."""
        if text_features is None:
            return x_bchw
        if isinstance(tsad_layer, nn.Identity):
            return x_bchw

        # Skip if entire batch has no valid text tokens
        if isinstance(text_features, torch.Tensor) and text_features.numel() > 0:
            tf = text_features
            if tf.dim() == 2:
                tf = tf.unsqueeze(0)
            if tf.dim() == 3:
                valid_any = (tf.abs().sum(dim=-1) > 0).any(dim=1)
                if not bool(valid_any.any()):
                    return x_bchw

        # BCHW → NHWC → TSA-D → NHWC → BCHW
        x_nhwc = x_bchw.permute(0, 2, 3, 1).contiguous()
        y_nhwc = tsad_layer(x_nhwc, text_features)
        return y_nhwc.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        inputs: Sequence[torch.Tensor],
        text_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """HSG decoder forward.

        Args:
            inputs: Multi-scale features from backbone (list of BCHW tensors).
            text_features: ``(B, T, Ct)`` text embeddings, or ``None``.

        Returns:
            ``(B, num_classes, H, W)`` segmentation logits at the highest
            input feature resolution.
        """
        feats = self._transform_inputs(inputs)  # list of (B, Ci, Hi, Wi)

        # Per-layer TSA-D + upsample to common resolution
        tgt_hw = feats[0].shape[2:]
        processed = []
        for i, f in enumerate(feats):
            if self.dec_tsad_enabled[i]:
                f = self._apply_tsad(f, self.dec_tsad_layers[i], text_features)
            f = F.interpolate(
                f, size=tgt_hw, mode="bilinear",
                align_corners=self.align_corners,
            )
            processed.append(f)

        # Concat → squeeze → Hamburger → align → classify
        x = torch.cat(processed, dim=1)     # (B, sum(Ci), H, W)
        x = self.squeeze(x)                 # (B, ham_channels, H, W)
        x = self.hamburger(x)               # (B, ham_channels, H, W)
        x = self.align(x)                   # (B, channels, H, W)
        return self.cls_seg(x)              # (B, num_classes, H, W)
