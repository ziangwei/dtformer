"""DTFormer encoder (backbone).
DTFormer 编码器（主干网络）。

Depth-aware hierarchical vision transformer with:
  - Geometry-prior attention (GeoPriorGen + decomposed/full GSA)
  - 4-stage hierarchical RGB-D-T Block structure
  - Per-block TSA-E (Text-Semantic Alignment — Encoder) injection
  - Configurable TSA-E injection per stage (default: stages 1, 2, 3)

Each RGB-D-T Block follows the pattern (Fig. 3b):
    GSA → TSA-E → FFN
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_

from ..modules.geometry_attention import (
    DWConv2d,
    DecomposedGSA,
    FeedForwardNetwork,
    FullGSA,
    GeoPriorGen,
)
from ..modules.semantic_alignment import TSAE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch embedding / merging
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """4× downsampling patch embedding (stride-2 × 2)."""

    def __init__(self, in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.SyncBatchNorm(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).permute(0, 2, 3, 1)  # NCHW → NHWC


class PatchMerging(nn.Module):
    """2× spatial downsampling between stages."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.SyncBatchNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.reduction(x)
        x = self.norm(x)
        return x.permute(0, 2, 3, 1)


# ---------------------------------------------------------------------------
# RGB-D-T Block  (Fig. 3b)
# ---------------------------------------------------------------------------
class RGBDTBlock(nn.Module):
    """Single RGB-D-T transformer block: GSA → TSA-E → FFN.

    When ``tsae_block`` and ``text_features`` are provided, TSA-E is
    applied between the geometry-aware self-attention and the FFN.
    """

    def __init__(
        self,
        split_or_not: bool,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        drop_path: float = 0.0,
        layerscale: bool = False,
        layer_init_values: float = 1e-5,
        init_value: float = 2.0,
        heads_range: float = 4.0,
    ):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.Attention = (
            DecomposedGSA(embed_dim, num_heads)
            if split_or_not
            else FullGSA(embed_dim, num_heads)
        )
        self.drop_path = DropPath(drop_path)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.cnn_pos_encode = DWConv2d(embed_dim, 3, 1, 1)
        self.Geo = GeoPriorGen(embed_dim, num_heads, init_value, heads_range)
        if layerscale:
            self.gamma_1 = nn.Parameter(
                layer_init_values * torch.ones(1, 1, 1, embed_dim),
                requires_grad=True,
            )
            self.gamma_2 = nn.Parameter(
                layer_init_values * torch.ones(1, 1, 1, embed_dim),
                requires_grad=True,
            )

    def forward(
        self,
        x: torch.Tensor,
        x_e: torch.Tensor,
        split_or_not: bool = False,
        tsae_block: Optional[nn.Module] = None,
        text_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.cnn_pos_encode(x)
        b, h, w, d = x.size()
        geo_prior = self.Geo((h, w), x_e, split_or_not=split_or_not)

        # GSA: geometry-aware self-attention
        out = self.Attention(self.layer_norm1(x), geo_prior, split_or_not)

        # TSA-E: text guidance injection between GSA and FFN
        if tsae_block is not None and text_features is not None:
            out = tsae_block(out, text_features)

        # Residual 1
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * out)
        else:
            x = x + self.drop_path(out)

        # FFN + Residual 2
        if self.layerscale:
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.layer_norm2(x)))
        else:
            x = x + self.drop_path(self.ffn(self.layer_norm2(x)))

        return x


# ---------------------------------------------------------------------------
# Stage (BasicLayer)
# ---------------------------------------------------------------------------
class BasicLayer(nn.Module):
    """One stage: a sequence of RGB-D-T Blocks + optional downsampling."""

    def __init__(
        self,
        embed_dim: int,
        out_dim: Optional[int],
        depth: int,
        num_heads: int,
        init_value: float,
        heads_range: float,
        ffn_dim: int = 96,
        drop_path: float | list = 0.0,
        split_or_not: bool = False,
        downsample: bool = False,
        use_checkpoint: bool = False,
        layerscale: bool = False,
        layer_init_values: float = 1e-5,
        tsae_share_factor: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.split_or_not = split_or_not
        self.tsae_share_factor = max(1, int(tsae_share_factor))

        self.blocks = nn.ModuleList([
            RGBDTBlock(
                split_or_not=split_or_not,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                drop_path=(
                    drop_path[i] if isinstance(drop_path, list) else drop_path
                ),
                layerscale=layerscale,
                layer_init_values=layer_init_values,
                init_value=init_value,
                heads_range=heads_range,
            )
            for i in range(depth)
        ])
        self.downsample = (
            PatchMerging(embed_dim, out_dim) if downsample and out_dim else None
        )

    def forward(
        self,
        x: torch.Tensor,
        x_e: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        tsae_blocks: Optional[nn.ModuleList] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for b_idx, blk in enumerate(self.blocks):
            # Resolve which TSA-E block to use (shared across groups)
            tsae_block = None
            if tsae_blocks is not None and len(tsae_blocks) > 0:
                pair_idx = b_idx // self.tsae_share_factor
                if pair_idx < len(tsae_blocks):
                    tsae_block = tsae_blocks[pair_idx]

            if self.use_checkpoint:
                x = checkpoint.checkpoint(
                    blk,
                    x, x_e,
                    self.split_or_not,
                    tsae_block,
                    text_features,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x, x_e,
                    split_or_not=self.split_or_not,
                    tsae_block=tsae_block,
                    text_features=text_features,
                )

        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        return x, x


# ---------------------------------------------------------------------------
# DTFormer Encoder
# ---------------------------------------------------------------------------
class DTFormerEncoder(nn.Module):
    """DTFormer encoder: depth-aware hierarchical transformer backbone.

    Per-block TSA-E is injected at configurable stages (default: 1, 2, 3).

    Args:
        embed_dims: Channel dimensions for each of the 4 stages.
        depths: Number of blocks per stage.
        num_heads: Number of attention heads per stage.
        tsae_stages: Which stages get TSA-E injection (default [1,2,3]).
        text_dim: Dimension of input text embeddings.
        drop_path_rate: Stochastic depth rate.
    """

    def __init__(
        self,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        embed_dims: List[int] = (64, 128, 256, 512),
        depths: List[int] = (3, 4, 18, 4),
        num_heads: List[int] = (4, 4, 8, 16),
        init_values: List[float] = (2, 2, 2, 2),
        heads_ranges: List[float] = (4, 4, 6, 6),
        mlp_ratios: List[float] = (4, 4, 3, 3),
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
        layerscales: List[bool] = (False, False, False, False),
        layer_init_values: float = 1e-6,
        norm_eval: bool = True,
        # Text / TSA-E config
        text_dim: int = 512,
        tsae_stages: Sequence[int] = (1, 2, 3),
        tsae_share_factors: Optional[Sequence[int]] = None,
        tsae_gamma_scale: float = 1.0,
        tsae_logit_init: float = 1.0,
    ):
        super().__init__()
        self.out_indices = out_indices
        self.num_layers = len(depths)
        self.num_heads = list(num_heads)
        self.norm_eval = norm_eval

        self._tsae_enabled = set(int(s) for s in tsae_stages)

        # Patch embedding (4× downsample)
        self.patch_embed = PatchEmbed(in_chans=3, embed_dim=embed_dims[0])

        # TSA-E block sharing factors per stage (Table 9)
        # Prefer explicit config; fall back to heuristic if not provided.
        if tsae_share_factors is not None:
            assert len(tsae_share_factors) == self.num_layers, (
                f"tsae_share_factors length ({len(tsae_share_factors)}) != "
                f"num_layers ({self.num_layers})"
            )
            self._tsae_share_factors = list(tsae_share_factors)
        else:
            self._tsae_share_factors = [
                self._resolve_share_factor(depths[i], embed_dims[i])
                for i in range(self.num_layers)
            ]

        # Stochastic depth schedule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        # Build stages
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i],
                out_dim=embed_dims[i + 1] if i < self.num_layers - 1 else None,
                depth=depths[i],
                num_heads=num_heads[i],
                init_value=init_values[i],
                heads_range=heads_ranges[i],
                ffn_dim=int(mlp_ratios[i] * embed_dims[i]),
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                split_or_not=(i != 3),  # stages 0-2: decomposed; 3: full
                downsample=(i < self.num_layers - 1),
                use_checkpoint=use_checkpoint,
                layerscale=layerscales[i],
                layer_init_values=layer_init_values,
                tsae_share_factor=self._tsae_share_factors[i],
            )
            self.layers.append(layer)

        # Per-block TSA-E modules
        self.encoder_tsae_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            if i in self._tsae_enabled:
                share = self._tsae_share_factors[i]
                num_units = (depths[i] + share - 1) // share
                ml = nn.ModuleList([
                    TSAE(
                        query_dim=embed_dims[i],
                        text_dim=text_dim,
                        num_heads=num_heads[i],
                        gamma_scale=tsae_gamma_scale,
                        logit_scale_init=tsae_logit_init,
                    )
                    for _ in range(num_units)
                ])
                self.encoder_tsae_blocks.append(ml)
            else:
                self.encoder_tsae_blocks.append(nn.ModuleList())

        # Extra norms between stages
        self.extra_norms = nn.ModuleList([
            nn.LayerNorm(embed_dims[i + 1])
            for i in range(len(embed_dims) - 1)
        ])

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Init / Loading
    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except AttributeError:
                pass

    def load_pretrained(self, path: str) -> None:
        """Load a pretrained backbone checkpoint.

        Handles both raw state dicts and wrapped ``{"model": ...}`` /
        ``{"state_dict": ...}`` formats.  Keys prefixed with ``backbone.``
        are stripped automatically.
        """
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if "model" in raw:
            raw = raw["model"]
        if "state_dict" in raw:
            raw = raw["state_dict"]

        state_dict = OrderedDict()
        for k, v in raw.items():
            if k.startswith("backbone."):
                state_dict[k[9:]] = v
            else:
                state_dict[k] = v

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading pretrained: {missing}")
        if unexpected:
            logger.info(f"Unexpected keys (ignored): {unexpected}")
        logger.info(f"Loaded pretrained backbone from {path}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        x_e: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass.

        Args:
            x: RGB input ``(B, 3, H, W)``.
            x_e: Depth input ``(B, 3, H, W)`` (3-channel replicated).
            text_features: ``(B, T, D)`` text embeddings or ``None``.

        Returns:
            Tuple of multi-scale features, one per ``out_indices`` stage,
            each ``(B, C_i, H_i, W_i)`` in NCHW layout.
        """
        # Patch embed
        x = self.patch_embed(x)  # (B, H/4, W/4, C0) NHWC

        # Depth: take single channel
        x_e = x_e[:, 0, :, :].unsqueeze(1)  # (B, 1, H, W)

        # Prepare text
        use_text = text_features is not None
        if use_text:
            if text_features.dim() == 2:
                text_features = text_features.unsqueeze(0)
            text_features = text_features.to(device=x.device, dtype=x.dtype)

        outs: List[torch.Tensor] = []
        for i in range(self.num_layers):
            tsae_blocks = (
                self.encoder_tsae_blocks[i]
                if (use_text and i in self._tsae_enabled)
                else None
            )
            x_out, x = self.layers[i](
                x, x_e,
                text_features=text_features,
                tsae_blocks=tsae_blocks,
            )

            if i in self.out_indices:
                if i != 0:
                    x_out = self.extra_norms[i - 1](x_out)
                outs.append(x_out.permute(0, 3, 1, 2).contiguous())

        return tuple(outs)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    # ------------------------------------------------------------------
    # Share factor heuristic (Table 9)
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_share_factor(depth: int, embed_dim: int) -> int:
        """Determine TSA-E block sharing factor for a stage.

        Controls how many consecutive blocks share one TSA-E module.
        Larger models use higher sharing to keep memory reasonable.
        """
        if depth >= 16:
            if embed_dim >= 448:   # L model
                return 8
            if embed_dim >= 320:   # B model
                return 4
            return 4               # S model deep stage

        if depth >= 6 and embed_dim >= 640:
            return 4
        if depth >= 6 and embed_dim >= 512:
            return 2

        return 2  # default

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------
def DTFormer_S(**kwargs) -> DTFormerEncoder:
    """DTFormer-Small encoder."""
    return DTFormerEncoder(
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        num_heads=[4, 4, 8, 16],
        heads_ranges=[4, 4, 6, 6],
        **kwargs,
    )


def DTFormer_B(**kwargs) -> DTFormerEncoder:
    """DTFormer-Base encoder."""
    return DTFormerEncoder(
        embed_dims=[80, 160, 320, 512],
        depths=[4, 8, 25, 8],
        num_heads=[5, 5, 10, 16],
        heads_ranges=[5, 5, 6, 6],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6,
        **kwargs,
    )


def DTFormer_L(**kwargs) -> DTFormerEncoder:
    """DTFormer-Large encoder."""
    return DTFormerEncoder(
        embed_dims=[112, 224, 448, 640],
        depths=[4, 8, 25, 8],
        num_heads=[7, 7, 14, 20],
        heads_ranges=[6, 6, 6, 6],
        layerscales=[False, False, True, True],
        layer_init_values=1e-6,
        **kwargs,
    )
