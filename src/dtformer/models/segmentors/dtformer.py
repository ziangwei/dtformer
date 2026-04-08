"""DTFormer segmentor (top-level model).
DTFormer 分割器（顶层模型）。

Assembles encoder (DTFormerEncoder + TSA-E), decoder (HSG + TSA-D),
and segmentation loss into a single nn.Module.

The model supports tri-modal input: RGB + Depth + Text.
Text guidance is always active when text_features are provided;
when text_features is None, all TSA modules are bypassed and the
model degrades to a pure RGB-D segmentor.

Forward modes:
  - Training: ``forward(rgb, depth, label=label, text_features=text)``
    → returns scalar loss
  - Inference: ``forward(rgb, depth, text_features=text)``
    → returns ``(B, C, H, W)`` logits at input resolution
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones.dtformer_encoder import (
    DTFormerEncoder,
    DTFormer_S,
    DTFormer_B,
    DTFormer_L,
)
from ..decoders.hsg import HSGHead

logger = logging.getLogger(__name__)

# Registry of backbone factory functions
_BACKBONE_REGISTRY = {
    "DTFormer_S": DTFormer_S,
    "DTFormer_B": DTFormer_B,
    "DTFormer_L": DTFormer_L,
}

# Default channel dims per backbone variant
_BACKBONE_CHANNELS = {
    "DTFormer_S": [64, 128, 256, 512],
    "DTFormer_B": [80, 160, 320, 512],
    "DTFormer_L": [112, 224, 448, 640],
}


class DTFormer(nn.Module):
    """DTFormer: Tri-modal semantic segmentation model.

    Args:
        backbone: Backbone variant name (``"DTFormer_S"`` / ``"DTFormer_B"`` / ``"DTFormer_L"``).
        num_classes: Number of segmentation classes.
        text_dim: Text embedding dimension Ct (default 512 for CLIP ViT-B-16).
        drop_path_rate: Stochastic depth rate for the encoder.
        decoder_embed_dim: Internal dimension of the HSG decoder.
        tsae_stages: Encoder stages with TSA-E (default [1,2,3]).
        tsad_stages: Decoder stages with TSA-D (default [1,2,3]).
        tsad_use_topk: Whether decoder TSA-D uses Top-K.
        tsad_top_m: K for decoder Top-K.
        decoder_in_index: Which backbone stages feed the decoder (default [1,2,3]).
        aux_rate: Auxiliary loss weight. 0 = no aux head.
        ignore_index: Label index to ignore in loss (default 255).
        pretrained: Path to pretrained backbone checkpoint (optional).
    """

    def __init__(
        self,
        backbone: str = "DTFormer_S",
        num_classes: int = 40,
        text_dim: int = 512,
        drop_path_rate: float = 0.25,
        decoder_embed_dim: int = 512,
        # TSA config
        tsae_stages: Sequence[int] = (1, 2, 3),
        tsad_stages: Sequence[int] = (1, 2, 3),
        tsad_use_topk: bool = False,
        tsad_top_m: int = 5,
        # Decoder config
        decoder_in_index: Sequence[int] = (1, 2, 3),
        aux_rate: float = 0.0,
        ignore_index: int = 255,
        # Pretrained weights
        pretrained: Optional[str] = None,
    ):
        super().__init__()

        if backbone not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{backbone}'. "
                f"Available: {list(_BACKBONE_REGISTRY.keys())}"
            )

        channels = _BACKBONE_CHANNELS[backbone]
        self.channels = channels
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.aux_rate = aux_rate

        # --- Encoder ---
        self.backbone = _BACKBONE_REGISTRY[backbone](
            text_dim=text_dim,
            drop_path_rate=drop_path_rate,
            tsae_stages=list(tsae_stages),
        )

        # --- Decoder (HSG) ---
        dec_in_index = list(decoder_in_index)
        dec_in_channels = [channels[i] for i in dec_in_index]
        backbone_num_heads = list(self.backbone.num_heads)

        self.decode_head = HSGHead(
            in_channels=dec_in_channels,
            in_index=dec_in_index,
            channels=decoder_embed_dim,
            num_classes=num_classes,
            text_dim=text_dim,
            tsad_stages=list(tsad_stages),
            tsad_use_topk=tsad_use_topk,
            tsad_top_m=tsad_top_m,
            backbone_num_heads=backbone_num_heads,
        )

        # --- Auxiliary head (optional FCN on stage 2) ---
        self.aux_head = None
        if aux_rate > 0:
            self.aux_index = 2
            self.aux_head = nn.Sequential(
                nn.Conv2d(channels[2], channels[2], 3, padding=1, bias=False),
                nn.SyncBatchNorm(channels[2]),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(channels[2], num_classes, 1),
            )

        # --- Loss ---
        self.criterion = nn.CrossEntropyLoss(
            reduction="none", ignore_index=ignore_index,
        )

        # --- Init weights ---
        self._init_decoder_weights()
        if pretrained:
            self.backbone.load_pretrained(pretrained)

    def _init_decoder_weights(self) -> None:
        """Kaiming init for decoder and aux head."""
        for module in [self.decode_head, self.aux_head]:
            if module is None:
                continue
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="relu",
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            rgb: ``(B, 3, H, W)`` RGB image.
            depth: ``(B, 3, H, W)`` depth (3-channel replicated).
            label: ``(B, H, W)`` or ``(B, 1, H, W)`` ground truth.
                   If provided, returns scalar loss; otherwise returns logits.
            text_features: ``(B, T, Ct)`` text embeddings, or ``None``.

        Returns:
            Training mode: scalar loss tensor.
            Inference mode: ``(B, num_classes, H, W)`` logits at input resolution.
        """
        ori_h, ori_w = rgb.shape[2:]

        # Sanitize text
        if text_features is not None:
            if torch.isnan(text_features).any():
                logger.warning("NaN in text_features — replacing with zeros")
                text_features = torch.nan_to_num(text_features, nan=0.0)
            if torch.isinf(text_features).any():
                logger.warning("Inf in text_features — clamping")
                text_features = torch.clamp(text_features, min=-1e6, max=1e6)

        # --- Encoder ---
        feats = self.backbone(rgb, depth, text_features=text_features)
        # feats: tuple of (B, Ci, Hi, Wi) for each stage

        # --- Decoder ---
        out = self.decode_head(feats, text_features=text_features)
        out = F.interpolate(
            out, size=(ori_h, ori_w), mode="bilinear", align_corners=False,
        )

        # --- Auxiliary ---
        aux_out = None
        if self.aux_head is not None:
            aux_out = self.aux_head(feats[self.aux_index])

        # --- Loss (training) ---
        if label is not None:
            return self._compute_loss(out, aux_out, label)

        return out

    def _compute_loss(
        self,
        out: torch.Tensor,
        aux_out: Optional[torch.Tensor],
        label: torch.Tensor,
    ) -> torch.Tensor:
        """Compute segmentation loss with valid-pixel masking and NaN safety.

        Args:
            out: ``(B, C, H, W)`` main logits.
            aux_out: ``(B, C, H', W')`` auxiliary logits or ``None``.
            label: ``(B, H, W)`` or ``(B, 1, H, W)`` ground truth.
        """
        if label.dim() == 4:
            label = label.squeeze(1)
        label = label.long()

        # Valid pixel mask (exclude ignore_index)
        valid_mask = label != self.ignore_index

        # Main loss
        main_loss_map = self.criterion(out, label)
        valid_loss = main_loss_map[valid_mask]

        if valid_loss.numel() > 0:
            loss = valid_loss.mean()
        else:
            logger.warning("Batch has no valid pixels — using zero loss")
            loss = main_loss_map.sum() * 0.0

        # Auxiliary loss
        if aux_out is not None and self.aux_rate > 0:
            aux_out_resized = F.interpolate(
                aux_out, size=label.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            aux_loss_map = self.criterion(aux_out_resized, label)
            valid_aux = aux_loss_map[valid_mask]

            if valid_aux.numel() > 0:
                loss = loss + self.aux_rate * valid_aux.mean()
            else:
                loss = loss + self.aux_rate * (aux_loss_map.sum() * 0.0)

        # NaN safety
        if torch.isnan(loss):
            logger.error("NaN loss detected — replacing with zero")
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

        return loss
