"""Text-Semantic Alignment (TSA) modules.
文本语义对齐模块。

Two variants of text-guided cross-attention for text-to-vision alignment:

  - **TSA-E** (Text-Semantic Alignment — Encoder):
    Lightweight TSCA + residual gating (β_e).  No Top-K, no FFN, no extra
    LayerNorm.  Applied per-block in encoder stages 2-4.  (Eq. 6 in paper.)

  - **TSA-D** (Text-Semantic Alignment — Decoder):
    Full Pre-LN + TSCA + MLP + residual gating (β_d).  Optional Top-K
    token selection.  Applied at every decoder level.  (Eq. 7-8 in paper.)

Both use **TSCA** (Temperature-Scaled Cosine Attention, Eq. 4):
    Q' = L2-norm(Q·W_q),  K' = L2-norm(T·W_k)
    A  = softmax(Q'·K'^T / τ),  where τ is a learnable temperature.

Cosine similarity and learnable temperature are always enabled (Table 5,
"Full (ours)" configuration).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def _ensure_batched_text(
    text_features: torch.Tensor, B: int
) -> torch.Tensor:
    """Normalise text to ``(B, T, Ct)``."""
    if text_features.dim() == 3:
        return text_features
    if text_features.dim() == 2:
        return text_features.unsqueeze(0).expand(B, -1, -1)
    if text_features.dim() == 1:
        return text_features.view(1, 1, -1).expand(B, 1, -1)
    raise ValueError(f"Unsupported text tensor shape: {text_features.shape}")


def _make_text_pad_mask(
    text_feats: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Return ``(B, T)`` bool mask: ``True`` for padding (zero-vector) tokens."""
    return text_feats.float().abs().sum(dim=-1) <= eps


# ===================================================================
# TSA-E  (Text-Semantic Alignment — Encoder)
# ===================================================================
class TSAE(nn.Module):
    """Lightweight encoder-side text alignment (Eq. 6).

    Performs Temperature-Scaled Cosine Attention (TSCA) between visual
    tokens and text embeddings, then applies a gated residual:

        V_out = V + β_e · TSCA(V, T)

    No FFN, no extra LayerNorm.  Designed to be inserted between GSA
    and FFN inside each RGB-D-T Block.

    Args:
        query_dim: Visual feature dimension Cv (current stage).
        text_dim: Text embedding dimension Ct (typically 512).
        num_heads: Number of attention heads (matches backbone stage).
        gamma_scale: Extra scaling factor for residual gate β_e.
        logit_scale_init: Initial value for the learnable temperature τ.
    """

    def __init__(
        self,
        query_dim: int,
        text_dim: int,
        num_heads: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        gamma_scale: float = 1.0,
        logit_scale_init: float = 1.0,
        clamp_logit: float = 2.0,
    ):
        super().__init__()

        # For compatibility with FLOPs counters (thop / ptflops)
        self.register_buffer("total_ops", torch.zeros(1))
        self.register_buffer("total_params", torch.zeros(1))

        self.clamp_logit = float(clamp_logit)

        # Multi-head config
        self.num_heads = int(num_heads)
        assert query_dim % self.num_heads == 0, (
            f"query_dim({query_dim}) must be divisible by num_heads({self.num_heads})"
        )
        self.head_dim = query_dim // self.num_heads
        self.d_k = float(self.head_dim)
        self.register_buffer(
            "_inv_sqrt_dk",
            torch.tensor(1.0 / math.sqrt(self.d_k), dtype=torch.float),
        )

        # TSCA projections
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(text_dim, query_dim)
        self.v_proj = nn.Linear(text_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Learnable temperature τ  (always learnable — Table 5 "Full (ours)")
        enc_init = math.log(max(logit_scale_init, 1e-6))
        self.logit_scale = nn.Parameter(
            torch.tensor(enc_init, dtype=torch.float),
            requires_grad=True,
        )

        # Residual gate β_e
        self.gamma = nn.Parameter(torch.tensor(0.5, dtype=torch.float))
        self.register_buffer(
            "gamma_scale",
            torch.tensor(float(gamma_scale), dtype=torch.float),
        )

        # Optional: save attention maps for visualization
        self.save_attention = False
        self.last_attention_map = None
        self.last_spatial_shape = None

        # Init
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _temperature_scale(self) -> torch.Tensor:
        """Compute attention scaling factor: exp(τ) / √d_k."""
        scale_log = torch.clamp(
            self.logit_scale, min=-self.clamp_logit, max=self.clamp_logit
        )
        return torch.exp(scale_log) * self._inv_sqrt_dk

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """TSA-E forward: lightweight multi-head TSCA + gated residual.

        Args:
            visual_features: ``(B, H, W, Cv)`` — NHWC layout.
            text_features: ``(B, T, Ct)`` / ``(T, Ct)`` / ``(B, Ct)``.

        Returns:
            ``(B, H, W, Cv)``
        """
        B, H, W, Cv = visual_features.shape
        x = visual_features.view(B, H * W, Cv)
        q_full = self.q_proj(x)

        # Unify text shape + cast to matching device/dtype
        text_b = _ensure_batched_text(text_features, B).to(
            device=visual_features.device, dtype=visual_features.dtype,
        )
        pad_mask = _make_text_pad_mask(text_b)

        # Trim to active token length
        valid_len = (~pad_mask).sum(dim=1)
        T_active = int(valid_len.max().item()) if valid_len.numel() > 0 else 0
        if T_active == 0:
            return visual_features

        if T_active < text_b.size(1):
            text_b = text_b[:, :T_active, :]
            pad_mask = pad_mask[:, :T_active]

        k_full = self.k_proj(text_b)
        v_full = self.v_proj(text_b)

        Hh, Dh = self.num_heads, self.head_dim
        q = q_full.view(B, -1, Hh, Dh)
        k = k_full.view(B, -1, Hh, Dh)

        # TSCA: L2-normalize Q and K (cosine attention — always on)
        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)

        q = q.permute(0, 2, 1, 3)  # [B, Hh, N, Dh]
        k = k.permute(0, 2, 1, 3)  # [B, Hh, T, Dh]
        v = v_full.view(B, -1, Hh, Dh).permute(0, 2, 1, 3)

        all_pad = pad_mask.all(dim=1)
        active_idx = (~all_pad).nonzero(as_tuple=False).squeeze(1)

        # Default: copy input (no text → unchanged)
        y = x.clone()

        if active_idx.numel() > 0:
            q_act = q.index_select(0, active_idx)
            k_act = k.index_select(0, active_idx)
            v_act = v.index_select(0, active_idx)
            pad_act = pad_mask.index_select(0, active_idx)

            # Temperature-scaled cosine attention (Eq. 4)
            attn_logits = (
                torch.matmul(q_act, k_act.transpose(-2, -1))
                * self._temperature_scale()
            )
            attn_logits = attn_logits.masked_fill(
                pad_act.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            attn = torch.softmax(attn_logits, dim=-1)

            if self.save_attention:
                full_attn = torch.zeros(
                    B, Hh, q.size(2), attn.size(-1),
                    device=attn.device, dtype=attn.dtype,
                )
                full_attn.index_copy_(0, active_idx, attn)
                self.last_attention_map = full_attn.permute(0, 2, 1, 3).detach()
                self.last_spatial_shape = (H, W)

            aligned = torch.matmul(attn, v_act)
            aligned = aligned.permute(0, 2, 1, 3).reshape(
                active_idx.numel(), -1, Hh * Dh,
            )
            aligned = self.out_proj(aligned)

            # Gated residual: V_out = V + β_e · F  (Eq. 6)
            y_active = (
                x.index_select(0, active_idx)
                + (self.gamma * self.gamma_scale) * aligned
            )
            y.index_copy_(0, active_idx, y_active)

        return y.view(B, H, W, Cv)


# ===================================================================
# TSA-D  (Text-Semantic Alignment — Decoder)
# ===================================================================
class TSAD(nn.Module):
    """Full decoder-side text alignment (Eq. 7-8).

    Performs Pre-LN + TSCA + MLP with gated residual:

        Ṽ = LN(V)
        F = TSCA(Ṽ, T)                        (Eq. 7)
        V_out = MLP(LN(V + β_d · F)) + LN(V + β_d · F)   (Eq. 8)

    Optional Top-K text token selection for efficiency.

    Args:
        query_dim: Visual feature dimension Cv.
        text_dim: Text embedding dimension Ct (typically 512).
        top_m: Top-K value for token selection.
        use_topk: Whether to apply Top-K in decoder mode.
        num_heads: Number of attention heads.
        alpha_init: Initial value for residual gate β_d.
        logit_scale_init: Initial value for the learnable temperature τ.
    """

    def __init__(
        self,
        query_dim: int,
        text_dim: int,
        top_m: int = 5,
        use_topk: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        add_residual: bool = True,
        alpha_init: float = 0.1,
        clamp_logit: float = 2.0,
        num_heads: int = 1,
        logit_scale_init: float = 1 / 0.07,
    ):
        super().__init__()

        # For compatibility with FLOPs counters (thop / ptflops)
        self.register_buffer("total_ops", torch.zeros(1))
        self.register_buffer("total_params", torch.zeros(1))

        self.top_m = top_m
        self.use_topk = use_topk
        self.add_residual = add_residual
        self.clamp_logit = float(clamp_logit)

        # Multi-head config
        self.num_heads = int(num_heads)
        assert query_dim % self.num_heads == 0, (
            f"query_dim({query_dim}) must be divisible by num_heads({self.num_heads})"
        )
        self.head_dim = query_dim // self.num_heads
        self.d_k = float(self.head_dim)
        self.register_buffer(
            "_inv_sqrt_dk",
            torch.tensor(1.0 / math.sqrt(self.d_k), dtype=torch.float),
        )

        # TSCA projections
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(text_dim, query_dim)
        self.v_proj = nn.Linear(text_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Pre-LN layers
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        # Learnable temperature τ  (always learnable — Table 5 "Full (ours)")
        dec_init = math.log(max(logit_scale_init, 1e-6))
        self.logit_scale = nn.Parameter(
            torch.tensor(dec_init, dtype=torch.float),
            requires_grad=True,
        )

        # MLP (Eq. 8)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(ffn_drop),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(ffn_drop),
        )

        # Residual gate β_d
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float))

        # Optional: save attention maps for visualization
        self.save_attention = False
        self.last_attention_map = None
        self.last_spatial_shape = None

        # Init
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _temperature_scale(self) -> torch.Tensor:
        """Compute attention scaling factor: exp(τ) / √d_k."""
        scale_log = torch.clamp(
            self.logit_scale, min=-self.clamp_logit, max=self.clamp_logit
        )
        return torch.exp(scale_log) * self._inv_sqrt_dk

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """TSA-D forward: Pre-LN + TSCA + MLP + gated residual.

        Args:
            visual_features: ``(B, H, W, Cv)`` — NHWC layout.
            text_features: ``(B, T, Ct)`` / ``(T, Ct)`` / ``(B, Ct)``.

        Returns:
            ``(B, H, W, Cv)``
        """
        B, H, W, Cv = visual_features.shape

        # Pre-LN + Q
        x = self.norm1(visual_features).view(B, H * W, Cv)
        q_full = self.q_proj(x)

        # Text → K/V
        text_b = _ensure_batched_text(text_features, B)
        k_full = self.k_proj(text_b)
        v_full = self.v_proj(text_b)
        pad_mask = _make_text_pad_mask(text_b)

        # Check for all-padding batch
        valid_len = (~pad_mask).sum(dim=1)
        T_active = int(valid_len.max().item()) if valid_len.numel() > 0 else 0
        if T_active == 0:
            y = self.norm2(x)
            y = y + self.ffn(y)
            return y.view(B, H, W, Cv)

        all_pad_per_sample = pad_mask.all(dim=1)

        # Multi-head split
        Hh, Dh = self.num_heads, self.head_dim
        q = q_full.view(B, -1, Hh, Dh)
        k = k_full.view(B, -1, Hh, Dh)

        # TSCA: L2-normalize Q and K (cosine attention — always on)
        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)

        v = v_full.view(B, -1, Hh, Dh)

        # Temperature-scaled cosine attention (Eq. 4)
        scale = self._temperature_scale()
        sim = torch.einsum("bnhd,bthd->bnht", q, k) * scale
        if pad_mask.any():
            sim = sim.masked_fill(
                pad_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # Top-K token selection
        if (
            self.use_topk
            and self.top_m is not None
            and self.top_m < sim.size(-1)
        ):
            _topv, topi = torch.topk(sim, k=self.top_m, dim=-1)
            mask = torch.zeros_like(sim).scatter_(-1, topi, 1.0)
            sim = sim.masked_fill(mask.eq(0), float("-inf"))

        attn = F.softmax(sim, dim=-1)
        if all_pad_per_sample.any():
            attn = attn.masked_fill(
                all_pad_per_sample.view(B, 1, 1, 1).expand_as(attn), 0.0,
            )
        attn = self.attn_drop(attn)

        if self.save_attention:
            self.last_attention_map = attn.detach()
            self.last_spatial_shape = (H, W)

        # Aggregate
        aligned_h = torch.einsum("bnht,bthd->bnhd", attn, v)
        aligned = self.proj_drop(aligned_h.reshape(B, -1, Hh * Dh))
        aligned = self.out_proj(aligned)

        # Gated residual + MLP (Pre-LN)  (Eq. 7-8)
        y = (
            (x + self.alpha * aligned)
            if self.add_residual
            else (self.alpha * aligned)
        )
        y = self.norm2(y)
        y = y + self.ffn(y)
        return y.view(B, H, W, Cv)
