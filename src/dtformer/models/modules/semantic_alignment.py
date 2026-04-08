"""Text-Semantic Alignment Module (TSAM).

Dual-mode cross-attention for text-to-vision alignment:
  - Encoder mode (SSA-lite): lightweight gamma-gated residual, per-block.
  - Decoder mode: full multi-head cross-attention + Top-K + FFN.

Stage injection is configurable; defaults: encoder [1,2,3], decoder all.
"""
