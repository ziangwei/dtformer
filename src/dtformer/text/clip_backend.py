"""CLIP encoding backend.
CLIP 文本编码后端。

Unified CLIP text encoder using ``open_clip_torch``.
Provides offline batch encoding and single-query encoding.
Internal module — only ``open_clip`` is supported; Jina-CLIP has been removed.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple  # Dict used by _GLOBAL

import torch
import torch.nn.functional as F

# Suppress HuggingFace tokenizer fork warnings in DataLoader workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Global singleton: keeps one CLIP model loaded to avoid redundant I/O.
# ---------------------------------------------------------------------------
_GLOBAL: Dict[str, object] = {
    "model": None,
    "tokenizer": None,
    "model_name": None,
    "device": None,
}


def _resolve_model_name(name: Optional[str]) -> Tuple[str, str]:
    """Map common aliases to ``(open_clip_model, pretrained_tag)``."""
    if not name:
        return "ViT-B-16", "openai"
    n = name.lower()
    if "vit-b-16" in n or "b/16" in n or "base-patch16" in n:
        return "ViT-B-16", "openai"
    if "vit-l-14" in n or "l/14" in n or "large-patch14" in n:
        return "ViT-L-14", "openai"
    if "vit-h-14" in n or "h/14" in n:
        return "ViT-H-14", "laion2b_s32b_b79k"
    return "ViT-B-16", "openai"


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------
def load_clip(
    model_name: Optional[str] = None,
    device: Optional[torch.device] = None,
):
    """Load (or reuse) the global CLIP model.

    Args:
        model_name: Open-CLIP model identifier.  ``None`` defaults to
            ``ViT-B-16`` with OpenAI weights.
        device: Target device.  ``None`` auto-selects CUDA when available.

    Returns:
        ``(model, tokenizer, device)``
    """
    import open_clip

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    oc_name, pretrained_tag = _resolve_model_name(model_name)
    cache_key = f"{oc_name}:{pretrained_tag}:{device}"

    if _GLOBAL["model"] is not None and _GLOBAL["model_name"] == cache_key:
        return _GLOBAL["model"], _GLOBAL["tokenizer"], _GLOBAL["device"]

    model, _, _ = open_clip.create_model_and_transforms(
        oc_name, pretrained=pretrained_tag, device=device,
    )
    tokenizer = open_clip.get_tokenizer(oc_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    _GLOBAL.update({
        "model": model,
        "tokenizer": tokenizer,
        "model_name": cache_key,
        "device": device,
    })
    return model, tokenizer, device


def unload_clip():
    """Release the global CLIP model and free GPU memory."""
    _GLOBAL.update({"model": None, "tokenizer": None, "model_name": None, "device": None})
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Encoding primitives
# ---------------------------------------------------------------------------
def encode_texts(
    texts: List[str],
    model_name: Optional[str] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 512,
) -> torch.Tensor:
    """Encode a list of strings into L2-normalised CLIP embeddings.

    Args:
        texts: Raw text strings.
        model_name: Passed to :func:`load_clip`.
        device: Passed to :func:`load_clip`.
        batch_size: Maximum batch size per forward pass.

    Returns:
        ``Tensor[N, D]`` on CPU, float32, L2-normalised.
    """
    import open_clip

    if len(texts) == 0:
        return torch.zeros(0, 512)

    model, _tok, device = load_clip(model_name, device)
    all_feats: List[torch.Tensor] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokens = open_clip.tokenize(batch).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)  # [B, D]
        feats = F.normalize(feats, dim=-1)
        # NaN / Inf safety
        if torch.isnan(feats).any() or torch.isinf(feats).any():
            feats = torch.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
            feats = F.normalize(feats, dim=-1)
        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0).float()


def encode_vocabulary(
    class_names: List[str],
    template_set: str = "clip",
    max_templates: int = 3,
    model_name: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Encode a full vocabulary into a stacked embedding table.

    Preserves the order of *class_names* (no deduplication).

    Returns:
        ``Tensor[C, D]`` on CPU.
    """
    from .templates import expand_label_to_prompts, normalize_label

    if not class_names:
        return torch.zeros(0, 512)

    all_prompts: List[str] = []
    group_sizes: List[int] = []
    for name in class_names:
        variants = expand_label_to_prompts(
            normalize_label(name), template_set, max_templates,
        )
        all_prompts.extend(variants)
        group_sizes.append(len(variants))

    all_embeds = encode_texts(all_prompts, model_name, device)

    rows: List[torch.Tensor] = []
    idx = 0
    for gsz in group_sizes:
        rows.append(all_embeds[idx : idx + gsz].mean(dim=0))
        idx += gsz

    return torch.stack(rows, dim=0)  # [C, D]
