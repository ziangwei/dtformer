"""Prompt templates for CLIP text encoding.
CLIP 文本模板。

Defines template sets used to expand class names into multiple textual
prompts before CLIP encoding.  Template expansion + averaging improves
zero-shot classification robustness.
"""

import re
from typing import List

from .vocabularies import LABEL_ALIASES

# ---------------------------------------------------------------------------
# Template sets
# ---------------------------------------------------------------------------
CLIP_TEMPLATES: List[str] = [
    "a photo of a {}.",
    "this is a photo of a {}.",
    "an image of a {}.",
]

TEMPLATE_REGISTRY = {
    "clip": CLIP_TEMPLATES,
}

# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------
def normalize_label(label: str) -> str:
    """Lowercase, collapse whitespace, apply alias mapping."""
    s = label.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return LABEL_ALIASES.get(s, s)


def _pick_article(noun: str) -> str:
    """Return 'an' for vowel-initial nouns, 'a' otherwise."""
    return "an" if noun[:1] in "aeiou" else "a"


# ---------------------------------------------------------------------------
# Template expansion
# ---------------------------------------------------------------------------
def expand_label_to_prompts(
    label: str,
    template_set: str = "clip",
    max_templates: int = 3,
) -> List[str]:
    """Expand a single label into a list of prompted strings.

    Args:
        label: Raw class name (will be normalized).
        template_set: Key in ``TEMPLATE_REGISTRY``.  ``"none"`` returns the
            bare label.
        max_templates: Maximum number of templates to use.

    Returns:
        List of prompt strings (length <= *max_templates*).
    """
    lbl = normalize_label(label)

    if template_set == "none" or template_set not in TEMPLATE_REGISTRY:
        return [lbl]

    templates = TEMPLATE_REGISTRY[template_set][:max_templates]
    prompts: List[str] = []
    for t in templates:
        if "{}" not in t:
            prompts.append(f"{t} {lbl}")
            continue
        # Handle article replacement: "a {}" -> "an {}" for vowel nouns
        if "a {}" in t:
            prompts.append(t.replace("a {}", f"{_pick_article(lbl)} {lbl}"))
        elif "an {}" in t:
            prompts.append(t.replace("an {}", f"{_pick_article(lbl)} {lbl}"))
        else:
            prompts.append(t.format(lbl))
    return prompts


def expand_labels_to_prompt_groups(
    labels: List[str],
    template_set: str = "clip",
    max_templates: int = 3,
) -> List[List[str]]:
    """Expand a list of labels into prompt groups (one group per label).

    Deduplicates labels while preserving order.

    Returns:
        ``List[List[str]]`` — outer list has one entry per unique label;
        inner list contains the template expansions.
    """
    seen: set = set()
    groups: List[List[str]] = []
    for lb in labels:
        norm = normalize_label(lb)
        if norm in seen:
            continue
        seen.add(norm)
        groups.append(expand_label_to_prompts(norm, template_set, max_templates))
    return groups
