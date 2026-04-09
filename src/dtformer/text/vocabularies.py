"""Dataset vocabulary definitions.
数据集词表定义。

Canonical class name lists for each supported dataset.
Maps dataset name -> ordered list of class strings.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# NYU Depth v2 — 40-class vocabulary (official NYU40 order)
# ---------------------------------------------------------------------------
NYU40_CLASSES: List[str] = [
    "wall", "floor", "cabinet", "bed", "chair",
    "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "blinds", "desk", "shelves",
    "curtain", "dresser", "pillow", "mirror", "floor mat",
    "clothes", "ceiling", "books", "refridgerator", "television",
    "paper", "towel", "shower curtain", "box", "whiteboard",
    "person", "night stand", "toilet", "sink", "lamp",
    "bathtub", "bag", "otherstructure", "otherfurniture", "otherprop",
]

# NYU37: 40 classes minus the 3 ambiguous "other*" categories (indices 37-39)
NYU37_CLASSES: List[str] = NYU40_CLASSES[:37]

# ---------------------------------------------------------------------------
# SUN RGB-D — 37-class vocabulary
# ---------------------------------------------------------------------------
SUNRGBD_CLASSES: List[str] = [
    "wall", "floor", "cabinet", "bed", "chair",
    "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "blinds", "desk", "shelves",
    "curtain", "dresser", "pillow", "mirror", "floor_mat",
    "clothes", "ceiling", "books", "fridge", "tv",
    "paper", "towel", "shower_curtain", "box", "whiteboard",
    "person", "night_stand", "toilet", "sink", "lamp",
    "bathtub", "bag",
]

# ---------------------------------------------------------------------------
# Label alias mapping: normalizes common spelling variants to canonical form.
# Used when matching VLM-generated labels back to the dataset vocabulary.
# ---------------------------------------------------------------------------
LABEL_ALIASES: Dict[str, str] = {
    # Spelling normalisation only — never change semantics.
    "refridgerator": "refrigerator",
    "nightstand": "night stand",
    "night_stand": "night stand",
    "floor_mat": "floor mat",
    "floormat": "floor mat",
    "shower_curtain": "shower curtain",
    "white board": "whiteboard",
}

# ---------------------------------------------------------------------------
# VLM normalisation maps: common VLM output variants -> dataset canonical name.
# Used by generate_tags_*.py to match VLM-generated text back to the
# official vocabulary.  Keys should be lowercase.
# ---------------------------------------------------------------------------
VLM_NORM_MAPS: Dict[str, Dict[str, str]] = {
    "nyu": {
        "nightstand": "night stand",
        "night_stand": "night stand",
        "floormat": "floor mat",
        "floor_mat": "floor mat",
        "tv": "television",
        "tv monitor": "television",
        "book shelf": "bookshelf",
        "bookshelves": "bookshelf",
        "white board": "whiteboard",
        "refridgerator": "refrigerator",
        "shower_curtain": "shower curtain",
        "closet": "cabinet",
        "wardrobe": "cabinet",
        "cloth": "clothes",
        "couch": "sofa",
        "book": "books",
    },
    "sun": {
        "nightstand": "night_stand",
        "night stand": "night_stand",
        "floormat": "floor_mat",
        "floor mat": "floor_mat",
        "television": "tv",
        "tv monitor": "tv",
        "refrigerator": "fridge",
        "fridgerator": "fridge",
        "book shelf": "bookshelf",
        "bookshelves": "bookshelf",
        "white board": "whiteboard",
        "shower curtain": "shower_curtain",
        "closet": "cabinet",
        "wardrobe": "cabinet",
        "cloth": "clothes",
        "couch": "sofa",
        "book": "books",
    },
}


def get_vlm_norm_map(dataset_key: str) -> Dict[str, str]:
    """Return the VLM normalisation map for *dataset_key* (``'nyu'`` or ``'sun'``)."""
    if dataset_key not in VLM_NORM_MAPS:
        raise KeyError(
            f"Unknown VLM norm map key '{dataset_key}'. "
            f"Available: {list(VLM_NORM_MAPS.keys())}"
        )
    return VLM_NORM_MAPS[dataset_key]


# ---------------------------------------------------------------------------
# Registry: dataset name -> class list
# ---------------------------------------------------------------------------
VOCABULARY_REGISTRY: Dict[str, List[str]] = {
    "NYUDepthv2": NYU40_CLASSES,
    "nyu40": NYU40_CLASSES,
    "nyu37": NYU37_CLASSES,
    "SUNRGBD": SUNRGBD_CLASSES,
    "sunrgbd": SUNRGBD_CLASSES,
}


def get_vocabulary(dataset_name: str) -> List[str]:
    """Return the canonical class name list for *dataset_name*.

    Raises ``KeyError`` if the dataset is not registered.
    """
    if dataset_name not in VOCABULARY_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(VOCABULARY_REGISTRY.keys())}"
        )
    return VOCABULARY_REGISTRY[dataset_name]
