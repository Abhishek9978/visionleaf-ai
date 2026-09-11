"""Pipeline stage and page metadata — single source of truth.

Purpose:
    Define, once, the 11 DIP pipeline stages (from the approved
    architecture) and how they group into the application's 8
    navigable pages. Every other UI module (sidebar, pipeline rail,
    dashboard) reads from here rather than hardcoding stage names, so
    they can never drift out of sync with each other.

Description:
    `STAGES` is the ordered, complete pipeline exactly as specified:
    Acquisition -> Enhancement -> Restoration -> Segmentation ->
    Morphological Operations -> ROI Extraction -> Feature Extraction ->
    PCA -> SVM Classification -> Analytics -> Experiment Mode.

    `PAGES` groups those 11 stages into the 8 pages the milestone list
    defines (e.g. Enhancement + Restoration both live on the "Image
    Processing Lab" page, matching Milestone 4's scope). Each page
    entry records which stage keys it covers, so the pipeline rail
    component knows which nodes to highlight when rendering a given
    page.

Dependencies:
    None (stdlib only) — deliberately framework-agnostic so it can be
    imported by components, pages, and tests alike without side effects.

Public functions:
    get_page(key: str) -> dict
        Look up a single page's metadata by key.
    stage_label(key: str) -> str
        Look up a single stage's display label by key.
"""

from __future__ import annotations

STAGES: tuple[dict[str, str], ...] = (
    {"key": "acquisition", "label": "Acquisition"},
    {"key": "enhancement", "label": "Enhancement"},
    {"key": "restoration", "label": "Restoration"},
    {"key": "segmentation", "label": "Segmentation"},
    {"key": "morphology", "label": "Morphology"},
    {"key": "roi", "label": "ROI Extraction"},
    {"key": "features", "label": "Feature Extraction"},
    {"key": "pca", "label": "PCA"},
    {"key": "svm", "label": "SVM Classification"},
    {"key": "analytics", "label": "Analytics"},
    {"key": "experiment", "label": "Experiment Mode"},
)

PAGES: tuple[dict[str, object], ...] = (
    {
        "key": "dashboard",
        "label": "Dashboard",
        "icon": "🏠",
        "stage_keys": (),
        "milestone": 2,
        "default": True,
    },
    {
        "key": "acquisition",
        "label": "Image Acquisition",
        "icon": "📷",
        "stage_keys": ("acquisition",),
        "milestone": 3,
        "default": False,
    },
    {
        "key": "processing_lab",
        "label": "Image Processing Lab",
        "icon": "🧪",
        "stage_keys": ("enhancement", "restoration"),
        "milestone": 5,
        "default": False,
    },
    {
        "key": "segmentation_pipeline",
        "label": "Segmentation Pipeline",
        "icon": "🧩",
        "stage_keys": ("segmentation", "morphology", "roi"),
        "milestone": 6,  # backend complete since M6; this page's UI itself is still pending
        "default": False,
    },
    {
        "key": "feature_extraction",
        "label": "Feature Extraction",
        "icon": "📐",
        "stage_keys": ("features",),
        "milestone": 7,
        "default": False,
    },
    {
        "key": "classification",
        "label": "Classification",
        "icon": "🧠",
        "stage_keys": ("pca", "svm"),
        "milestone": 8,
        "default": False,
    },
    {
        "key": "analytics",
        "label": "Analytics",
        "icon": "📊",
        "stage_keys": ("analytics",),
        "milestone": 9,
        "default": False,
    },
    {
        "key": "experiment_mode",
        "label": "Experiment Mode",
        "icon": "🔬",
        "stage_keys": ("experiment",),
        "milestone": 10,
        "default": False,
    },
)

_PAGES_BY_KEY = {page["key"]: page for page in PAGES}
_STAGES_BY_KEY = {stage["key"]: stage["label"] for stage in STAGES}


def get_page(key: str) -> dict[str, object]:
    """Look up a page's metadata by its key.

    Args:
        key: One of the `key` values in `PAGES` (e.g. "processing_lab").

    Returns:
        The page's metadata dict.

    Raises:
        KeyError: If `key` does not match any page.
    """
    return _PAGES_BY_KEY[key]


def stage_label(key: str) -> str:
    """Look up a stage's display label by its key.

    Args:
        key: One of the `key` values in `STAGES` (e.g. "morphology").

    Returns:
        The stage's human-readable label (e.g. "Morphology").

    Raises:
        KeyError: If `key` does not match any stage.
    """
    return _STAGES_BY_KEY[key]
