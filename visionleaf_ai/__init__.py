"""VisionLeaf AI — Professional Digital Image Processing & ML Workbench.

Purpose:
    Root package for the VisionLeaf AI application. Exposes the package
    version and initializes nothing at import time (no side effects),
    so importing `visionleaf_ai` is always safe and cheap.

Description:
    VisionLeaf AI implements a full digital image processing pipeline —
    acquisition, enhancement, restoration, segmentation, morphology, ROI
    extraction, feature extraction, PCA, SVM classification, analytics,
    and experiment mode — as a modular Python package with a Streamlit
    front end.

Dependencies:
    None at package init time.

Public API:
    __version__ (str): Semantic version of the package.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
