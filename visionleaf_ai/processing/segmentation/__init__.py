"""Thresholding (Segmentation) algorithms.

Purpose:
    Concrete `PipelineStage` implementations that split an image into
    foreground/background regions: Binary, Binary Inverse, Truncate,
    To Zero, Otsu, Adaptive Mean, and Adaptive Gaussian thresholding.
    Output goes to `session.binary_image`, read next by
    `processing.morphology` (cleanup) or `processing.roi` (contour
    finding).

Description:
    Every module here registers its algorithm via `@register_algorithm`
    at import time — `pipeline.bootstrap.ensure_registered()` imports
    all of them. `_common.py` is a private helper (grayscale
    conversion) shared by every thresholding algorithm, not itself
    registered.

Dependencies:
    visionleaf_ai.processing.segmentation.{binary_threshold,
    binary_inverse_threshold,truncate_threshold,to_zero_threshold,
    otsu_threshold,adaptive_mean_threshold,adaptive_gaussian_threshold}

Public functions:
    None at the package level — see pipeline.bootstrap.ensure_registered().
"""
