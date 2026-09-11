"""Morphological Operations algorithms.

Purpose:
    Concrete `PipelineStage` implementations that clean up or analyze
    a binary mask's shape: Erosion, Dilation, Opening, Closing,
    Morphological Gradient, Top Hat, Black Hat. Read from
    `session.active_mask` (binary_image or, if already run once,
    segmentation_mask), write to `session.segmentation_mask`.

Description:
    Every module here registers its algorithm via `@register_algorithm`
    at import time — `pipeline.bootstrap.ensure_registered()` imports
    all of them. `_common.py` holds private helpers (mask retrieval,
    structuring-element construction) shared by every operation here,
    not itself registered.

Dependencies:
    visionleaf_ai.processing.morphology.{erosion,dilation,opening,
    closing,gradient,top_hat,black_hat}

Public functions:
    None at the package level — see pipeline.bootstrap.ensure_registered().
"""
