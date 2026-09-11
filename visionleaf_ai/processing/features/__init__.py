"""Feature Extraction algorithms.

Purpose:
    Concrete `PipelineStage` implementations that transform a
    segmented/cropped leaf region into numerical features: Color,
    GLCM Texture, LBP Texture, Shape, and Edge extractors, plus the
    `FeatureVectorBuilder` that combines whichever of them have run
    into one flat vector in `session.feature_vector`.

Description:
    Every module here registers its algorithm via `@register_algorithm`
    at import time — `pipeline.bootstrap.ensure_registered()` imports
    all of them. `_common.py` holds private helpers (input-image
    resolution, contour reuse from `processing.roi`, deterministic
    flattening for the vector builder) shared across this package.

    This is the first `processing` subpackage whose stages write
    non-image data (dicts, then a flat vector) to `ImageSession` —
    everything before Milestone 7 produced images or image-derived
    structures (masks, contours). Milestone 8 (PCA/SVM) reads
    `session.feature_vector` exclusively and never touches a pixel.

Dependencies:
    visionleaf_ai.processing.features.{color_features,glcm_features,
    lbp_features,shape_features,edge_features,feature_vector_builder}

Public functions:
    None at the package level — see pipeline.bootstrap.ensure_registered().
"""
