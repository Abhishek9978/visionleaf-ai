"""Image Enhancement algorithms.

Purpose:
    Concrete `PipelineStage` implementations that improve an image's
    perceptual quality (brightness, contrast, dynamic range) without
    modeling how the image was degraded — see `processing.restoration`
    for algorithms that do model and invert a specific degradation.

Description:
    Every module here registers its algorithm via `@register_algorithm`
    at import time — `pipeline.bootstrap.ensure_registered()` imports
    all of them, and `PipelineEngine`/`AlgorithmRegistry` are the
    intended way to reach them. This `__init__` intentionally does NOT
    re-export the classes directly, to keep "go through the registry"
    the natural path — import the classes directly only from tests.

Dependencies:
    visionleaf_ai.processing.enhancement.{brightness,contrast,
    gamma_correction,histogram_equalization,clahe}

Public functions:
    None at the package level — see pipeline.bootstrap.ensure_registered().
"""
