"""Image Restoration algorithms.

Purpose:
    Concrete `PipelineStage` implementations that smooth/denoise an
    image — see `processing.enhancement` for algorithms that instead
    improve perceptual quality without a noise/degradation model.

Description:
    Every module here registers its algorithm via `@register_algorithm`
    at import time — `pipeline.bootstrap.ensure_registered()` imports
    all of them. As with `processing.enhancement`, this `__init__`
    intentionally does not re-export the classes directly.

Dependencies:
    visionleaf_ai.processing.restoration.{gaussian_blur,median_filter,
    bilateral_filter}

Public functions:
    None at the package level — see pipeline.bootstrap.ensure_registered().
"""
