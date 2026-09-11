"""Bootstrap — guarantees every algorithm module has been imported.

Purpose:
    Python only runs a module's top-level code (including
    `@register_algorithm` decorators) the first time it's imported.
    If nothing ever imports `processing.enhancement.gamma_correction`
    directly, its class never registers, and `AlgorithmRegistry.get(...)`
    would raise "unknown algorithm" even though the file exists.
    `ensure_registered()` is the one place that imports every algorithm
    module for its registration side effect, so callers never need to
    remember to import a specific algorithm module themselves.

Description:
    Idempotent: Python caches imports, so calling this multiple times
    (e.g. once per `PipelineEngine()` construction) costs nothing after
    the first call. `PipelineEngine.__init__` calls this automatically,
    so code that only ever goes through the engine never needs to call
    it directly.

Dependencies:
    Every concrete algorithm module under
    visionleaf_ai.processing.enhancement and
    visionleaf_ai.processing.restoration.

Public functions:
    ensure_registered() -> None
"""

from __future__ import annotations

_bootstrapped = False


def ensure_registered() -> None:
    """Import every algorithm module so its `@register_algorithm` runs.

    Safe to call repeatedly; only imports on the first call.
    """
    global _bootstrapped
    if _bootstrapped:
        return

    # Imported for side effect (class-decorator registration) only —
    # the imported names themselves are intentionally unused here.
    from visionleaf_ai.processing.enhancement import (  # noqa: F401
        brightness,
        clahe,
        contrast,
        gamma_correction,
        histogram_equalization,
    )
    from visionleaf_ai.processing.features import (  # noqa: F401
        color_features,
        edge_features,
        feature_vector_builder,
        glcm_features,
        lbp_features,
        shape_features,
    )
    from visionleaf_ai.processing.morphology import (  # noqa: F401
        black_hat,
        closing,
        dilation,
        erosion,
        gradient,
        opening,
        top_hat,
    )
    from visionleaf_ai.processing.restoration import (  # noqa: F401
        bilateral_filter,
        gaussian_blur,
        median_filter,
    )
    from visionleaf_ai.processing.roi import (  # noqa: F401
        bounding_box_visualization,
        bounding_rectangle,
        convex_hull,
        filter_contours,
        find_contours,
        min_area_rectangle,
        roi_crop,
        roi_mask,
    )
    from visionleaf_ai.processing.segmentation import (  # noqa: F401
        adaptive_gaussian_threshold,
        adaptive_mean_threshold,
        binary_inverse_threshold,
        binary_threshold,
        otsu_threshold,
        to_zero_threshold,
        truncate_threshold,
    )

    _bootstrapped = True
