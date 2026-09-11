"""Shared internal helpers for feature extraction stages.

Purpose:
    Every color/texture/edge extractor needs the same input-resolution
    logic ("operate on the segmented/ROI image, not the raw photo, and
    ignore masked-out background pixels where possible") and the same
    validation. Shape features need the largest contour (reusing
    `processing.roi`'s existing helper rather than duplicating it).
    `FeatureVectorBuilder` needs deterministic, fixed-order flattening
    for each feature group. Centralizing all of this here avoids
    duplicating it across five extractor modules.

Description:
    Not `PipelineStage`s themselves, not registered with
    `AlgorithmRegistry` — private helpers used only by sibling modules
    in `processing.features`.

    `require_feature_input_image()` deliberately requires that
    segmentation/ROI has already run — it does NOT fall back to
    `active_image` — because feature extraction is meant to describe
    "the leaf region," not an arbitrary uncropped photo that may be
    mostly background. This is why "missing segmentation" and "missing
    ROI" are real, expected validation failures for every extractor in
    this package, not internal invariants.

Dependencies:
    numpy, opencv-python (cv2); visionleaf_ai.core;
    visionleaf_ai.processing.roi._common (reused, not duplicated).

Public functions:
    require_feature_input_image(session) -> tuple[np.ndarray, np.ndarray | None]
    require_grayscale_input(session) -> np.ndarray
    require_nonzero_area_contour(session) -> np.ndarray
    flatten_color_features(color_features) -> list[float]
    flatten_texture_features(texture_features) -> list[float]
    flatten_shape_features(shape_features) -> list[float]
    flatten_edge_features(edge_features) -> list[float]
"""

from __future__ import annotations

import cv2
import numpy as np

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.processing.roi._common import require_largest_contour


def require_feature_input_image(session: ImageSession) -> tuple[np.ndarray, np.ndarray | None]:
    """Return the image feature extractors should operate on, plus an optional foreground mask.

    Prefers `session.segmented_image` (background already zeroed by ROI
    Masking) over `session.roi_image` (a plain rectangular crop that
    may still include some background in its corners), since the
    former lets statistics be computed over foreground pixels only.

    Args:
        session: The `ImageSession` to read from.

    Returns:
        `(image, foreground_mask)`. `foreground_mask` is a boolean
        array (`True` = foreground) the same height/width as `image`
        if `segmented_image` was used, otherwise `None` (meaning:
        compute over every pixel — the best available when only a
        plain crop exists).

    Raises:
        ValidationError: If neither `segmented_image` nor `roi_image`
            exists yet ("run ROI Masking or ROI Cropping first"), or
            if the resolved image is degenerate (zero-sized).
    """
    image = session.segmented_image if session.segmented_image is not None else session.roi_image
    if image is None:
        raise ValidationError(
            "No segmented or ROI image available",
            details="run ROI Masking or ROI Cropping before extracting features",
        )
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValidationError("Segmented/ROI image is empty", details=f"shape={image.shape}")

    foreground_mask = None
    if session.segmented_image is not None and image is session.segmented_image:
        foreground_mask = np.asarray(image.any(axis=-1) if image.ndim == 3 else image > 0)
        if not foreground_mask.any():
            raise ValidationError(
                "Segmented image has no foreground pixels",
                details="the mask applied during ROI Masking left nothing behind",
            )
    return image, foreground_mask


def require_grayscale_input(session: ImageSession) -> np.ndarray:
    """Return a grayscale version of the feature-extraction input image.

    Args:
        session: The `ImageSession` to read from.

    Returns:
        A single-channel `np.ndarray`.

    Raises:
        ValidationError: Same conditions as `require_feature_input_image`.
    """
    image, _ = require_feature_input_image(session)
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def require_nonzero_area_contour(session: ImageSession) -> np.ndarray:
    """Return the largest contour, rejecting degenerate (zero-area) ones.

    Reuses `processing.roi._common.require_largest_contour` rather
    than duplicating its "no contour found" check, then adds the
    zero-area check Shape Features specifically needs (a degenerate
    contour would make aspect ratio, solidity, and circularity all
    divide by zero).

    Args:
        session: The `ImageSession` to read from.

    Returns:
        The largest contour, guaranteed to enclose a positive area.

    Raises:
        ValidationError: If no contour exists, or if it encloses zero area.
    """
    contour = require_largest_contour(session)
    if cv2.contourArea(contour) <= 0:
        raise ValidationError(
            "Largest contour has zero area",
            details="cannot compute shape descriptors from a degenerate contour",
        )
    return contour


def flatten_color_features(color_features: dict) -> list[float]:
    """Flatten a color-features dict into a fixed-order list of floats.

    Order: rgb_mean (3), rgb_std (3), hsv_mean (3), hsv_std (3),
    histogram_normalized (however many bins were configured, R then G
    then B channel concatenated).

    Args:
        color_features: `session.color_features`.

    Returns:
        A flat list of floats.
    """
    values: list[float] = []
    values.extend(float(v) for v in color_features["rgb_mean"])
    values.extend(float(v) for v in color_features["rgb_std"])
    values.extend(float(v) for v in color_features["hsv_mean"])
    values.extend(float(v) for v in color_features["hsv_std"])
    for channel_hist in color_features["histogram_normalized"]:
        values.extend(float(v) for v in channel_hist)
    return values


def flatten_texture_features(texture_features: dict) -> list[float]:
    """Flatten a texture-features dict into a fixed-order list of floats.

    Order: GLCM properties (contrast, dissimilarity, homogeneity,
    energy, correlation, ASM — only the sub-keys present are included,
    in that fixed order), then the LBP normalized histogram (only if
    present).

    Args:
        texture_features: `session.texture_features`.

    Returns:
        A flat list of floats.
    """
    values: list[float] = []
    glcm = texture_features.get("glcm")
    if glcm:
        for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "asm"):
            if prop in glcm:
                values.append(float(glcm[prop]))
    lbp = texture_features.get("lbp")
    if lbp:
        values.extend(float(v) for v in lbp["histogram_normalized"])
    return values


def flatten_shape_features(shape_features: dict) -> list[float]:
    """Flatten a shape-features dict into a fixed-order list of floats.

    Order: area, perimeter, aspect_ratio, extent, solidity,
    circularity, equivalent_diameter, then the 7 (log-scaled) Hu moments.

    Args:
        shape_features: `session.shape_features`.

    Returns:
        A flat list of floats.
    """
    values = [
        float(shape_features["area"]),
        float(shape_features["perimeter"]),
        float(shape_features["aspect_ratio"]),
        float(shape_features["extent"]),
        float(shape_features["solidity"]),
        float(shape_features["circularity"]),
        float(shape_features["equivalent_diameter"]),
    ]
    values.extend(float(v) for v in shape_features["hu_moments"])
    return values


def flatten_edge_features(edge_features: dict) -> list[float]:
    """Flatten an edge-features dict into a fixed-order list of floats.

    Order: canny_edge_density, sobel_gradient_magnitude_mean,
    laplacian_response_variance.

    Args:
        edge_features: `session.edge_features`.

    Returns:
        A flat list of floats.
    """
    return [
        float(edge_features["canny_edge_density"]),
        float(edge_features["sobel_gradient_magnitude_mean"]),
        float(edge_features["laplacian_response_variance"]),
    ]
