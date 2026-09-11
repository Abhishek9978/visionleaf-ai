"""Pixel statistics component.

Purpose:
    Compute and display basic descriptive statistics for an image
    array: resolution, mean, standard deviation, min, max, and dynamic
    range.

Description:
    This is analytics/display computation, not a pipeline algorithm —
    the same category as the Histogram Viewer. It reads an already-
    produced NumPy array (from `ImageSession`) and summarizes it; it
    never transforms the array or writes back to the session, so it
    doesn't belong in `processing/` as a `PipelineStage`.

Dependencies:
    numpy.

Public classes/functions:
    PixelStatistics (TypedDict)
    compute_pixel_statistics(image) -> PixelStatistics
    render_pixel_statistics(image, title="Pixel Statistics") -> None
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from visionleaf_ai.ui.components.cards import section_card
from visionleaf_ai.ui.components.metrics import render_metric_row


class PixelStatistics(TypedDict):
    """Descriptive statistics for one image array.

    A `TypedDict` rather than a plain `dict[str, float | tuple[int, int]]`
    so each key has its own precise type — `resolution` is a 2-tuple of
    ints, every other key is a float — instead of a union type that
    forces every reader to re-narrow it (e.g. unpacking `resolution`
    from a `float | tuple[int, int]`-typed value is a real type error,
    not just an overly-cautious one).
    """

    resolution: tuple[int, int]
    mean: float
    std: float
    min: float
    max: float
    dynamic_range: float


def compute_pixel_statistics(image: np.ndarray) -> PixelStatistics:
    """Compute descriptive statistics for an image array.

    Args:
        image: An (H, W) or (H, W, C) NumPy array.

    Returns:
        A `PixelStatistics` dict: "resolution" (width, height), "mean",
        "std", "min", "max", "dynamic_range" (max - min).
    """
    float_image = image.astype(np.float64)
    height, width = image.shape[0], image.shape[1]
    minimum = float(float_image.min())
    maximum = float(float_image.max())
    return PixelStatistics(
        resolution=(width, height),
        mean=float(float_image.mean()),
        std=float(float_image.std()),
        min=minimum,
        max=maximum,
        dynamic_range=maximum - minimum,
    )


def render_pixel_statistics(image: np.ndarray, title: str = "Pixel Statistics") -> None:
    """Render a card of pixel statistics for an image.

    Args:
        image: An (H, W) or (H, W, C) NumPy array.
        title: Card title, so the same component can label an
            "Original" vs. "Processed" statistics card differently.
    """
    stats = compute_pixel_statistics(image)
    width, height = stats["resolution"]
    with section_card(title):
        render_metric_row(
            [
                ("Resolution", f"{width} × {height}"),
                ("Mean", f"{stats['mean']:.2f}"),
                ("Std Dev", f"{stats['std']:.2f}"),
            ]
        )
        render_metric_row(
            [
                ("Min", f"{stats['min']:.0f}"),
                ("Max", f"{stats['max']:.0f}"),
                ("Dynamic Range", f"{stats['dynamic_range']:.0f}"),
            ]
        )
