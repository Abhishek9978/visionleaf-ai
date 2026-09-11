"""Image viewer component.

Purpose:
    Consistently display an `ImageSession`'s image(s) — used by
    Acquisition today, and by every later stage that wants to show
    "the image at this point in the pipeline" alongside its own result.

Description:
    Offers both a single-image render (`render_image`) and a two-up
    comparison (`render_image_comparison`) for "before vs after" views
    that Enhancement, Restoration, and Segmentation will need starting
    Milestone 4. Neither function contains any image-processing logic
    — they only display arrays that other modules have already produced.

Dependencies:
    streamlit; numpy.

Public functions:
    render_image(image, caption=None) -> None
    render_image_comparison(left_image, right_image, left_caption, right_caption) -> None
"""

from __future__ import annotations

import numpy as np
import streamlit as st


def render_image(image: np.ndarray, caption: str | None = None) -> None:
    """Render a single image array.

    Args:
        image: An RGB (H, W, 3) or grayscale (H, W) NumPy array.
        caption: Optional caption shown under the image.
    """
    st.image(image, caption=caption, width="stretch")


def render_image_comparison(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_caption: str = "Before",
    right_caption: str = "After",
) -> None:
    """Render two images side by side for comparison.

    Args:
        left_image: Image array for the left column.
        right_image: Image array for the right column.
        left_caption: Caption for the left image.
        right_caption: Caption for the right image.
    """
    left_col, right_col = st.columns(2)
    with left_col:
        render_image(left_image, caption=left_caption)
    with right_col:
        render_image(right_image, caption=right_caption)
