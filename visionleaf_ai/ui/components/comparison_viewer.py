"""Interactive comparison viewer.

Purpose:
    Let the user compare the original and processed image — side by
    side, or as a single before/after composite with an adjustable
    split point — with zoom and fit-to-window controls.

Description:
    Zoom and fit-to-window are pure *display* parameters: they set the
    pixel width passed to `st.image`, which the browser scales via a
    plain HTML `width` attribute. No resampling library is invoked, so
    this never touches OpenCV (or even Pillow) — exactly the boundary
    the architecture requires ("the UI should never call OpenCV
    functions directly").

    The before/after composite is built with a NumPy boolean-mask
    column split (`composite[:, :split_col] = left; ... = right`) —
    array indexing, not an image-processing algorithm. It requires
    both images to share a shape, which holds for every Milestone 4
    algorithm (none of them resize the image).

Dependencies:
    streamlit; numpy.

Public functions:
    render_comparison_viewer(original, processed) -> None
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import streamlit as st

from visionleaf_ai.ui.components.cards import section_card


def _build_before_after_composite(
    original: np.ndarray, processed: np.ndarray, split_percent: int
) -> np.ndarray:
    """Build a single image: `original` left of the split, `processed` right.

    Args:
        original: Left-hand image.
        processed: Right-hand image. Must share `original`'s shape.
        split_percent: 0-100; where the vertical split line sits.

    Returns:
        A new array, `original`'s shape, composited from both inputs.
    """
    width = original.shape[1]
    split_col = int(width * split_percent / 100)
    composite = processed.copy()
    composite[:, :split_col] = original[:, :split_col]
    # A thin divider line at the split, for visibility.
    line_col = min(split_col, width - 1)
    composite[:, line_col : line_col + 2] = 255 if composite.ndim == 2 else (255, 255, 255)
    return composite


def render_comparison_viewer(original: np.ndarray, processed: np.ndarray) -> None:
    """Render the interactive comparison section.

    Args:
        original: The session's `original_image`.
        processed: The session's `active_image`.
    """
    with section_card("Interactive Comparison"):
        mode_col, fit_col, zoom_col = st.columns([2, 1, 2])
        with mode_col:
            mode = st.radio(
                "Mode",
                ["Side by Side", "Before / After"],
                horizontal=True,
                key="comparison_mode",
                label_visibility="collapsed",
            )
        with fit_col:
            fit_to_window = st.checkbox("Fit to window", value=True, key="comparison_fit")
        with zoom_col:
            zoom_percent = st.slider(
                "Zoom",
                min_value=25,
                max_value=300,
                value=100,
                step=25,
                disabled=fit_to_window,
                key="comparison_zoom",
                label_visibility="collapsed",
            )

        display_width: Literal["stretch"] | int = (
            "stretch" if fit_to_window else int(original.shape[1] * zoom_percent / 100)
        )

        if mode == "Side by Side":
            left, right = st.columns(2)
            with left:
                st.image(original, caption="Original", width=display_width)
            with right:
                st.image(processed, caption="Processed", width=display_width)
        else:
            if original.shape != processed.shape:
                st.info(
                    "Before/After comparison requires matching dimensions; "
                    "showing Side by Side instead."
                )
                left, right = st.columns(2)
                with left:
                    st.image(original, caption="Original", width=display_width)
                with right:
                    st.image(processed, caption="Processed", width=display_width)
            else:
                split_percent = st.slider(
                    "Split position", min_value=0, max_value=100, value=50, key="comparison_split"
                )
                composite = _build_before_after_composite(original, processed, split_percent)
                st.image(
                    composite,
                    caption=f"Original (left) / Processed (right) — split at {split_percent}%",
                    width=display_width,
                )
