"""Histogram viewer component.

Purpose:
    Compute and display side-by-side intensity histograms for the
    original and processed image, so the effect of an applied
    algorithm is visible in the pixel-value distribution, not just the
    image preview.

Description:
    Like `pixel_statistics`, this computes analytics from an
    already-produced NumPy array — it never transforms image data.
    Color images show one overlaid trace per RGB channel; grayscale
    images show a single trace. Built with Plotly per the Blueprint's
    requirement.

Dependencies:
    streamlit; numpy; plotly.graph_objects.

Public functions:
    render_histogram_comparison(original, processed) -> None
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from visionleaf_ai.ui.components.cards import section_card

_CHANNEL_COLORS = {
    0: "rgba(227, 92, 92, 0.65)",  # red
    1: "rgba(63, 184, 143, 0.65)",  # green (matches brand primary)
    2: "rgba(96, 150, 227, 0.65)",  # blue
}
_CHANNEL_NAMES = {0: "Red", 1: "Green", 2: "Blue"}
_GRAYSCALE_COLOR = "rgba(140, 163, 155, 0.75)"


def _build_histogram_figure(image: np.ndarray, title: str) -> go.Figure:
    """Build a Plotly figure of an image's intensity histogram(s)."""
    figure = go.Figure()

    if image.ndim == 2:
        counts, bin_edges = np.histogram(image, bins=64, range=(0, 255))
        figure.add_trace(
            go.Bar(
                x=bin_edges[:-1],
                y=counts,
                marker_color=_GRAYSCALE_COLOR,
                name="Intensity",
            )
        )
    else:
        channel_count = min(image.shape[2], 3)
        for channel_index in range(channel_count):
            counts, bin_edges = np.histogram(
                image[..., channel_index], bins=64, range=(0, 255)
            )
            figure.add_trace(
                go.Bar(
                    x=bin_edges[:-1],
                    y=counts,
                    marker_color=_CHANNEL_COLORS[channel_index],
                    name=_CHANNEL_NAMES[channel_index],
                )
            )

    figure.update_layout(
        title=title,
        barmode="overlay",
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8CA39B", family="IBM Plex Sans, sans-serif"),
        xaxis=dict(title="Pixel Intensity", gridcolor="#2A3438"),
        yaxis=dict(title="Count", gridcolor="#2A3438"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
    )
    return figure


def render_histogram_comparison(original: np.ndarray, processed: np.ndarray) -> None:
    """Render side-by-side Original/Processed histograms.

    Args:
        original: The session's `original_image`.
        processed: The session's `active_image` (current state, after
            whatever processing has been applied so far).
    """
    with section_card(
        "Histogram", "Pixel intensity distribution — updates after every operation."
    ):
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                _build_histogram_figure(original, "Original"),
                key="histogram_original",
            )
        with right:
            st.plotly_chart(
                _build_histogram_figure(processed, "Processed"),
                key="histogram_processed",
            )
