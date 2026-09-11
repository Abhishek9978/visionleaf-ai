"""Metric row component.

Purpose:
    Render a row of labeled numeric readouts consistently across pages
    (e.g. later: image dimensions, PSNR/SSIM values, class probabilities).

Description:
    Thin wrapper over `st.columns` + native `st.metric`, so values get
    Streamlit's built-in accessibility/semantics for free while still
    picking up the app's monospace styling for metric values (defined
    in `theme.py`, targeting `[data-testid="stMetricValue"]`).

Dependencies:
    streamlit.

Public functions:
    render_metric_row(metrics) -> None
"""

from __future__ import annotations

import streamlit as st


def render_metric_row(metrics: list[tuple[str, str]] | list[tuple[str, str, str]]) -> None:
    """Render a row of metrics side by side.

    Args:
        metrics: A list of `(label, value)` or `(label, value, delta)`
            tuples, e.g. `[("Width", "512 px"), ("Height", "384 px")]`.
            Rendered in equal-width columns, left to right.
    """
    if not metrics:
        return
    columns = st.columns(len(metrics))
    for column, metric in zip(columns, metrics):
        with column:
            if len(metric) == 3:
                label, value, delta = metric
                st.metric(label, value, delta)
            else:
                label, value = metric
                st.metric(label, value)
