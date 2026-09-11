"""Experiment Mode page.

Purpose:
    Milestone 9 will implement degradation/restoration experiments with
    PSNR, SSIM, and MSE performance comparison here.

Dependencies:
    streamlit; visionleaf_ai.ui.components.

Public functions:
    render() -> None
"""

from __future__ import annotations

from visionleaf_ai.ui.components import (
    learning_note,
    page_header,
    render_pipeline_rail,
    render_stage_placeholder,
)


def render() -> None:
    """Render the Experiment Mode page."""
    page_header(
        "Experiment Mode",
        subtitle="Quantitatively compare restoration techniques.",
        eyebrow="Milestone 10",
    )
    render_pipeline_rail(active_stage_keys=("experiment",))

    learning_note(
        "PSNR, SSIM, and MSE are the standard ways to measure how close a "
        "restored image is to the original — PSNR and MSE focus on pixel-"
        "level error, while SSIM tries to approximate perceived structural "
        "similarity, which correlates better with what a human would notice."
    )

    render_stage_placeholder(
        "Experiment Mode",
        status_note="Arrives in Milestone 10",
        will_do=[
            "Apply a chosen degradation to a clean image",
            "Apply one or more restoration techniques to recover it",
            "Compute PSNR, SSIM, and MSE against the original",
            "Compare multiple restoration techniques side by side",
        ],
    )
