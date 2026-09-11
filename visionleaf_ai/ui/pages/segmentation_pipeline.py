"""Segmentation Pipeline page (Segmentation + Morphology + ROI).

Purpose:
    Placeholder for the Segmentation Lab UI. Unlike the other
    placeholder pages, the backend for this page is NOT future work:
    all 22 Thresholding/Morphology/Contour/ROI algorithms were built
    and fully tested in Milestone 6 (see docs/MILESTONES.md) and are
    already runnable through `PipelineEngine.run_by_name(...)` — only
    the interactive controls on this page (sliders, live preview,
    parameter panel) remain to be built, the same way Milestone 5
    built the UI for Milestone 4's Enhancement/Restoration engine.

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

_UI_PENDING_NOTE = "Backend complete since Milestone 6 — UI integration pending"


def render() -> None:
    """Render the Segmentation Pipeline page."""
    page_header(
        "Segmentation Pipeline",
        subtitle="Isolate the diseased region from the leaf background.",
        eyebrow="Milestone 6 (backend)",
    )
    render_pipeline_rail(active_stage_keys=("segmentation", "morphology", "roi"))

    learning_note(
        "Segmentation splits the image into foreground (the region of "
        "interest) and background using thresholding. Morphological "
        "operations (erosion, dilation, opening, closing) clean up noise "
        "in that mask. ROI extraction then crops to just the region that "
        "matters, so later feature extraction ignores irrelevant background."
    )

    render_stage_placeholder(
        "Thresholding & Segmentation",
        status_note=_UI_PENDING_NOTE,
        will_do=[
            "Apply global (Otsu) and adaptive thresholding",
            "Try color-space-based segmentation (HSV, Lab)",
            "Preview the resulting binary mask",
        ],
    )
    render_stage_placeholder(
        "Morphological Operations",
        status_note=_UI_PENDING_NOTE,
        will_do=[
            "Apply erosion, dilation, opening, and closing to the mask",
            "Adjust structuring element shape and size interactively",
        ],
    )
    render_stage_placeholder(
        "ROI Extraction",
        status_note=_UI_PENDING_NOTE,
        will_do=[
            "Find contours and bounding regions from the cleaned mask",
            "Crop the image to the extracted region of interest",
            "Visualize the full segmentation pipeline end to end",
        ],
    )
