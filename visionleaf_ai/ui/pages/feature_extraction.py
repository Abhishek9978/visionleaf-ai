"""Feature Extraction page.

Purpose:
    Milestone 6 will implement color histogram, GLCM, LBP, and shape
    feature extraction, plus feature vector generation, here.

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
    """Render the Feature Extraction page."""
    page_header(
        "Feature Extraction",
        subtitle="Turn the extracted region into a numeric feature vector.",
        eyebrow="Milestone 7",
    )
    render_pipeline_rail(active_stage_keys=("features",))

    learning_note(
        "Classifiers don't see pixels — they see numbers. Feature "
        "extraction summarizes color (histograms), texture (GLCM, LBP), "
        "and shape into a fixed-length vector, which is what PCA and the "
        "SVM will actually operate on in later stages."
    )

    render_stage_placeholder(
        "Feature Extraction",
        status_note="Arrives in Milestone 7",
        will_do=[
            "Compute color histograms per channel and color space",
            "Compute GLCM texture features (contrast, homogeneity, energy)",
            "Compute Local Binary Pattern (LBP) texture descriptors",
            "Compute shape features (area, perimeter, eccentricity)",
            "Assemble everything into a single feature vector, ready for ML",
        ],
    )
