"""Analytics page.

Purpose:
    Milestone 8 will implement dataset analysis, confusion matrix,
    accuracy/precision/recall/F1, ROC curves, and PCA visualization here.

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
    """Render the Analytics page."""
    page_header(
        "Analytics",
        subtitle="Evaluate how well the classifier actually performs.",
        eyebrow="Milestone 9",
    )
    render_pipeline_rail(active_stage_keys=("analytics",))

    learning_note(
        "Accuracy alone can be misleading on imbalanced disease datasets. "
        "Precision, recall, and F1 per class, plus a confusion matrix, "
        "show exactly which diseases the model confuses with each other."
    )

    render_stage_placeholder(
        "Analytics",
        status_note="Arrives in Milestone 9",
        will_do=[
            "Summarize the dataset: class balance, image counts, sample previews",
            "Show a confusion matrix for the trained classifier",
            "Report accuracy, precision, recall, and F1 per class",
            "Plot ROC curves for each class",
            "Visualize the dataset in PCA-reduced 2D/3D space",
        ],
    )
