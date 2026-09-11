"""Classification page (PCA + SVM).

Purpose:
    Milestone 7 will implement feature scaling, PCA dimensionality
    reduction, SVM classification, and prediction/probability output
    here. Two pipeline stages share this one page.

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
    """Render the Classification page."""
    page_header(
        "Classification",
        subtitle="Reduce, then classify: PCA followed by an SVM.",
        eyebrow="Milestone 8",
    )
    render_pipeline_rail(active_stage_keys=("pca", "svm"))

    learning_note(
        "Feature vectors are often high-dimensional and correlated. PCA "
        "projects them onto a smaller set of uncorrelated components that "
        "capture most of the variance, which typically makes the SVM both "
        "faster to train and less prone to overfitting."
    )

    render_stage_placeholder(
        "Scaling & PCA",
        status_note="Arrives in Milestone 8",
        will_do=[
            "Standardize feature vectors before dimensionality reduction",
            "Apply PCA and choose how many components to keep",
            "Visualize explained variance per component",
        ],
    )
    render_stage_placeholder(
        "SVM Classification",
        status_note="Arrives in Milestone 8",
        will_do=[
            "Train an SVM classifier on the reduced feature space",
            "Predict the disease class for a new leaf image",
            "Show class probabilities, not just the top prediction",
        ],
    )
