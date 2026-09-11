"""Image Processing Lab page — the Interactive Image Processing Laboratory.

Purpose:
    The centerpiece of the application: lets the user apply Enhancement
    (Brightness, Contrast, Gamma, Histogram Equalization, CLAHE) and
    Restoration (Gaussian Blur, Median Filter, Bilateral Filter)
    algorithms to the loaded image, see the effect immediately via an
    interactive comparison and live-updating histogram, and inspect
    the processing history, pixel statistics, and (when Learning Mode
    is on) each algorithm's underlying theory.

Description:
    Architecture is strictly UI -> PipelineEngine -> PipelineStage ->
    ImageSession: this page never calls OpenCV, never imports an
    algorithm class, and never computes a pixel transformation itself.
    Every "Apply" button does exactly one thing —
    `engine.run_by_name(selected_algorithm, session, **params)` — and
    every other section on the page (comparison, histogram, stats,
    history) only *reads* `ImageSession` or computes display-only
    analytics (histogram bins, mean/std/min/max) from its arrays.

    Design note: the Blueprint lists "Original Image," "Processed
    Image," and "Interactive Comparison" as three separate sections.
    Rather than rendering the original and processed images twice each
    (once plainly, once inside the comparison viewer), this
    implementation treats the Interactive Comparison viewer as the
    single place both images are shown — its Side-by-Side mode
    literally *is* "Original Image" + "Processed Image" displayed
    together, with Before/After, zoom, and fit-to-window as additional
    ways to view the same two images. This avoids showing every image
    redundantly while still satisfying all three requirements.

Dependencies:
    streamlit; visionleaf_ai.core; visionleaf_ai.processing.acquisition;
    visionleaf_ai.processing.pipeline; visionleaf_ai.ui.components;
    visionleaf_ai.ui.session.

Public functions:
    render() -> None
"""

from __future__ import annotations

import streamlit as st

from visionleaf_ai.core import get_logger
from visionleaf_ai.processing.acquisition import get_image_manager
from visionleaf_ai.processing.pipeline import AlgorithmRegistry, get_pipeline_engine
from visionleaf_ai.ui.components import (
    ALGORITHM_PARAM_SPECS,
    learning_note,
    page_header,
    render_comparison_viewer,
    render_empty_state,
    render_export_panel,
    render_histogram_comparison,
    render_history_panel,
    render_learning_panel,
    render_metadata_card,
    render_parameter_panel,
    render_pipeline_rail,
    render_pixel_statistics,
    section_card,
)
from visionleaf_ai.ui.session import get_image_session, set_stage_status

logger = get_logger(__name__)


def render() -> None:
    """Render the Image Processing Lab page."""
    page_header(
        "Image Processing Lab",
        subtitle="Apply enhancement and restoration algorithms and see the effect immediately.",
        eyebrow="Milestone 5",
    )
    render_pipeline_rail(active_stage_keys=("enhancement", "restoration"))

    learning_note(
        "Enhancement improves perceptual quality (contrast, sharpness) "
        "without modeling how the image was degraded. Restoration goes "
        "further: it assumes a specific degradation model (e.g. blur or "
        "noise) and tries to invert it. Every algorithm here runs through "
        "the same Pipeline Engine, chaining onto whatever the previous "
        "stage produced."
    )

    session = get_image_session()
    if not session.is_loaded():
        render_empty_state(
            "No Image Loaded",
            "Load an image on the Image Acquisition page before using the "
            "Processing Lab.",
            icon="🧪",
        )
        return

    engine = get_pipeline_engine()
    original = session.require_original_image()
    processed = session.require_active_image()

    render_comparison_viewer(original, processed)

    left_col, right_col = st.columns(2)
    with left_col:
        _render_algorithm_controls(
            engine, session, category="enhancement", title="Enhancement Controls"
        )
    with right_col:
        _render_algorithm_controls(
            engine, session, category="restoration", title="Restoration Controls"
        )

    render_histogram_comparison(original, processed)

    stats_left, stats_right = st.columns(2)
    with stats_left:
        render_pixel_statistics(original, title="Original — Pixel Statistics")
    with stats_right:
        render_pixel_statistics(processed, title="Processed — Pixel Statistics")

    info_col, history_col = st.columns(2)
    with info_col:
        current_stage = (
            session.processing_history[-1].label if session.processing_history else None
        )
        render_metadata_card(session.require_metadata(), current_stage=current_stage)
    with history_col:
        render_history_panel(session.processing_history)

    action_col, export_col = st.columns(2)
    with action_col:
        with section_card("Reset", "Undo every edit and start over from the original image."):
            if st.button("↺ Reset to Original", width="stretch"):
                session.reset_processing()
                set_stage_status("enhancement", "pending")
                set_stage_status("restoration", "pending")
                st.rerun()
    with export_col:
        render_export_panel(get_image_manager(), session)


def _render_algorithm_controls(engine, session, category: str, title: str) -> None:
    """Render the algorithm picker, parameter panel, and Apply button for one category.

    Args:
        engine: The `PipelineEngine` to run the selected algorithm through.
        session: The current `ImageSession`.
        category: "enhancement" or "restoration" — matches each
            algorithm's `stage_key` in the registry.
        title: Card title, e.g. "Enhancement Controls".
    """
    algorithm_names = AlgorithmRegistry.list_algorithms(category)
    with section_card(title):
        selected = st.selectbox(
            "Algorithm",
            options=algorithm_names,
            format_func=lambda name: AlgorithmRegistry.get_info(name).name,
            key=f"algorithm_select__{category}",
        )
        params = render_parameter_panel(selected)

        if st.session_state.get("learning_mode"):
            render_learning_panel(
                AlgorithmRegistry.get_info(selected),
                ALGORITHM_PARAM_SPECS.get(selected, []),
            )

        if st.button(f"▶ Apply {AlgorithmRegistry.get_info(selected).name}", key=f"apply__{category}"):
            _, result = engine.run_by_name(selected, session, **params)
            if result.success:
                set_stage_status(category, "complete")
                logger.info("Applied %s via Processing Lab UI", result.algorithm_name)
                st.rerun()
            else:
                st.error(f"⚠️ {result.error_message}")
