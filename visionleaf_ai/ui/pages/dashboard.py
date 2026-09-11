"""Dashboard page — application home and pipeline overview.

Purpose:
    First thing a user sees: what VisionLeaf AI is, the full pipeline
    at a glance, and a status card per stage-group page.

Description:
    Uses the pipeline rail with no stage active (this page sits above
    the pipeline, not inside it), then one `section_card` per non-
    dashboard page in `pipeline_stages.PAGES`, each showing its
    milestone number, status badge, and covered stages. Navigation
    into a stage happens via the sidebar (built once in `app.py` from
    real `st.Page` objects); this page is read-only overview content.

Dependencies:
    streamlit; visionleaf_ai.ui.components; visionleaf_ai.ui.pipeline_stages;
    visionleaf_ai.ui.session.

Public functions:
    render() -> None
"""

from __future__ import annotations

import streamlit as st

from visionleaf_ai.ui.components import (
    learning_note,
    page_header,
    render_metric_row,
    render_pipeline_rail,
    section_card,
    status_badge,
)
from visionleaf_ai.ui.pipeline_stages import PAGES, STAGES, stage_label
from visionleaf_ai.ui.session import get_stage_status


def render() -> None:
    """Render the Dashboard page."""
    page_header(
        "Dashboard",
        subtitle="A complete digital image processing pipeline, from raw leaf photo to disease classification.",
        eyebrow="VisionLeaf AI",
    )
    render_pipeline_rail(active_stage_keys=())

    learning_note(
        "This dashboard is the entry point to an 11-stage image processing "
        "pipeline. Each card below groups related stages — for example, "
        "Enhancement and Restoration are both part of the 'Image Processing "
        "Lab' page. Turn Learning Mode off in the sidebar once you don't "
        "need these explanations anymore."
    )

    complete_count = sum(
        1 for stage in STAGES if get_stage_status(stage["key"]) == "complete"
    )
    render_metric_row(
        [
            ("Pipeline Stages", str(len(STAGES))),
            ("Stages Complete", f"{complete_count}/{len(STAGES)}"),
            # NOTE: hardcoded — update this string each time a milestone
            # completes (see docs/MILESTONES.md for the current one).
            # This was found stale during the Milestone 6 stabilization
            # review (still read "2 · Application Shell" after 4 more
            # milestones shipped); nothing else on this page derives it
            # automatically, so it will drift again unless updated by hand.
            ("Current Milestone", "6 · Segmentation & ROI Engine"),
        ]
    )

    st.markdown("### Pipeline Stages")
    non_dashboard_pages = [p for p in PAGES if p["key"] != "dashboard"]

    for row_start in range(0, len(non_dashboard_pages), 2):
        columns = st.columns(2)
        pair = non_dashboard_pages[row_start : row_start + 2]
        for column, page in zip(columns, pair):
            with column:
                stage_names = ", ".join(stage_label(k) for k in page["stage_keys"])
                with section_card(
                    f'{page["icon"]}  {page["label"]}',
                    f"Stages: {stage_names} · Milestone {page['milestone']}",
                ):
                    overall = (
                        "complete"
                        if all(
                            get_stage_status(k) == "complete"
                            for k in page["stage_keys"]
                        )
                        else "pending"
                    )
                    status_badge(overall)
                    st.caption("Open this stage from the sidebar.")
