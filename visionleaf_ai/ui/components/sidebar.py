"""Sidebar: brand header, navigation links, and Learning Mode toggle.

Purpose:
    Render everything in the sidebar: brand identity, a link to every
    page (each annotated with that page's pipeline status), and the
    global Learning Mode toggle pinned at the bottom.

Description:
    Navigation uses `st.page_link`, Streamlit's sanctioned way to link
    to an `st.Page` object — this keeps routing/URL handling owned by
    `st.navigation` in `app.py` while letting us fully control the
    surrounding markup (status dots, section grouping) that a default
    auto-generated nav wouldn't offer.

    A page's status is derived from its stage keys via
    `session.get_stage_status`: a page shows "complete" only once every
    stage it covers is complete, "active" if any stage is in progress,
    and "pending" otherwise. Milestone 2 has no working stages yet, so
    every page correctly shows "Not started" until later milestones
    flip statuses in `st.session_state["pipeline_status"]`.

Dependencies:
    streamlit; visionleaf_ai.ui.session; visionleaf_ai.ui.components.cards.

Public functions:
    render_sidebar(pages) -> None
"""

from __future__ import annotations

import streamlit as st

from visionleaf_ai.ui.components.cards import status_badge
from visionleaf_ai.ui.session import get_stage_status


def _page_status(stage_keys: tuple[str, ...]) -> str:
    """Roll up a page's overall status from its constituent stage keys."""
    if not stage_keys:
        return "complete"  # Dashboard itself has no pipeline stage to await
    statuses = [get_stage_status(key) for key in stage_keys]
    if all(status == "complete" for status in statuses):
        return "complete"
    if any(status in ("active", "complete") for status in statuses):
        return "active"
    return "pending"


def render_sidebar(pages: list, page_meta: tuple[dict, ...]) -> None:
    """Render the full sidebar: brand, nav links with status, Learning Mode.

    Args:
        pages: The list of `st.Page` objects created in `app.py`, in
            the same order as `page_meta`.
        page_meta: The corresponding metadata dicts from
            `pipeline_stages.PAGES` (same order as `pages`).
    """
    with st.sidebar:
        st.markdown(
            '<div style="font-family: var(--vl-font-display); '
            'font-weight: 700; font-size: 1.3rem;">🍃 VisionLeaf AI</div>'
            '<div style="color: #A9BDB4; font-size: 0.78rem; margin-bottom: 1rem;">'
            "Digital Image Processing Workbench</div>",
            unsafe_allow_html=True,
        )

        for page, meta in zip(pages, page_meta):
            status = _page_status(meta["stage_keys"])
            st.page_link(page, label=meta["label"], icon=meta["icon"])
            if meta["stage_keys"]:
                status_badge(status)

        st.markdown("---")
        st.toggle(
            "Learning Mode",
            key="learning_mode",
            help="Show plain-language explanations for each pipeline stage.",
        )
        st.caption("VisionLeaf AI v0.1.0 · Milestone 2")
