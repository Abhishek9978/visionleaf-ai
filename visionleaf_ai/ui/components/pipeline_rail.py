"""The pipeline rail — VisionLeaf AI's signature navigational element.

Purpose:
    Render all 11 DIP pipeline stages as a connected, numbered row,
    highlighting whichever stage(s) the current page covers. This is
    the one element every stage page shares, reinforcing that the
    whole application is one continuous pipeline rather than a set of
    unrelated tools.

Description:
    Numbering here is meaningful, not decorative: the stages *are* a
    fixed, ordered sequence (Acquisition through Experiment Mode), and
    a viewer benefits from seeing both where they are and how many
    steps remain. Reads stage data from `pipeline_stages.STAGES` so it
    never drifts out of sync with the sidebar or dashboard.

Dependencies:
    streamlit; visionleaf_ai.ui.pipeline_stages.

Public functions:
    render_pipeline_rail(active_stage_keys) -> None
"""

from __future__ import annotations

import streamlit as st

from visionleaf_ai.ui.pipeline_stages import STAGES


def render_pipeline_rail(active_stage_keys: tuple[str, ...] = ()) -> None:
    """Render the 11-stage pipeline rail with the given stages highlighted.

    Args:
        active_stage_keys: Stage keys (from `pipeline_stages.STAGES`)
            belonging to the current page. Pass an empty tuple (e.g.
            on the Dashboard) to render the rail with nothing active.
    """
    nodes_html = []
    for index, stage in enumerate(STAGES, start=1):
        is_active = stage["key"] in active_stage_keys
        active_class = " vl-rail-active" if is_active else ""
        nodes_html.append(
            f'<div class="vl-rail-node{active_class}">'
            f'<div class="vl-rail-dot">{index:02d}</div>'
            f'<div class="vl-rail-label">{stage["label"]}</div>'
            f"</div>"
        )
    st.markdown(f'<div class="vl-rail">{"".join(nodes_html)}</div>', unsafe_allow_html=True)
