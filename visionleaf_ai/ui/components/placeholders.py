"""Stage placeholder component.

Purpose:
    Give every not-yet-implemented pipeline stage a professional,
    informative placeholder — never a bare "TODO" or blank page —
    that's explicit about what's coming and clearly ready for later
    milestones to slot real controls into.

Description:
    Used by every stage page whose UI hasn't been built yet. Each
    placeholder names the stage, a status note, and a short bullet
    list of what the finished stage will let the user do — set once
    per page, describing planned functionality rather than
    fabricating fake results.

    `status_note` is a caller-supplied full sentence (e.g. "Arrives in
    Milestone 7") rather than a bare milestone number: the project's
    actual milestone order didn't stay linear (Milestone 5 was
    redirected mid-project — see docs/MILESTONES.md), so a rigid
    "Arrives in Milestone {N}" template couldn't accurately describe a
    stage like Segmentation, whose backend algorithms already exist
    (Milestone 6) while its UI remains unbuilt. Letting each caller
    supply the exact wording avoids the component silently drifting
    out of sync with reality as the roadmap evolves.

Dependencies:
    streamlit; visionleaf_ai.ui.components.cards.

Public functions:
    render_stage_placeholder(stage_label, status_note, will_do) -> None
"""

from __future__ import annotations

import streamlit as st

from visionleaf_ai.ui.components.cards import section_card, status_badge


def render_stage_placeholder(
    stage_label: str, status_note: str, will_do: list[str]
) -> None:
    """Render a placeholder card for a stage whose UI is awaiting implementation.

    Args:
        stage_label: Human-readable stage name, e.g. "Segmentation".
        status_note: A full sentence describing when/whether this
            stage's UI is coming, e.g. "Arrives in Milestone 7" or
            "Backend complete since Milestone 6 — UI integration pending".
        will_do: Short bullet points describing planned functionality,
            e.g. ["Apply Otsu and adaptive thresholding", "Compare
            binary masks side by side"].
    """
    with section_card(stage_label, status_note):
        status_badge("pending")
        st.markdown("")
        st.markdown("**This stage will let you:**")
        for item in will_do:
            st.markdown(f"- {item}")
