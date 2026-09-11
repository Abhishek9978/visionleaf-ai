"""History panel component.

Purpose:
    Render an `ImageSession`'s `processing_history` as a timeline —
    reusable as-is by every later stage, since every stage appends to
    the same list on the same session object.

Dependencies:
    streamlit; visionleaf_ai.core.image_session.ProcessingEvent.

Public functions:
    render_history_panel(events) -> None
"""

from __future__ import annotations

import streamlit as st

from visionleaf_ai.core.image_session import ProcessingEvent
from visionleaf_ai.ui.components.cards import section_card


def render_history_panel(events: list[ProcessingEvent]) -> None:
    """Render the processing history as a simple timeline.

    Args:
        events: `session.processing_history`, in chronological order.
    """
    with section_card("Processing History", "What has been done to this image so far."):
        if not events:
            st.caption("No processing steps recorded yet.")
            return
        for event in events:
            timestamp = event.timestamp.strftime("%H:%M:%S")
            line = f"`{timestamp}` — **{event.label}**"
            if event.details:
                line += f" _{event.details}_"
            st.markdown(line)
