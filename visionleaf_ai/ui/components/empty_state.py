"""Empty state component.

Purpose:
    One consistent "there's nothing here yet" panel, reusable by any
    future stage — not just Acquisition. Segmentation before an image
    is loaded, Analytics before a model is trained, etc. all show the
    same visual pattern instead of each inventing its own.

Dependencies:
    streamlit.

Public functions:
    render_empty_state(title, description, icon="📭") -> None
"""

from __future__ import annotations

import streamlit as st

from visionleaf_ai.ui.components.cards import section_card


def render_empty_state(title: str, description: str, icon: str = "📭") -> None:
    """Render a consistent empty-state panel.

    Args:
        title: Short heading, e.g. "No Image Loaded".
        description: One or two sentences on what to do next.
        icon: A single emoji shown above the title.
    """
    with section_card(f"{icon}  {title}"):
        st.markdown(description)
