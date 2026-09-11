"""Core reusable UI components: cards, page headers, badges, learning notes.

Purpose:
    The small set of building blocks every page is built from, so
    pages compose these rather than hand-writing markup — a change to
    how a "card" looks happens once, here.

Description:
    - `page_header()`: consistent title/eyebrow/subtitle block at the
      top of every page.
    - `section_card()`: a context manager wrapping Streamlit's native
      bordered container, with a styled title/description header.
      Usage: `with section_card("Title"): st.write(...)`.
    - `status_badge()`: small pill showing a stage's pending/active/
      complete status.
    - `learning_note()`: renders an info callout only when Learning
      Mode is on — the mechanism behind the "Learning Mode must
      globally enable educational explanations without changing
      application behavior" requirement. Pages call this freely;
      it's a no-op when Learning Mode is off.

Dependencies:
    streamlit; contextlib.

Public functions:
    page_header(title, subtitle=None, eyebrow=None) -> None
    section_card(title, description=None) -> AbstractContextManager
    status_badge(status) -> None
    learning_note(text) -> None
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import streamlit as st

_BADGE_LABELS = {
    "pending": "Not started",
    "active": "In progress",
    "complete": "Complete",
}


def page_header(title: str, subtitle: str | None = None, eyebrow: str | None = None) -> None:
    """Render the standard page title block.

    Args:
        title: Main page title, e.g. "Image Processing Lab".
        subtitle: Optional one-line description shown under the title.
        eyebrow: Optional small uppercase label above the title, e.g.
            "MILESTONE 4".
    """
    if eyebrow:
        st.markdown(f'<div class="vl-page-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f'<div class="vl-page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


@contextmanager
def section_card(title: str, description: str | None = None) -> Iterator[None]:
    """Context manager rendering a bordered card with a styled header.

    Args:
        title: Card title, shown in the display font.
        description: Optional muted one-line description under the title.

    Yields:
        None. Put any Streamlit widgets/content inside the `with` block;
        they render inside the card.

    Example:
        >>> with section_card("Upload", "Bring your own leaf image"):
        ...     st.file_uploader("Image")
    """
    container = st.container(border=True)
    with container:
        st.markdown(f'<div class="vl-card-title">{title}</div>', unsafe_allow_html=True)
        if description:
            st.markdown(f'<div class="vl-card-desc">{description}</div>', unsafe_allow_html=True)
        yield


def status_badge(status: str) -> None:
    """Render a small status pill.

    Args:
        status: One of "pending", "active", "complete". Unknown values
            fall back to the "pending" style so a typo never raises.
    """
    label = _BADGE_LABELS.get(status, _BADGE_LABELS["pending"])
    css_status = status if status in _BADGE_LABELS else "pending"
    st.markdown(
        f'<span class="vl-badge vl-badge-{css_status}">{label}</span>',
        unsafe_allow_html=True,
    )


def learning_note(text: str) -> None:
    """Render an educational callout, but only when Learning Mode is on.

    Args:
        text: Plain-language explanation of the concept relevant to
            wherever this is called from.

    Notes:
        This is the entire Learning Mode mechanism: it's read from
        `st.session_state["learning_mode"]` and renders nothing at all
        when that flag is False, so Learning Mode can never change
        what a stage actually computes — only whether an explanation
        appears alongside it.
    """
    if st.session_state.get("learning_mode"):
        st.info(text, icon="🎓")
