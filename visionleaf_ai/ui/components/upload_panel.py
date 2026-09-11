"""Upload panel component: file upload + sample gallery.

Purpose:
    The one place that renders "how do I get an image in" — a file
    uploader (Streamlit's uploader natively supports drag-and-drop) and
    a sample gallery of thumbnail buttons. Returns raw data; the
    calling page is responsible for handing it to `ImageManager`.

Description:
    Deliberately does NOT call `ImageManager` itself — this component
    only collects input (bytes + filename, or a chosen sample path) and
    hands it back to the page, keeping the "load and validate" logic
    centralized in `ImageManager` as the architecture requires.

Dependencies:
    streamlit; pathlib.

Public functions:
    render_upload_panel(allowed_extensions, max_size_mb) -> tuple[bytes, str] | None
    render_sample_gallery(sample_paths) -> Path | None
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from visionleaf_ai.ui.components.cards import section_card


def render_upload_panel(
    allowed_extensions: tuple[str, ...], max_size_mb: float
) -> tuple[bytes, str] | None:
    """Render a file upload control (drag-and-drop is native to Streamlit).

    Args:
        allowed_extensions: Extensions to accept, e.g. (".png", ".jpg").
        max_size_mb: Shown to the user as a hint; the actual enforced
            limit is applied later by `ImageManager`.

    Returns:
        `(file_bytes, filename)` if a file is currently uploaded,
        otherwise `None`.
    """
    with section_card(
        "Upload Image",
        f"Drag and drop, or browse — up to {max_size_mb:.0f} MB.",
    ):
        extensions_hint = [ext.lstrip(".") for ext in allowed_extensions]
        uploaded_file = st.file_uploader(
            "Choose a leaf image",
            type=extensions_hint,
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            return uploaded_file.getvalue(), uploaded_file.name
    return None


def render_sample_gallery(sample_paths: list[Path]) -> Path | None:
    """Render a gallery of sample images as selectable thumbnails.

    Args:
        sample_paths: Paths returned by `ImageManager.list_sample_images()`.

    Returns:
        The `Path` of the sample the user clicked "Use this image" for
        this rerun, otherwise `None`.
    """
    selected: Path | None = None
    with section_card("Sample Gallery", "Or start from a built-in sample image."):
        if not sample_paths:
            st.caption("No sample images are available.")
            return None

        columns = st.columns(min(len(sample_paths), 4))
        for column, sample_path in zip(columns, sample_paths):
            with column:
                st.image(str(sample_path), width="stretch")
                if st.button("Use this image", key=f"sample_{sample_path.name}"):
                    selected = sample_path
    return selected
