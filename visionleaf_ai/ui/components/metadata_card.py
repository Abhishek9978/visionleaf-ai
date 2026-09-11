"""Metadata card component.

Purpose:
    Consistently display an image's descriptive metadata: filename,
    format, resolution, channels, file size, and upload timestamp.

Dependencies:
    streamlit; visionleaf_ai.core.image_session.ImageMetadata.

Public functions:
    render_metadata_card(metadata) -> None
"""

from __future__ import annotations

import streamlit as st

from visionleaf_ai.core.image_session import ImageMetadata
from visionleaf_ai.ui.components.cards import section_card
from visionleaf_ai.ui.components.metrics import render_metric_row


def _format_file_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable size string."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def render_metadata_card(metadata: ImageMetadata, current_stage: str | None = None) -> None:
    """Render a card showing an image's metadata fields.

    Args:
        metadata: The `ImageMetadata` from the active `ImageSession`.
        current_stage: Optional label of the most recent processing
            stage applied (e.g. the last `processing_history` entry's
            label), shown as an extra field. Omitted if `None`.
    """
    channel_label = "Grayscale (1)" if metadata.channels == 1 else f"Color ({metadata.channels})"
    with section_card("Image Metadata", metadata.filename):
        render_metric_row(
            [
                ("Resolution", f"{metadata.width} × {metadata.height} px"),
                ("Format", metadata.file_format),
                ("File Size", _format_file_size(metadata.file_size_bytes)),
            ]
        )
        row_two = [
            ("Channels", channel_label),
            ("Source", metadata.source.capitalize()),
            ("Uploaded", metadata.uploaded_at.strftime("%H:%M:%S")),
        ]
        render_metric_row(row_two)
        if current_stage:
            st.caption(f"**Current stage:** {current_stage}")
        if metadata.was_resized:
            st.caption(
                "ℹ️ This image was automatically resized to fit the "
                "configured maximum dimension."
            )
