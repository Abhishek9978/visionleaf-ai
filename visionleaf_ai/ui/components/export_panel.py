"""Export panel component.

Purpose:
    Let the user download the current processed image. Encoding is
    delegated entirely to `ImageManager.export_image` — this component
    only renders the format choice and the download button.

Dependencies:
    streamlit; visionleaf_ai.core.image_session.ImageSession;
    visionleaf_ai.processing.acquisition.ImageManager.

Public functions:
    render_export_panel(manager, session) -> None
"""

from __future__ import annotations

import streamlit as st

from visionleaf_ai.core.exceptions import VisionLeafError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.processing.acquisition.image_manager import ImageManager
from visionleaf_ai.ui.components.cards import section_card

_FORMAT_MIME_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg"}


def render_export_panel(manager: ImageManager, session: ImageSession) -> None:
    """Render format selection and a download button for the active image.

    Args:
        manager: The `ImageManager` used to encode the export (never
            called by this component directly for pixel decoding —
            only its `export_image` encoding method).
        session: The current `ImageSession`.
    """
    with section_card("Export", "Download the current processed image."):
        image_format = st.selectbox("Format", options=list(_FORMAT_MIME_TYPES), key="export_format")
        try:
            image_bytes = manager.export_image(session, image_format=image_format)
        except VisionLeafError as exc:
            st.error(f"⚠️ Could not prepare export: {exc.message}")
            return

        base_name = session.metadata.filename.rsplit(".", 1)[0] if session.metadata else "image"
        st.download_button(
            "⬇️ Download Processed Image",
            data=image_bytes,
            file_name=f"{base_name}_processed.{image_format.lower()}",
            mime=_FORMAT_MIME_TYPES[image_format],
            width="stretch",
        )
