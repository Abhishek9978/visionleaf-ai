"""Image Acquisition page.

Purpose:
    The user-facing entry point to the Image Management System: upload
    or pick a sample image, preview it, see its metadata, and see the
    processing history begin. Every image entering the pipeline passes
    through this page and `ImageManager` — no other page loads images.

Description:
    Renders one of two states:
      - No image loaded: an empty state, the upload panel, and the
        sample gallery.
      - Image loaded: the image viewer, metadata card, history panel,
        and a "Load a different image" reset action.
    All loading goes through `ImageManager`; this page only collects
    input via the upload/gallery components and hands it off, then
    updates the single authoritative `ImageSession` via
    `session.set_image_session()`.

Dependencies:
    streamlit; visionleaf_ai.core; visionleaf_ai.processing.acquisition;
    visionleaf_ai.ui.components; visionleaf_ai.ui.session.

Public functions:
    render() -> None
"""

from __future__ import annotations

import streamlit as st

from visionleaf_ai.core import ImageLoadError, ValidationError, get_logger
from visionleaf_ai.processing.acquisition import get_image_manager
from visionleaf_ai.ui.components import (
    learning_note,
    page_header,
    render_empty_state,
    render_history_panel,
    render_image,
    render_metadata_card,
    render_pipeline_rail,
    render_sample_gallery,
    render_upload_panel,
    section_card,
)
from visionleaf_ai.ui.session import get_image_session, set_image_session

logger = get_logger(__name__)


def render() -> None:
    """Render the Image Acquisition page."""
    page_header(
        "Image Acquisition",
        subtitle="Bring a leaf image into the pipeline — from your device or the sample gallery.",
        eyebrow="Milestone 3",
    )
    render_pipeline_rail(active_stage_keys=("acquisition",))

    learning_note(
        "Acquisition is the pipeline's entry point: every later stage "
        "operates on whatever image is loaded here. Real systems validate "
        "format, size, and resolution before doing anything else, so bad "
        "input fails fast instead of corrupting results downstream."
    )

    manager = get_image_manager()
    session = get_image_session()

    if session.is_loaded():
        _render_loaded_state(session)
    else:
        _render_empty_state(manager)


def _render_empty_state(manager) -> None:
    """Render the upload panel and sample gallery when no image is loaded."""
    render_empty_state(
        "No Image Loaded",
        "Upload an image from your device, or choose one from the sample "
        "gallery below, to begin the pipeline.",
        icon="🖼️",
    )

    upload_result = render_upload_panel(
        allowed_extensions=manager.image_settings.allowed_extensions,
        max_size_mb=manager.image_settings.max_upload_size_mb,
    )
    if upload_result is not None:
        file_bytes, filename = upload_result
        _try_load(lambda: manager.load_from_upload(file_bytes, filename))

    sample_paths = manager.list_sample_images()
    selected_sample = render_sample_gallery(sample_paths)
    if selected_sample is not None:
        _try_load(lambda: manager.load_sample(selected_sample))

    if sample_paths:
        st.caption(
            "🔬 Sample gallery images are synthetic placeholders generated "
            "for this scaffold — swap in real, licensed leaf images before "
            "using this project for actual disease classification."
        )


def _render_loaded_state(session) -> None:
    """Render the preview, metadata, and history once an image is loaded."""
    left, right = st.columns([3, 2])
    with left:
        with section_card("Preview", session.metadata.filename):
            render_image(session.active_image)
        if st.button("🔄 Load a different image", width="stretch"):
            manager = get_image_manager()
            set_image_session(manager.reset_session())
            st.rerun()

    with right:
        render_metadata_card(session.metadata)
        render_history_panel(session.processing_history)


def _try_load(load_fn) -> None:
    """Run a load callable, surfacing validation/decode errors as st.error."""
    try:
        new_session = load_fn()
    except ValidationError as exc:
        logger.warning("Image validation failed: %s", exc)
        st.error(f"⚠️ {exc.message}" + (f" ({exc.details})" if exc.details else ""))
        return
    except ImageLoadError as exc:
        logger.warning("Image decode failed: %s", exc)
        st.error(f"⚠️ {exc.message}" + (f" ({exc.details})" if exc.details else ""))
        return

    set_image_session(new_session)
    st.rerun()
