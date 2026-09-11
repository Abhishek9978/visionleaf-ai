"""Test harness: renders the Processing Lab page with a tiny (10x10) image.

Not a real application entry point — used only by
tests/ui/test_processing_lab_page.py to exercise error handling: a
10x10 image is small enough that the parameter panel's normal slider
range (kernel sizes up to 31) can genuinely exceed the image's
dimensions, triggering a real ValidationError through the real
PipelineEngine — not a mocked one.
"""

from datetime import datetime

import numpy as np
import streamlit as st

from visionleaf_ai.core.image_session import ImageMetadata, ImageSession
from visionleaf_ai.ui.pages import processing_lab
from visionleaf_ai.ui.session import init_session_state, set_image_session
from visionleaf_ai.ui.theme import inject_theme

init_session_state()
inject_theme()

if "_test_seeded" not in st.session_state:
    st.session_state["_test_seeded"] = True
    tiny_image = (np.random.default_rng(1).random((10, 10, 3)) * 255).astype(np.uint8)
    session = ImageSession(
        original_image=tiny_image.copy(),
        current_image=tiny_image.copy(),
        metadata=ImageMetadata(
            filename="tiny.png",
            file_format="PNG",
            width=10,
            height=10,
            channels=3,
            file_size_bytes=100,
            uploaded_at=datetime.now(),
            was_resized=False,
            source="upload",
        ),
    )
    session.record_event("acquisition", "Image Loaded")
    set_image_session(session)

processing_lab.render()
