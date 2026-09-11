"""VisionLeaf AI — Streamlit application entry point.

Purpose:
    Wire together configuration, logging, theming, session state, and
    page navigation into a running app. Run with `streamlit run app.py`.

Description:
    Uses `st.navigation` with `position="hidden"` so routing/URL
    handling is owned by Streamlit while the actual sidebar UI (brand,
    status-annotated links, Learning Mode toggle) is fully custom,
    built by `components.sidebar.render_sidebar`. Each page is a plain
    Python function (`render()` in `ui/pages/<name>.py`), wrapped in an
    `st.Page` here — the mapping from page key to render function is
    kept explicit in `_RENDERERS` so it's obvious which module backs
    which page.

Dependencies:
    streamlit; visionleaf_ai.core; visionleaf_ai.config; visionleaf_ai.ui.*

Public functions:
    main() -> None
        Configure and run the app. Called at module scope so
        `streamlit run app.py` executes it.
"""

from __future__ import annotations

import streamlit as st

from visionleaf_ai.config import get_settings
from visionleaf_ai.core import get_logger
from visionleaf_ai.ui.components.sidebar import render_sidebar
from visionleaf_ai.ui.pages import (
    acquisition,
    analytics,
    classification,
    dashboard,
    experiment_mode,
    feature_extraction,
    processing_lab,
    segmentation_pipeline,
)
from visionleaf_ai.ui.pipeline_stages import PAGES
from visionleaf_ai.ui.session import init_session_state
from visionleaf_ai.ui.theme import inject_theme

logger = get_logger(__name__)

# Explicit key -> render function mapping. Keys must match PAGES in
# pipeline_stages.py exactly; a mismatch raises KeyError at startup
# rather than silently rendering the wrong page.
_RENDERERS = {
    "dashboard": dashboard.render,
    "acquisition": acquisition.render,
    "processing_lab": processing_lab.render,
    "segmentation_pipeline": segmentation_pipeline.render,
    "feature_extraction": feature_extraction.render,
    "classification": classification.render,
    "analytics": analytics.render,
    "experiment_mode": experiment_mode.render,
}


def main() -> None:
    """Configure and run the VisionLeaf AI Streamlit application."""
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app.name,
        page_icon="🍃",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    inject_theme()

    pages = [
        st.Page(
            _RENDERERS[meta["key"]],
            title=meta["label"],
            icon=meta["icon"],
            url_path=meta["key"],
            default=meta["default"],
        )
        for meta in PAGES
    ]

    render_sidebar(pages, PAGES)

    navigation = st.navigation(pages, position="hidden")
    navigation.run()


main()
