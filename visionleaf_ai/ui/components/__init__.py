"""UI components package — reusable building blocks for every page.

Purpose:
    Re-export the component functions pages actually use, so page
    modules can write a single import line.

Description:
    Every component here is presentation-only: none contain DIP logic.
    Milestone 5 adds the Image Processing Laboratory's components
    (comparison viewer, parameter panel, histogram viewer, pixel
    statistics, learning panel, export panel) — all reusable by later
    stages, not just the Processing Lab page.

Dependencies:
    visionleaf_ai.ui.components.*

Public functions:
    page_header, section_card, status_badge, learning_note,
    render_pipeline_rail, render_metric_row, render_stage_placeholder,
    render_sidebar, render_image, render_image_comparison,
    render_metadata_card, render_upload_panel, render_sample_gallery,
    render_history_panel, render_empty_state, render_comparison_viewer,
    render_parameter_panel, ParamSpec, ALGORITHM_PARAM_SPECS,
    render_histogram_comparison, render_pixel_statistics,
    compute_pixel_statistics, render_learning_panel, render_export_panel
"""

from visionleaf_ai.ui.components.cards import (
    learning_note,
    page_header,
    section_card,
    status_badge,
)
from visionleaf_ai.ui.components.comparison_viewer import render_comparison_viewer
from visionleaf_ai.ui.components.empty_state import render_empty_state
from visionleaf_ai.ui.components.export_panel import render_export_panel
from visionleaf_ai.ui.components.histogram_viewer import render_histogram_comparison
from visionleaf_ai.ui.components.history_panel import render_history_panel
from visionleaf_ai.ui.components.image_viewer import (
    render_image,
    render_image_comparison,
)
from visionleaf_ai.ui.components.learning_panel import render_learning_panel
from visionleaf_ai.ui.components.metadata_card import render_metadata_card
from visionleaf_ai.ui.components.metrics import render_metric_row
from visionleaf_ai.ui.components.parameter_panel import (
    ALGORITHM_PARAM_SPECS,
    ParamSpec,
    render_parameter_panel,
)
from visionleaf_ai.ui.components.pipeline_rail import render_pipeline_rail
from visionleaf_ai.ui.components.pixel_statistics import (
    compute_pixel_statistics,
    render_pixel_statistics,
)
from visionleaf_ai.ui.components.placeholders import render_stage_placeholder
from visionleaf_ai.ui.components.sidebar import render_sidebar
from visionleaf_ai.ui.components.upload_panel import (
    render_sample_gallery,
    render_upload_panel,
)

__all__ = [
    "page_header",
    "section_card",
    "status_badge",
    "learning_note",
    "render_pipeline_rail",
    "render_metric_row",
    "render_stage_placeholder",
    "render_sidebar",
    "render_image",
    "render_image_comparison",
    "render_metadata_card",
    "render_upload_panel",
    "render_sample_gallery",
    "render_history_panel",
    "render_empty_state",
    "render_comparison_viewer",
    "render_parameter_panel",
    "ParamSpec",
    "ALGORITHM_PARAM_SPECS",
    "render_histogram_comparison",
    "render_pixel_statistics",
    "compute_pixel_statistics",
    "render_learning_panel",
    "render_export_panel",
]
