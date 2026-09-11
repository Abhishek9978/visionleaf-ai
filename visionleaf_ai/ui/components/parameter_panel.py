"""Algorithm parameter panel.

Purpose:
    Render only the widgets relevant to whichever algorithm is
    currently selected (e.g. CLAHE shows Clip Limit + Tile Grid Size;
    Gaussian Blur shows Kernel Size + Sigma), and collect their current
    values into a plain dict ready to `**`-unpack into
    `PipelineEngine.run_by_name(name, session, **params)`.

Description:
    The parameter *schema* (labels, ranges, defaults, help text) is
    owned entirely by the UI layer here — deliberately not imported
    from `processing.enhancement`/`processing.restoration`, keeping the
    UI/processing boundary exactly where the architecture draws it:
    the UI knows how to ask a human for a number, the processing layer
    knows what to do with it. Registered algorithm *names* still come
    from the real registry (via `AlgorithmRegistry`), so this schema
    can never reference an algorithm that doesn't actually exist.

Dependencies:
    streamlit; dataclasses.

Public classes/functions:
    ParamSpec
    ALGORITHM_PARAM_SPECS
    render_parameter_panel(algorithm_name) -> dict
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class ParamSpec:
    """Describes one widget in the parameter panel.

    Attributes:
        key: The keyword-argument name expected by the algorithm's
            constructor (and therefore by `run_by_name(..., **params)`).
        label: Human-readable widget label.
        widget: One of "int_slider", "float_slider", "odd_slider"
            (a select_slider restricted to odd integers), or
            "tile_grid" (a single square-grid-size slider that expands
            to a `(n, n)` tuple).
        min_value, max_value, default, step: Passed to the underlying
            Streamlit widget.
        help: Tooltip text — also reused verbatim as this parameter's
            explanation in the Learning Mode panel, so the two never
            drift out of sync.
    """

    key: str
    label: str
    widget: str
    min_value: float
    max_value: float
    default: float
    step: float
    help: str


ALGORITHM_PARAM_SPECS: dict[str, list[ParamSpec]] = {
    "brightness_adjustment": [
        ParamSpec(
            "delta", "Delta", "int_slider", -150, 150, 30, 1,
            "Amount added to every pixel. Positive brightens, negative darkens.",
        ),
    ],
    "contrast_adjustment": [
        ParamSpec(
            "alpha", "Alpha (Contrast Factor)", "float_slider", 0.1, 3.0, 1.5, 0.1,
            "Multiplies every pixel value. Above 1.0 increases contrast, below 1.0 reduces it.",
        ),
    ],
    "gamma_correction": [
        ParamSpec(
            "gamma", "Gamma", "float_slider", 0.1, 5.0, 1.5, 0.1,
            "Above 1.0 brightens midtones, below 1.0 darkens them, non-linearly.",
        ),
    ],
    "histogram_equalization": [],
    "clahe": [
        ParamSpec(
            "clip_limit", "Clip Limit", "float_slider", 0.5, 10.0, 2.0, 0.5,
            "Caps how much contrast any single tile can be stretched by, limiting noise amplification.",
        ),
        ParamSpec(
            "tile_grid_size", "Tile Grid Size (N x N)", "tile_grid", 2, 16, 8, 1,
            "How many tiles the image is divided into per side before each tile is equalized independently.",
        ),
    ],
    "gaussian_blur": [
        ParamSpec(
            "kernel_size", "Kernel Size", "odd_slider", 3, 31, 5, 2,
            "Size of the smoothing neighborhood. Larger values blur more.",
        ),
        ParamSpec(
            "sigma", "Sigma", "float_slider", 0.0, 10.0, 0.0, 0.1,
            "Spread of the Gaussian weighting. 0 lets OpenCV derive it from the kernel size.",
        ),
    ],
    "median_filter": [
        ParamSpec(
            "kernel_size", "Kernel Size", "odd_slider", 3, 31, 5, 2,
            "Size of the neighborhood each pixel's median is computed from.",
        ),
    ],
    "bilateral_filter": [
        ParamSpec(
            "diameter", "Diameter", "int_slider", 1, 25, 9, 1,
            "Diameter of each pixel's neighborhood.",
        ),
        ParamSpec(
            "sigma_color", "Sigma Color", "float_slider", 10.0, 200.0, 75.0, 5.0,
            "How much an intensity difference reduces a neighbor's influence — smaller preserves edges more aggressively.",
        ),
        ParamSpec(
            "sigma_space", "Sigma Space", "float_slider", 10.0, 200.0, 75.0, 5.0,
            "How far spatially a pixel's influence reaches.",
        ),
    ],
}


def render_parameter_panel(algorithm_name: str) -> dict[str, object]:
    """Render the parameter widgets for one algorithm and collect their values.

    Args:
        algorithm_name: Registered algorithm key, e.g. "clahe".

    Returns:
        A dict of `{param_key: current_value}`, ready to `**`-unpack
        into `PipelineEngine.run_by_name(algorithm_name, session, **result)`.
        Empty dict if the algorithm has no tunable parameters.
    """
    specs = ALGORITHM_PARAM_SPECS.get(algorithm_name, [])
    if not specs:
        st.caption("This algorithm has no adjustable parameters.")
        return {}

    values: dict[str, object] = {}
    for spec in specs:
        widget_key = f"param__{algorithm_name}__{spec.key}"
        if spec.widget == "int_slider":
            values[spec.key] = st.slider(
                spec.label,
                min_value=int(spec.min_value),
                max_value=int(spec.max_value),
                value=int(spec.default),
                step=int(spec.step),
                help=spec.help,
                key=widget_key,
            )
        elif spec.widget == "float_slider":
            values[spec.key] = st.slider(
                spec.label,
                min_value=float(spec.min_value),
                max_value=float(spec.max_value),
                value=float(spec.default),
                step=float(spec.step),
                help=spec.help,
                key=widget_key,
            )
        elif spec.widget == "odd_slider":
            options = list(range(int(spec.min_value), int(spec.max_value) + 1, int(spec.step)))
            values[spec.key] = st.select_slider(
                spec.label,
                options=options,
                value=int(spec.default),
                help=spec.help,
                key=widget_key,
            )
        elif spec.widget == "tile_grid":
            grid_n = st.slider(
                spec.label,
                min_value=int(spec.min_value),
                max_value=int(spec.max_value),
                value=int(spec.default),
                step=int(spec.step),
                help=spec.help,
                key=widget_key,
            )
            values[spec.key] = (grid_n, grid_n)
        else:  # pragma: no cover — defensive: every spec above uses a known widget
            raise ValueError(f"Unknown widget type: {spec.widget!r}")

    return values
