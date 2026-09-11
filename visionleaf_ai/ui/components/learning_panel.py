"""Algorithm Learning Mode panel.

Purpose:
    When Learning Mode is on, show an algorithm's structured
    documentation — Purpose, Theory, Advantages, Limitations, and an
    explanation of each of its current parameters — sourced directly
    from the `AlgorithmInfo` each `PipelineStage` already provides via
    `get_info()`. No separate copy of this content is maintained in
    the UI layer.

Description:
    This is the payoff of Milestone 4's `AlgorithmInfo`/`get_info()`
    design: the UI never re-describes what an algorithm does — it only
    renders what the algorithm itself reports. Parameter explanations
    come from the same `ParamSpec.help` text already shown as each
    widget's tooltip in the Algorithm Parameter Panel, so the two can
    never disagree with each other.

Dependencies:
    streamlit; visionleaf_ai.processing.pipeline.stage.AlgorithmInfo;
    visionleaf_ai.ui.components.parameter_panel.

Public functions:
    render_learning_panel(info, param_specs) -> None
"""

from __future__ import annotations

import streamlit as st

from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo
from visionleaf_ai.ui.components.cards import section_card
from visionleaf_ai.ui.components.parameter_panel import ParamSpec


def render_learning_panel(info: AlgorithmInfo, param_specs: list[ParamSpec]) -> None:
    """Render an algorithm's structured documentation for Learning Mode.

    Args:
        info: The `AlgorithmInfo` returned by the selected algorithm's
            `get_info()`.
        param_specs: The `ParamSpec` list for the same algorithm (from
            `parameter_panel.ALGORITHM_PARAM_SPECS`), used only for its
            `label`/`help` text — this panel doesn't render widgets.
    """
    with section_card(f"📚 {info.name} — Learning Mode", info.purpose):
        st.markdown('<div class="vl-learning-section-title">Theory</div>', unsafe_allow_html=True)
        st.markdown(info.theory)

        st.markdown(
            '<div class="vl-learning-section-title">Mathematical Intuition</div>',
            unsafe_allow_html=True,
        )
        st.code(info.math_intuition, language="text")

        adv_col, lim_col = st.columns(2)
        with adv_col:
            st.markdown(
                '<div class="vl-learning-section-title">Advantages</div>', unsafe_allow_html=True
            )
            for advantage in info.advantages:
                st.markdown(f"- {advantage}")
        with lim_col:
            st.markdown(
                '<div class="vl-learning-section-title">Limitations</div>', unsafe_allow_html=True
            )
            for limitation in info.limitations:
                st.markdown(f"- {limitation}")

        if info.complexity:
            st.caption(f"**Complexity:** {info.complexity}")

        if param_specs:
            st.markdown(
                '<div class="vl-learning-section-title">Parameters</div>', unsafe_allow_html=True
            )
            for spec in param_specs:
                st.markdown(f"- **{spec.label}** — {spec.help}")
