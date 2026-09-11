"""Design tokens and global CSS injection.

Purpose:
    Single place defining VisionLeaf AI's visual identity (colors,
    type, spacing) and the one function, `inject_theme()`, that applies
    it to the running Streamlit app.

Description:
    Colors and fonts are expressed as CSS custom properties (`--vl-*`)
    on `:root`, so component modules style against named tokens
    (`var(--vl-primary)`) instead of repeating hex codes — change a
    color once here and every component picks it up. Native Streamlit
    theming (colors for built-in widgets) is handled separately in
    `.streamlit/config.toml`; this module layers custom component
    styling (cards, badges, the pipeline rail) on top via a single
    injected `<style>` block.

    Milestone 5 switched the palette to a dark, scientific-software
    theme (per the approved Blueprint's requirement for the Image
    Processing Laboratory) — applied app-wide rather than to a single
    page, so every page shares one consistent look. `sidebar_bg` is a
    dedicated token distinct from `bg`/`surface`, since with a dark
    canvas the sidebar can no longer simply reuse a "dark ink" token
    the way the original light theme did.

    Fonts are loaded from Google Fonts with a full system-font fallback
    stack, so the app still renders correctly if the client has no
    network access to fonts.googleapis.com.

Dependencies:
    streamlit.

Public functions:
    inject_theme() -> None
        Inject the global stylesheet into the current Streamlit page.
        Safe to call on every rerun (idempotent from the user's
        perspective — Streamlit simply re-renders the same <style> tag).
"""

from __future__ import annotations

import streamlit as st

TOKENS: dict[str, str] = {
    "bg": "#11171A",
    "surface": "#1B2226",
    "surface_alt": "#222B30",
    "sidebar_bg": "#0B0F11",
    "ink": "#E7EEEA",
    "muted": "#8CA39B",
    "primary": "#3FB88F",
    "primary_soft": "#173832",
    "accent": "#E3A857",
    "border": "#2A3438",
    "danger": "#E2694F",
}

FONT_DISPLAY = "'Space Grotesk', ui-sans-serif, system-ui, sans-serif"
FONT_BODY = "'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif"
FONT_MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, monospace"

_CSS_TEMPLATE = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {{
    --vl-bg: {bg};
    --vl-surface: {surface};
    --vl-surface-alt: {surface_alt};
    --vl-sidebar-bg: {sidebar_bg};
    --vl-ink: {ink};
    --vl-muted: {muted};
    --vl-primary: {primary};
    --vl-primary-soft: {primary_soft};
    --vl-accent: {accent};
    --vl-border: {border};
    --vl-danger: {danger};
    --vl-font-display: {font_display};
    --vl-font-body: {font_body};
    --vl-font-mono: {font_mono};
}}

.stApp {{
    background-color: var(--vl-bg);
    font-family: var(--vl-font-body);
    color: var(--vl-ink);
}}

section[data-testid="stSidebar"] {{
    background-color: var(--vl-sidebar-bg);
    border-right: 1px solid var(--vl-border);
}}
section[data-testid="stSidebar"] * {{
    color: var(--vl-ink) !important;
}}
section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {{
    border-radius: 8px;
    margin-bottom: 2px;
}}
section[data-testid="stSidebar"] a[aria-current="page"] {{
    background-color: rgba(255, 255, 255, 0.08) !important;
}}

h1, h2, h3, h4 {{
    font-family: var(--vl-font-display) !important;
    letter-spacing: -0.01em;
    color: var(--vl-ink);
}}

div[data-testid="stMetricValue"] {{
    font-family: var(--vl-font-mono);
    color: var(--vl-ink);
}}
div[data-testid="stMetricLabel"] {{
    color: var(--vl-muted);
}}

/* Card surfaces: Streamlit's native bordered container */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: var(--vl-surface);
    border: 1px solid var(--vl-border);
    border-radius: 12px;
}}

/* Widgets: dark-mode-appropriate contrast tweaks */
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label,
div[data-testid="stRadio"] label, div[data-testid="stCheckbox"] label {{
    color: var(--vl-muted) !important;
}}

.vl-page-eyebrow {{
    font-family: var(--vl-font-mono);
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--vl-muted);
    margin-bottom: 0.15rem;
}}
.vl-page-subtitle {{
    color: var(--vl-muted);
    margin-top: -0.4rem;
    margin-bottom: 0.5rem;
}}

.vl-card-title {{
    font-family: var(--vl-font-display);
    font-weight: 600;
    font-size: 1.05rem;
    margin-bottom: 0.1rem;
    color: var(--vl-ink);
}}
.vl-card-desc {{
    color: var(--vl-muted);
    font-size: 0.9rem;
    margin-bottom: 0.6rem;
}}

.vl-badge {{
    display: inline-block;
    font-family: var(--vl-font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    padding: 2px 9px;
    border-radius: 999px;
    border: 1px solid transparent;
}}
.vl-badge-pending {{
    background-color: var(--vl-primary-soft);
    color: var(--vl-primary);
    border-color: var(--vl-primary);
}}
.vl-badge-active {{
    background-color: var(--vl-accent);
    color: #11171A;
}}
.vl-badge-complete {{
    background-color: var(--vl-primary);
    color: #11171A;
}}

/* Pipeline rail — the signature element */
.vl-rail {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 2px;
    padding: 14px 4px 6px 4px;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid var(--vl-border);
}}
.vl-rail-node {{
    flex: 1;
    text-align: center;
    position: relative;
}}
.vl-rail-dot {{
    width: 22px;
    height: 22px;
    line-height: 22px;
    margin: 0 auto 6px auto;
    border-radius: 50%;
    background-color: var(--vl-surface);
    border: 2px solid var(--vl-border);
    color: var(--vl-muted);
    font-family: var(--vl-font-mono);
    font-size: 0.72rem;
    font-weight: 500;
}}
.vl-rail-node.vl-rail-active .vl-rail-dot {{
    background-color: var(--vl-primary);
    border-color: var(--vl-primary);
    color: #11171A;
}}
.vl-rail-label {{
    font-size: 0.68rem;
    color: var(--vl-muted);
    line-height: 1.15;
}}
.vl-rail-node.vl-rail-active .vl-rail-label {{
    color: var(--vl-ink);
    font-weight: 600;
}}
.vl-rail-node:not(:last-child)::after {{
    content: "";
    position: absolute;
    top: 11px;
    left: 50%;
    width: 100%;
    height: 2px;
    background-color: var(--vl-border);
    z-index: -1;
}}

/* Learning Mode panel */
.vl-learning-section-title {{
    font-family: var(--vl-font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--vl-primary);
    margin-top: 0.6rem;
    margin-bottom: 0.1rem;
}}
</style>
"""


def inject_theme() -> None:
    """Inject VisionLeaf AI's global stylesheet into the current page.

    Call once near the top of `app.py`, after `st.set_page_config()`.
    Safe to call on every Streamlit rerun.
    """
    css = _CSS_TEMPLATE.format(
        bg=TOKENS["bg"],
        surface=TOKENS["surface"],
        surface_alt=TOKENS["surface_alt"],
        sidebar_bg=TOKENS["sidebar_bg"],
        ink=TOKENS["ink"],
        muted=TOKENS["muted"],
        primary=TOKENS["primary"],
        primary_soft=TOKENS["primary_soft"],
        accent=TOKENS["accent"],
        border=TOKENS["border"],
        danger=TOKENS["danger"],
        font_display=FONT_DISPLAY,
        font_body=FONT_BODY,
        font_mono=FONT_MONO,
    )
    st.markdown(css, unsafe_allow_html=True)
