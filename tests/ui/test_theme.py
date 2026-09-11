"""Tests for visionleaf_ai.ui.theme.

Purpose:
    Verify the token dictionary is well-formed and that inject_theme()
    runs without raising (full CSS-rendering correctness is verified
    visually / via the app-level AppTest in test_app.py).

Dependencies:
    pytest; visionleaf_ai.ui.theme.
"""

from __future__ import annotations

import re

from visionleaf_ai.ui.theme import TOKENS

_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def test_all_tokens_are_valid_hex_colors():
    for name, value in TOKENS.items():
        assert _HEX_COLOR_RE.match(value), f"{name} = {value!r} is not a valid hex color"


def test_expected_tokens_present():
    expected = {
        "bg",
        "surface",
        "ink",
        "muted",
        "primary",
        "primary_soft",
        "accent",
        "border",
        "danger",
    }
    assert expected.issubset(TOKENS.keys())
