"""Configuration package.

Purpose:
    Public entry point for reading validated application settings.

Description:
    Re-exports `get_settings` so callers can do
    `from visionleaf_ai.config import get_settings` rather than reaching
    into `visionleaf_ai.config.settings` directly.

Dependencies:
    visionleaf_ai.config.settings

Public functions:
    get_settings() -> Settings
"""

from visionleaf_ai.config.settings import Settings, get_settings

__all__ = ["get_settings", "Settings"]
