"""Tests for visionleaf_ai.config.settings.

Purpose:
    Verify that get_settings() loads config.yaml correctly, resolves
    paths, applies environment overrides, and fails loudly on bad input.

Dependencies:
    pytest; visionleaf_ai.config.

Public functions:
    None — this is a pytest test module, discovered by function name.
"""

from __future__ import annotations

import pytest

from visionleaf_ai.config.settings import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure each test loads settings fresh rather than reusing the
    process-wide lru_cache from a previous test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_get_settings_loads_successfully():
    settings = get_settings()
    assert settings.app.name == "VisionLeaf AI"
    assert settings.app.version
    assert settings.logging.level in {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }


def test_paths_are_absolute_and_exist():
    settings = get_settings()
    for directory in (
        settings.paths.assets_dir,
        settings.paths.sample_images_dir,
        settings.paths.models_dir,
        settings.paths.logs_dir,
    ):
        assert directory.is_absolute()
        assert directory.exists()


def test_image_settings_have_expected_defaults():
    settings = get_settings()
    assert settings.image.max_upload_size_mb > 0
    assert ".png" in settings.image.allowed_extensions
    assert settings.image.max_dimension_px > 0


def test_env_override_for_debug_flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VISIONLEAF_APP__DEBUG", "true")
    settings = get_settings()
    assert settings.app.debug is True


def test_env_override_for_logging_level(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VISIONLEAF_LOGGING__LEVEL", "DEBUG")
    settings = get_settings()
    assert settings.logging.level == "DEBUG"


def test_settings_is_cached_across_calls():
    first = get_settings()
    second = get_settings()
    assert first is second
