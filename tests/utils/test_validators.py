"""Tests for visionleaf_ai.utils.validators.

Purpose:
    Verify each validator raises ValidationError on bad input and
    returns a resolved Path on success.

Dependencies:
    pytest; visionleaf_ai.utils; visionleaf_ai.core.exceptions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.utils.validators import (
    ensure_extension_allowed,
    ensure_file_size_within_limit,
    ensure_path_exists,
)


def test_ensure_path_exists_success(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello")
    result = ensure_path_exists(file_path)
    assert result == file_path.resolve()


def test_ensure_path_exists_raises_for_missing_path(tmp_path: Path):
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(ValidationError):
        ensure_path_exists(missing)


def test_ensure_extension_allowed_success(tmp_path: Path):
    file_path = tmp_path / "leaf.png"
    file_path.write_text("fake image bytes")
    result = ensure_extension_allowed(file_path, (".png", ".jpg"))
    assert result.suffix == ".png"


def test_ensure_extension_allowed_is_case_insensitive(tmp_path: Path):
    file_path = tmp_path / "leaf.PNG"
    file_path.write_text("fake image bytes")
    ensure_extension_allowed(file_path, (".png",))  # should not raise


def test_ensure_extension_allowed_rejects_disallowed_extension(tmp_path: Path):
    file_path = tmp_path / "leaf.exe"
    file_path.write_text("not an image")
    with pytest.raises(ValidationError):
        ensure_extension_allowed(file_path, (".png", ".jpg"))


def test_ensure_file_size_within_limit_success(tmp_path: Path):
    file_path = tmp_path / "small.txt"
    file_path.write_text("x" * 100)
    ensure_file_size_within_limit(file_path, max_size_mb=1)  # should not raise


def test_ensure_file_size_within_limit_rejects_oversized_file(tmp_path: Path):
    file_path = tmp_path / "big.txt"
    file_path.write_bytes(b"0" * (2 * 1024 * 1024))  # 2 MB
    with pytest.raises(ValidationError):
        ensure_file_size_within_limit(file_path, max_size_mb=1)
