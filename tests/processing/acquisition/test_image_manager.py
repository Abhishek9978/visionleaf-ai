"""Tests for visionleaf_ai.processing.acquisition.image_manager.

Purpose:
    Verify loading, validation, resizing, metadata generation, and the
    sample gallery — the whole Milestone 3 loading contract — using
    small synthetic images generated in-memory with Pillow, no real
    dataset needed.

Dependencies:
    pytest; Pillow; numpy; visionleaf_ai.processing.acquisition.image_manager.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visionleaf_ai.config.settings import ImageSettings
from visionleaf_ai.core.exceptions import ImageLoadError, ValidationError
from visionleaf_ai.core.image_session import ImageSession
from visionleaf_ai.processing.acquisition.image_manager import ImageManager


def _png_bytes(width: int = 64, height: int = 48, color: tuple = (10, 120, 60)) -> bytes:
    """Build in-memory PNG bytes for a solid-color test image."""
    image = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def default_settings() -> ImageSettings:
    return ImageSettings(
        max_upload_size_mb=10,
        allowed_extensions=(".png", ".jpg", ".jpeg"),
        max_dimension_px=4096,
    )


@pytest.fixture
def small_max_dim_settings() -> ImageSettings:
    """Settings with a tiny max dimension, to cheaply exercise resizing."""
    return ImageSettings(
        max_upload_size_mb=10,
        allowed_extensions=(".png", ".jpg", ".jpeg"),
        max_dimension_px=32,
    )


def test_load_from_upload_success(tmp_path: Path, default_settings: ImageSettings):
    manager = ImageManager(image_settings=default_settings, sample_dir=tmp_path)
    session = manager.load_from_upload(_png_bytes(64, 48), "leaf.png")

    assert session.is_loaded()
    assert session.original_image.shape[:2] == (48, 64)
    assert session.current_image is not None
    assert session.metadata.filename == "leaf.png"
    assert session.metadata.file_format == "PNG"
    assert session.metadata.width == 64
    assert session.metadata.height == 48
    assert session.metadata.source == "upload"
    assert session.metadata.was_resized is False


def test_load_from_upload_records_history_event(tmp_path: Path, default_settings: ImageSettings):
    manager = ImageManager(image_settings=default_settings, sample_dir=tmp_path)
    session = manager.load_from_upload(_png_bytes(), "leaf.png")

    assert len(session.processing_history) == 1
    assert session.processing_history[0].label == "Image Loaded"
    assert session.processing_history[0].stage_key == "acquisition"


def test_load_from_upload_rejects_disallowed_extension(
    tmp_path: Path, default_settings: ImageSettings
):
    manager = ImageManager(image_settings=default_settings, sample_dir=tmp_path)
    with pytest.raises(ValidationError):
        manager.load_from_upload(_png_bytes(), "leaf.exe")


def test_load_from_upload_rejects_oversized_file(tmp_path: Path):
    tiny_limit_settings = ImageSettings(
        max_upload_size_mb=0.0001,  # ~100 bytes
        allowed_extensions=(".png",),
        max_dimension_px=4096,
    )
    manager = ImageManager(image_settings=tiny_limit_settings, sample_dir=tmp_path)
    with pytest.raises(ValidationError):
        manager.load_from_upload(_png_bytes(64, 48), "leaf.png")


def test_load_from_upload_rejects_corrupt_image_bytes(
    tmp_path: Path, default_settings: ImageSettings
):
    manager = ImageManager(image_settings=default_settings, sample_dir=tmp_path)
    with pytest.raises(ImageLoadError):
        manager.load_from_upload(b"not a real image", "leaf.png")


def test_resize_downscales_oversized_image(
    tmp_path: Path, small_max_dim_settings: ImageSettings
):
    manager = ImageManager(image_settings=small_max_dim_settings, sample_dir=tmp_path)
    session = manager.load_from_upload(_png_bytes(200, 100), "wide.png")

    assert session.metadata.was_resized is True
    assert max(session.metadata.width, session.metadata.height) <= 32
    # Aspect ratio preserved (200:100 == 2:1)
    assert session.metadata.width == 2 * session.metadata.height
    assert "resized" in session.processing_history[0].details


def test_resize_leaves_small_image_unchanged(
    tmp_path: Path, default_settings: ImageSettings
):
    manager = ImageManager(image_settings=default_settings, sample_dir=tmp_path)
    session = manager.load_from_upload(_png_bytes(64, 48), "leaf.png")
    assert session.metadata.was_resized is False
    assert session.metadata.width == 64
    assert session.metadata.height == 48


def test_load_sample_success(tmp_path: Path, default_settings: ImageSettings):
    sample_path = tmp_path / "sample_leaf.png"
    sample_path.write_bytes(_png_bytes())

    manager = ImageManager(image_settings=default_settings, sample_dir=tmp_path)
    session = manager.load_sample(sample_path)

    assert session.is_loaded()
    assert session.metadata.source == "sample"
    assert session.metadata.filename == "sample_leaf.png"


def test_load_sample_missing_file_raises(tmp_path: Path, default_settings: ImageSettings):
    manager = ImageManager(image_settings=default_settings, sample_dir=tmp_path)
    with pytest.raises(ValidationError):
        manager.load_sample(tmp_path / "does_not_exist.png")


def test_list_sample_images_returns_only_allowed_extensions(
    tmp_path: Path, default_settings: ImageSettings
):
    (tmp_path / "a.png").write_bytes(_png_bytes())
    (tmp_path / "b.jpg").write_bytes(_png_bytes())
    (tmp_path / "notes.txt").write_text("not an image")

    manager = ImageManager(image_settings=default_settings, sample_dir=tmp_path)
    results = manager.list_sample_images()

    assert {p.name for p in results} == {"a.png", "b.jpg"}


def test_list_sample_images_empty_dir_returns_empty_list(
    tmp_path: Path, default_settings: ImageSettings
):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    manager = ImageManager(image_settings=default_settings, sample_dir=empty_dir)
    assert manager.list_sample_images() == []


def test_list_sample_images_missing_dir_returns_empty_list(
    tmp_path: Path, default_settings: ImageSettings
):
    manager = ImageManager(
        image_settings=default_settings, sample_dir=tmp_path / "does_not_exist"
    )
    assert manager.list_sample_images() == []


def test_reset_session_returns_fresh_empty_session(
    tmp_path: Path, default_settings: ImageSettings
):
    manager = ImageManager(image_settings=default_settings, sample_dir=tmp_path)
    fresh = manager.reset_session()
    assert fresh.is_loaded() is False
    assert fresh.processing_history == []


def test_grayscale_upload_reports_one_channel(tmp_path: Path, default_settings: ImageSettings):
    manager = ImageManager(image_settings=default_settings, sample_dir=tmp_path)
    grayscale_image = Image.new("L", (40, 30), 128)
    buffer = io.BytesIO()
    grayscale_image.save(buffer, format="PNG")

    session = manager.load_from_upload(buffer.getvalue(), "gray.png")
    assert session.metadata.channels == 1
    assert isinstance(session.original_image, np.ndarray)


def test_export_image_returns_valid_png_bytes(tmp_path: Path, default_settings: ImageSettings):
    manager = ImageManager(image_settings=default_settings, sample_dir=tmp_path)
    session = manager.load_from_upload(_png_bytes(32, 24), "leaf.png")

    exported = manager.export_image(session, image_format="PNG")

    assert exported.startswith(b"\x89PNG\r\n\x1a\n")
    round_tripped = Image.open(io.BytesIO(exported))
    assert round_tripped.size == (32, 24)


def test_export_image_supports_jpeg(tmp_path: Path, default_settings: ImageSettings):
    manager = ImageManager(image_settings=default_settings, sample_dir=tmp_path)
    session = manager.load_from_upload(_png_bytes(32, 24), "leaf.png")

    exported = manager.export_image(session, image_format="JPEG")

    round_tripped = Image.open(io.BytesIO(exported))
    assert round_tripped.format == "JPEG"


def test_export_image_raises_when_no_image_loaded(tmp_path: Path, default_settings: ImageSettings):
    manager = ImageManager(image_settings=default_settings, sample_dir=tmp_path)
    with pytest.raises(ValidationError):
        manager.export_image(ImageSession.empty())


def test_export_image_raises_on_bad_format(tmp_path: Path, default_settings: ImageSettings):
    manager = ImageManager(image_settings=default_settings, sample_dir=tmp_path)
    session = manager.load_from_upload(_png_bytes(32, 24), "leaf.png")
    with pytest.raises(ImageLoadError):
        manager.export_image(session, image_format="NOT_A_REAL_FORMAT")
