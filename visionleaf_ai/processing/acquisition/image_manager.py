"""ImageManager — the only service that loads images into the pipeline.

Purpose:
    Own every step of getting an image into a usable `ImageSession`:
    validating it, decoding it, resizing it if it's too large, building
    its metadata, and recording the initial "Image Loaded" history
    event. No other module should open image bytes directly.

Description:
    `ImageManager` is deliberately Streamlit-agnostic (it only depends
    on PIL, NumPy, and `visionleaf_ai.core`/`config`), so its logic —
    including resizing and validation — is fully unit-testable with
    plain pytest, independent of the UI layer.

Dependencies:
    Pillow (PIL.Image), NumPy; visionleaf_ai.config, visionleaf_ai.core.

Public functions/classes:
    ImageManager
        load_from_upload(file_bytes, filename) -> ImageSession
        load_sample(sample_path) -> ImageSession
        list_sample_images() -> list[Path]
        reset_session() -> ImageSession
    get_image_manager() -> ImageManager
        Module-level convenience accessor using default settings.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from visionleaf_ai.config.settings import ImageSettings, get_settings
from visionleaf_ai.core.exceptions import ImageLoadError, ValidationError
from visionleaf_ai.core.image_session import ImageMetadata, ImageSession
from visionleaf_ai.core.logging_config import get_logger
from visionleaf_ai.utils.validators import ensure_extension_allowed

logger = get_logger(__name__)

_STAGE_KEY = "acquisition"


class ImageManager:
    """Loads, validates, resizes, and describes images for the pipeline.

    Args:
        image_settings: Constraints to validate against (allowed
            extensions, max upload size, max dimension). Defaults to
            `get_settings().image` — pass an explicit `ImageSettings`
            in tests to exercise validation/resizing cheaply without
            depending on the real config file.
        sample_dir: Directory to search for sample gallery images.
            Defaults to `get_settings().paths.sample_images_dir`.
    """

    def __init__(
        self,
        image_settings: ImageSettings | None = None,
        sample_dir: Path | None = None,
    ) -> None:
        if image_settings is None or sample_dir is None:
            defaults = get_settings()
            image_settings = image_settings or defaults.image
            sample_dir = sample_dir or defaults.paths.sample_images_dir
        self._image_settings = image_settings
        self._sample_dir = sample_dir

    @property
    def image_settings(self) -> ImageSettings:
        """The validation constraints this manager enforces (read-only).

        Exposed so callers (e.g. the upload panel) can display accurate
        hints — allowed extensions, size limits — without reaching into
        the manager's private state.
        """
        return self._image_settings

    def load_from_upload(self, file_bytes: bytes, filename: str) -> ImageSession:
        """Validate, decode, and build a session from an uploaded file.

        Args:
            file_bytes: Raw file content, as provided by
                `st.file_uploader(...).getvalue()`.
            filename: Original filename, used for extension validation
                and metadata (e.g. "leaf_sample.jpg").

        Returns:
            A new `ImageSession` with `original_image`, `current_image`,
            `metadata` populated and one "Image Loaded" history event.

        Raises:
            ValidationError: If the extension is disallowed or the file
                exceeds the configured maximum size.
            ImageLoadError: If the file passes validation but its
                content cannot be decoded as an image.
        """
        self._validate_upload(filename, len(file_bytes))
        return self._build_session(file_bytes, filename, source="upload")

    def load_sample(self, sample_path: str | Path) -> ImageSession:
        """Load one of the built-in sample gallery images.

        Args:
            sample_path: Path to a sample image file, typically one
                returned by `list_sample_images()`.

        Returns:
            A new `ImageSession`, as with `load_from_upload`.

        Raises:
            ValidationError: If the path does not exist or has a
                disallowed extension.
            ImageLoadError: If the file cannot be decoded as an image.
        """
        path = Path(sample_path)
        if not path.is_file():
            raise ValidationError("Sample image not found", details=str(path))
        ensure_extension_allowed(path, self._image_settings.allowed_extensions)
        file_bytes = path.read_bytes()
        return self._build_session(file_bytes, path.name, source="sample")

    def list_sample_images(self) -> list[Path]:
        """List available sample gallery images, sorted by filename.

        Returns:
            Paths to every file in the sample directory whose extension
            is in the allowed set. Empty list if the directory has no
            matching files (never raises for an empty gallery).
        """
        if not self._sample_dir.is_dir():
            return []
        allowed = {ext.lower() for ext in self._image_settings.allowed_extensions}
        return sorted(
            p for p in self._sample_dir.iterdir() if p.suffix.lower() in allowed
        )

    def reset_session(self) -> ImageSession:
        """Return a brand-new, empty `ImageSession`.

        This is the entire "reset" operation: because every downstream
        field (segmented_image, feature_vector, prediction, ...) lives
        on the session object, discarding it and starting a fresh one
        clears the whole pipeline's results in one step.
        """
        logger.info("Image session reset")
        return ImageSession.empty()

    def export_image(self, session: ImageSession, image_format: str = "PNG") -> bytes:
        """Encode the session's active image to bytes for download.

        The symmetric counterpart to `load_from_upload`/`load_sample`:
        just as those are the only path images enter the pipeline
        through, this is the only path an image leaves it through — the
        UI never encodes image bytes itself.

        Args:
            session: The `ImageSession` whose `active_image` should be
                exported.
            image_format: Output format Pillow understands, e.g. "PNG"
                or "JPEG".

        Returns:
            Encoded image bytes, ready for `st.download_button` or
            writing to disk.

        Raises:
            ValidationError: If no image is loaded.
            ImageLoadError: If encoding fails for an unexpected reason
                (e.g. an unsupported format string).
        """
        if session.active_image is None:
            raise ValidationError(
                "Cannot export: no image is loaded", details="session.active_image is None"
            )
        try:
            pil_image = Image.fromarray(session.active_image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format=image_format)
            return buffer.getvalue()
        except (ValueError, OSError, KeyError) as exc:
            raise ImageLoadError(
                f"Failed to export image as {image_format}", details=str(exc)
            ) from exc

    # -- internal helpers ---------------------------------------------------

    def _validate_upload(self, filename: str, file_size_bytes: int) -> None:
        """Run extension and size checks before attempting to decode."""
        ensure_extension_allowed(filename, self._image_settings.allowed_extensions)
        size_mb = file_size_bytes / (1024 * 1024)
        if size_mb > self._image_settings.max_upload_size_mb:
            raise ValidationError(
                f"File size {size_mb:.2f} MB exceeds limit of "
                f"{self._image_settings.max_upload_size_mb} MB",
                details=filename,
            )

    def _build_session(self, file_bytes: bytes, filename: str, source: str) -> ImageSession:
        """Decode bytes, resize if needed, build metadata, and assemble a session."""
        try:
            with Image.open(io.BytesIO(file_bytes)) as pil_image:
                pil_image.load()  # force full decode now, not lazily later
                image_format = pil_image.format or Path(filename).suffix.lstrip(".").upper()
                rgb_image: Image.Image = (
                    pil_image.convert("RGB") if pil_image.mode not in ("RGB", "L") else pil_image
                )
                original_dimensions = rgb_image.size
                resized_image, was_resized = self._resize_if_needed(rgb_image)
        except UnidentifiedImageError as exc:
            raise ImageLoadError(
                "File could not be decoded as an image", details=filename
            ) from exc
        except OSError as exc:
            raise ImageLoadError(
                "File is corrupt or truncated", details=filename
            ) from exc

        image_array = np.array(resized_image)
        channels = 1 if image_array.ndim == 2 else image_array.shape[2]

        metadata = ImageMetadata(
            filename=filename,
            file_format=image_format,
            width=resized_image.width,
            height=resized_image.height,
            channels=channels,
            file_size_bytes=len(file_bytes),
            uploaded_at=datetime.now(),
            was_resized=was_resized,
            source=source,
        )

        session = ImageSession(
            original_image=image_array.copy(),
            current_image=image_array.copy(),
            metadata=metadata,
        )

        detail = None
        if was_resized:
            detail = f"resized from {original_dimensions[0]}x{original_dimensions[1]} to {resized_image.width}x{resized_image.height}"
        session.record_event(_STAGE_KEY, "Image Loaded", details=detail)

        logger.info(
            "Loaded image '%s' (%dx%d, %s, source=%s)%s",
            filename,
            resized_image.width,
            resized_image.height,
            image_format,
            source,
            f" [{detail}]" if detail else "",
        )
        return session

    def _resize_if_needed(self, pil_image: Image.Image) -> tuple[Image.Image, bool]:
        """Downscale an image if it exceeds the configured max dimension.

        Args:
            pil_image: Decoded image to check.

        Returns:
            A `(image, was_resized)` tuple. `image` is unchanged if no
            resize was needed, preserving aspect ratio otherwise.
        """
        max_dim = self._image_settings.max_dimension_px
        width, height = pil_image.size
        if max(width, height) <= max_dim:
            return pil_image, False

        scale = max_dim / max(width, height)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        return resized, True


def get_image_manager() -> ImageManager:
    """Return an `ImageManager` configured from the application's settings.

    Convenience accessor for pages — equivalent to
    `ImageManager()` but named for readability at call sites.
    """
    return ImageManager()
