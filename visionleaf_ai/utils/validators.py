"""Stage-agnostic validation helpers.

Purpose:
    Provide small, reusable validation functions that don't belong to
    any specific pipeline stage (image-specific validation belongs in
    `processing.acquisition`, added in Milestone 3).

Description:
    Currently covers filesystem-path and file-size/extension checks,
    the kind of validation multiple future subpackages (acquisition,
    ml model loading, analytics export) will all need. Every function
    raises `visionleaf_ai.core.exceptions.ValidationError` with a clear
    message on failure rather than returning booleans or None, so
    callers can't accidentally ignore a failed validation.

Dependencies:
    stdlib pathlib; visionleaf_ai.core.exceptions.

Public functions:
    ensure_path_exists(path) -> Path
    ensure_extension_allowed(path, allowed_extensions) -> Path
    ensure_file_size_within_limit(path, max_size_mb) -> Path
"""

from __future__ import annotations

from pathlib import Path

from visionleaf_ai.core.exceptions import ValidationError


def ensure_path_exists(path: str | Path) -> Path:
    """Validate that a filesystem path exists.

    Args:
        path: Path to check (str or Path).

    Returns:
        The path, resolved to an absolute `Path`.

    Raises:
        ValidationError: If the path does not exist.
    """
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise ValidationError("Path does not exist", details=str(resolved))
    return resolved


def ensure_extension_allowed(
    path: str | Path, allowed_extensions: tuple[str, ...] | list[str]
) -> Path:
    """Validate that a file's extension is in an allowed set.

    Args:
        path: File path to check.
        allowed_extensions: Extensions to allow, e.g. (".png", ".jpg").
            Matching is case-insensitive.

    Returns:
        The path, resolved to an absolute `Path`.

    Raises:
        ValidationError: If the extension is not in `allowed_extensions`.
    """
    resolved = Path(path).resolve()
    normalized_allowed = {ext.lower() for ext in allowed_extensions}
    if resolved.suffix.lower() not in normalized_allowed:
        raise ValidationError(
            f"File extension '{resolved.suffix}' is not allowed",
            details=f"allowed: {sorted(normalized_allowed)}",
        )
    return resolved


def ensure_file_size_within_limit(path: str | Path, max_size_mb: float) -> Path:
    """Validate that a file's size does not exceed a maximum, in megabytes.

    Args:
        path: File path to check. Must already exist.
        max_size_mb: Maximum allowed size in megabytes.

    Returns:
        The path, resolved to an absolute `Path`.

    Raises:
        ValidationError: If the file does not exist or exceeds the limit.
    """
    resolved = ensure_path_exists(path)
    size_mb = resolved.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValidationError(
            f"File size {size_mb:.2f} MB exceeds limit of {max_size_mb} MB",
            details=str(resolved),
        )
    return resolved


def ensure_odd_positive_int(value: int, name: str) -> int:
    """Validate that a parameter is a positive odd integer (e.g. a kernel size).

    Args:
        value: The value to check.
        name: Parameter name, used in the error message, e.g. "kernel_size".

    Returns:
        `value`, unchanged, if valid.

    Raises:
        ValidationError: If `value` is not a positive odd integer.
    """
    if not isinstance(value, int) or value <= 0 or value % 2 == 0:
        raise ValidationError(
            f"{name} must be a positive odd integer", details=f"got {value!r}"
        )
    return value


def ensure_kernel_fits_image(kernel_size: int, image_shape: tuple[int, ...]) -> None:
    """Validate that a square kernel is not larger than the image itself.

    Args:
        kernel_size: The kernel's side length in pixels.
        image_shape: The image array's `.shape` (height, width, ...).

    Raises:
        ValidationError: If `kernel_size` exceeds the image's smaller
            dimension.
    """
    height, width = image_shape[0], image_shape[1]
    if kernel_size > min(height, width):
        raise ValidationError(
            f"kernel_size {kernel_size} exceeds image dimensions "
            f"({width}x{height})",
            details="kernel must be no larger than the image's smaller side",
        )


def ensure_threshold_in_range(value: int, name: str = "threshold") -> int:
    """Validate that a threshold value is a valid 8-bit intensity, [0, 255].

    Args:
        value: The value to check.
        name: Parameter name, used in the error message.

    Returns:
        `value`, unchanged, if valid.

    Raises:
        ValidationError: If `value` is outside [0, 255].
    """
    if not isinstance(value, int) or not (0 <= value <= 255):
        raise ValidationError(
            f"{name} must be an integer in [0, 255]", details=f"got {value!r}"
        )
    return value
