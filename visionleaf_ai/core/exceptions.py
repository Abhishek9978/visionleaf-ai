"""Project-wide exception hierarchy.

Purpose:
    Define a small, specific exception hierarchy so every subpackage
    raises catchable, meaningful errors instead of bare `Exception` or
    silent failures.

Description:
    All custom exceptions in VisionLeaf AI inherit from `VisionLeafError`.
    Callers (e.g. the Streamlit UI) can catch `VisionLeafError` to handle
    any application-specific failure, or catch a specific subclass (e.g.
    `ConfigurationError`) to handle one failure mode precisely.

Dependencies:
    None (stdlib only).

Public functions:
    None — this module exposes classes, not functions.
"""

from __future__ import annotations


class VisionLeafError(Exception):
    """Base class for all VisionLeaf AI application errors.

    Args:
        message: Human-readable description of what went wrong.
        details: Optional extra context (e.g. a file path, a bad value)
            useful for logging/debugging without cluttering the message
            shown to end users.
    """

    def __init__(self, message: str, details: str | None = None) -> None:
        self.message = message
        self.details = details
        full_message = f"{message} ({details})" if details else message
        super().__init__(full_message)


class ConfigurationError(VisionLeafError):
    """Raised when application configuration is missing, malformed, or invalid.

    Examples:
        - `config.yaml` is missing a required key.
        - A configured path does not exist and cannot be created.
        - An environment variable override has the wrong type.
    """


class ValidationError(VisionLeafError):
    """Raised when input validation fails for a value, file, or parameter.

    This is intentionally generic and reused across subpackages (image
    validation in Milestone 3, parameter validation in later milestones)
    rather than each subpackage defining its own duplicate error type.
    Use this for validation that runs *before* attempting to decode or
    process a file (bad extension, oversized file). Use `ImageLoadError`
    for failures that occur while actually decoding image *content*.
    """


class ImageLoadError(VisionLeafError):
    """Raised when image bytes cannot be decoded into a usable image.

    Distinct from `ValidationError`: validation checks (extension, file
    size) happen before an image is ever opened and catch cheap,
    predictable problems. `ImageLoadError` covers the case where a file
    passed those checks (e.g. has a `.png` extension) but its contents
    are corrupt, truncated, or not actually a decodable image.
    """


class AlgorithmExecutionError(VisionLeafError):
    """Raised when a pipeline algorithm fails while actually running.

    Distinct from `ValidationError`: validation catches bad input
    *before* any processing is attempted (missing image, out-of-range
    parameter, wrong dtype). `AlgorithmExecutionError` covers the case
    where input passed validation but the underlying operation (e.g. an
    OpenCV call) still failed at runtime — so callers can tell "you gave
    me something invalid" apart from "something went wrong while I was
    working on valid input."
    """
