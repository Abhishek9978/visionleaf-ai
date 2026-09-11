"""Core package — cross-cutting concerns shared by every subpackage.

Purpose:
    Home for functionality that isn't specific to any single pipeline
    stage: the exception hierarchy and logging configuration.

Description:
    Re-exports the most commonly used symbols so callers can write
    `from visionleaf_ai.core import get_logger, VisionLeafError` instead
    of importing from the individual submodules.

Dependencies:
    visionleaf_ai.core.exceptions, visionleaf_ai.core.logging_config

Public functions / classes:
    get_logger(name: str) -> logging.Logger
    VisionLeafError, ConfigurationError, ValidationError
"""

from visionleaf_ai.core.exceptions import (
    AlgorithmExecutionError,
    ConfigurationError,
    ImageLoadError,
    ValidationError,
    VisionLeafError,
)
from visionleaf_ai.core.image_session import ImageMetadata, ImageSession, ProcessingEvent
from visionleaf_ai.core.logging_config import get_logger, setup_logging

__all__ = [
    "VisionLeafError",
    "ConfigurationError",
    "ValidationError",
    "ImageLoadError",
    "AlgorithmExecutionError",
    "get_logger",
    "setup_logging",
    "ImageSession",
    "ImageMetadata",
    "ProcessingEvent",
]
