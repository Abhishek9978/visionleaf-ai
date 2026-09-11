"""Tests for visionleaf_ai.core (exceptions and logging).

Purpose:
    Verify the exception hierarchy behaves as documented and that
    logging configuration is idempotent and produces a working logger.

Dependencies:
    pytest; visionleaf_ai.core.
"""

from __future__ import annotations

import logging

from visionleaf_ai.core.exceptions import (
    ConfigurationError,
    ValidationError,
    VisionLeafError,
)
from visionleaf_ai.core.logging_config import get_logger, setup_logging


def test_configuration_error_is_a_visionleaf_error():
    assert issubclass(ConfigurationError, VisionLeafError)


def test_validation_error_is_a_visionleaf_error():
    assert issubclass(ValidationError, VisionLeafError)


def test_error_message_includes_details():
    err = VisionLeafError("Something failed", details="extra context")
    assert "Something failed" in str(err)
    assert "extra context" in str(err)


def test_error_message_without_details_is_clean():
    err = VisionLeafError("Something failed")
    assert str(err) == "Something failed"


def test_setup_logging_is_idempotent():
    setup_logging(force=True)
    logger = logging.getLogger("visionleaf_ai")
    handler_count_first = len(logger.handlers)

    setup_logging()  # should be a no-op the second time
    assert len(logger.handlers) == handler_count_first


def test_get_logger_returns_named_logger():
    logger = get_logger("visionleaf_ai.some.module")
    assert logger.name == "visionleaf_ai.some.module"
