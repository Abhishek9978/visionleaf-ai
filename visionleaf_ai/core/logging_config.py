"""Central logging configuration.

Purpose:
    Configure the `visionleaf_ai` root logger exactly once per process,
    so every module can simply do `logging.getLogger(__name__)` and
    automatically inherit console + rotating file handlers.

Description:
    `setup_logging()` reads levels/paths/format from `Settings`
    (visionleaf_ai.config) and attaches:
      - a console (stream) handler, always
      - a rotating file handler, if `logging.log_to_file` is true
    It is idempotent: calling it multiple times will not duplicate
    handlers, so Streamlit's rerun-on-every-interaction model (which
    could otherwise re-execute module-level setup code repeatedly)
    won't accumulate duplicate log lines.

Dependencies:
    stdlib logging, logging.handlers; visionleaf_ai.config.

Public functions:
    setup_logging(force: bool = False) -> None
        Configure the root `visionleaf_ai` logger.
    get_logger(name: str) -> logging.Logger
        Convenience wrapper ensuring logging is configured before
        returning a named logger.

Note on import structure:
    `get_settings` is imported lazily, inside `setup_logging()`, rather
    than at module level. `config.settings` imports
    `core.exceptions`, and Python fully executes a package's
    `__init__.py` (which imports this module) before the submodule
    that triggered the import finishes — importing `visionleaf_ai.config`
    at module level here would create a circular import. Deferring the
    import to call time breaks the cycle without weakening the module's
    public API.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

_ROOT_LOGGER_NAME = "visionleaf_ai"
_configured = False


def setup_logging(force: bool = False) -> None:
    """Configure the `visionleaf_ai` logger with console + file handlers.

    Args:
        force: If True, reconfigure even if already configured this
            process (removes and re-adds handlers). Useful in tests.

    Notes:
        Safe to call multiple times — a no-op after the first call
        unless `force=True`.
    """
    from visionleaf_ai.config import get_settings  # local import: see module note above

    global _configured
    if _configured and not force:
        return

    settings = get_settings()
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(settings.logging.level.upper())
    logger.propagate = False

    formatter = logging.Formatter(settings.logging.format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if settings.logging.log_to_file:
        log_path = settings.paths.logs_dir / settings.logging.log_filename
        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=settings.logging.max_bytes,
            backupCount=settings.logging.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _configured = True
    logger.debug("Logging configured (level=%s)", settings.logging.level)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring root configuration has run first.

    Args:
        name: Typically `__name__` of the calling module, so log lines
            are traceable to their source (e.g.
            `visionleaf_ai.processing.enhancement.filters`).

    Returns:
        A configured `logging.Logger` instance that writes through the
        `visionleaf_ai` root logger's handlers.
    """
    setup_logging()
    return logging.getLogger(name)
