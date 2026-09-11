"""Milestone 1 verification script.

Purpose:
    A tiny, standalone script (not part of the Streamlit app, which
    arrives in Milestone 2) that proves the foundation — config,
    logging, exceptions, utils — actually works together.

Description:
    Run this after `pip install -r requirements.txt` to confirm your
    environment and project layout are set up correctly before moving
    on to Milestone 2.

Dependencies:
    visionleaf_ai.config, visionleaf_ai.core, visionleaf_ai.utils

Public functions:
    main() -> None
"""

from __future__ import annotations

from visionleaf_ai.config import get_settings
from visionleaf_ai.core import get_logger
from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.utils import ensure_extension_allowed


def main() -> None:
    """Load settings, log a few messages, and exercise a validator."""
    settings = get_settings()
    logger = get_logger(__name__)

    logger.info("Starting %s v%s", settings.app.name, settings.app.version)
    logger.info("Debug mode: %s", settings.app.debug)
    logger.info("Assets directory: %s", settings.paths.assets_dir)
    logger.info("Logs directory: %s", settings.paths.logs_dir)
    logger.info(
        "Allowed image extensions: %s", settings.image.allowed_extensions
    )

    try:
        ensure_extension_allowed("leaf.txt", settings.image.allowed_extensions)
    except ValidationError as exc:
        logger.warning("Validation correctly rejected a bad file: %s", exc)

    logger.info("Milestone 1 foundation verified successfully.")
    print("\n✅ VisionLeaf AI foundation is set up correctly.")
    print(f"   Check {settings.paths.logs_dir / settings.logging.log_filename} for log output.")


if __name__ == "__main__":
    main()
