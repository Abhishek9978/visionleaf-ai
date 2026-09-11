"""Image acquisition package.

Purpose:
    The only subpackage allowed to load raw image bytes into the
    pipeline. Re-exports `ImageManager` and `get_image_manager` for
    convenient importing.

Description:
    Implemented in Milestone 3 as part of the Image Management System:
    validation, decoding, resizing, and metadata generation all live in
    `image_manager.py`.

Dependencies:
    visionleaf_ai.processing.acquisition.image_manager

Public functions/classes:
    ImageManager, get_image_manager
"""

from visionleaf_ai.processing.acquisition.image_manager import (
    ImageManager,
    get_image_manager,
)

__all__ = ["ImageManager", "get_image_manager"]
