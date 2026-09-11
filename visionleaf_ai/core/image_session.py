"""ImageSession — the single source of truth for one image's pipeline state.

Purpose:
    Define the central data model that every pipeline stage (from
    Milestone 3's acquisition through Milestone 9's experiment mode)
    reads from and writes to. This module is the foundation the
    approved architecture calls for: one authoritative image object
    instead of each page/stage inventing its own image storage.

Description:
    `ImageSession` is a plain dataclass with no Streamlit dependency,
    so it — and every stage's logic that operates on it — can be unit
    tested with plain pytest. It holds every intermediate image the
    pipeline produces (original, current, grayscale, binary, segmented,
    ROI), the derived feature vector and prediction, metadata about the
    source image, and an ordered processing history.

    Contract for future modules (Milestones 4-9):
      - Never construct or load an image yourselves — only
        `ImageManager` creates `ImageSession` instances (see
        `processing.acquisition.image_manager`).
      - Read whatever input field you need (typically `current_image`,
        via `session.active_image` below) and write your stage's
        result to its own field (e.g. Segmentation writes
        `segmented_image`, never overwrites `original_image`).
      - Call `session.record_event(...)` after producing a result, so
        the History Panel stays accurate without each stage needing
        its own bookkeeping.

Dependencies:
    numpy (image arrays); stdlib dataclasses, datetime.

Public classes:
    ImageMetadata, ProcessingEvent, ImageSession
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from visionleaf_ai.core.exceptions import ValidationError


@dataclass(frozen=True)
class ImageMetadata:
    """Descriptive information about the source image.

    Attributes:
        filename: Original filename as uploaded, or the sample's name.
        file_format: Image format, e.g. "PNG", "JPEG".
        width: Width in pixels of the (possibly resized) loaded image.
        height: Height in pixels of the (possibly resized) loaded image.
        channels: Number of color channels (1 = grayscale, 3 = RGB).
        file_size_bytes: Size of the original uploaded/sample file, in bytes.
        uploaded_at: When the image was loaded into the session.
        was_resized: Whether ImageManager downscaled the image on load
            because it exceeded the configured maximum dimension.
        source: Where the image came from — "upload" or "sample".
    """

    filename: str
    file_format: str
    width: int
    height: int
    channels: int
    file_size_bytes: int
    uploaded_at: datetime
    was_resized: bool
    source: str


@dataclass(frozen=True)
class ProcessingEvent:
    """One entry in an image's processing timeline.

    Attributes:
        stage_key: Pipeline stage key from `ui.pipeline_stages.STAGES`,
            e.g. "acquisition", "enhancement".
        label: Human-readable label shown in the History Panel, e.g.
            "Image Loaded", "Enhanced".
        timestamp: When this event occurred.
        details: Optional short extra context, e.g. "resized to 4096px".
    """

    stage_key: str
    label: str
    timestamp: datetime
    details: str | None = None


@dataclass(eq=False)
class ImageSession:
    """Single source of truth for one image's full pipeline state.

    `eq=False` is deliberate: dataclass-generated equality would try to
    compare `numpy.ndarray` fields with `==`, which raises instead of
    returning a bool. Sessions are identified by reference (there is
    exactly one authoritative instance, held in
    `st.session_state["image_session"]`), so equality comparison is
    never a meaningful operation here.

    Attributes:
        original_image: The image exactly as loaded (never mutated by
            any later stage — always the ground truth to compare against).
        current_image: The "working" image later stages should read
            from and, if they modify it, write back to. Starts as a
            copy of `original_image` on load.
        grayscale_image: Grayscale conversion of the working image —
            populated by Enhancement/Restoration (Milestone 4) or by
            Thresholding (Milestone 6), whichever runs first.
        binary_image: Raw output of a thresholding stage (Milestone 6).
        segmentation_mask: The mask after morphological cleanup
            (Milestone 6) — `None` until a morphology stage runs.
            Read via `active_mask` (below), which falls back to
            `binary_image` if this isn't set yet.
        segmented_image: Full-size visualization with the background
            masked out (Milestone 6's ROI Masking stage).
        roi_image: The cropped rectangular sub-image (Milestone 6's
            ROI Cropping stage).
        contours: Every contour found by Find Contours (Milestone 6),
            each an (N, 1, 2) NumPy array of point coordinates.
        largest_contour: The single largest contour by area — also
            what Convex Hull (Milestone 6) reads and overwrites.
        bounding_box: A dict describing the region of interest's
            bounding rectangle. Either
            `{"type": "axis_aligned", "x", "y", "w", "h"}` (from
            Bounding Rectangle) or
            `{"type": "min_area_rect", "center", "size", "angle", "box_points"}`
            (from Minimum Area Rectangle) — both stages write the same
            field under a `"type"` discriminator rather than each
            claiming a separate field, since both answer the same
            question ("what's the bounding region?").
        feature_vector: Populated by Feature Extraction (Milestone 7).
        color_features: Dict of RGB/HSV mean, std, and normalized
            histograms — populated by the Color Features stage
            (Milestone 7).
        texture_features: Dict with "glcm" and "lbp" sub-keys — GLCM
            (Haralick) properties and the LBP histogram, populated by
            two separate stages that each update their own sub-key
            without clobbering the other's (Milestone 7).
        shape_features: Dict of contour-derived shape descriptors
            (area, perimeter, circularity, Hu moments, ...) —
            populated by the Shape Features stage (Milestone 7).
        edge_features: Dict of Canny/Sobel/Laplacian edge statistics —
            populated by the Edge Features stage (Milestone 7).
        prediction: Populated by SVM Classification (Milestone 8) —
            e.g. {"label": "Healthy", "probabilities": {...}}.
        metadata: Descriptive info about the loaded image.
        processing_history: Ordered list of every stage that has run.
    """

    original_image: np.ndarray | None = None
    current_image: np.ndarray | None = None
    grayscale_image: np.ndarray | None = None
    binary_image: np.ndarray | None = None
    segmentation_mask: np.ndarray | None = None
    segmented_image: np.ndarray | None = None
    roi_image: np.ndarray | None = None
    contours: list[np.ndarray] | None = None
    largest_contour: np.ndarray | None = None
    bounding_box: dict | None = None
    feature_vector: np.ndarray | None = None
    color_features: dict | None = None
    texture_features: dict | None = None
    shape_features: dict | None = None
    edge_features: dict | None = None
    prediction: dict | None = None
    metadata: ImageMetadata | None = None
    processing_history: list[ProcessingEvent] = field(default_factory=list)

    def is_loaded(self) -> bool:
        """Return whether an image has actually been loaded yet."""
        return self.original_image is not None

    @property
    def active_image(self) -> np.ndarray | None:
        """The image later stages should treat as "the current input".

        Returns:
            `current_image` if set, otherwise `original_image`, otherwise
            `None`. Centralizing this fallback here means every future
            stage reads `session.active_image` instead of separately
            deciding whether to prefer `current_image` or
            `original_image`.
        """
        if self.current_image is not None:
            return self.current_image
        return self.original_image

    @property
    def active_mask(self) -> np.ndarray | None:
        """The mask later stages should treat as "the current mask".

        Returns:
            `segmentation_mask` if set (i.e. morphological cleanup has
            run), otherwise `binary_image` (raw thresholding output),
            otherwise `None`. Directly parallel to `active_image`: it
            lets morphology and contour-finding stages chain onto
            whatever mask exists so far, the same way Enhancement and
            Restoration already chain onto `active_image`.
        """
        if self.segmentation_mask is not None:
            return self.segmentation_mask
        return self.binary_image

    def require_active_image(self) -> np.ndarray:
        """Return `active_image`, guaranteed non-`None`.

        Every `PipelineStage.process()` calls `self.validate(session)`
        first, which already guarantees an image is loaded — but a
        type checker has no way to know that a `validate()` call
        narrows `active_image` from `np.ndarray | None` down to
        `np.ndarray` afterward. This method is the single place that
        narrowing happens explicitly, so `process()` methods can write
        `image = session.require_active_image()` and get a properly
        typed `np.ndarray` back, instead of every call site repeating
        an `assert` or triggering a `mypy` "Item None has no attribute"
        error on the very next line that uses `.shape`, `.copy()`, etc.

        Returns:
            `active_image`, typed as a plain `np.ndarray`.

        Raises:
            ValidationError: If no image is loaded. In correct usage
                (validate-then-process) this should be unreachable —
                it exists as a defensive backstop, not a substitute for
                calling `validate()` first.
        """
        if self.active_image is None:
            raise ValidationError(
                "No image is loaded", details="session.active_image is None"
            )
        return self.active_image

    def require_active_mask(self) -> np.ndarray:
        """Return `active_mask`, guaranteed non-`None`.

        The mask counterpart to `require_active_image()` — see that
        method's docstring for why this narrowing helper exists.

        Returns:
            `active_mask`, typed as a plain `np.ndarray`.

        Raises:
            ValidationError: If no mask exists yet (no thresholding
                stage has run). In correct usage this is a real,
                reachable error — unlike `require_active_image()`,
                callers should expect and handle it, since "run
                Segmentation before Morphology" is a genuine user
                workflow requirement, not just an internal invariant.
        """
        if self.active_mask is None:
            raise ValidationError(
                "No segmentation mask available",
                details="run a thresholding stage first",
            )
        return self.active_mask

    def require_original_image(self) -> np.ndarray:
        """Return `original_image`, guaranteed non-`None`.

        Used by UI code (e.g. the Processing Lab's "Original" pane)
        after already checking `session.is_loaded()` — the same
        narrowing gap `require_active_image()` closes for
        `PipelineStage.process()`, just at the UI layer instead of the
        processing layer.

        Returns:
            `original_image`, typed as a plain `np.ndarray`.

        Raises:
            ValidationError: If no image is loaded.
        """
        if self.original_image is None:
            raise ValidationError(
                "No image is loaded", details="session.original_image is None"
            )
        return self.original_image

    def require_metadata(self) -> ImageMetadata:
        """Return `metadata`, guaranteed non-`None`.

        Returns:
            `metadata`, typed as a plain `ImageMetadata`.

        Raises:
            ValidationError: If no image (and therefore no metadata)
                has been loaded yet.
        """
        if self.metadata is None:
            raise ValidationError(
                "No image metadata available", details="session.metadata is None"
            )
        return self.metadata

    def record_event(self, stage_key: str, label: str, details: str | None = None) -> None:
        """Append an entry to the processing history.

        Args:
            stage_key: Pipeline stage key, e.g. "acquisition".
            label: Human-readable event label, e.g. "Image Loaded".
            details: Optional short extra context.
        """
        self.processing_history.append(
            ProcessingEvent(
                stage_key=stage_key,
                label=label,
                timestamp=datetime.now(),
                details=details,
            )
        )

    def clear_mask_results(self) -> None:
        """Clear every result derived from a specific binary mask.

        Called by thresholding stages before writing a new
        `binary_image`: a fresh threshold invalidates any morphological
        cleanup, contours, bounding box, or ROI images built on the
        *previous* mask — the same cascading-invalidation principle
        `reset_processing` already applies at the whole-session level.
        """
        self.segmentation_mask = None
        self.contours = None
        self.largest_contour = None
        self.bounding_box = None
        self.roi_image = None
        self.segmented_image = None

    def clear_contour_results(self) -> None:
        """Clear results derived from contours, without touching masks.

        Called by morphology stages before writing a new
        `segmentation_mask`: the mask changed, so any previously-found
        contours/bounding box/ROI images are stale — but `binary_image`
        itself (the raw threshold) is untouched, since morphology
        operates on top of it rather than replacing it.
        """
        self.contours = None
        self.largest_contour = None
        self.bounding_box = None
        self.roi_image = None
        self.segmented_image = None

    def clear_roi_results(self) -> None:
        """Clear only the bounding-box/ROI-image results.

        Called by Find Contours before writing new `contours`/
        `largest_contour`: a fresh contour search invalidates any
        bounding box or ROI image computed from the *previous*
        contour, but the contours themselves are about to be replaced
        by this same call, so they aren't cleared here.
        """
        self.bounding_box = None
        self.roi_image = None
        self.segmented_image = None

    def reset_processing(self) -> None:
        """Undo every processing stage, keeping the original image loaded.

        Restores `current_image` from `original_image` and clears every
        field a pipeline stage could have written to (grayscale, binary,
        segmentation mask, contours, bounding box, ROI images, all four
        feature-group dicts, the feature vector, prediction), then
        trims `processing_history` back to only its Acquisition
        ("Image Loaded") events.

        This is distinct from `ImageManager.reset_session()` (Milestone
        3), which discards the image entirely and returns an empty
        session — appropriate for "load a different image." This method
        is for "undo my edits, keep the same photo," which is what a
        Processing Lab reset button needs: the user shouldn't have to
        re-upload just to discard a bad Gamma Correction.
        """
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
        self.grayscale_image = None
        self.binary_image = None
        self.segmentation_mask = None
        self.segmented_image = None
        self.roi_image = None
        self.contours = None
        self.largest_contour = None
        self.bounding_box = None
        self.feature_vector = None
        self.color_features = None
        self.texture_features = None
        self.shape_features = None
        self.edge_features = None
        self.prediction = None
        self.processing_history = [
            event for event in self.processing_history if event.stage_key == "acquisition"
        ]

    @classmethod
    def empty(cls) -> "ImageSession":
        """Construct a fresh, empty session (no image loaded yet).

        Used both as the initial `st.session_state["image_session"]`
        value and by `ImageManager.reset_session()` to discard all
        current results when the user resets or loads a new image.
        """
        return cls()
