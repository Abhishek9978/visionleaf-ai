"""PipelineStage — the interface every DIP algorithm implements.

Purpose:
    Define the abstract contract that makes every processing algorithm
    (Brightness, CLAHE, Gaussian Blur, ...) interchangeable from the
    engine's point of view: same four methods, regardless of what the
    algorithm actually does to pixels.

Description:
    A concrete stage implements `process()` (the actual transformation),
    `get_name()`, `get_description()`, and `get_info()` (structured
    documentation that will power Learning Mode). `validate()` has a
    sensible default (require a loaded image) that subclasses extend
    with `super().validate(session)` plus their own parameter/dimension
    checks — see any concrete algorithm (e.g.
    `processing.enhancement.gamma_correction`) for the pattern.

    Each concrete `process()` is expected to be self-contained: it
    validates, transforms, updates the session, records a processing
    history event, and logs — all inside the method. This makes every
    algorithm independently correct and testable with plain pytest,
    with or without `PipelineEngine` involved. `PipelineEngine` (see
    `pipeline.engine`) adds an *outer* layer — overall timing, uniform
    error handling, and multi-stage cancellation — on top of that.

Dependencies:
    abc, dataclasses, numpy; visionleaf_ai.core.image_session.

Public classes:
    AlgorithmInfo, PipelineStage
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.core.image_session import ImageSession


@dataclass(frozen=True)
class AlgorithmInfo:
    """Structured documentation for one algorithm.

    This is the machine-readable counterpart to the algorithm's Python
    docstring: the docstring is for developers reading the source, this
    is for end users via a future Learning Mode panel.

    Attributes:
        name: Human-readable algorithm name, e.g. "Gamma Correction".
        category: "enhancement", "restoration", "segmentation",
            "morphology", or "roi" (matches `stage_key`).
        purpose: One or two sentences on what problem this solves.
        theory: A short explanation of how it works.
        math_intuition: The core formula or mathematical idea, in
            plain terms (not necessarily LaTeX — this is prose).
        advantages: Short bullet-style strings.
        limitations: Short bullet-style strings.
        complexity: Optional Big-O or informal complexity note.
        working_principle: Optional step-by-step description of the
            algorithm's procedure, distinct from `theory`'s conceptual
            explanation — added in Milestone 6, defaults to empty so
            every Milestone 4/5 algorithm's `get_info()` still works
            unchanged.
        typical_applications: Optional short bullet-style strings
            naming real use cases (e.g. "Separating a leaf from a
            plain background"). Added in Milestone 6.
        opencv_reference: Optional name of the underlying OpenCV
            function, e.g. "cv2.adaptiveThreshold". Added in Milestone 6.
    """

    name: str
    category: str
    purpose: str
    theory: str
    math_intuition: str
    advantages: tuple[str, ...]
    limitations: tuple[str, ...]
    complexity: str | None = None
    working_principle: str = ""
    typical_applications: tuple[str, ...] = ()
    opencv_reference: str | None = None


class PipelineStage(ABC):
    """Abstract base class for every DIP pipeline algorithm.

    Subclasses must implement `process()`, `get_name()`,
    `get_description()`, and `get_info()`. `validate()` may be
    extended but does not have to be overridden if the default
    (image-loaded) check is sufficient.
    """

    #: Which pipeline stage this algorithm belongs to (e.g.
    #: "enhancement", "restoration"). Plain string, not an import from
    #: the UI layer — processing must never depend on ui.
    stage_key: str = "unspecified"

    def validate(self, session: ImageSession) -> None:
        """Validate that `session` is fit to be processed by this stage.

        Default check: an image must actually be loaded. Subclasses
        should call `super().validate(session)` first, then add their
        own parameter and dimension checks.

        Args:
            session: The `ImageSession` about to be processed.

        Raises:
            ValidationError: If no image is loaded, or (in a subclass
                override) if a parameter or image dimension is invalid.
        """
        if session.active_image is None:
            raise ValidationError(
                f"{self.get_name()} requires a loaded image",
                details="session.active_image is None",
            )

    @abstractmethod
    def process(self, session: ImageSession) -> ImageSession:
        """Run this algorithm against `session` and return the updated session.

        Implementations must: call `self.validate(session)` first,
        perform the transformation, write the result to the
        appropriate `ImageSession` field, call
        `session.record_event(...)`, log execution, and return
        `session`.

        Args:
            session: The `ImageSession` to process. Mutated in place
                and also returned, matching the architecture's
                "UI -> Engine -> Stage -> ImageSession -> Updated
                ImageSession" data flow.

        Returns:
            The same `ImageSession`, updated.

        Raises:
            ValidationError: If `validate()` fails.
            AlgorithmExecutionError: If the transformation itself fails
                on otherwise-valid input.
        """

    @abstractmethod
    def get_name(self) -> str:
        """Return this algorithm's human-readable name."""

    @abstractmethod
    def get_description(self) -> str:
        """Return a one-sentence description of what this algorithm does."""

    @abstractmethod
    def get_info(self) -> AlgorithmInfo:
        """Return this algorithm's structured documentation (for Learning Mode)."""
