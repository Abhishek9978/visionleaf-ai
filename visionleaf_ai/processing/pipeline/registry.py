"""AlgorithmRegistry — decorator-based lookup for every pipeline algorithm.

Purpose:
    Let `PipelineEngine` (and, eventually, the UI) look up and
    instantiate an algorithm by string name, without an if/elif chain
    anywhere and without importing algorithm classes by hand.

Description:
    Algorithm modules register themselves with the `@register_algorithm`
    class decorator at class-definition time:

        @register_algorithm("gamma_correction")
        class GammaCorrection(PipelineStage):
            ...

    Adding a new algorithm later means writing one class with one
    decorator — nothing else in the codebase changes. Because
    registration happens at import time, `bootstrap.ensure_registered()`
    imports every algorithm module once so the registry is fully
    populated before anything looks something up in it (see that
    module's docstring for why this indirection is needed).

Dependencies:
    visionleaf_ai.processing.pipeline.stage.PipelineStage;
    visionleaf_ai.core.exceptions.

Public functions/classes:
    register_algorithm(name) -> decorator
    AlgorithmRegistry
        get(name) -> type[PipelineStage]
        create(name, **kwargs) -> PipelineStage
        list_algorithms(category=None) -> list[str]
        get_info(name) -> AlgorithmInfo
"""

from __future__ import annotations

from visionleaf_ai.core.exceptions import ValidationError
from visionleaf_ai.processing.pipeline.stage import AlgorithmInfo, PipelineStage

_REGISTRY: dict[str, type[PipelineStage]] = {}


def register_algorithm(name: str):
    """Class decorator that registers a `PipelineStage` subclass by name.

    Args:
        name: Unique algorithm key, e.g. "gamma_correction". Used by
            `AlgorithmRegistry.get`/`create` and by
            `PipelineEngine.run_by_name`.

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If `name` is already registered to a different
            class (catches accidental duplicate keys at import time).
    """

    def decorator(stage_cls: type[PipelineStage]) -> type[PipelineStage]:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not stage_cls:
            raise ValueError(
                f"Algorithm name '{name}' is already registered to "
                f"{existing.__name__}; cannot also register {stage_cls.__name__}"
            )
        _REGISTRY[name] = stage_cls
        return stage_cls

    return decorator


class AlgorithmRegistry:
    """Lookup and instantiation for registered `PipelineStage` classes.

    This class has no instance state — the registry itself is the
    module-level `_REGISTRY` dict, shared process-wide (analogous to
    `config.get_settings()`'s single cached instance). All methods are
    classmethods so callers don't need to construct an `AlgorithmRegistry()`.
    """

    @classmethod
    def get(cls, name: str) -> type[PipelineStage]:
        """Look up a registered algorithm class by name.

        Args:
            name: Algorithm key, e.g. "clahe".

        Returns:
            The registered `PipelineStage` subclass (not an instance).

        Raises:
            ValidationError: If no algorithm is registered under `name`.
        """
        stage_cls = _REGISTRY.get(name)
        if stage_cls is None:
            raise ValidationError(
                f"Unknown algorithm '{name}'",
                details=f"available: {sorted(_REGISTRY.keys())}",
            )
        return stage_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> PipelineStage:
        """Instantiate a registered algorithm by name.

        Args:
            name: Algorithm key, e.g. "gaussian_blur".
            **kwargs: Forwarded to the algorithm class's constructor
                (typically parameter dataclass fields, e.g.
                `kernel_size=5, sigma=1.2`).

        Returns:
            A new instance of the registered `PipelineStage` subclass.

        Raises:
            ValidationError: If no algorithm is registered under `name`.
        """
        stage_cls = cls.get(name)
        return stage_cls(**kwargs)

    @classmethod
    def list_algorithms(cls, category: str | None = None) -> list[str]:
        """List registered algorithm names, optionally filtered by category.

        Args:
            category: If given, only return algorithms whose
                `stage_key` matches (e.g. "enhancement").

        Returns:
            Sorted list of algorithm names.
        """
        if category is None:
            return sorted(_REGISTRY.keys())
        return sorted(
            name for name, stage_cls in _REGISTRY.items()
            if stage_cls.stage_key == category
        )

    @classmethod
    def get_info(cls, name: str) -> AlgorithmInfo:
        """Return an algorithm's structured documentation without instantiating it.

        Args:
            name: Algorithm key, e.g. "median_filter".

        Returns:
            The `AlgorithmInfo` for that algorithm.

        Raises:
            ValidationError: If no algorithm is registered under `name`.
        """
        stage_cls = cls.get(name)
        return stage_cls().get_info()

    @classmethod
    def _clear_for_testing(cls) -> None:
        """Clear the registry. Test-only — never call from application code."""
        _REGISTRY.clear()
