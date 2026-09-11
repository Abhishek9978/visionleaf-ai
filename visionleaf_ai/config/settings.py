"""Typed application settings loaded from config.yaml.

Purpose:
    Provide a single, validated, typed entry point (`get_settings`) for
    every other module to read configuration from — no module should
    open or parse config.yaml directly.

Description:
    Settings are represented as nested, frozen dataclasses so values are
    immutable after load (preventing accidental mutation at runtime) and
    fully type-hinted (so IDEs/type-checkers catch typos like
    `settings.aap.debug`). Values are read from `config.yaml` at the
    project root, then optionally overridden by environment variables of
    the form `VISIONLEAF_SECTION__KEY` (double underscore separates
    nesting), which is useful for deployment environments (e.g. CI,
    Docker) where editing a YAML file isn't practical.

    The loader is cached with `functools.lru_cache` so the file is only
    read and parsed once per process; call `get_settings.cache_clear()`
    in tests if you need to reload after changing config.yaml.

Dependencies:
    PyYAML (yaml), stdlib (dataclasses, functools, os, pathlib).

Public functions:
    get_settings() -> Settings
        Load (once, cached) and return the validated application settings.
    find_project_root() -> Path
        Locate the project root directory (where config.yaml lives).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from visionleaf_ai.core.exceptions import ConfigurationError

_CONFIG_FILENAME = "config.yaml"
_ENV_PREFIX = "VISIONLEAF_"


@dataclass(frozen=True)
class AppSettings:
    """Application identity and top-level behavior flags."""

    name: str
    version: str
    debug: bool
    learning_mode_default: bool


@dataclass(frozen=True)
class PathSettings:
    """Filesystem locations used across the application, resolved to
    absolute paths relative to the project root."""

    assets_dir: Path
    sample_images_dir: Path
    models_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class LoggingSettings:
    """Logging configuration consumed by core.logging_config."""

    level: str
    log_to_file: bool
    log_filename: str
    max_bytes: int
    backup_count: int
    format: str


@dataclass(frozen=True)
class ImageSettings:
    """Constraints applied to uploaded/processed images."""

    max_upload_size_mb: int
    allowed_extensions: tuple[str, ...]
    max_dimension_px: int


@dataclass(frozen=True)
class Settings:
    """Top-level, fully-validated application settings.

    Attributes:
        app: Application identity/behavior flags.
        paths: Resolved filesystem locations.
        logging: Logging configuration.
        image: Image validation constraints.
        project_root: Absolute path to the project root directory.
    """

    app: AppSettings
    paths: PathSettings
    logging: LoggingSettings
    image: ImageSettings
    project_root: Path = field(repr=False)


def find_project_root() -> Path:
    """Locate the project root by walking up from this file until
    `config.yaml` is found.

    Returns:
        Absolute path to the directory containing `config.yaml`.

    Raises:
        ConfigurationError: If no `config.yaml` is found in any parent
            directory (project layout has been moved or corrupted).
    """
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / _CONFIG_FILENAME).is_file():
            return candidate
    raise ConfigurationError(
        f"Could not locate {_CONFIG_FILENAME} in any parent directory.",
        details=f"started search at {current}",
    )


def _env_override(section: str, key: str, default: object) -> object:
    """Return an environment variable override for `section.key`, cast
    to the type of `default`, or `default` if no override is set."""
    env_name = f"{_ENV_PREFIX}{section.upper()}__{key.upper()}"
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigurationError(
                f"Environment override {env_name} must be an integer",
                details=f"got {raw!r}",
            ) from exc
    return raw


def _require(section: dict, key: str, section_name: str) -> object:
    """Fetch `key` from `section`, raising ConfigurationError if absent."""
    if key not in section:
        raise ConfigurationError(
            f"Missing required config key '{key}' in section '{section_name}'"
        )
    return section[key]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load, validate, and return the application settings (cached).

    Reads `config.yaml` from the project root, applies any
    `VISIONLEAF_*` environment variable overrides, resolves all paths
    to absolute paths, and creates directories declared under `paths`
    if they do not already exist.

    Returns:
        A fully populated, immutable `Settings` instance.

    Raises:
        ConfigurationError: If `config.yaml` is missing, malformed, or
            missing required keys.
    """
    root = find_project_root()
    config_path = root / _CONFIG_FILENAME

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Failed to parse {config_path}", details=str(exc)
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"{config_path} did not parse to a mapping at the top level"
        )

    try:
        app_raw = _require(raw, "app", "root")
        paths_raw = _require(raw, "paths", "root")
        logging_raw = _require(raw, "logging", "root")
        image_raw = _require(raw, "image", "root")

        app = AppSettings(
            name=str(_require(app_raw, "name", "app")),
            version=str(_require(app_raw, "version", "app")),
            debug=bool(_env_override("app", "debug", app_raw.get("debug", False))),
            learning_mode_default=bool(
                app_raw.get("learning_mode_default", False)
            ),
        )

        paths = PathSettings(
            assets_dir=(root / _require(paths_raw, "assets_dir", "paths")).resolve(),
            sample_images_dir=(
                root / _require(paths_raw, "sample_images_dir", "paths")
            ).resolve(),
            models_dir=(root / _require(paths_raw, "models_dir", "paths")).resolve(),
            logs_dir=(root / _require(paths_raw, "logs_dir", "paths")).resolve(),
        )

        logging_settings = LoggingSettings(
            level=str(
                _env_override(
                    "logging", "level", logging_raw.get("level", "INFO")
                )
            ),
            log_to_file=bool(logging_raw.get("log_to_file", True)),
            log_filename=str(logging_raw.get("log_filename", "visionleaf.log")),
            max_bytes=int(logging_raw.get("max_bytes", 1_048_576)),
            backup_count=int(logging_raw.get("backup_count", 3)),
            format=str(
                logging_raw.get(
                    "format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
                )
            ),
        )

        image = ImageSettings(
            max_upload_size_mb=int(image_raw.get("max_upload_size_mb", 10)),
            allowed_extensions=tuple(
                image_raw.get(
                    "allowed_extensions", [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
                )
            ),
            max_dimension_px=int(image_raw.get("max_dimension_px", 4096)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(
            "Invalid or malformed config.yaml", details=str(exc)
        ) from exc

    for directory in (
        paths.assets_dir,
        paths.sample_images_dir,
        paths.models_dir,
        paths.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return Settings(
        app=app,
        paths=paths,
        logging=logging_settings,
        image=image,
        project_root=root,
    )
