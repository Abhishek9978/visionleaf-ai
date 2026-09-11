# Development Guide

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ is required (the codebase uses `from __future__ import annotations`
and modern type-hint syntax like `str | None`).

## Coding Standards

- **PEP8** throughout; format with `black` and lint with `ruff` if
  available in your environment (not enforced by CI in Milestone 1, but
  recommended).
- **Type hints on every function signature.** Prefer `str | None` over
  `Optional[str]`.
- **Docstrings on every module and public function**, following the
  format used throughout this codebase:
  ```python
  def my_function(arg: int) -> bool:
      """One-line summary.

      Args:
          arg: What this parameter means.

      Returns:
          What gets returned.

      Raises:
          SomeError: When and why.
      """
  ```
- **Single Responsibility Principle**: one module = one job. If a file
  is doing two unrelated things, split it.
- **No bare `except:`** and no silent failures. Catch specific
  exceptions; if you don't know what to do with an error, let it
  propagate. Application-specific errors should raise from the
  `visionleaf_ai.core.exceptions` hierarchy (add a new subclass of
  `VisionLeafError` if an existing one doesn't fit).
- **No print() in library code.** Use
  `from visionleaf_ai.core import get_logger` →
  `logger = get_logger(__name__)`. `print()` is acceptable only inside
  Streamlit UI code (Milestone 2+) where it's actually `st.write`/etc.,
  not real console output.
- **No hardcoded values** that a user might reasonably want to change —
  put them in `config.yaml` and read them via
  `visionleaf_ai.config.get_settings()`.

## Adding a New Module

1. Put it in the subpackage that matches its pipeline stage (see the
   folder structure in the README). Don't create new top-level folders
   without updating the README's structure diagram and explaining why.
2. Give it a module docstring with Purpose / Description / Dependencies
   / Public functions, matching the existing modules in `core/` and
   `config/`.
3. Add a mirrored test file under `tests/<same-subpackage-path>/`.
4. If it needs configuration values, add them to `config.yaml` under a
   clearly-named section and read them through `get_settings()` —
   never `open("config.yaml")` directly from a new module.

## Testing

```bash
# Run everything
PYTHONPATH=. pytest tests/ -v

# Run one subpackage's tests
PYTHONPATH=. pytest tests/config/ -v

# With coverage
PYTHONPATH=. pytest tests/ --cov=visionleaf_ai --cov-report=term-missing
```

Each milestone's deliverables should be independently testable — i.e.
you should be able to write and run tests for a subpackage without any
later milestone's code existing yet. That's why, e.g., `tests/config/`
and `tests/core/` don't depend on anything in `processing/` or `ml/`.

## Logging

```python
from visionleaf_ai.core import get_logger

logger = get_logger(__name__)
logger.info("Loaded %d images", count)
logger.warning("Skipping malformed file: %s", path)
logger.error("Failed to save model", exc_info=True)
```

Logs go to console always, and additionally to `logs/visionleaf.log`
(rotating, 1 MB × 3 backups by default) if `logging.log_to_file` is
`true` in `config.yaml`.

## Configuration

Read settings once per call site via `get_settings()` — it's cached
(`functools.lru_cache`), so calling it repeatedly is cheap:

```python
from visionleaf_ai.config import get_settings

settings = get_settings()
max_size = settings.image.max_upload_size_mb
```

If you're writing a test that changes an environment variable affecting
config, call `get_settings.cache_clear()` before and after (see
`tests/config/test_settings.py` for the pattern).
