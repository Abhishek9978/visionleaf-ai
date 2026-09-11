# VisionLeaf AI — Engineering Review Report

**Scope:** Full-codebase engineering quality review following Milestone 6 and the
post-M6 stabilization pass. No architectural redesign, no user-facing behavior
changes, no new features (Feature Extraction, ML, and Analytics remain
out of scope). This document is the record of that review.

**Method:** Every finding below was verified with a tool or a direct code
read — not asserted from memory. Tooling used: `pyflakes` (unused
imports/names), `mypy` (type-hint correctness), `vulture` (dead code), the
full `pytest` suite (behavior), and a real `streamlit run` smoke test
(end-to-end sanity). Baseline before this sprint: 323 tests passing.
After this sprint: **331 tests passing**, `pyflakes` clean, `mypy`
errors reduced from ~45 across ~10 files to 23 across 2 files.

---

## Strengths

- **The core architectural pattern held up under real pressure.**
  `PipelineStage` → `PipelineEngine` → `AlgorithmRegistry` → `ImageSession`
  was designed in Milestone 4 and has now absorbed 30 algorithms across
  5 categories, a UI page, and a mid-project scope change (Milestone 5
  being redirected from Segmentation to the Processing Lab) without a
  single structural change. That's a meaningful validation of the
  original design.
- **`PipelineEngine` is genuinely safe.** `run_stage`/`run_pipeline`
  never let an algorithm exception propagate uncaught — `VisionLeafError`
  and bare `Exception` are both caught and converted into a
  `StageResult`, and multi-stage cancellation (`stop_on_error=True`)
  is real: verified live that a failing stage leaves `ImageSession`
  exactly as it was after the last success, never partially applied.
- **`ImageSession`'s chaining/invalidation model is consistent and now
  more complete.** `active_image`/`active_mask` let every stage read
  "whatever came before" uniformly; `clear_mask_results`/
  `clear_contour_results`/`clear_roi_results`/`reset_processing` give
  precise, non-overlapping invalidation instead of one blunt reset.
  This sprint extended the same philosophy with `require_active_image`/
  `require_active_mask`/`require_original_image`/`require_metadata`,
  closing the last gap in that pattern (type narrowing).
- **Exception handling is disciplined.** Exactly one broad
  `except Exception` in the entire codebase (in `PipelineEngine`,
  deliberately, and documented with a `noqa` explaining why); zero bare
  `except:` clauses; a clean three-tier hierarchy
  (`ValidationError` / `ImageLoadError` / `AlgorithmExecutionError`)
  used consistently.
- **Logging is uniform.** Every module that logs does so via
  `core.get_logger(__name__)` — no stray `print()` statements anywhere
  in `processing/`, `core/`, `config/`, or `utils/`.
- **Test coverage is real, not decorative.** Tests exercise actual
  failure modes (oversized kernels, missing masks, invalid parameters)
  through the real `PipelineEngine`, not mocks — and the test suite has
  already caught genuine bugs during development (e.g., a Bilateral
  Filter test that correctly identified the algorithm's real weakness
  against impulse noise; an Opening test that was checking the wrong
  pixel).
- **Documentation-as-data is a strong pattern.** `AlgorithmInfo`
  (Purpose/Theory/Math/Advantages/Limitations/Complexity/Working
  Principle/Typical Applications/OpenCV reference) means Learning Mode
  will render real per-algorithm content instead of hand-maintained UI
  copy that could drift from the implementation.

## Weaknesses

- **Two remaining clusters of loosely-typed data structures.**
  `config/settings.py`'s YAML-loading path and `ui/pipeline_stages.py`'s
  `PAGES`/`STAGES` tuples both use `dict[str, object]`-shaped data,
  which is why 23 `mypy` errors remain (see Recommended Improvements).
  Neither is a runtime bug — both are covered by passing tests — but
  both would benefit from `TypedDict`s or small dataclasses.
- **One hardcoded UI string had drifted silently for four milestones.**
  The Dashboard's "Current Milestone" metric read "2 · Application
  Shell" from Milestone 2 onward, never updated through Milestones 3–6.
  Nothing enforces that this string stays in sync with reality — it's
  now correct and commented, but it's still a manual value with no
  guardrail.
- **`AlgorithmExecutionError` is unused.** Defined in Milestone 4,
  exported, documented — but no algorithm across all 30 has ever
  actually raised it, because `PipelineEngine`'s blanket exception
  handling already covers the runtime-failure case it was meant for.
  Not a bug, but dead-in-practice API surface.
- **No pinned lockfile.** `requirements.txt` and `pyproject.toml` both
  use range pins (`>=X,<Y`), which is appropriate for a library/app
  under active development, but means two contributors installing on
  different days could get different exact versions. Acceptable for
  now; worth a `requirements-lock.txt` or `pip-compile` once the
  dependency set stabilizes further.

## Risks

- **The `PAGES`/`STAGES` milestone-number drift is structurally likely
  to recur.** It already happened once (this sprint fixed 8 stale
  references across 6 files) because nothing ties those hardcoded
  numbers to `docs/MILESTONES.md`. The next scope change will
  reintroduce the same class of bug unless the data is restructured or
  a test is added asserting page milestone numbers are monotonically
  non-decreasing (a partial guardrail, not a full fix).
- **`config/settings.py`'s loose typing is a latent risk in the
  highest-blast-radius code in the project.** Every page, every
  algorithm, and the app entry point all depend on `get_settings()`
  succeeding. Right now a malformed `config.yaml` produces a `mypy`-
  invisible `AttributeError` deep in `_require`/`.get()` chains rather
  than a clean, typed failure — the code raises `ConfigurationError`
  correctly at the top level, but the type checker can't confirm the
  internals are actually safe.
- **Synthetic sample images remain synthetic.** Not a code-quality
  risk, but a project risk worth restating for anyone about to build
  Feature Extraction/Classification on top of this: the bundled
  `assets/samples/` images are procedurally generated placeholders, not
  real leaf photographs, and will produce meaningless features/
  predictions if used as real training or demo data.

## Recommended Improvements *(not implemented this sprint — see rationale)*

1. **Introduce `TypedDict`s for `config/settings.py`'s raw YAML
   sections and for `ui/pipeline_stages.py`'s `PAGES`/`STAGES`
   entries.** Would close the remaining 23 `mypy` errors. Not done
   this sprint because both touch enough call sites (settings.py's
   `_require` helper is generic-typed and used for every config
   section; `PAGES` is read by `sidebar.py`, `dashboard.py`, and
   `app.py`) that the risk/effort ratio favored a follow-up with
   dedicated test coverage over folding it into this review.
2. **Add a test asserting `PAGES` milestone numbers are internally
   consistent with `docs/MILESTONES.md`** (e.g., parse the roadmap
   table and cross-check), so the exact bug this sprint fixed can't
   silently reappear. Deferred because it requires deciding how much
   coupling between a test and a markdown doc's exact formatting is
   acceptable — a design decision, not a quick fix.
3. **Decide the fate of `AlgorithmExecutionError`.** Either start using
   it (wrap each algorithm's core OpenCV call in a try/except that
   raises it on unexpected failure — a real change to 30 files) or
   remove it. Left as a recommendation because both options are
   judgment calls about future error-reporting granularity, not
   something to decide unilaterally during a stabilization-focused
   sprint.
4. **Consider a `requirements-lock.txt`** once Feature Extraction pins
   `scikit-learn`/`pandas` usage concretely — locking now would just
   need re-locking immediately after.

## Changes Implemented This Sprint

**Type-hint correctness (the bulk of this sprint's work):**
- Added `ImageSession.require_active_image()`, `require_active_mask()`,
  `require_original_image()`, and `require_metadata()` — narrowed,
  non-`None` accessors used after `validate()`/`is_loaded()` checks
  already guarantee the data exists, closing a systemic
  `np.ndarray | None` type-narrowing gap that `mypy` had no way to
  resolve across function-call boundaries.
- Updated all 11 call sites across `enhancement/`, `restoration/`,
  `roi/`, and `segmentation/_common.py` that read `session.active_image`
  directly, plus `processing_lab.py` (the UI-layer equivalent), to use
  the new accessors.
- Fixed `morphology/_common.py` to reuse the mask `require_mask()`
  already returns instead of re-reading the `Optional`-typed property.
- Introduced a `PixelStatistics` `TypedDict` (replacing a
  `dict[str, float | tuple[int, int]]` union that made unpacking
  `resolution` a real type error, not just an overcautious one).
- Fixed an `Image`/`ImageFile` type mismatch in `ImageManager` by
  giving the post-`.convert()` value its own correctly-typed variable.
- Fixed a `Literal["stretch"] | int` narrowing issue in
  `comparison_viewer.py`'s zoom/fit-to-window width calculation.
- **Result:** `mypy` errors reduced from ~45 (across ~10 files) to 23
  (across 2 files); zero behavior change (331 tests passing throughout).

**Duplicated code removed:**
- All 7 morphology algorithms shared an identical 8-line `validate()`
  body — extracted into `validate_morphology_params()`.
- Both adaptive-threshold algorithms shared an identical 5-line
  validation block — extracted into `validate_adaptive_threshold_params()`.

**Dead/unused code removed:**
- 6 unused imports across 2 production modules and 6 test files,
  found via a project-wide `pyflakes` sweep (previously never run).

**Stale documentation and UI-text fixes** *(carried over from the
prior stabilization review's momentum, verified again this sprint)*:
- Corrected 8 stale milestone-number references across
  `pipeline_stages.py` and 5 placeholder pages.
- Fixed the Dashboard's hardcoded "Current Milestone: 2" string (stale
  since Milestone 2) and added a comment flagging it as a manual value.

**Test coverage added:**
- 8 new tests for the 4 new `ImageSession` narrowing accessors
  (success and failure cases for each).
- (From the prior review, still in the suite): 2 cross-module
  integration tests verifying every registered algorithm's `stage_key`
  matches a real UI stage — the kind of typo a future contributor
  could otherwise introduce silently.

**Project packaging:**
- Added `pyproject.toml` (PEP 621 project metadata, build-system
  declaration, `pytest`/`mypy` tool config) — the project previously
  had no packaging metadata at all, meaning `pip install .` didn't
  work and there was no canonical place recording supported Python
  versions or the project's own identity. Dependencies are mirrored
  from `requirements.txt` (documented as the source of truth if the
  two ever drift) rather than read dynamically, so `pip install .`
  works with zero extra tooling.

**Test hygiene:**
- Cleaned up an inline `__import__("datetime")` hack in
  `test_image_session.py`, replaced with a proper top-level import.

## Intentionally Left Unchanged

- **`AlgorithmExecutionError`** — unused but harmless; see Recommended
  Improvements #3. Removing or wiring it up is a judgment call about
  future error granularity, not a stabilization fix.
- **`config/settings.py`'s loose YAML typing** — real, but fixing it
  properly means introducing `TypedDict`s across a file every page
  depends on; deferred to a focused follow-up rather than rushed here.
- **`PAGES`/`STAGES` dict typing in `pipeline_stages.py`** — same
  reasoning; the milestone-number *values* were fixed (the actual bug
  the prior review found), but the *structure* (loosely-typed dicts)
  that made the bug possible in the first place is a larger, separate
  change.
- **Sample gallery images** — still synthetic placeholders, as
  documented since Milestone 3; not something an engineering-quality
  pass should silently "fix" by fabricating fake real-looking data.
- **No pinned lockfile** — appropriate for the project's current
  stage; see Recommended Improvements #4.
- **Everything else about the architecture** — per the sprint's
  explicit scope, no redesign was performed. `PipelineStage`,
  `PipelineEngine`, `AlgorithmRegistry`, and `ImageManager` were
  reviewed carefully (items 13/15/16 of the sprint checklist) and
  found to already meet the bar for an open-source release as-is —
  the only changes made to them were the additive, backward-compatible
  `ImageSession` accessor methods described above.
