# Milestone Roadmap

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Project Foundation | ✅ Complete |
| 2 | Application Shell | ✅ Complete |
| 3 | Image Management System | ✅ Complete |
| 4 | Core Digital Image Processing Engine (Enhancement & Restoration algorithms) | ✅ Complete |
| 5 | Image Processing Laboratory (Interactive UI for Enhancement & Restoration) | ✅ Complete |
| 6 | Segmentation & ROI Engine (Thresholding, Morphology, Contours, ROI) | ✅ Complete |
| 7 | Feature Extraction Engine | ✅ Complete |
| 8 | Classification (Scaling, PCA, SVM) | ⬜ Not started |
| 9 | Analytics | ⬜ Not started |
| 10 | Experiment Mode | ⬜ Not started |
| 11 | Final Polish | ⬜ Not started |

*Note: Milestone 5 was originally scoped as "Segmentation Pipeline" in
the initial plan. Per client direction, it was redirected to the
Image Processing Laboratory instead, since Milestone 4 had just built
the underlying engine and needed a user-facing home before moving on.
Segmentation and every subsequent milestone shifted down by one as a
result — numbers above reflect the actual build order.*

## Milestone 1 — Project Foundation ✅

**Delivered:**
- Full folder structure for every pipeline stage (populated now,
  implemented incrementally — the layout won't change shape later).
- Typed, validated configuration system (`visionleaf_ai/config/`) backed
  by `config.yaml`, with environment-variable override support.
- Exception hierarchy (`visionleaf_ai/core/exceptions.py`).
- Idempotent logging setup with console + rotating file handlers
  (`visionleaf_ai/core/logging_config.py`).
- Stage-agnostic validation helpers (`visionleaf_ai/utils/validators.py`).
- 19 passing unit tests across config, core, and utils.
- `requirements.txt`, `.gitignore`, README, developer guide, git
  workflow recommendations, and a setup-verification script.

**Verified by:**
```bash
PYTHONPATH=. python scripts/verify_setup.py
PYTHONPATH=. pytest tests/ -v   # 19 passed
```

**Not yet included (by design — later milestones):** any Streamlit UI,
any OpenCV/scikit-image usage, any ML code. Milestone 1 is intentionally
UI-free and image-processing-free per the approved architecture.

## Milestone 2 — Application Shell ✅

**Delivered:**
- `app.py` entry point wiring config, session state, theming, and
  `st.navigation` into a single running app.
- Custom design system (`visionleaf_ai/ui/theme.py`): named color
  tokens, Space Grotesk / IBM Plex Sans / IBM Plex Mono type system,
  injected as one global stylesheet.
- Signature "pipeline rail" component showing all 11 DIP stages as a
  connected, numbered row, highlighting the current page's stage(s).
- Custom sidebar (`components/sidebar.py`): brand header, a status-
  annotated link to every page, and the global Learning Mode toggle.
- Reusable components: `page_header`, `section_card`, `status_badge`,
  `learning_note`, `render_metric_row`, `render_stage_placeholder`.
- Idempotent session-state initialization (`ui/session.py`) with a
  `pipeline_status` dict later milestones will update as stages
  actually run.
- 8 wired pages (Dashboard + 7 stage pages spanning all 11 pipeline
  stages), each using the shared components and a clear, informative
  placeholder — no DIP algorithms implemented yet, as scoped.
- 20 new automated tests (pipeline-stage metadata consistency, session
  state, theme tokens, and full-app integration tests via Streamlit's
  `AppTest` framework) — 39 passing total across both milestones.

**Verified by:**
```bash
PYTHONPATH=. pytest tests/ -v          # 39 passed
PYTHONPATH=. streamlit run app.py      # real server, confirmed HTTP 200
```
Every page was also rendered in isolation via `AppTest` to confirm no
page raises on render, and the Learning Mode toggle was verified to
actually show/hide the educational callouts.

**Not yet included (by design — later milestones):** any real image
upload/processing/ML logic. Every stage page currently shows a
professional placeholder describing what that stage will do —
Milestone 3 onward replace these placeholders with working controls.

## Milestone 3 — Image Management System ✅

**Architectural decision (superseding the original "Image Acquisition
only" scope):** rather than building acquisition in isolation,
Milestone 3 built the centralized Image Management System every later
stage depends on.

**Why centralize instead of letting each page store its own image:**
1. **Single source of truth** — `st.session_state["image_session"]` is
   the only place an image lives; no page can disagree with another
   about which image, or which result, is current.
2. **Testable without Streamlit** — `ImageSession` and `ImageManager`
   are plain Python (dataclasses + PIL/NumPy). All 28 of their tests
   run with plain pytest, no UI test harness needed.
3. **Consistent provenance** — every stage appends to the same
   `processing_history` list on the same object, so the History Panel
   is automatically correct with zero per-page bookkeeping.
4. **Cheap invalidation** — loading a new image means `ImageManager`
   returns a brand-new `ImageSession`; every downstream field
   (`segmented_image`, `feature_vector`, `prediction`, ...) starts
   `None` automatically, and `set_image_session()` resets every
   stage's status to "pending" (except Acquisition) so a stale
   segmentation result can never survive a new image.
5. **One validation/resize path** — extension, size, and dimension
   policy live in exactly one place (`ImageManager`), so it can't drift
   between pages.

**Delivered:**
- `ImageSession` (`core/image_session.py`): the central model — every
  intermediate image field the full pipeline will use, plus
  `feature_vector`, `prediction`, `metadata`, `processing_history`,
  and the `active_image` property later stages read from.
- `ImageManager` (`processing/acquisition/image_manager.py`): the only
  path by which images enter the pipeline — validates extension/size,
  decodes, resizes if oversized (preserving aspect ratio), builds
  metadata, and records the initial "Image Loaded" history event.
  Fully Streamlit-agnostic and dependency-injectable for testing.
- `ImageLoadError` added to the exception hierarchy, distinct from
  `ValidationError` (pre-decode checks vs. actual decode failures).
- Upload + sample gallery flow, image preview, metadata card, and
  processing history — all via 5 new reusable components (Image
  Viewer, Metadata Card, Upload Panel, History Panel, Empty State),
  usable as-is by every later stage.
- 4 synthetic placeholder sample images (clearly flagged as
  synthetic — not real leaf photographs) plus the generator script
  that made them.
- `ui/session.py` extended with `get_image_session()` /
  `set_image_session()` as the single authoritative accessor pair.
- Fixed a Streamlit API deprecation discovered during testing
  (`use_container_width` → `width=`), bumping the `streamlit`
  requirement to `>=1.40`.
- 33 new tests (28 pure-Python unit tests on `ImageSession`/
  `ImageManager`, 5 full-page integration tests via `AppTest`) — 72
  passing total across all three milestones.

**Verified by:**
```bash
PYTHONPATH=. pytest tests/ -v          # 72 passed
PYTHONPATH=. streamlit run app.py      # real server, confirmed HTTP 200
```
Upload, sample-gallery selection, reset, and the resulting
`pipeline_status`/`ImageSession` state were all exercised through the
real page code (not mocked), including the case where loading a new
image must invalidate downstream stage statuses.

**Not yet included (by design — later milestones):** any actual
enhancement, restoration, segmentation, or classification logic.
Milestone 4 will read `session.active_image`, write to
`session.current_image`/`grayscale_image`, and call
`session.record_event("enhancement", "Enhanced")` — using the
foundation built here without needing to touch loading/validation code
at all.

## Milestone 4 — Core Digital Image Processing Engine ✅

**Scope note:** per the approved architecture, this milestone is
explicitly NOT about Streamlit — no UI, sliders, or pages were built.
The goal was a reusable, UI-agnostic processing engine every future
stage (this one and Milestones 5–9) will run through.

**Architecture — UI → Pipeline Engine → Pipeline Stage → ImageSession:**
1. **`PipelineStage`** (`processing/pipeline/stage.py`) — abstract base
   class every algorithm implements: `process()`, `validate()`,
   `get_name()`, `get_description()`, `get_info()` (structured
   Purpose/Theory/Math/Advantages/Limitations/Complexity documentation
   that will power Learning Mode). Every algorithm is interchangeable
   through this interface.
2. **`PipelineEngine`** (`processing/pipeline/engine.py`) — the only
   execution surface: `run_stage()`, `run_pipeline()` (with real
   cancellation — a failing stage stops every stage after it),
   `run_by_name()` (look up + run by string name). Adds an outer layer
   of timing/logging/error-handling on top of each stage's own
   self-contained validation/logging/history-recording.
3. **`AlgorithmRegistry`** (`processing/pipeline/registry.py`) —
   decorator-based (`@register_algorithm("name")`), no if/elif chains.
   Adding an algorithm means writing one class with one decorator line.
4. **`bootstrap.ensure_registered()`** — guarantees every algorithm
   module is imported (and therefore registered) without any caller
   needing to import algorithm classes directly; `PipelineEngine.__init__`
   calls it automatically.

**Why this is scalable:** every future milestone (Segmentation, ROI,
Feature Extraction, PCA, SVM) is the same pattern — a new
`PipelineStage` subclass, registered, unit-tested with plain pytest
(no Streamlit needed), callable through the same engine. When the
Streamlit UI for these stages is eventually built, it will contain
zero algorithm code: just widgets collecting parameters, a call to
`engine.run_by_name(...)`, and rendering whatever `ImageSession` field
came back.

**Delivered:**
- 5 Enhancement algorithms: Brightness Adjustment, Contrast Adjustment,
  Gamma Correction, Histogram Equalization (YCrCb luminance-only, to
  avoid color distortion), CLAHE (LAB L-channel).
- 3 Restoration algorithms: Gaussian Blur, Median Filter, Bilateral
  Filter.
- Each with a frozen parameter dataclass, full validation (missing
  image, invalid parameters, oversized kernels, wrong dtype), logging
  (algorithm name, parameters, image dimensions, duration,
  success/failure), and history recording — all self-contained inside
  `process()`.
- `AlgorithmExecutionError` added to the exception hierarchy, distinct
  from `ValidationError` (invalid input vs. runtime failure on valid
  input).
- 101 new tests (22 on the pipeline infrastructure itself, 79 across
  the 8 algorithms — normal cases, invalid parameters, missing images,
  and edge cases like clipping and oversized kernels) — **173 passing
  total across all four milestones.**

**A genuinely useful test failure, not just a passing suite:** an
early Bilateral Filter test assumed it would remove salt-and-pepper
noise. It didn't — verified directly against raw OpenCV, not just this
codebase. The reason is exactly the algorithm's documented limitation:
bilateral filtering is edge-preserving, so an outlier pixel's own
zero-distance self-weight dominates its neighbors' contributions,
making it specifically *weak* against impulse noise despite being
strong against smooth (Gaussian) noise. The test was corrected to use
Gaussian noise, which the algorithm is actually good at — the kind of
distinction this milestone's documentation requirements (Theory,
Limitations) exist to surface.

**Verified by:**
```bash
PYTHONPATH=. pytest tests/ -v          # 173 passed
```
Also verified live: a hand-run pipeline of
[Brightness → Gaussian Blur (deliberately invalid kernel) → Median
Filter] correctly executed the first stage, failed the second, and
**cancelled** the third — leaving `ImageSession` exactly as it was
after the last successful stage.

**Not yet included (by design — this milestone was explicitly
engine-only):** no Streamlit page renders any of these 8 algorithms
yet. That UI work — sliders, before/after previews — is deferred to
whichever future milestone builds the Image Processing Lab page, and
will call `PipelineEngine.run_by_name(...)` exclusively.

## Milestone 5 — Image Processing Laboratory ✅

**Scope note:** redirected here per client direction from the
originally-planned Segmentation Pipeline — Milestone 4 had just built
the DIP engine and needed its first real, user-facing consumer before
moving further down the pipeline.

**Architecture — strictly UI → PipelineEngine → PipelineStage →
ImageSession:** the page contains zero pixel-manipulation code. Every
"Apply" button does exactly one thing:
`engine.run_by_name(selected_algorithm, session, **params)`. Every
other section (comparison, histogram, stats, history) only reads
`ImageSession` or computes display-only analytics from its arrays —
never a pipeline algorithm.

**Two design decisions made explicit and implemented:**
1. **Apply, not live-drag.** Sliders stage parameter values only;
   nothing runs until "Apply" is clicked — verified necessary because
   `ImageSession.active_image` chains onto whatever the previous stage
   produced, so live-running on every slider drag would silently
   double-apply. This is also what makes the History timeline
   meaningful and matches the required example exactly (Image Loaded
   → Brightness → CLAHE → Median Filter → Histogram Equalization).
2. **New, narrower reset + new export capability.**
   `ImageSession.reset_processing()` restores `current_image` from
   `original_image` and trims history back to Acquisition only —
   distinct from Milestone 3's `ImageManager.reset_session()`, which
   discards the image entirely. `ImageManager.export_image()` was
   added as the symmetric encode-out counterpart to the existing
   decode-in load methods, so the UI still never touches image
   encoding directly.

**Dark theme applied app-wide** (not just this page) per the approved
Blueprint — `theme.py` and `.streamlit/config.toml` re-themed with a
near-black canvas and brightened teal/amber accents; every existing
page inherits it automatically through the same CSS injection.

**Delivered — all 15 required sections:**
- Interactive Comparison: Side-by-Side and Before/After (a NumPy
  column-split composite, not an algorithm) modes, zoom and
  fit-to-window (pure `st.image(width=...)` display parameters — no
  resampling library touches the array). Serves as the single place
  "Original Image" and "Processed Image" are both shown (documented
  assumption — see `processing_lab.py`'s module docstring).
- Enhancement Controls / Restoration Controls, each with an algorithm
  picker sourced from `AlgorithmRegistry.list_algorithms(category)`
  and a dynamic Algorithm Parameter Panel showing only that
  algorithm's relevant parameters.
- Histogram Viewer (Plotly, per-channel, Original vs. Processed) and
  Pixel Statistics (resolution, mean, std, min, max, dynamic range),
  both recomputed fresh on every render — no caching to go stale.
- Processing History (reused from Milestone 3's component unchanged).
- Learning Mode: renders each algorithm's real `AlgorithmInfo` from
  `get_info()` — Purpose, Theory, Math Intuition, Advantages,
  Limitations — plus per-parameter explanations sourced from the same
  `ParamSpec.help` text used as each slider's tooltip, so the two can
  never disagree.
- Image Metadata (extended with a "Current Stage" field), Reset,
  Export (PNG/JPEG download), and graceful error handling throughout.
- 6 new reusable components: Comparison Viewer, Parameter Panel,
  Histogram Viewer, Pixel Statistics, Learning Panel, Export Panel.
- 44 new tests (pure-logic tests for statistics/composite/parameter-
  schema consistency, plus 7 full-page `AppTest` integration tests) —
  **204 passing total across all five milestones.**

**Verified by:**
```bash
PYTHONPATH=. pytest tests/ -v          # 204 passed
PYTHONPATH=. streamlit run app.py      # real server, confirmed HTTP 200
```
Also verified live through the actual page, not just unit tests: a
2-stage chain (Brightness → CLAHE) correctly recorded both history
events in order; Reset correctly trimmed history back to one entry and
reset `pipeline_status` for both stages to "pending"; and — using a
dedicated tiny (10x10) synthetic-image test harness — a genuinely
oversized kernel parameter produced a real `ValidationError` from the
real `PipelineEngine`, rendered as `st.error(...)` without crashing the
page, with the session left untouched by the failed stage.

**Not yet included (by design):** Segmentation, Morphology, and ROI
Extraction — deferred to Milestone 6, which will follow this exact
pattern (new `PipelineStage` subclasses, a new page consuming them
through the same `PipelineEngine`).

## Milestone 6 — Segmentation & ROI Engine ✅

**Scope note:** engine-only, following the same M4-then-M5 pattern —
no Streamlit page was built this milestone (the spec had no "UI
Requirements" section, unlike M5). A future Segmentation Lab page will
consume these algorithms exactly as the Processing Lab consumes
Enhancement/Restoration.

**Architecture — no redesign, pure extension:** every algorithm here
is another `PipelineStage` subclass registered via the same
`@register_algorithm` decorator, run through the same
`PipelineEngine`, reading/writing the same `ImageSession`. Two
additive extensions were made, both flagged before implementing them:

1. **`ImageSession.active_mask`** — a new property directly parallel
   to `active_image`: returns `segmentation_mask` if set, else
   `binary_image`, else `None`. Lets masks chain across stages the
   same way images already do.
2. **Cascading invalidation** — three new narrow methods
   (`clear_mask_results`, `clear_contour_results`, `clear_roi_results`)
   so re-running a threshold, a morphological op, or Find Contours
   correctly invalidates exactly what it should downstream, without
   ever needing a full session reset. Verified live: re-running
   Binary Threshold after a full Otsu→Closing→Contours→BBox→Crop
   chain correctly cleared every one of those five downstream fields
   in one call.
3. **`AlgorithmInfo` gained three optional fields**
   (`working_principle`, `typical_applications`, `opencv_reference`),
   additive with defaults, so every Milestone 4 algorithm's
   `get_info()` still works unchanged.

**Delivered — 22 new algorithms, 30 total registered:**
- **Thresholding** (`stage_key="segmentation"`): Binary, Binary
  Inverse, Truncate, To Zero, Otsu, Adaptive Mean, Adaptive Gaussian —
  all writing to `session.binary_image` via a shared `_common.py`
  grayscale-conversion helper.
- **Morphology** (`stage_key="morphology"`): Erosion, Dilation,
  Opening, Closing, Morphological Gradient, Top Hat, Black Hat — all
  reading `session.active_mask`, writing `session.segmentation_mask`,
  with configurable kernel size and iterations, via a shared
  `_common.py` (mask retrieval + structuring-element construction).
- **Contour Processing** (`stage_key="roi"`): Find Contours (also
  identifies `largest_contour`), Contour Filtering by Area, Bounding
  Rectangle, Minimum Area Rectangle, Convex Hull (refines
  `largest_contour` in place).
- **ROI Extraction** (`stage_key="roi"`): ROI Cropping (→ `roi_image`),
  ROI Masking (→ `segmented_image`, reusing a field that had sat
  unused since Milestone 3), Bounding Box Visualization (→
  `current_image`, following Enhancement/Restoration's own convention
  for where visual output goes).
- **Automatic ROI Extraction** — deliberately NOT a new monolithic
  `PipelineStage` (that would duplicate the five stages above).
  Instead, `processing.roi.auto_extract.run_automatic_roi_extraction()`
  is a thin function that runs Otsu → Closing → Find Contours →
  Bounding Rectangle → ROI Cropping through the existing
  `PipelineEngine.run_pipeline()`, reusing every stage as-is.
- `ImageSession` extended with `segmentation_mask`, `contours`,
  `largest_contour`, `bounding_box` (a dict, discriminated by a
  `"type"` field, since both Bounding Rectangle and Minimum Area
  Rectangle answer the same question) — no existing field removed.
- 117 new tests (18 on `ImageSession`'s new fields/methods, 35 on
  thresholding, 30 on morphology, 45 on contour/ROI/auto-extraction) —
  **321 passing total across all six milestones.**

**A genuinely useful test failure caught during this milestone:** an
Opening test initially asserted the wrong pixel — accidentally
checking a point *inside* the synthetic mask's internal hole, which
Opening correctly does not fill (only Closing does). Diagnosed by
re-reading the fixture's coordinates, not by loosening the assertion.
A similar issue in the Automatic ROI Extraction cancellation test (a
uniform gray image doesn't produce "no contours" — Otsu's degenerate
threshold of 0 makes every pixel foreground) was fixed by switching to
a genuinely all-black image, which does.

**Verified by:**
```bash
PYTHONPATH=. pytest tests/ -v          # 321 passed
PYTHONPATH=. streamlit run app.py      # real server, confirmed HTTP 200 (untouched by this milestone)
```
Also verified live: a full Otsu → Closing → Find Contours → Bounding
Rectangle → ROI Cropping chain on a synthetic square correctly
identified its exact location and size; "no contours found" and
"no largest contour available" both surface as clean, gracefully
handled `ValidationError`s rather than crashes.

**Not yet included (by design):** Feature Extraction — Milestone 7
will read `session.roi_image`/`segmented_image` and
`largest_contour`/`bounding_box` as its primary inputs, the first
milestone where the pipeline stops producing images and starts
producing numbers.

## Post-Milestone-6 Stabilization Review

Not a milestone — a full-application review requested after M6, with
no new features. Findings and fixes:

**Dead code removed:** 6 unused imports across 2 production modules
(`pixel_statistics.py`, `brightness.py`, `clahe.py`) and 6 test files,
found via a full `pyflakes` sweep (previously never run project-wide).

**Duplication eliminated:**
- All 7 morphology algorithms had a byte-identical 8-line `validate()`
  body. Extracted into `morphology/_common.py`'s new
  `validate_morphology_params()`; each algorithm's `validate()` is now
  one line.
- Both adaptive thresholding algorithms had an identical 5-line
  block_size/max_value validation block. Extracted into
  `segmentation/_common.py`'s new `validate_adaptive_threshold_params()`.

**Stale-reference bugs found and fixed** — the project's actual
milestone order didn't stay linear (M5 was redirected from
Segmentation to the Processing Lab UI), and several places still
reflected the *original* linear plan instead of what actually
shipped:
- `ui/pipeline_stages.py`: `processing_lab`'s milestone number was 4
  (wrong — it shipped in M5); `segmentation_pipeline`,
  `feature_extraction`, `classification`, `analytics`, and
  `experiment_mode` were all off by one.
- 5 placeholder pages (`segmentation_pipeline.py`,
  `feature_extraction.py`, `classification.py`, `analytics.py`,
  `experiment_mode.py`) had matching stale `eyebrow=`/milestone-number
  values.
- `dashboard.py` had hardcoded `"Current Milestone: 2 · Application
  Shell"` — untouched since Milestone 2, still showing after 4 more
  milestones shipped.
- `render_stage_placeholder()`'s `milestone_number: int` parameter
  couldn't honestly describe Segmentation's actual state (backend
  complete since M6, UI still unbuilt) — relaxed to a caller-supplied
  `status_note: str` so each page states its real status instead of
  being forced into a single-number template that assumed a strictly
  linear roadmap.
- `README.md` referenced "Milestones 6–7" for disease classification
  readiness; corrected to 7–8 (Feature Extraction, Classification).

**Missing integration test added:** nothing previously verified that
every registered algorithm's `stage_key` actually matches a real
`ui.pipeline_stages.STAGES` key — a typo here would have silently
broken that algorithm's pipeline-rail/sidebar-status display with no
test failure anywhere. Added
`tests/test_cross_module_integration.py` at the top level of `tests/`
(deliberately outside both `tests/processing/` and `tests/ui/`, since
it tests the seam between them).

**Verified, not just fixed:** full test suite (323 tests, +2 new),
`pyflakes` across `visionleaf_ai/` and `tests/`, every one of the 5
placeholder pages re-rendered via `AppTest` to confirm the
`status_note` refactor didn't break anything, and a real
`streamlit run` smoke test (HTTP 200, clean logs).

**Noted but deliberately not changed:** `AlgorithmExecutionError`
(added in M4) has never actually been raised by any of the 30
algorithms — `PipelineEngine`'s blanket exception handling already
covers the gap, so this is unused forward-looking infrastructure, not
a functional bug. Retrofitting try/except into every `process()`
method to exercise it would be a substantial, riskier change better
suited to a deliberate follow-up than a stabilization pass.

## Milestone 7 — Feature Extraction Engine ✅

**Scope note:** engine-only, per the established M4/M6 pattern — this
milestone's spec had no "UI Requirements" section, so no Streamlit
page was built. A future Feature Extraction Lab page (following M5's
role for M4) would consume these exactly as the Processing Lab
consumes Enhancement/Restoration.

**Architecture — no redesign, pure extension, per your "architecture
frozen" instruction:** every extractor is another `PipelineStage`,
registered via `@register_algorithm` into the `AlgorithmRegistry`, run
through the unmodified `PipelineEngine`, reading/writing the
unmodified `ImageSession` (only additively extended with 5 new
fields, exactly as instructed — nothing removed). This is the first
milestone where a stage's output is numbers, not images — the
mechanism for getting there was identical to every prior milestone.

**Delivered — 6 new algorithms across 4 feature groups + 1 builder:**
- **Color Features** — RGB/HSV mean, std, and normalized per-channel
  histograms, computed over foreground pixels only when a mask is
  available (verified live: an all-black-background/all-white-
  foreground test image produced an RGB mean of exactly 255, proving
  background pixels are correctly excluded from the statistics).
- **GLCM Texture Features** and **Local Binary Pattern** — both write
  into the same `session.texture_features` dict under their own
  `"glcm"`/`"lbp"` sub-keys, verified to never clobber each other
  regardless of run order.
- **Shape Features** — area, perimeter, aspect ratio, extent,
  solidity, circularity, equivalent diameter, and log-scaled Hu
  moments, reusing `processing.roi`'s existing `require_largest_contour`
  rather than duplicating contour-validation logic. Verified against
  closed-form theory: a synthetic square's circularity computed to
  exactly π/4, matching the textbook formula.
- **Edge Features** — Canny edge density, Sobel gradient magnitude,
  Laplacian response variance.
- **Feature Vector Builder** — concatenates whichever groups have run,
  in a fixed documented order, into one `np.ndarray`, min-max
  normalized to `[0, 1]`. Verified the exact expected vector length
  (131 = 12 + 96 + 6 + 14 + 3) when all four groups are present.
- `ImageSession` extended with `color_features`, `texture_features`,
  `shape_features`, `edge_features`, and (already existing since M3)
  `feature_vector` — `reset_processing()` updated to clear all of them.
- 68 new tests (67 for the features package, 1 for `ImageSession`'s
  extension) — **399 passing total across all seven milestones.**
  `pyflakes` and `mypy` both clean on the new package.

**Validation coverage (per spec item 10), all verified with real
failing inputs, not just written:** missing segmentation/ROI, an
all-zero (fully-masked-out) segmented image, a zero-area contour, and
"no features extracted yet" for the builder all produce clean
`ValidationError`s through the real `PipelineEngine`, never a crash.

**Verified by:**
```bash
PYTHONPATH=. pytest tests/ -v          # 399 passed
PYTHONPATH=. streamlit run app.py      # real server, confirmed HTTP 200 (untouched by this milestone)
```
Also verified live: a full pipeline — Otsu → Closing → Find Contours →
Bounding Rectangle → ROI Cropping → ROI Masking → all 5 feature
extractors → Feature Vector Builder — run end to end on a synthetic
leaf-like image, producing a 141-dimensional vector correctly
normalized to `[0, 1]` with a complete, correctly-ordered processing
history.

**Not yet included (by design):** PCA and SVM — Milestone 8 will read
`session.feature_vector` exclusively and will be the first milestone
that never touches a pixel.
