# Handoff — Visual AI Workbench (project-void)

You are taking over an in-progress build. Read this whole file before doing anything.

## 1. What this project is

- **Product spec (authoritative):** `docs/VISION.md`, the same as the original `README.md` the user wrote. It covers 9 milestones (0–8) and acceptance tests A01–A64 (§24).
- **Plan and rules:** `docs/PLAN.md`.
- **What actually works:** `docs/CAPABILITIES.md`, the honest ledger. Update it with every change.
- **Design decisions:** `docs/adr/0001…0011`. Read them before changing an area.
- **How to run it:** the root `README.md`. It lists only commands that were actually run.

**Repo:** `/Users/mazenkhaled/project-void` (local git, no remote).

**Stack:**
- Python 3.13 in `.venv`, with PyTorch CPU, FastAPI, SQLite, scikit-learn, SciPy, LangGraph, Gymnasium.
- Editor: Vite, React, TypeScript, React Flow in `apps/editor` (pnpm).

**Layout:**

| Path | Contents |
|---|---|
| `python/graph_core` | schema, registry, validation, hashing, lowering, codegen |
| `python/operations` | block definitions |
| `python/worker` | runs in a separate process |
| `python/artifact_store` | SQLite metadata and content-addressed artifacts |
| `python/tabular`, `connectors`, `studies`, `training`, `debugger`, `codeblocks`, `agent`, `rl`, `unsup`, `domain`, `vision`, `nlp`, `speech` | one package per milestone area |
| `services/control` | FastAPI app (`control.app:create_app`) |
| `examples/` | example projects and fixture generators |
| `tests/` | pytest suite |

## 2. Status (as of 2026-10-04)

| Milestone | State | Commit |
|---|---|---|
| 0–1 Graph foundation and visual CNN workbench | done, verified | dd883eb |
| 2a Tabular, leakage checks, regression, statistics | done, verified | 9cb014c |
| 2b PostgreSQL/S3/DVC connectors, snapshots, studies and sweeps | done, verified | 8c83047 |
| 3 Composites, sharing, training procedure, debugger, transformer, code blocks | done, verified | b30a197 |
| 4 LangGraph agents, memory, context inspector | done, verified | 4201da2 |
| 5 RL (DQN/CartPole, grid world), unsupervised | done, verified | 5674566 |
| 6a Keras/TensorFlow and JAX backends, compatibility reports, coverage ledger, benchmarks | done, verified | 503ff9c |
| 6b Vision detection/segmentation, NLP, speech workflows | done, verified (bounded scope; see §7) | this milestone commit; see git log |
| 7 Registry and production investigation | not started | none |
| 8 Scale, integrations, community | not started | none |

After 6a: `pytest -q` → 707 passed, 1 skipped (live Anthropic test; no API key); `pytest -q -m live` → 6 passed (local Ollama).

After 6b: `pytest -q -o faulthandler_timeout=240` → **864 passed, 1 skipped, 6 deselected**; `pytest -q -m live` → **6 passed**. Editor build/type check, curl and real Chrome domain journeys passed. The skip is the existing Anthropic test without an API key. See §7 for commands, results, scope and push status.

### Check for in-progress work first
Run `git status`. If there are uncommitted files, a previous session was cut off mid-milestone: inspect them with `git diff`, do **not** discard them, finish that milestone, verify (§4), then commit.

## 3. Environment facts

- macOS arm64. There is no `timeout` command; use pytest `-o faulthandler_timeout=240` to catch hangs.
- **Port 8000:** a pre-existing uvicorn (PID 8258) is the user's. Don't kill it; use other ports and stop every server you start.
- **Docker:** not running. These local real engines are used instead, all already wired into tests:
  - PostgreSQL: `pgserver`, which needs `python -m connectors.install_pgserver` on Python 3.13
  - S3: `moto` server
  - DVC: the `dvc` package
- **Ollama:** running at `localhost:11434` with `qwen3.5:0.8b/2b/4b`, `gemma4:12b` and `nomic-embed-text`. Live tests are marked `@pytest.mark.live` and deselected by default.
- **API keys:** none. The Anthropic adapter exists, but its live test skips.
- **Full suite:** about 4–5 minutes with domain workflows. In Codex's filesystem/network sandbox, local service tests may fail or skip and worker tests may hang: run the required suite with local service access, not inside that restricted sandbox.

## 4. Working rules (non-negotiable, from VISION §26)

- No mock data, simulated charts or fabricated "learned" values. Illustrative/teaching values must be labelled as such. Values never recorded show "not recorded", never a guess.
- No placeholder buttons for unbuilt features. Unbuilt means absent, or labelled "not implemented" in `CAPABILITIES.md`.
- Native library semantics; numerical tests compare against native references.
- **Never weaken existing tests.** If a registry-count assertion must change, keep the original op set required and only allow the new families.
- The README lists only commands you actually ran.
- **Every milestone ends with this verification:**
  ```bash
  source .venv/bin/activate
  pytest -q -o faulthandler_timeout=240
  pytest -q -m live
  pnpm -C apps/editor build
  pnpm -C apps/editor exec tsc --noEmit
  ```
  Then run a curl smoke against a backend you start, and a real-browser check (puppeteer-core with the installed Chrome; keep scripts outside the repo).
- **Commit after each verified milestone.** Use the message style of `git log`, ending with a co-author line naming your own model.
- Update `docs/CAPABILITIES.md`, `README.md` and add an ADR under `docs/adr/` for notable decisions.

## 5. Remaining work

### 6b — completed domain scope (VISION §9.7, §9.8, §23 Milestone 6 "Domain evidence", A56, A57, A58)

Implemented and verified; §7 records the actual scope and limits. Retained here as the acceptance reference, not an instruction to rebuild the milestone.

- **Vision:** detection/segmentation workflow.
  - Transforms must keep boxes, masks and keypoints consistent with the image (A56). Use torchvision.
  - Small labelled synthetic fixture; inspect predictions against ground truth.
- **NLP:** tokenization with source spans, subwords, masks and label alignment, using the same evaluation convention as `seqeval` (A58).
  - A small token-classification example.
- **Speech:** audio workflow with duration/sample rate, framing, sequence lengths, decoding and error alignment (A57; the teaching signal has 32,000 samples per channel — find the exact spec in VISION).
  - Use torchaudio, or numpy/scipy if torchaudio is unavailable.
- Each domain needs a runnable example, a real inspection journey, tests, and editor views.

### 7 — Registry and production (VISION §17.5–17.8, §19.5, A50, A59–A63)

- **Registry and serving:**
  - Model registry versions.
  - Pinned inference pipelines (exact model, transforms, schema, labels/tokenizer, environment).
  - Local and staging serving through a local FastAPI inference adapter.
- **Traffic and isolation:**
  - Traffic builder and bounded load tests reporting offered vs achieved load, latency, errors and resource use.
  - Session isolation under concurrency.
- **Release management:**
  - Request traces from prediction → release → run → data → preprocessing.
  - Monitoring and drift, keeping label-based quality separate from input drift.
  - Rollout and rollback with lifecycle events.
- **Labelling:** simulated deployment behaviour is visibly distinct from live operation. There is no remote infrastructure, so any remote deployment adapter stays not-implemented unless it can be truly tested.

### 8 — Scale and community (VISION §20, §23 Milestone 8, A49, A64)

- Remote worker protocol: a second local process over HTTP counts as "remote" only if labelled honestly.
- MLflow and W&B bridges: MLflow can run locally and must be tested; W&B offline mode. Sync resumes without duplicates (A49).
- Plugin/operation SDK with conformance checks; reusable project packages.
- Keyboard-accessibility pass (A20).
- The full connected-researcher journey end to end (A64).
- An end-to-end acceptance checklist mapping A01–A64 to tests, or to an honest "not implemented".

## 6. Definition of "finished"

Every acceptance test A01–A64 in VISION §24 is one of two things:
- covered by a passing test or a recorded real run, or
- explicitly listed in `docs/CAPABILITIES.md` as not implemented, with the reason (for example, needs GPU or cloud infrastructure).

The full verification in §4 passes, and everything is committed.

## 7. Milestone 6b continuation — Codex, 2026-10-04

The user asked Codex to finish Claude's interrupted Milestone 6b **without subagents**, maintain this handoff, then commit and push. This overrides the pasted historical prompt's "Do not commit". The repo was on `master`, commit `503ff9c` (6a), with uncommitted domain Python modules, fixtures/examples and 121 passing domain tests. Those files were preserved and extended.

### Implementation and files

- `python/vision/`: annotated image/box/mask/keypoint contracts, declared continuous coordinates and visibility/flip pairs, native torchvision v2 transforms (resize/crop/flip/rotate90/pad/affine), joint clipping/removal, tiny FCN training and native reference metrics. Predictions are segmentation masks; connected-component boxes are explicitly labelled derived detections.
- `python/nlp/`: source character spans, offline train-only WordPiece, first/all-subword label policy, ignored special/padding/truncated-word labels, BIO/IOB2 validation, packed tiny BiGRU training and seqeval-compatible default/strict span evaluation. Token artifacts preserve the fitted tokenizer JSON and a real padded validation batch.
- `python/speech/`: per-channel audio contracts, exact 16 kHz / 2 s / 32,000-sample teaching sine, native resampling/STFT/mel, frame lengths/masks, stride-2 convolution + packed BiGRU with torch CTCLoss, greedy decoding and edit-distance CER/WER with S/D/I alignment.
- `python/domain/`: eleven typed operations and metadata validation. `graph_core/validate.py`, `operations/__init__.py`, worker and control service integrate `kind: domain` with the existing immutable node executor, stable diagnostics and CAS inspection. Native library versions and source hashes are recorded.
- `apps/editor/src/components/DomainWorkspace.tsx` and `DomainViews.tsx`: working stage settings, vision before/after and prediction/ground-truth overlays, selectable NLP original-span/subword/label alignment and actual padding, speech waveform/mel/frame-time/greedy/error inspection. Picker links open each workspace; the typed graph canvas and coverage remain available. Values identify their run/graph/node/summary/sample and recorded configuration; absent values say "not recorded".
- Shared editor fixes: `ConfigForm.tsx` supports structured JSON and schema references without changing values on invalid JSON; `hooks.ts` prevents a previous project's polling result from selecting an unrelated run; `TabularPanels.tsx` refreshes a selected node when the run records its result. Model-only controls are absent from domain graphs.
- `examples/make_domain_fixtures.py`, `make_domain_examples.py`, three `.project.json` / `.ui.json` pairs, `fixtures/synthetic_ner.jsonl`, and `domain_journey.py` are runnable. Regenerate binary fixtures under gitignored `examples/data/domain/`; they are deterministic, labelled SYNTHETIC, and intentionally excluded from Git.
- `tests/test_domain_vision.py`, `test_domain_nlp.py`, `test_domain_speech.py`, `test_domain_api.py`: native numerical references, hand calculations, complete annotation consistency, deterministic fixtures, tokenizer truncation/padding identity, odd FFTs, ceiling resampling, stable contract refusals, and three real workers with read-only provenance inspection. Existing tests were preserved. ADR 0011 explains decisions; CAPABILITIES and generated COVERAGE describe the actual scope.

### Environment and corrections

All requested native libraries were available on CPython 3.13/macOS arm64, already installed during the interrupted Claude work and now pinned: torchvision **0.29.1**, torchaudio **2.11.0**, tokenizers **0.23.2**, seqeval **1.2.2**, torchmetrics **1.9.0**, pycocotools **2.0.11**, jiwer **4.0.0**. Existing torch is **2.14.1**. Torchaudio imports and executes; no fallback was necessary.

Corrections made during continuation: exclude a word whose final subword is replaced by SEP on truncation; centered odd-FFT frame counts use `1 + floor((N + 2*floor(n_fft/2) - n_fft)/hop)`; native resampling lengths use ceiling; invalid mel bounds and too-short reflect padding are refused. Keypoint flips use the declared continuous `W-x` convention, with flip-pair coordinates/visibility swapped together. See ADR 0011 before changing these conventions.

### Verification record

- `.venv/bin/pytest -q -o faulthandler_timeout=240`: final run after all code/test changes **864 passed, 1 skipped, 6 deselected, 1833 warnings**, 271.47 s (exit 0). Earlier complete run also passed in 275.07 s. Warnings are native TensorFlow/gast deprecations and existing Torch warnings. A restricted-sandbox rerun was stopped after local service failures/skips; it is not accepted verification.
- `.venv/bin/pytest -q -m live`: final run **6 passed, 865 deselected**, 16.11 s (local Ollama; earlier run passed in 17.72 s). Latest `.venv/bin/pytest -q tests/test_domain_api.py -o faulthandler_timeout=240`: **13 passed**, 9.17 s.
- `pnpm -C apps/editor build` and `pnpm -C apps/editor exec tsc --noEmit`: **exit 0** after final UI changes. Vite's existing large-chunk warning remains.
- `.venv/bin/python -m backends.coverage --write` and `--check`: regenerated/current. `git diff --check`: passed.
- Both fixture/example generators ran successfully. `.venv/bin/python examples/domain_journey.py --workbench /private/tmp/void-m6b-cli` trained all three full default examples in real workers. Recorded validation metrics on labelled SYNTHETIC fixtures: vision meanIoU **0.486806**, meanDice **0.547543**, pixel accuracy **0.947144**, derived mAP **0.078532**, AP50 **0.204015**; NLP microF1 **0.992701**, macroF1 **0.993648**; speech CER **0.013889**, WER **0.047619**. These are fixture evidence, not real-world benchmark claims.
- Curl smoke against temporary backend 8766: all three examples listed, NLP graph validation returned `ok: true`, and all three domain runs completed their three nodes with `kind: domain`.
- Real installed Chrome via temporary Puppeteer script `/private/tmp/void-m6b-browser.mjs`: all three picker → run → recorded stage inspections passed; mask toggles, selected subword/source highlight, keyboard speech-frame selection and 32,000-sample teaching signal checked. **No browser runtime errors**. Final log `/private/tmp/void-m6b-browser-final.log`; screenshots `/private/tmp/void-m6b-{vision,nlp,speech}.png`. Scripts/screenshots are temporary local evidence, outside the repo as §4 requires; do not depend on them surviving machine cleanup.
- Temporary backend **8766** (PID 46946) and editor **5291** (PID 46915) were stopped and confirmed absent. Temporary workbench `/private/tmp/void-m6b-smoke` is isolated. Browser process closed. The user's port 8000 service was never stopped.

### Known limits and next work

Source blocks support the declared synthetic fixture formats; general COCO/audio/text importers are not implemented. Domain models train per run: recorded summaries/predictions persist, but model/optimizer checkpoint/resume/export/serving do not. Cancellation is checked between nodes. There is no pretrained/dedicated detector, language generation, real speech recognizer, streaming/beam decoding, audio playback, forced alignment or multimodal fusion. Greedy token onsets and synthetic gold regions are labelled honestly. Native tokenizers can break vocabulary ties differently across library/platform versions; the persisted tokenizer JSON is authoritative.

Next planned milestone is **7**, with scope in §5. Build on these typed contracts and immutable recorded identities when adding pinned inference pipelines, registry and release traces; do not advertise domain serving until weights/tokenizer/transforms/environment are actually persisted and tested. Milestone 8 remains untouched.

### Git / continuation

The verified milestone is committed on the existing `master` as **Milestone 6b: typed vision, NLP and speech workflows with recorded domain inspection**, using §4's style and a Codex co-author line; use `git log -1` for its hash. No Git remote was configured. GitHub authentication works for active account `mazenDDr` (also `mazenkhaledZC` is available), but neither account has an existing `project-void` repo. The user has been asked for a destination URL or instruction to create a private repo. **Push remains pending that answer; do not invent a destination or claim a push happened.**

Next agent: read this handoff, inspect `git status` / `git log -1` / `git remote -v`, preserve any new user work, and push the verified milestone to the user's specified destination. Do not rerun completed verification unless code or environment changes require it. Keep the user's port 8000 process intact and stop every temporary server you start.
