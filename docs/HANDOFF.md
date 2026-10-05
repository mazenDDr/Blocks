# Handoff — Visual AI Workbench (project-void)

You are taking over an in-progress build. Read this whole file before doing anything.

> **Latest completed continuation: §22 — reviewed conversation reset/fork.** Read §22 for current work, verification, cleanup and remaining scope. §21 records persistent native agent conversations. §20 records isolated native agent serving. §19 records domain imports and their earlier checkpoint compatibility effects. §18 verified the cloud merges; §17 records their earlier compatibility effects/priorities. §11–§16 record the cloud implementation.

## 1. What this project is

- **Product spec (authoritative):** `docs/VISION.md`, the same as the original `README.md` the user wrote. It covers 9 milestones (0–8) and acceptance tests A01–A64 (§24).
- **Plan and rules:** `docs/PLAN.md`.
- **What actually works:** `docs/CAPABILITIES.md`, the honest ledger. Update it with every change.
- **Design decisions:** `docs/adr/0001…0025`. Read them before changing an area.
- **How to run it:** the root `README.md`. It lists only commands that were actually run.

**Repo:** `/Users/mazenkhaled/project-void`; private GitHub repository https://github.com/mazenDDr/project-void. `master` tracks `origin/master`.

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
| `python/tabular`, `connectors`, `studies`, `training`, `debugger`, `codeblocks`, `agent`, `rl`, `unsup`, `domain`, `vision`, `nlp`, `speech`, `production`, `scale`, `tracking`, `extensions` | one package per milestone area |
| `services/control` | FastAPI app (`control.app:create_app`) |
| `examples/` | example projects and fixture generators |
| `tests/` | pytest suite |

## 2. Status (as of 2026-10-05)

| Milestone | State | Commit |
|---|---|---|
| 0–1 Graph foundation and visual CNN workbench | done, verified | dd883eb |
| 2a Tabular, leakage checks, regression, statistics | done, verified | 9cb014c |
| 2b PostgreSQL/S3/DVC connectors, snapshots, studies and sweeps | done, verified | 8c83047 |
| 3 Composites, sharing, training procedure, debugger, transformer, code blocks | done, verified | b30a197 |
| 4 LangGraph agents, memory, context inspector | done, verified | 4201da2 |
| 5 RL (DQN/CartPole, grid world), unsupervised | done, verified | 5674566 |
| 6a Keras/TensorFlow and JAX backends, compatibility reports, coverage ledger, benchmarks | done, verified | 503ff9c |
| 6b Vision detection/segmentation, NLP, speech workflows | done, verified (bounded scope; see §7) | 927e9c3 |
| 7 Registry and production investigation | done, verified (bounded local tabular adapter; see §8) | 302cc50 |
| 8 Scale, integrations, community | done, verified (bounded local CPU/offline scope; see §9) | cc8b4d0 |
| Domain checkpoint/inference continuation | done, verified (native CPU completed-epoch continuation/local inference; see §10) | 90d6721 |
| A09 node cache, A44 repository import, inspection refresh, cache retention (Claude, cloud) | done, verified on Linux x86 and Mac arm64 (see §11–§13, §18) | PR #1 → 0d25b02 |
| Production serving for domain, CNN, RL, unsupervised; run-status race fix | done, verified on Linux x86 and Mac arm64 (see §14, §18) | PR #2 → 6c39878 |
| Multi-file repository imports | done, verified on Linux x86 and Mac arm64 (see §15, §18) | PR #3 → 14b7b27 |
| Optional bearer token | done, verified on Linux x86 and Mac arm64 (see §16, §18) | PR #4 → 1f5d026 |
| Bounded local COCO / CoNLL / WAV dataset imports | done, verified on Mac arm64 (see §19, ADR 0022) | 3b639e2 |
| Isolated native agent serving; worker cancellation race fix | done, verified on Mac arm64 (see §20, ADR 0023) | bfc503a |
| Persistent native agent conversations | done, verified on Mac arm64 (see §21, ADR 0024) | c7e1f18 |
| Reviewed native conversation reset/fork | done, verified on Mac arm64 (see §22, ADR 0025) | this release; `git log -1` |

After 6a: `pytest -q` → 707 passed, 1 skipped (live Anthropic test; no API key); `pytest -q -m live` → 6 passed (local Ollama).

After 6b: `pytest -q -o faulthandler_timeout=240` → **864 passed, 1 skipped, 6 deselected**; `pytest -q -m live` → **6 passed**. Editor build/type check, curl and real Chrome domain journeys passed. The skip is the existing Anthropic test without an API key. See §7 for commands, results, scope and Git details.

After 7: full suite → **894 passed, 1 skipped, 6 deselected**; live → **6 passed, 895 deselected**. Editor build/TypeScript, real HTTP CLI, curl readiness and installed Chrome train/register/serve/replay/labels/load/monitor/rollout/rollback journeys passed. See §8.

After 8: full suite → **919 passed, 1 skipped, 6 deselected**; live → **6 passed, 920 deselected**. Editor build/TypeScript, curl readiness, real native worker/tracker/connected CLI journeys, and keyboard-only installed Chrome inspection/integration journeys passed. See §9.

After domain checkpoints/inference: full suite → **931 passed, 1 skipped, 6 deselected**; live → **6 passed, 932 deselected**. Editor build/TypeScript, native HTTP CLI, curl and installed Chrome all-three-domain inference/continuation journeys passed. See §10.

After the Claude cloud sessions (§11–§16, Linux x86_64 container, not the Mac): full suite → **1000 passed, 1 skipped, 6 deselected** on `1f5d026`. Live tests were **not run** (no Ollama there). Re-verify on the Mac per §17.

After pulling the cloud merges to Mac `37368c5`: full suite → **1000 passed, 1 skipped, 6 deselected**, **438.09 s**; live Ollama → **6 passed, 1001 deselected**, **16.96 s**. Final editor/curl/Chrome evidence and scope are in §18.

After bounded domain imports: full suite → **1032 passed, 1 skipped, 6 deselected**, **447.24 s**; live Ollama → **6 passed, 1033 deselected**, **16.33 s**. Final build/curl/installed Chrome evidence is in §19.

After isolated agent serving and the worker cancellation fix: full suite → **1053 passed, 1 skipped, 8 deselected**, **444.45 s**; live Ollama → **8 passed, 1054 deselected**, **16.72 s**. Final build/curl/installed Chrome evidence and remaining scope are in §20.

After persistent native conversations: full suite → **1064 passed, 1 skipped, 9 deselected**, **447.84 s**; live Ollama → **9 passed, 1065 deselected**, **16.38 s**. Final build/curl/installed Chrome, crash/real restart and earlier stateless version compatibility evidence are in §21.

After reviewed conversation reset/fork: full suite → **1082 passed, 1 skipped, 10 deselected**, **457.54 s**; live Ollama → **10 passed, 1083 deselected**, **18.09 s**. Final native/HTTP/installed Chrome, transactional crash/restart, compatibility and cleanup evidence is in §22.

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
- **Full suite:** about 6 minutes with domain/checkpoint workflows. In Codex's filesystem/network sandbox, local service tests may fail or skip and worker tests may hang: run the required suite with local service access, not inside that restricted sandbox.

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

> **Superseded: the current remaining-work list is §22, supplemented by §18 and the unfinished items in §17.4.** The entries below are the historical milestone acceptance references, all completed.

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

### 7 — completed bounded registry/production scope (VISION §17.5–17.8, §19.5, A50, A59–A63)

Implemented and verified through the native tabular local adapter; §8 records scope/limits. Retained as acceptance reference, not an instruction to rebuild. Remote deployment remains unavailable without actual infrastructure.

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

### 8 — completed bounded scale and community scope (VISION §20, §23 Milestone 8, A49, A64)

Implemented and verified; §9 records scope, limits and continuation priorities. Retained as acceptance reference, not an instruction to rebuild. `docs/ACCEPTANCE.md` maps all scenarios and remaining gaps.

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

Milestone **7** followed this release and is recorded in §8. Build on these typed contracts and immutable recorded identities when extending serving; do not advertise domain serving until weights/tokenizer/transforms/environment are actually persisted and tested. At that historical release, Milestone 8 was untouched; its current status is in §9.

### Git / continuation

The verified milestone is **927e9c3**, committed on `master` as **Milestone 6b: typed vision, NLP and speech workflows with recorded domain inspection**, with a Codex co-author line. The user then explicitly requested `gh repo create project-void --private --source=. --remote=origin --push`. Created the **private** repository https://github.com/mazenDDr/project-void and pushed the complete committed history; `master` now tracks `origin/master`.

The initial automatic push failed because global Git config rewrites `https://github.com/` to SSH, whose identity lacks access to this account. Only this repo's remote was changed to `https://mazenDDr@github.com/mazenDDr/project-void.git`, which bypasses that rewrite and uses the existing `gh auth git-credential` helper. No token is stored in the URL and no global config was changed. Active GitHub account is `mazenDDr`; `mazenkhaledZC` is also authenticated. `git push -u origin master` succeeded. This handoff update is committed and pushed separately after the milestone.

Next agent: read this handoff, inspect `git status` / `git log -1` / `git remote -v`, preserve any new user work, and follow the latest status in §9. Do not rerun completed verification unless code or environment changes require it. Keep the user's port 8000 process intact and stop every temporary server you start. Continue committing verified milestones and pushing their handoffs to `origin` as the user requested.

## 8. Milestone 7 — Codex continuation, 2026-10-04

The user requested continued work after the private GitHub push. Continue without subagents. Starting point: clean `master`, `233ee72`, synchronized with `origin/master`.

### Implementation and invariants

`python/production/{models,pipeline,store,runtime,monitor,traffic}.py`, `services/control/production_api.py`, control app registration, and tabular worker fitted-pipeline capture implement this milestone. Native scikit-learn estimators + fitted imputer/one-hot/scaler objects are captured during the real run, with raw-input schema, ordered transforms/features, labels, source/graph/evaluation identities and the declared native environment. Adapter/shared transform code hashes are pinned and checked even when objects are cached. Only internal hash-verified worker pickle artifacts load; no pickle-upload API. Unsupported paths get an explicit refusal artifact. Local/staging are distinct real routes in the same single CPU FastAPI process; there is no remote infrastructure.

Registry versions and release candidates are immutable CAS records. Deployment/rollback uses compare-and-swap on the prior route; lifecycle events persist. Requests pin their route/version at admission, have bounded queues/timeouts/batches, idempotency and recorded traces. Optional application state is a durable counter isolated by release/user/session with serialized updates; cancellation prevents commit even if the native forward pass finishes. Caller-declared user IDs are not authentication. Monitoring separates native input/prediction drift from label-based quality. Traffic uses bounded real loopback HTTP and records generator drops, observed latency/errors and combined server/generator resource use.

`apps/editor/src/components/ProductionWorkspace.tsx` adds Registry, Release, Requests, Traffic and Monitoring views, exposed by App's Production tab. Working controls register exact completed runs, move aliases, edit supported serving settings, preview/explicitly deploy/rollback, load recorded input batches, send and cancel requests, inspect lineage, isolate replay, submit ground truth, generate/cancel measured HTTP load and inspect drift/quality. Traffic and monitoring show readable observed metric tables with full evidence in expandable records. Shared schema errors are displayed with field locations. No unsupported cloud/streaming control is presented. `onOpenRun` opens the existing research inspector.

`examples/make_production_example.py` generates 160 deterministic labelled SYNTHETIC sensor rows, `production_sensors.project.json` and `.ui.json`. It is a train-only fitted scaler plus real LogisticRegression, with held-out native validation metrics. `examples/production_journey.py` performs the entire workflow via real HTTP against a running loopback workbench. `tests/test_production.py` has **30 passing cases**, covering native classifier/regressor agreement, imputation/one-hot/scaling order, raw schema errors, artifact/environment integrity, source independence, readonly fitted artifacts, actual different-weight rollout with in-flight pinning, alias/staging isolation, idempotency, 20 concurrent stateful requests, active native-call cancellation, queue/deadline behavior, single-owner restart, label/drift references, and all four real HTTP traffic patterns including saturation, cancel/drain and persisted history. No existing tests were weakened.

Reference data is limited to the first 2,000 rows in the recorded training order, with count/truncation policy; the aligned training labels have their own CAS identity. Loaded reference labels are clearly labelled in-sample evidence, not held-out production accuracy. Monitoring's observed window is the newest 1,000 requests for the selected release/time. Capture defaults off; uncaptured inputs cannot be replayed or used for input drift. Counter identity includes the release, user and session; rollback recovers the prior release's counter namespace. On a single-owner restart, requests with no committed trace become failed (503) and never resume state updates. Native calls are not forcibly interrupted; cancellation/timeouts discard their result before state commit.

### Commands and final evidence

- `.venv/bin/python examples/make_production_example.py`: **generated 160 labelled SYNTHETIC rows and project**.
- `.venv/bin/pytest -q tests/test_tabular_api.py tests/test_tabular_ops.py -o faulthandler_timeout=240`: **40 passed**, 14.66 s after fitted artifact capture.
- `.venv/bin/pytest -q tests/test_production.py --tb=short -o faulthandler_timeout=240`: **30 passed**, 6.93 s; final pre-commit rerun **30 passed**, 7.16 s after normalizing generated CSV line endings to LF. Fixture values are unchanged; prior source hashes identify the original recorded bytes.
- `.venv/bin/python -m backends.coverage --write`, then `--check`: **current**. The first full run found only two stale-ledger assertions (892 other tests passed); regenerated and reran the full suite successfully. Do not weaken the ledger-current tests.
- `.venv/bin/pytest -q -o faulthandler_timeout=240`: final **894 passed, 1 skipped, 6 deselected, 1833 warnings**, **279.85 s**, exit 0. Skip/warnings remain the existing Anthropic-without-key / native Torch and TensorFlow-gast cases.
- `.venv/bin/pytest -q -m live`: **6 passed, 895 deselected**, **16.09 s**, exit 0.
- `pnpm -C apps/editor build`: **245 modules**, exit 0 after final UI changes; `pnpm -C apps/editor exec tsc --noEmit`: exit 0. Existing Vite large-chunk warning remains. `git diff --check`: passed.
- Temporary backend command: `VOID_WORKBENCH=/private/tmp/void-m7-smoke .venv/bin/python -m uvicorn control.app:create_app --factory --app-dir services --host 127.0.0.1 --port 8767`. Temporary editor: `VOID_API=http://127.0.0.1:8767 pnpm -C apps/editor dev --host 127.0.0.1 --port 5292`.
- `.venv/bin/python examples/production_journey.py --base http://127.0.0.1:8767 --namespace m7-cli`: **exit 0**. Real worker run **ba715ec85cad**; version **646bdec3bc227c3aa37429b29410fe0937704e5d5019b32d6bc612c4abf51be5**; release **1df1625afe0df168f3f260df66a0537afda90e7f83f33df67d94f4789b3b81dc**; request trace **ca75da9ac6f794ce3440aa3ad4c45bf926c670c98765dc0f5dce680e3fc23f33**; traffic result **4d37ac5f9b24ecd2c23372c5869bdd4fc9c4d45e9bb2caad57300f6bef22642f**. Test offered/sent 10 requests, **10 succeeded / 0 errors / 0 generator drops**, achieved **9.99265 successful requests/s**, p50 **8.71175 ms**, p95 **9.92723 ms**, p99 **9.97121 ms**. Warmup, isolated replay, explicit labels and known-release rollback verified. Quality 1.0 on 3 recorded training-reference rows is **in-sample SYNTHETIC evidence**, not a production benchmark.
- Required curl smoke: `curl -fsS http://127.0.0.1:8767/api/serve/local/m7-cli/health`: **ready true**, exact release/version above, one CPU replica and pinned serving limits.
- Installed Chrome using `/private/tmp/void-m7-browser.mjs` (temporary Puppeteer script outside repo): **passed** picker → actual train → register → alias → release preview/deploy → real request/counter → source/fitted/evaluation lineage → explicit labels → isolated replay → observed monitoring → measured HTTP → rollout/rollback → restored counter continuity. **No browser runtime errors**. Final screenshots `/private/tmp/void-m7-{requests,monitor,traffic}.png` inspected. Browser test had temporary text-case/input-replacement issues; corrected the test script and completed the full journey. Screenshots/scripts are temporary local evidence, not required for a fresh clone.
- All native dependencies were already installed/pinned; **no new libraries installed**. Adapter records Python 3.13.12, scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.6. Monitoring uses existing SciPy 1.18.1; resource measurements use existing psutil 7.2.2. Existing FastAPI/httpx/uvicorn serve and generate the real traffic.
- Temporary backend **PID 50602 / port 8767** and editor **PID 50617 / port 5292** were stopped. Browser closed; test-created ephemeral servers close in fixture cleanup. Workbench evidence remains isolated under `/private/tmp/void-m7-smoke`. The user's port 8000 service was never stopped.

### Scope, limitations and next agent

Read ADR **0012**, CAPABILITIES, README and regenerated COVERAGE for the detailed contract. This completes A50/A59–A63 through a bounded native tabular representative. There is **no** PyTorch CNN/domain/agent/RL/unsupervised/Keras/JAX registry-serving adapter, remote deployment, authenticated multi-user boundary, multi-process replicas, autoscaling, canary/shadow allocation, dynamic batching, streaming, asynchronous batch jobs, online learning, automatic retraining/rollback or retention/garbage collection. Optional state is a durable application request counter, not LLM memory. Use one owning control process; declared user names are isolation keys, not authentication. Admission bounds cover the adapter; traffic resource scope includes both local server/generator and warmup/drain, with sampled RSS and no cost estimate.

The milestone, ADR, docs and this handoff are committed together on `master` with the §4 style / Codex co-author and pushed to private `origin`; use `git log -1` / `git status` for the exact commit. At this historical release, the next planned milestone was 8, with §5's scope. Preserve the current implementation. Begin with the remote-worker and tracker integration contracts, honest local-vs-remote labels, idempotent external-ID mapping, extension conformance and an A01–A64 acceptance checklist. No work on 8 was started in the Milestone 7 release; see §9 for the subsequent implementation. Continue without subagents unless the user changes that preference; keep the handoff current and push each verified milestone.

## 9. Milestone 8 — continuation record, Codex 2026-10-04

Starting point: clean `302cc50` on `master`, pushed. The user requested continued implementation **without subagents**, with verified commits and pushed handoffs so Claude can resume. Preserve that preference and the private Git remote configuration in §7. This is bounded Milestone 8 evidence, not a claim that every behavior in VISION is implemented.

### Implementation and file map

- `python/scale/{common,state,remote,worker_server}.py`: authenticated `void-worker/1` HTTP with a real separate local CPU process. Native tabular CSV, selection/profile/cleanup, train/validation split, fitted imputation/one-hot/scaling, linear/logistic regression and metrics only. Original graph and input bytes are hashed before transfer; CSV paths are privately materialized, with a distinct execution graph identity. No shared source filesystem is assumed. Durable intentions and known worker IDs are saved before result transfer; reconnect reuses the same job. Terminal artifacts are membership/size/SHA checked before atomic idempotent metadata import. Imported pickle artifacts **remain untrusted for local serving**. Cancellation is between graph nodes.
- `python/tracking/{bridge,native}.py`, `requirements.txt`, `lock.txt`: terminal-run export intentions and immutable CAS payloads. Default shares only run ID/hash/status; numeric metrics and native sklearn evaluation-summary artifacts are explicit opt-ins. No raw rows, graph/config/code or model pickle export. Real MLflow SQLite checks native run tag, metric history and content-addressed artifact path on recovery. Native confirmation with lost local acknowledgement does not duplicate mappings/history/artifacts. Real W&B offline writes one immutable snapshot, disables git/code/system metadata capture, finishes before confirming its directory, rebuilds only unconfirmed owned pending directories, and reuses confirmed mappings. This is not live offline resume or cloud upload.
- `python/extensions/{sdk,packages}.py`: strict trusted native operation manifests pin version/source/settings/ports/dependencies/state/effects/inspector/docs/example/numerical cases. Pure tabular activation is explicit startup `VOID_PLUGIN_MANIFESTS` (paths separated by `os.pathsep`); graph import never installs or activates Python. Native conformance checks independently declared expected outputs, schema, determinism and input immutability. It is trusted code, **not a security sandbox**. Workers record activated package identities for used operations. Inert CAS project packages contain graph/UI/exact operation dependencies/environment requirements, optionally explicit bounded CSV snapshots. Unknown operations remain readable; missing/different dependencies block execution with `E_PACKAGE_DEPENDENCY`. Hash/size/path/symlink checks protect owned package resources. Structured secret fields and credential-bearing URLs are refused; freeform researcher text is not a universal secret scanner. Environment requirements are declared, not automatically installed/enforced.
- `services/control/scale_api.py` and app registration: working integration/worker/export/package/conformance/acceptance endpoints, plus recorded comparison/conclusion evidence. Baseline/variant summary values come from verified native node-summary artifacts; conclusions bind exact run/source/fitted/evaluation CAS identities. `graph_core/registry.py`, `validate.py` and `worker/tabular_run.py` integrate explicit activation, dependency validation and provenance without weakening existing validation.
- `apps/editor/src/components/{ScaleWorkspace,KeyboardGraphTools}.tsx`, `App.tsx`, styles: actual worker submit/retrieve/cancel/inspect, native tracker queue/reconnect/sharing, package export/inspect/import, trusted bundled SDK checks, measured comparison table with full evidence, conclusion recording and 64-scenario acceptance view. Keyboard tools select graph nodes/wires and connect ports through the existing mutation/validation path; controlled selection retains the chosen item. Native forms and visible focus support the tested reference workflow.
- `examples/plugins/offset/{operation.py,manifest.json}`: pure native pandas constant offset teaching operation with usable defaults and two declared numerical cases; source hash `6cb3c37510039bb6e56f062718a7c95bf1fd4e56ea189e687cc8276de31a8b3e`. This is not a learned transform. `examples/scale_journey.py` measures real local vs HTTP work and tracker recovery. `examples/connected_production_journey.py` starts/stops actual local PostgreSQL, seeds labelled SYNTHETIC sensors, then uses normal HTTP connector/query/project/run/snapshot/comparison/conclusion/registry/serving/traffic APIs; no manual data extraction into an IDE is required after fixture setup.
- `tests/test_scale.py`: 25 cases including real child-worker HTTP, local/native agreement, auth/admission/subset/input/output refusals, lost-ack recovery and atomic imports, native MLflow history/artifacts, W&B persisted binary records (CRC plus native SDK protobuf), explicit sharing, inert packages/dependency/hash/path/credential refusals, trusted SDK activation/conformance and the complete PostgreSQL-to-serving evidence chain. The new coverage was regenerated; no old test was weakened.
- `docs/acceptance.json` / `ACCEPTANCE.md`: all A01–A64 titles mapped to tested bounded scope or explicit unimplemented behavior. A09 dependency-scoped numerical caching and A44 repository source-code import remain not implemented. A20 and A64 have precise limits, not full accessibility/onboarding claims. ADR 0013, CAPABILITIES, README and PLAN explain contracts. `benchmarks/results/m8_local_worker.json` retains measured inputs, run IDs, timings and native agreement.

### Native environment and operating bounds

MLflow-skinny **3.16.1**, W&B **0.30.0**, SQLAlchemy **2.0.48**, Alembic **1.18.4** installed in isolated **`.venv-trackers`**, with complete installed lock in `python/tracking/lock.txt`. Tracker protobuf is **6.33.6**; main `.venv` remains verified **7.36.2**. `.venv-trackers` is gitignored. Do not install these tracker dependencies into the main training environment. The bridge uses the isolated native subprocess with JSON input and 90-second timeout. Missing runtime leaves exports pending and reports an error, with no pretend confirmation.

The worker binds 127.0.0.1; client accepts only explicit `http://127.0.0.1:PORT`. Bearer credentials are environment references, never persisted in bundles. Limits: 64 graph nodes, 8 MiB input bytes, 12 MiB streamed HTTP request, two active jobs, 32 MiB aggregate output transfer. Package resources are at most 8 MiB/64 files. Bounds do not guarantee native wall time or GPU isolation. One owning control/worker process per workbench; no multi-process replica support.

An intention without a recorded worker run requires investigation/new identity rather than silent execution. A worker restart does not promise adoption/cancellation of an orphaned active native child; investigate nonterminal jobs explicitly. Deleted/externally modified confirmed tracker destinations require investigation; no background reconciliation or upload policy is implied. Cross-host TLS/provisioning, GPU, distributed scheduling, online trackers/credential UI and authenticated team access are unavailable.

### Verification commands and recorded results

- `.venv/bin/pytest -q tests/test_scale.py --tb=short -o faulthandler_timeout=240`: final **25 passed, 32.20 s**, exit 0.
- `.venv/bin/pytest -q -o faulthandler_timeout=240`: final **919 passed, 1 skipped, 6 deselected, 1833 warnings**, **313.25 s (5:13)**, exit 0. Existing skip is Anthropic without an API key; warnings remain native TensorFlow/gast/Torch cases. Log `/private/tmp/void-m8-pytest.log` is temporary evidence.
- `.venv/bin/pytest -q -m live`: **6 passed, 920 deselected**, **17.48 s**, exit 0, real local Ollama.
- `pnpm -C apps/editor build`: **247 modules**, exit 0 after final keyboard/metric-table changes; `pnpm -C apps/editor exec tsc --noEmit`: exit 0. Existing Vite large-chunk warning remains.
- `.venv/bin/python -m backends.coverage --write` and `--check`: regenerated/current. `git diff --check`: passed. `.venv/bin/python -m extensions.sdk examples/plugins/offset/manifest.json`: **passed, two native numerical cases**, exact source identity above.
- `.venv/bin/python examples/scale_journey.py --base http://127.0.0.1:8768 --worker http://127.0.0.1:8778 --output benchmarks/results/m8_local_worker.json`: exit 0. Three seed-controlled local/HTTP native evaluations matched exactly. Seeds 71/72/73 local wall times **1.860775 / 1.903135 / 1.887405 s**; HTTP wall times **3.264782 / 1.955246 / 1.976406 s**. Actual timing includes HTTP/polling/process startup/input/result transfer; first-worker cold effects are not controlled away. This establishes no remote-hardware performance advantage. Local runs `89dd58bc84c5`, `adcf7dcb624d`, `b5c56daaad9e`; complete remote identities and MLflow/W&B confirmation mappings are in the committed JSON. Repeat retrieval/exports reused confirmed identities.
- `.venv/bin/python examples/connected_production_journey.py --base http://127.0.0.1:8768`: exit 0. Real runs **9c5cc0df25bd / 8aa097419cb0** use one recorded PostgreSQL source snapshot and 120 train/40 validation rows; actual C=1 vs C=0.1 accuracy both **1.0**, log loss **0.1643845884 vs 0.3416004170**. Conclusion CAS **99b359947b3f68021ae41ef21ac23b41440bfe740ec47d35d897f2adb06c5c34**; version **9d0c5df1fac8e930e7636a21a2ca88c2235a710191171dc26a494a6771844337**; release **f702d2c1efcf85a0c89c3887e47cf6be33be1af4131e2aadbbfc30e0b6dc6c09**; trace **220ae962c830315b1b70852aac7fbadf4fe01e1109bf484285ae30bd78ec86a6**. Five real HTTP requests succeeded, zero errors, achieved **4.997928 successful requests/s**, p50 **8.593916 ms**, p95 **8.959241 ms**, p99 **8.998781 ms**. PostgreSQL stopped in `finally`; serving uses pinned artifacts independently. All quality/traffic evidence is explicitly SYNTHETIC local fixture evidence, not a production benchmark.

### Browser, cleanup and Git

- Required curl: `curl -fsS http://127.0.0.1:8768/api/serve/local/m8-connected/health` → **ready true**, exact release/version above, one local CPU replica. The actual PostgreSQL fixture was already stopped; the pinned pipeline still served independently.
- Installed real Chrome with temporary Puppeteer script `/private/tmp/void-m8-browser.mjs`: **all passed, no browser runtime errors**. Keyboard only: new/reference graph → save → native settings change (24 then 32 filters) → connect ports → actual training → recorded learned weights → select real validation sample → wire activation values/feature maps with run/graph/checkpoint/sample provenance. Final model run **2c1510abfcbb**, checkpoint step **20**. The script used 964 Tab presses for the reference flow; this is a functional proof, not a claim of optimized navigation or broad accessibility certification.
- The same Chrome journey ran native tabular training (**e5260ee1a0a9**), separate-worker submit/retrieve (import **remote-de44c45a4da1cbf2578e8032**), actual MLflow queue/confirmation, bundled native SDK conformance, recorded connected baseline/variant comparison and conclusion, source snapshot/120-row train-owned fitted state/40-row held-out metrics inspections, all 64 acceptance rows, and CSV package export/dependency inspection/import/open. Actual baseline log loss **0.164385** appeared with provenance. The temporary PostgreSQL connection error on the current draft was expected after fixture cleanup; completed-run snapshots and fitted/evaluation values remained readable. No source-error value was invented.
- Final log `/private/tmp/void-m8-browser.log`; screenshots `/private/tmp/void-m8-{keyboard,connected,acceptance}.png` inspected. Temporary test-script issues were corrected (native select typeahead, waiting for asynchronous project loads/worker completion/API records, exact displayed metric names); final checks all passed. A real controlled-select retention problem was fixed in KeyboardGraphTools before final build/full tests. Scripts/screenshots/logs are outside the repo as §4 requires; do not depend on machine-temporary evidence surviving cleanup.
- After correcting the A05 checklist reference to the actual tabular leakage tests, `.venv/bin/pytest -q tests/test_scale.py::test_acceptance_checklist_covers_all_scenarios_and_has_real_references` → **1 passed, 0.69 s**. This changed documentation only; final Python/editor implementation remained the full-suite/build-verified code. Coverage `--check` remained current.
- Temporary backend **PID 53595 / port 8768**, editor **PID 53613 / port 5293**, and worker **PID 53624 / port 8778** were stopped and confirmed absent. Browser closed; test/CLI PostgreSQL servers stop in fixture/finally cleanup. Workbenches remain isolated in `/private/tmp/void-m8-smoke` and `/private/tmp/void-m8-worker`. Only a fixture worker token was used; no real credentials were accessed. The user's port 8000 service was never stopped.
- Milestone implementation, benchmark, ADR, capabilities/checklist/README/PLAN/COVERAGE and this handoff are committed together on `master` using §4 style and a Codex co-author line, then pushed to the private `origin`. Inspect `git log -1`, `git status` and `origin/master` for exact commit/push state. No generated virtual environments, temporary browser files or workbench data are committed.

### Remaining work for Claude

Read ADR 0013/0014 and ACCEPTANCE before extending this milestone. Domain state persistence, completed-epoch child continuation and separate local native inference/export subsequently shipped in §10. Native tabular debugger interventions, broad external-user onboarding/accessibility, cross-host/GPU operation, online tracker sharing, A09 numerical dependency caches and A44 repository source-code import remain gaps. General production registry/release adapters for domain/CNN/agent/RL/Keras/JAX remain unavailable. Model inspection opened before its checkpoint appears used to retain “not recorded” until the tab/context was reopened; fixed in §13 (it re-asks while the run is active).

Next agent: inspect `git status`, `git log -1` and `git remote -v`; preserve new user work. Select any next scope from the actual acceptance gaps and latest user direction rather than rebuilding completed milestones. Continue alone, keep this handoff current, run §4 verification for substantive changes, commit verified work and push to the private origin. Never stop the user's port 8000 service; close every temporary server/browser you start.

## 10. Domain checkpoints and local inference — Codex, 2026-10-04

The user requested continuation after the remaining-gap review. Starting point clean, pushed `cc8b4d0`. Worked **without subagents**; keep that preference, preserve port 8000, and continue committing/pushing verified implementation and handoff updates. This completes a bounded follow-on release: persistent native domain models, exact completed-epoch CPU child continuation, checkpoint/manifest export and source-independent local inference. It does not complete domain production registry/releases or the entire VISION.

### Implementation and invariants

- `python/domain/checkpoints.py`: actual native model/Adam/shuffle-generator/PyTorch-RNG/curve/epoch state dictionaries. Speech includes train-only per-mel mean/std. Manifests pin original source, prepared-data/training signature, architecture/preprocessing/tokenizer/labels, native Python/library versions, implementation file hashes, model/run/graph/node/parent identities and a recorded held-out input reference. Max checkpoint 32 MiB. SHA, completed source run and internal run/node artifact membership checks precede loading; explicit `torch.load(..., weights_only=True, map_location="cpu")`, no whole-module pickle, uploads or arbitrary paths. Nonfinite weights are refused.
- `python/vision/segment.py`, `nlp/model.py`, `speech/ctc.py` extend the existing trainers, preserving native algorithms. Restoring optimizer, shuffle/RNG and prior curve starts at the completed epoch; epochs is the **total target**, not additional epochs. `domain/{vision_ops,nlp_ops,speech_ops}.py` records checkpoints during real worker execution. `resume_model_id` references the parent manifest. Prepared tensors/labels/source/transforms, model/optimizer/split settings and CPU thread count must match. Parents remain immutable; continuation creates a new run with recorded parent. Failure/cancel/mid-batch resume is not offered.
- NLP tokenizer node adds `fitted_model_id` for explicit reuse of authoritative tokenizer JSON. Native vocabulary tie-breaking can differ on refit, so continuation reloads the parent's fitted tokenizer and verifies corpus and tokenizer/split settings. Tagger checks actual prepared IDs/labels/words and vocabulary identity. No fitted vocabulary is reconstructed from validation text.
- `python/domain/inference.py`, `services/control/domain_api.py` and app registration: `/api/domain/models`, `/models/{id}`, `/checkpoint`, `/example`, `/predict`. Only recorded completed internal models can run. Inference never rereads source files, fits, trains or appends run events/artifacts; unavailable model identities stay visible with specific errors. Native architecture is reconstructed locally from the pinned manifest and strict state dict. Original run/source/graph/node/checkpoint/epoch provenance accompanies predictions.
- Vision requests are RGB PNG **at the pinned post-geometry size**, max 512 per dimension, with native saved numerical normalization; geometry is explicit, never implicitly resized/cropped/random-flipped. Outputs are class masks and honestly labelled derived-component detections. NLP requests are original nonempty text ≤4000 characters; saved tokenizer/truncation/attention/label/readout semantics retain character spans and report invalid predicted IOB2 transitions. Speech requests are finite normalized PCM in [-1,1], 1–4 equal-length channels, exact pinned rate, ≤32,000 samples/channel and ≤4096 feature frames; mean-to-mono/pinned framing/mel/saved train-only normalization/greedy decoding remain explicit. Onsets are not forced alignment. Request batches 1–4 within 1.5 MB decoded JSON; max two concurrent native requests. These are input/admission bounds, not wall-time or security guarantees.
- `apps/editor/src/components/DomainModels.tsx`, `DomainWorkspace.tsx` and scoped styles: real checkpoint/manifest links, additional-epoch draft preparation, loaded recorded SYNTHETIC held-out input, editable inference JSON and native prediction. Continuation changes the draft; Run creates the child. Vision mask, NLP original-word/span/label table and speech text/onsets are readable; full records/provenance are expandable with bounded display height. NLP text is labelled input, not generated text.
- `examples/domain_checkpoint_journey.py`: actual three-family HTTP training/export/predict/continuation/parent-immutability workflow. `tests/test_domain_checkpoints.py`: 12 parameterized cases across all three families, native recorded prediction/forward agreement, batch agreement, restart with absent sources, exact continued vs uninterrupted weights/Adam/RNG/curves, stable input/hash/trust/environment/source/config/epoch refusals and read-only events/artifacts. No existing tests were weakened.
- ADR **0014**, CAPABILITIES, README, PLAN, ACCEPTANCE JSON/Markdown and this handoff updated. Old contradictory capability exclusions (local trackers/packages, tensor primitives, connected sources, supported checkpoint resume) were consolidated; later bounded milestone sections remain authoritative. A56–A58 now reference native checkpoint evidence. Coverage is current. No new native libraries installed; main and tracker environments remain separate.

### Final verification and concrete evidence

- `.venv/bin/pytest -q tests/test_domain_checkpoints.py --tb=short -o faulthandler_timeout=240`: **12 passed, 60.88 s** after checkpoint/example/trust updates. Initial native run also passed (59.44 s). A subsequent stricter-membership test-fixture error was corrected by providing the correct node metadata, preserving the intended environment-refusal assertion. Full suite below includes final batch assertions.
- `.venv/bin/pytest -q -o faulthandler_timeout=240`: **931 passed, 1 skipped, 6 deselected, 1833 warnings**, **375.04 s (6:15)**, exit 0. Existing skip is Anthropic without key; native TensorFlow/gast/Torch warnings unchanged. Log `/private/tmp/void-domain-checkpoint-pytest.log` is temporary evidence.
- `.venv/bin/pytest -q -m live`: **6 passed, 932 deselected**, **16.36 s**, exit 0, actual Ollama. Log `/private/tmp/void-domain-checkpoint-live.log`.
- `pnpm -C apps/editor build`: **248 modules**, exit 0; `pnpm -C apps/editor exec tsc --noEmit`: exit 0 after final UI changes. Existing Vite large-chunk warning remains. `.venv/bin/python -m backends.coverage --write` and `--check`: current; `git diff --check`: passed.
- `.venv/bin/python examples/domain_checkpoint_journey.py --base http://127.0.0.1:8769`: **exit 0**, all three real native CPU families, epochs 2→4, exported SHA checks and unchanged parent predictions. These short runs test persistence/continuity on SYNTHETIC fixtures, not useful model quality or real-world performance. Output `/private/tmp/void-domain-checkpoint-cli.json` is temporary. Source files are only needed for training/continuation, not saved-model inference.

| Family | Parent run | Model manifest ID | Child run | Checkpoint export bytes |
|---|---|---|---|---|
| Vision | b07770d2d577 | 9ca38791fe3d67efc4a50ac8f5548d6077d1fb5738c68c54fbc2f7a02add5995 | 46745fa45baf | 43,533 |
| NLP | d1520249e668 | 69a3792f6c7a739d8d20e7dfb7ae647c2f01fd97ff593bb366a911acee404a7c | ac2ed6d9fee6 | 55,257 |
| Speech | 048b162c8e0d | cb666f6d0303b19e50e766fa04fa463ffbd9f0e6cda0a3d57e16c143d7ff1f98 | 50775f7515a6 | 48,071 |

- Curl `GET http://127.0.0.1:8769/api/domain/models/9ca38791fe3d67efc4a50ac8f5548d6077d1fb5738c68c54fbc2f7a02add5995`: native manifest **void-domain-model/1**, family vision, parent run above, epoch 2, checkpoint **d43f43b8d8f4eb7837126682a2880544792d934d682c58741d81241c124e9eb3**. Unknown identity returns 422. Real prediction HTTP is verified by CLI/Chrome/tests.
- Installed Chrome through temporary `/private/tmp/void-domain-checkpoint-browser.mjs`: all three saved projects → selected completed parent → load held-out input → predict with saved model → visible mask/span-label table/greedy output with exact provenance → prepare draft continuation → verify parent manifest and total epoch 4 in native settings → Run → completed child checkpoint. **All passed, no browser runtime errors.** Temporary script load timing and JavaScript argument-scoping issues were fixed; no app behavior was bypassed. Final screenshot/log paths `/private/tmp/void-domain-checkpoint-{vision,nlp,speech}.png` and `void-domain-checkpoint-browser.log`; snapshots inspected. These temporary files need not survive machine cleanup.

### Cleanup, commit and next scope

Temporary backend **PID 56926 / port 8769** and editor **PID 56959 / port 5294** were stopped with SIGTERM and confirmed absent with `ps`. Browser closed in `finally`; no separate worker HTTP server was started. Tests finished their spawned native workers. The user's port 8000 was left untouched. Isolated workbench `/private/tmp/void-domain-checkpoint-smoke` contains native evidence. Final Chrome continuation child runs: vision `34387dc3d5b1`, NLP `0f4bfbd22bc0`, speech `6007b9f003f7`. An additional final NLP browser check confirmed the original text is labelled correctly, with zero runtime errors.

This release is committed on `master` and pushed to the existing private origin together with this handoff; find its exact commit using `git log -1`. The final report records the verified pushed commit. Keep subsequent work documented and pushed so another agent can resume from Git rather than temporary local evidence.

Next scope: integrate supported domain models into real registry/release/trace/monitoring contracts, expand dataset importers, or improve editor/inspection/worker recovery per user direction. This release is **not** domain registry rollout/production monitoring, intermediate checkpoint cadence, mid-node pause/cancel/recovery, portable checkpoint upload/import/migration, raw-image geometry inference, GPU/cross-version exactness, or cloud/team deployment. Models pin the implementation/environment; editing those files can intentionally make earlier manifests unavailable until migration/rerun. No cost, model-quality or security guarantee follows from the small local fixture proof. A09/A44 and the other ACCEPTANCE/CAPABILITIES gaps remain.

Continue without subagents. Read ADR 0014 before extending checkpoint semantics; inspect Git/status/remote and preserve user work. Update handoff/capabilities, verify §4, commit and push to the existing private origin. Never stop port 8000, and stop every temporary server/browser you start.

## 11. A09 dependency-scoped node cache — Claude, 2026-10-04

The user asked Claude to continue Codex's work using this handoff. Starting point: clean `90d6721`. This session ran in a **Linux x86_64 cloud container** (not the macOS machine in §3), on branch `claude/laughing-bell-la46nd`, which was pushed to the private origin. Port 8000 / Ollama do not exist here. Scope chosen from the remaining ACCEPTANCE gaps: **A09**. A44 remains not implemented.

### Implementation

- `python/tabular/cache.py` (new): per-node cache keys over operation type/version, node id, resolved config (after run seed override), SHA of every tabular implementation file, Python/native library versions, and input identities. Cacheable producers pass their key downstream. Sources pass a content hash of what they actually read, so editing a node invalidates it and its dependents only. Explicit allowlist of deterministic tabular/sklearn/scipy/unsupervised ops. CSV/connector sources, joins, code blocks, plugins and `domain` nodes always run (`bypass`). Miss explanations name `settings`, `implementation`, `environment` or `input <port> (from <node>)` against the node's latest entry in the same project.
- `artifact_store/store.py`: `node_cache` SQLite index (written only by the local worker; no import/upload path). Entries are pickles in the CAS store, SHA-, format- and key-verified before loading; failures become a miss and the node runs. 64 MiB entry cap.
- `tabular/engine.py`, `worker/tabular_run.py`: `TabularRunConfig.cache: "off" | "reuse"` (default off: unchanged behavior, nothing recorded). `run_started.cache` records the mode and hashes; `node_finished.cache` records the decision. Hits still record ordinary node_output/summary artifacts in the new run.
- `services/control/app.py` run summary exposes `cache` per run and node. Editor `TabularPanels.tsx`: **Reuse unchanged node results (cache)** checkbox, reused/ran badges, and a **Node cache** explanation table in the Run record.
- ADR **0015**; CAPABILITIES, ACCEPTANCE (A09 → bounded evidence), README updated; COVERAGE regenerated.
- `tests/test_node_cache.py` (9 cases): first-run misses, identical-rerun hits, byte-identical outputs vs original and vs uncached execution, single-node edit invalidates exactly dependents, late edits, revert reuse, changed source bytes, run seed override, implementation/environment change, corrupted entry, cache-off unchanged, non-cacheable/domain bypass, HTTP API summary and 422 for an invalid mode.
- `tests/test_scale.py` acceptance assertion: A44 must still be `not implemented`; A09 must now be `bounded evidence` citing `tests/test_node_cache.py` (strengthened, not weakened).

### Portability fixes found on Linux

- `connectors/install_pgserver.py`: the compound manylinux tag was passed as one `--platform` value, which pip rejects. It now passes one `--platform` per tag part. Installed and verified here.
- `tests/test_debugger.py` conditional-breakpoint test: on Linux x86 the CE/`seq_model` fixture at lr=1e30 never produced a non-finite loss (losses ~1e29, bounded by tanh and stable log-softmax). It also failed on the base commit. Replaced the fixture with a dense→tanh→dense MSE regression model, where one step leaves weights finite and the next squared prediction overflows float32 by ~20 orders of magnitude. All original assertions are kept, plus `step == 2`. Debugger file: 19 passed; repeated 3× stable.

### Verification (this container)

- Environment set up from scratch: `.venv` (CPython 3.13.x, `pip install --extra-index-url https://download.pytorch.org/whl/cpu -r python/requirements.txt`, `pip install -e . --no-deps`), `.venv-trackers` from `python/tracking/requirements.txt`, `python -m connectors.install_pgserver`.
- `.venv/bin/pytest -q -o faulthandler_timeout=240`: **939 passed, 1 skipped, 6 deselected, 1 failed** (961 s). The single failure was the debugger test fixed afterwards (not rerun in a whole-suite pass after the fix; `tests/test_debugger.py` alone: 19 passed). The skip is Anthropic without a key. An earlier full run before the trackers venv/pgserver existed had environment-only failures; that run is not accepted evidence.
- `pytest -m live`: **not run**: no Ollama in this container.
- `pnpm -C apps/editor build` and `exec tsc --noEmit`: exit 0 (existing large-chunk warning). `backends.coverage --write`/`--check`: current.
- Curl smoke on a temporary backend at 127.0.0.1:8770: the tabular_regression example ran with `cache: reuse`. Then `sc_fit` was edited: 10 upstream nodes reused, `sc_fit` + 6 dependents executed with exact `changed` reasons, `housing` bypassed.
- Real headless Chromium (Playwright, `/opt/pw-browsers`; script in scratchpad, outside the repo) against editor 5295 on a fresh workbench: open example → tick cache → Run (0 reused / 16 executed / 1 always run) → untick `fit_intercept` on `ols` in the form → Run (**13 reused / 3 executed / 1 always run**), explanation table rendered. The only console error was the browser's automatic `/favicon.ico` 404, which predates this change (no API errors).
- Temporary backend/editor were stopped and confirmed absent.

### Next

A44 (pinned repository code import) is the last unimplemented acceptance row. Other gaps: cache retention/GC, caching for other graph kinds, and the §5/§9/§10 limits. Keep using a separate branch per session if the harness requires it; `master` was not changed by this session.

## 12. A44 pinned repository code import — Claude, 2026-10-04

Continuation in the same Linux x86_64 cloud session as §11, on branch `claude/laughing-bell-la46nd` (starting from `ddf2606`). This closes the last `not implemented` acceptance row. **All A01–A64 rows are now bounded evidence**; read each row's scope before claiming more than it states.

### Implementation

- `python/repos/core.py` (new package): `Repos` fetches a remote (absolute path, file, https, ssh or scp-style; credential-bearing URLs, `ext::`, other schemes, relative paths and leading-dash arguments are refused) into a bare mirror `repos/<sha256(url)[:24]>.git`. It uses hooks off, a protocol allowlist, `GIT_CONFIG_NOSYSTEM`, no prompts and no submodule recursion. `resolve` maps branch/tag/HEAD/SHA to a commit; `tree` classifies entries; `read` returns raw blobs (no filters, capped at 256 KiB). `dependencies` parses requirements/pyproject/setup.cfg statically; setup.py is reported, not run. `license` gives an SPDX keyword guess; `inspect_python` is ast-only (functions, params, imports, local imports, wrappability). `compare` uses `--no-ext-diff --no-textconv`. `import_function` wraps one top-level function of a single-file module into a code-block definition. It writes the immutable, hash-verified `repos/imports/<sha256>.json` record and sets the block's `origin`, which is part of the semantic hash. `origin_status` reports local modifications.
- `services/control/repos_api.py` (registered in `app.py`): `/api/repos/resolve`, `/{id}/tree`, `/{id}/file`, `/{id}/python`, `/{id}/compare`, `/{id}/import`, `/imports`, `/imports/{id}`, `/origin-status`.
- Editor: `RepoImport.tsx` (Modules & code ▸ **Import from repository…**: resolve, classified tree with filter, license, dependency pins, file view, compare-with-revision patch, function/parameter-role/output selection, import). The `CodeBlockEditor` origin banner shows the pin and **unmodified pinned import** / **locally modified since import**. `ModuleLibrary` lists the origin.
- ADR **0016**; ACCEPTANCE (A44 → bounded evidence), CAPABILITIES, README updated; COVERAGE regenerated. The `test_scale.py` checklist assertion now requires A09 and A44 to be bounded evidence citing their tests.
- `tests/test_repos.py` (16 cases): a real local Git fixture whose `setup.py`, `conftest.py`, `pkg/__init__.py`, `install.sh`, source-side `post-checkout` hook and `.gitattributes` diff/filter drivers would each write a marker file. Resolve/browse/read/inspect/compare/import never create it. It also covers: rev resolution, unsafe URLs, classification incl. LFS pointer and submodule gitlink, static dependency/license parsing, local-import refusal, interface mismatch refusals, tamper-detected import records, sandbox execution agreeing with a native torch reference, semantic hash changing with the pinned commit, local-modification status, and the HTTP journey.

### Verification (this container)

- `.venv/bin/pytest -q -o faulthandler_timeout=240`: **956 passed, 1 skipped, 6 deselected, 0 failed** (1071 s). This includes the §11 debugger fix; the skip is Anthropic without a key. `pytest -m live` was **not run** (no Ollama here).
- `pnpm -C apps/editor build` and `exec tsc --noEmit`: exit 0. `backends.coverage --check`: current. `git diff --check`: clean.
- Curl smoke on temporary backend 127.0.0.1:8771 against the trap fixture: resolve v2 → commit and subject; tree counts with 3 installation scripts and MIT license; import list shows the pinned `stats_utils.py@d7f58797ab1f standardize`. No marker file.
- Headless Chromium (Playwright; script in scratchpad, outside the repo) against editor 5296: Modules & code → Import from repository → resolve v1 → license/“setup.py … NOT run” shown → open setup.py (ast only) → open stats_utils.py → choose `standardize`, `eps` as node setting → compare with v2 (patch shows `unbiased=True`) → import → origin banner “unmodified pinned import” → Test with fixtures → **all fixtures pass**. Zero runtime/API errors; marker absent. Screenshots inspected.
- Temporary backend/editor stopped and confirmed absent.

### Next

No acceptance row is `not implemented`, but every row is bounded. The largest remaining gaps are in CAPABILITIES: multi-file repository packages, LFS/submodules, dependency installation, cache retention, domain/agent production adapters, cross-host/GPU, and team features. The Mac-specific evidence (live Ollama tests, port 8000 service) should be re-run on the user's machine when convenient.

## 13. Inspection refresh fix and continuation — Claude, 2026-10-04

PR [mazenDDr/project-void#1](https://github.com/mazenDDr/project-void/pull/1) was opened from `claude/laughing-bell-la46nd` for §11–§12. It is mergeable, has no review comments and no CI checks (the repository has no workflows).

**Fixed: inspection stuck on “not recorded”.** `useInspect` (apps/editor/src/hooks.ts) asked once per request key. With “latest checkpoint” selected, the key does not change when the first checkpoint appears, so the Weights/Activations/wire views kept showing “not recorded” until reopened. Unavailable inspection answers now carry `runStatus` (`services/control/inspection.py`). While an answer is `not_recorded` for a queued/preparing/running/paused/cancelling run, the hook re-asks every 3 s. The retry count is tied to the request key, so a new selection starts fresh and terminal runs are not polled. `tests/test_api.py` asserts `runStatus`. Real Chromium (reference CNN, conv_1 ▸ Weights, then Run): “not recorded” at 1.6 s, real filters rendered at 8.1 s without reopening, zero errors. The same journey with the previous hook stayed “not recorded” for the full 60 s window, which confirms the root cause.

**Added: node-cache retention.** `tabular/cache.py` `cache_summary`/`prune`, `ArtifactStore.node_cache_entries`/`delete_node_cache`, `services/control/cache_api.py` (`GET /api/cache/nodes`, `POST /api/cache/nodes/prune`), and editor controls in the tabular Run panel. Pruning needs an explicit project or `allProjects` and is dry-run by default. It frees only entry bytes no index row or run artifact references; run artifacts are never touched, and pruned results are recomputed on the next miss. ADR 0015, CAPABILITIES and README are updated. `tests/test_node_cache.py` grew to 14 cases (dry run, keep-latest with identical recomputation, older-than, project scope, shared-bytes protection, API). The browser check (two cached runs, then "keep only the latest per node" previews and removes 3 entries) found a real refresh bug: the size only updated when a run started, not when it finished. Fixed by refreshing on run status changes; the rerun passed with zero errors.

**Fixed: repository credential helpers.** `repos/core.py` overrode `HOME`, which hid the user's `~/.gitconfig` credential helpers despite ADR 0016 promising the environment's own Git configuration. HOME is now preserved. `test_user_git_config_cannot_reenable_hooks_or_external_diff` shows a user-level hooksPath/diff.external still cannot execute anything; a mutation check removing the hooksPath override makes it fail.

## 14. Production registry for domain models and CNN classifiers — Claude, 2026-10-04

PR [mazenDDr/project-void#1](https://github.com/mazenDDr/project-void/pull/1) (§11–§13) was merged into `master` as `0d25b02` at the user's request. The branch `claude/laughing-bell-la46nd` was restarted from that merge for this work.

- **Domain models (ADR 0017).** `production/domain_adapter.py` registers a recorded vision/NLP/speech model as a version and serves it through the existing release/route/admission/trace/replay/label/traffic machinery. `domain/inference.py` was split into `load`/`run`, with `predict` unchanged. Monitoring adds per-family descriptive drift and native label-based quality: NLP word accuracy + seqeval span F1 (undefined when no spans exist, not 0), speech corpus CER/WER, vision pixel accuracy/IoU/Dice. `tests/test_production_domain.py` has 13 cases, including predictions equal to `/api/domain/models/{id}/predict` and refusal under a changed implementation. A Chromium journey trained `nlp_token_classification`, then ran Production register → release → deploy → load held-out example → request → labels → monitoring with zero errors. The browser screenshot exposed seqeval's misleading 0 span F1 for span-free labels; fixed and tested.
- **Model-graph image classifiers (ADR 0018).** `production/model_adapter.py` pins the stored graph, latest complete checkpoint, training preprocessing, classes, environment and implementation hashes. It freezes up to 64 held-out images only when the source folder matches its recorded dataset identity (`E_REFERENCE_SOURCE` otherwise) and never rereads the folder when serving. Predictions equal `/api/infer` on the same checkpoint. `tests/test_production_model.py` has 3 cases. Candidate detection is memoized per completed run because the overview is polled. A Chromium journey trained the reference CNN in the editor, then ran Production register → release → deploy → held-out images → labels → monitoring with zero errors. Accuracy 0 matched the run's recorded val_acc 0.025 (the model collapsed to "x" after the editor's short default run), so it is a real measurement.
- ACCEPTANCE A50, CAPABILITIES, README and both ADRs are updated. Remaining serving gaps: agent, RL, unsupervised, Keras/JAX and non-image model graphs.
- **Greedy DQN policies (ADR 0019).** `production/rl_adapter.py` pins the final Q-network, network graph, environment spec, full Box bounds, evaluation report, environment and implementation hashes, plus a frozen replay-buffer reference. Serving returns argmax Q and the Q-values, equal to the recorded network loaded independently. Action agreement is measured only against supplied actions; behaviour-policy actions are not treated as labels. `tests/test_production_rl.py` has 6 cases. A Chromium journey (tiny CartPole via the API, then everything in the UI) passed with zero errors.
- **Shared serving fix.** A JSON body with NaN/Infinity made the serving route fail while echoing the value in its JSON trace. The HTTP route now refuses non-finite numbers before admission. The in-process trace behaviour, which an existing test relies on, is unchanged. A NaN-safe 422 handler covers request-model errors. There is a regression test in `test_production.py`.
- **Unsupervised models (ADR 0020).** `production/unsup_adapter.py` captures k-means/GMM/PCA during tabular runs (refusing DBSCAN, t-SNE and fitted upstream preprocessing) without touching the tabular adapter identity. Serving equals the fitted native objects; monitoring uses ARI/NMI from supplied labels or label-free PCA reconstruction error. `tests/test_production_unsup.py` has 7 cases. A Chromium journey ran k-means through register → release → deploy → fitted rows → labels → ARI with zero errors.
- **Production serving now covers** tabular, domain, model-graph image classifiers, RL policies and k-means/GMM/PCA. Agent graphs (provider/tool effects) and Keras/JAX exports remain unserved.
- **Repository mirror cleanup.** `GET /api/repos/mirrors` and `DELETE /api/repos/{id}` (`Repos.mirrors`/`remove_mirror`); import records and blocks are kept. Test in `test_repos.py`.
- **Not done, needs a decision: general domain dataset importers (COCO/WAV/CoNLL).** The domain source operations hard-code `synthetic: True` in their provenance. Real imported data would be mislabelled unless those op files change, and the domain checkpoint implementation hash covers every file in `vision/`, `nlp/`, `speech/` and `domain/`, so any edit makes all previously saved domain models unavailable until retrained (ADR 0014). Ask the user before doing this.
- **Fixed: run-status race (pre-existing).** `ArtifactStore.set_status` read the status and then updated it separately. A control-process cancel could read `running`, the worker could then move the run to `cancelling` → `cancelled`, and the cancel's late write left a finished run stuck at `cancelling`. `test_worker.py::test_cancel_running_worker_process_yields_partial_checkpoint` failed about 1 in 3 runs here. The check and update now share one `BEGIN IMMEDIATE` transaction. The cancel test passed 12/12 afterwards. The new `test_status_transitions_are_atomic_across_store_instances` races 150 rounds through two store instances; it fails 3/3 on the old code and passes with the fix.

## 15. Multi-file repository imports — Claude, 2026-10-04

PR [mazenDDr/project-void#2](https://github.com/mazenDDr/project-void/pull/2) (§14) was merged as `6c39878`, and the branch was restarted from it. A44 now bundles pure-Python package closures (ADR 0016 addendum). `repos/core.py` gains `_closure`, `_module_index` and `_bundle_preamble`, and the import record/origin list every bundled module. `test_repos.py` has 20 cases: a three-module relative-import package computes correctly in the sandbox even with the source moved away, and a trapped package `__init__` is refused by the sandbox guard. The previous assertion that multi-file imports are refused was replaced by stronger assertions of the new behaviour. A Chromium import/test journey passed with zero errors.

## 16. Optional bearer-token boundary — Claude, 2026-10-04

PR [mazenDDr/project-void#3](https://github.com/mazenDDr/project-void/pull/3) (§15) was merged as `14b7b27`. ADR 0021 adds `services/control/auth.py`: with `VOID_API_TOKEN` set, every request needs the bearer token, and weak tokens are refused when the app is created (validated eagerly, because Starlette builds middleware lazily). The Vite proxy injects the header server-side; `production/traffic.py` and the example journeys authenticate. `tests/test_auth.py` has 4 cases, including a real loopback traffic run with the token. In Chromium, the editor with the proxy token loaded normally (API 200); without it every call got 401; the token is absent from the client scripts.

## 17. Handoff to Codex on the user's Mac — written by Claude, 2026-10-04

The user asked for this section so Codex can continue on their PC. Everything below is on `origin/master` at **`1f5d026`** (PRs #1–#4 merged). The Claude sessions ran in a Linux x86_64 cloud container, so Mac-specific evidence (live Ollama tests, port 8000 service, installed Chrome) has **not** been re-collected for this code.

### 17.1 Get the code

```bash
cd /Users/mazenkhaled/project-void
git status                      # preserve any local work first (see §2 "Check for in-progress work")
git fetch origin && git checkout master && git pull --ff-only origin master
git log -1                      # expect 1f5d026 or later
```

The remote setup from §7 (HTTPS remote using the `gh` credential helper) is unchanged. The branch `claude/laughing-bell-la46nd` is identical to `master` and can be ignored or deleted. The Claude sessions used one PR per verified round and merged it at the user's request; the user previously had Codex commit verified work directly to `master` and push. Either works; follow the user's latest instruction.

### 17.2 Compatibility effects on the existing local workbench (read before restarting anything)

- **Dependencies: none changed.** `python/requirements.txt`, `python/tracking/{requirements,lock}.txt`, `apps/editor/package.json` and `pnpm-lock.yaml` are identical to `90d6721`, so no reinstall is needed. `connectors/install_pgserver.py` only changed its Linux path (compound manylinux tag); macOS is unaffected.
- **Saved domain models (vision/NLP/speech) become unavailable.** `python/domain/inference.py` was refactored into `load`/`run` (behaviour unchanged, tested). ADR 0014's implementation hash covers every file in `domain/`, `vision/`, `nlp/` and `speech/`, so models trained before `6c39878` are refused with `E_DOMAIN_CHECKPOINT_ENVIRONMENT`. This is the intended safety behaviour. Tell the user, then retrain the examples (`examples/domain_checkpoint_journey.py` or the editor) if they want saved models. Do not weaken the check.
- **Registered tabular production versions keep serving.** The tabular adapter identity files (`production/pipeline.py`, `operations/tabular_ops.py`, `tabular/core.py`) are unchanged. The unsupervised capture was deliberately put in its own module (ADR 0020), and a test pins that file set.
- **Workbench schema.** `meta.db` gains a `node_cache` table, created automatically (`CREATE TABLE IF NOT EXISTS`). `production.sqlite` is unchanged. Repository mirrors go under `<workbench>/repos/`.
- **Run status updates** are now one SQLite write transaction (§14). Behaviour is the same except that the cancel race is gone.
- **Port 8000.** The user's own uvicorn (§3) is still running old code. Do not stop it; ask the user whether and when to restart it. Use other ports for your own servers and stop them afterwards.

### 17.3 Verify first (§4, on the Mac)

```bash
source .venv/bin/activate
pytest -q -o faulthandler_timeout=240        # cloud result: 1000 passed, 1 skipped (Anthropic without key)
pytest -q -m live                            # NOT yet run for this code; needs local Ollama (expect 6 passed)
pnpm -C apps/editor build && pnpm -C apps/editor exec tsc --noEmit
.venv/bin/python -m backends.coverage --check
```

Then curl and installed-Chrome journeys (puppeteer-core, scripts outside the repo, as before) for the new surfaces:
1. Tabular run with **Reuse unchanged node results** twice, edit one node, check the node-cache table (§11), then **keep only the latest per node** (§13).
2. Graph ▸ Modules & code ▸ **Import from repository…** on a local Git repo, including a multi-file package (§12, §15).
3. Production for each new family: an NLP example, the reference CNN trained in the editor, a CartPole run, and `unsupervised_cells` k-means. Register → release → deploy → request → labels → monitoring (§14).
4. Weights tab opened before the first checkpoint fills in by itself (§13).
5. Backend and editor with `VOID_API_TOKEN` (§16).

Watch on macOS arm64 specifically:
- `tests/test_debugger.py::test_conditional_breakpoint_is_labelled_execution_changing_and_stops_resumably` now uses an MSE regression fixture (§11) designed to overflow on any platform. Confirm it passes on arm64.
- `test_status_transitions_are_atomic_across_store_instances` stresses SQLite locking with threads. The macOS note in `artifact_store/store.py` mentions an earlier libsqlite deadlock in one process; the test uses two store instances, each with its own lock. If it hangs on macOS, investigate rather than skip.
- Domain and production tests retrain tiny models; arm64 numerics can differ slightly. Tests compare served outputs with the same native objects, not fixed numbers, so differences should not matter.

Record results in a new HANDOFF section, as previous sessions did.

### 17.4 Remaining work, in priority order

> Historical priority list: Mac verification and the bounded dataset-import scope are now complete (§18–§19). Use §19 for the latest scope/compatibility and next priorities; do not repeat the completed importer approval request.

Everything in ACCEPTANCE is bounded evidence (no `not implemented` rows). These are the real gaps, from `docs/CAPABILITIES.md` "Not implemented" lines and the ADRs:

1. **Mac verification (§17.3).** Required before anything else.
2. **Domain dataset importers (COCO / WAV + transcripts / CoNLL): needs the user's decision first.** The domain source ops (`domain/{vision,nlp,speech}_ops.py`) and summaries hard-code `synthetic: True` and the SYNTHETIC notes, so real data would be mislabelled. Correct provenance means editing those files, which (like §17.2) makes every saved domain model unavailable until retrained. If the user agrees, a suggested design:
   - converters that write each source's existing canonical format (span JSONL, the vision `.npz` contract in `vision/contract.py`, the speech `.npz` in `speech/audio.py`) into the workbench, content-addressed;
   - a recorded `synthetic` flag and source license carried from the converter into the source summaries;
   - COCO polygons/RLE to instance masks via the pinned `pycocotools`, WAV decoding via `torchaudio`, CoNLL BIO to character spans;
   - tests comparing against native library readers.
3. **Agent graph serving: feasible on the Mac because Ollama runs there.** This is the one family still missing from production serving (ADR 0017–0020 cover the rest). It needs an ADR first:
   - what to pin: graph, model provider, model name, and the digest from Ollama `/api/show`;
   - memory/index identities, and tool effects with their approval policy;
   - per-session state, provider non-determinism, and which state may be written by serving.
   Follow the shared runtime pattern in `production/runtime.py` (adapters: tabular, `domain`, `model`, `rl`, `unsup`). Live tests must be `@pytest.mark.live`.
4. **Anthropic live test.** It skips without `ANTHROPIC_API_KEY`. Only run it if the user provides a key through the environment; never store it.
5. **Online MLflow/W&B destinations.** The bridges are local/offline (ADR 0013). Online use needs the user's accounts and credentials through environment references only.
6. **Serving gaps inside implemented families:**
   - non-image or multi-input model graphs, and procedure-trained sequence models (ADR 0018);
   - continuous-action, image-observation or recurrent policies (ADR 0019);
   - fitted preprocessing upstream of unsupervised estimators (ADR 0020);
   - a larger domain monitoring reference than the single recorded held-out example (ADR 0017).
7. **Node cache.**
   - Extend it to model/procedure graphs: needs a determinism contract per op; ADR 0015 explicitly excludes training activations across updates.
   - Optional automatic retention policy; explicit prune exists.
8. **Repository import (ADR 0016):** notebooks as entry points, data files a package reads at import time (would need a declared `file_read` effect), LFS object download, submodule fetch, dependency installation into an isolated environment, and hosted-provider (GitHub/GitLab) browsing with credentials.
9. **Access control beyond the shared token (ADR 0021):** user accounts, roles, per-user audit, and TLS deployment guidance or tests.
10. **Infrastructure-bound, keep honest:** GPU (Apple MPS could be explored but numerical-equivalence tests would need new tolerances), cloud/remote deployment, cross-host worker TLS, distributed training, multi-agent RL, external-user onboarding and accessibility certification. Leave these `not implemented` unless real infrastructure exists to test against.

### 17.5 Conventions that still apply

- §4 working rules are unchanged:
  - no mock or fabricated values;
  - "not recorded" when absent;
  - never weaken tests: replace an assertion only with a stronger one when behaviour intentionally changes, and say so in HANDOFF;
  - README lists only commands actually run;
  - update CAPABILITIES/ACCEPTANCE/README and add an ADR for notable decisions (next number: **0022**);
  - commit message style from `git log`, with a co-author line naming your own model.
- Identity hashes are deliberate. Editing files covered by an implementation hash (domain ADR 0014, tabular adapter ADR 0012, node cache ADR 0015, serving adapters ADR 0018–0020) invalidates saved artifacts that depend on them. Say so in HANDOFF whenever you do it.
- Production adapters share one interface: `manifest`, `validate_records(records, max_batch)`, `predict(records) -> (result, timings)`, plus `reference_records()` for warmup and monitoring. They re-verify environment and implementation on every use, and all labels/quality are measured only from supplied ground truth.
- Keep the user's port 8000 service running, stop every server and browser you start, and keep the HANDOFF current so the next agent can resume from Git alone.

## 18. Cloud merges verified on the Mac — Codex, 2026-10-04

The user supplied Claude's cloud handoff and asked for an assessment of remaining work. Started clean at `90d6721`; `git pull --ff-only origin master` brought this Mac to PR #5 merge `37368c5`. Worked alone, without subagents. No dependencies changed or were installed. Compatibility statements in §17.2 were checked against the actual Git diff: tabular adapter identities and dependency manifests are unchanged; the cloud domain inference refactor invalidates earlier domain models by the deliberate implementation pin. The user was told. All verification runs below use an isolated workbench, not the user's saved models or registered versions. No historical artifact was rewritten and no compatibility checks were weakened.

### Mac verification

- `.venv/bin/pytest -q -o faulthandler_timeout=240`: **1000 passed, 1 skipped, 6 deselected, 1833 warnings**, **438.09 s (7:18)**, exit 0. Includes both arm64 watch cases in §17.3: the execution-changing MSE conditional breakpoint and atomic cross-store status transitions. No test changes. Existing Anthropic-without-key skip and native TensorFlow/gast/Torch warnings remain. Temporary log `/private/tmp/void-cloud-mac-pytest.log`.
- `.venv/bin/pytest -q -m live`: **6 passed, 1001 deselected**, **16.96 s**, exit 0, actual local Ollama. Temporary log `/private/tmp/void-cloud-mac-live.log`. No Anthropic key or tracker account credentials supplied.
- `pnpm -C apps/editor build` and `pnpm -C apps/editor exec tsc --noEmit`: exit 0 after the final monitoring label correction, **249 modules**; existing Vite large-chunk warning. `.venv/bin/python -m backends.coverage --check`: current. `git diff --check`: clean.
- `VOID_API_TOKEN="$(cat /private/tmp/void-cloud-mac-token)" .venv/bin/python examples/domain_checkpoint_journey.py --base http://127.0.0.1:8775`: exit 0, all three freshly trained native families, actual epoch 2→4 continuation, checkpoint export SHA verification and immutable parent predictions. Tiny labelled SYNTHETIC fixtures verify mechanics, not useful model quality. No additional domain implementation files changed during this Mac continuation, so this session does not introduce another domain identity invalidation.
- Curl against isolated backend **8775** and editor **5298**: unauthenticated `/api/registry` **401**, authenticated backend **200**, editor's authenticated proxy **200**. The test token was generated locally only for these servers and is never committed. Native HTTP training, checkpoint inference and production are exercised through that same token boundary.

Fresh checkpoint parents/children: vision `b5f7bdac6e3b` → `d1ae2296baba`, NLP `85ed499734be` → `08ccd9774825`, speech `18d4fafe7f39` → `7fb0963a2998`. Temporary native CLI evidence `/private/tmp/void-cloud-mac-domain-cli.json`.

Installed Chrome / temporary Puppeteer scripts (outside the repo, per §4):
- Token-authenticated editor loaded and API calls succeeded; token absent from the DOM and loaded browser scripts.
- Tabular regression: cache enabled → first run `ddc7a009e1cc` **0 reused**, identical rerun `b6ed0315e021` **16 reused**, change `ols.fit_intercept` in the actual editor form → `93d6f5dcc054` **13 reused / 3 executed**, with settings/dependency reasons; explicit keep-latest preview/confirmation removed **3 entries**. Recorded runs retained. Evidence `/private/tmp/void-cloud-mac-browser.log` and cache screenshot.
- Real local Git fixture with setup/conftest/hook/filter traps: resolve v1 → static license/dependencies → import `uses_lib.py:scaled` with three pinned `lib/` modules → compare v2 → unmodified-origin banner → **all native sandbox fixtures pass**. No trap marker created. Temporary repository fixture commit `e44ff0e9eaee4873efa2e25cf2f3464180a86ae1` is test data, not a project source commit. Evidence `/private/tmp/void-cloud-mac-browser-final.log` and repository screenshot.
- CNN in the editor: open `conv_1` Weights before Run, observe the unavailable inspection response, then real filter heatmaps appear **without reopening the tab**. Native training run **`a5a8b528dcdb`** completed. No fake checkpoint or direct API training shortcut for this journey.
- All four new production surfaces completed **register → release → deploy → load recorded reference → native request → supply labels → replay → monitoring** in Chrome. Final run `/private/tmp/void-cloud-mac-production-browser.mjs` exited 0 with **zero runtime errors and zero API errors**; log/evidence JSON/screenshots are under `/private/tmp/void-cloud-mac-*`. Final screenshots of NLP and k-means monitoring were visually inspected, including the corrected reference heading.

| Family | Training run | Release (prefix) | Native labelled result on the tiny recorded request |
|---|---|---|---|
| CNN | a5a8b528dcdb | 913ecee6dadd | accuracy 0 (three actual held-out image labels) |
| NLP | 85ed499734be | 4059b6800035 | word accuracy 0, span F1 0 (actual source character-span labels aligned to returned original words) |
| DQN | 09a257b6295b | cc4fe00ad074 | action agreement 0 against an explicitly supplied constant-action reference (all 0); not ground truth from behaviour actions or environment return |
| k-means | fcbbdde601c6 | 036591c86535 | ARI 1, NMI 1 on three fitted rows with `true_group` labels from the actual SYNTHETIC CSV; in-sample, not a benchmark |

These low-quality/very-small-request results are reported as measured, not replaced by optimistic values. The domain reference-input endpoint intentionally supplies input only; its NLP ground truth was recovered from the exact labelled fixture, not inferred from predictions. Initial temporary browser scripts required fixes for text entry, CSS-transformed text, initial project loading, asynchronous response/DOM waits and supplying those labels. They did not require weakening tests or bypassing app controls. The accepted evidence is the successful native journeys above, not the failed temporary-script attempts.

Cleanup completed: backend **PID 61792 / port 8775** and editor **PID 61811 / port 5298** stopped with SIGTERM and confirmed absent with `ps`; all Chrome processes started by these scripts closed in `finally`. Generated test-token/config files deleted. The user's port 8000 service was untouched and remains a separate old-code service until the user authorizes its restart. Workbench `/private/tmp/void-cloud-mac-smoke` and logs/screenshots are temporary evidence and need not survive cleanup. This verified correction/assessment is committed and pushed with the handoff; use `git log -1` for its exact commit.

### Documentation and small UI correction

Production monitoring's shared heading called every non-tabular reference a “held-out example.” RL reference observations actually come from the replay buffer, and unsupervised references are fitted rows. Changed only that heading to **“Input changes against the recorded reference”**; native reference records, quality definitions and identities are unchanged. CAPABILITIES now removes contradictory historical exclusions for domain/RL/unsupervised serving and shared-token authentication, and states that package imports include pinned pure-Python closures. README links this verification record. No new architecture decision is needed for these label/documentation corrections; **0022 remains the next ADR**.

### Remaining work assessment (supplements §17.4)

All A01–A64 rows have **bounded evidence**. That means the documented scenarios work; it does not mean the entire VISION or a generally deployable product is finished. Read the scope of each row and the full CAPABILITIES ledger rather than treating the acceptance count as feature completeness.

§17.4's priorities remain useful, with these additional product gaps already recorded elsewhere in CAPABILITIES:
- Tabular/research: broader estimator families (trees/forests/boosting/SVM/kNN), sample weights, richer split/held-out final-test workflows, Parquet/JSON and larger-data profiling; broader statistics, bootstrap/permutation/power/multiple-comparison methods; richer connectors and study/experiment management.
- Editor/inspection: undo/redo, copy/paste, grouping/outline/layout and richer large-graph navigation, teaching/saliency/training-distribution views, wider multi-input training and device/precision/optimizer workflows. Check earlier milestone sections before implementing a named gap, because some narrower variants already exist.
- Operational readiness: browser checks currently live in temporary scripts, not repository-owned end-to-end tests; no GitHub CI workflow exists. A reproducible test runner/CI, measured workloads beyond fixtures, backup-and-restore drills, versioned migrations preserving old runs, and broader disconnect/crash/disk-pressure recovery are proposed next work, not implemented claims. Current strict domain code identity is safe but forces retraining after refactors; a deliberate compatibility/migration contract would improve everyday use without removing integrity checks.
- Security/team readiness: the shared token does not provide accounts/roles/audit or transport encryption. The imported-code guard is best-effort, not hostile-code isolation. Production beyond loopback needs an explicit tested deployment and execution boundary. Artifact/trace/checkpoint retention and dependency-aware deletion remain broader gaps despite explicit node-cache and mirror cleanup.
- Serving: **Keras/JAX also remain unserved**, along with agents and the narrower model/procedure/RL/preprocessing gaps in §17.4. Do not describe agent serving as the only remaining model backend.

Recommended next scope after this Mac verification: real user dataset import with explicit source/provenance and the checkpoint-compatibility decision from §17.4, then bounded agent serving with real Ollama and ADR 0022. A CI/browser smoke baseline and restore/upgrade tests should accompany work toward everyday use. Do not require credentials for unrelated work or fabricate remote/GPU/online evidence. The user's current message asks for this assessment, not automatic implementation of every gap. Keep working alone and keep verified changes and the handoff pushed.


## 19. Bounded domain dataset imports — Codex, 2026-10-04 (verified, committed and pushed)

The user said “okay continue work” after §18 recommended domain dataset imports and described their retraining effect. This is authorization to continue that scope. Worked alone, no subagents, from clean pushed `606cc29`. Before implementation the user was told again that changes in pinned domain source/contract files invalidate previously saved vision/NLP/speech models and registered domain versions. No compatibility bypass, historical artifact rewrite or automatic migration. No dependencies installed, no database migration, no user datasets supplied or overwritten. Tabular/CNN/RL/unsupervised identities are unchanged. Tiny local CPU fixtures are §4 acceptance evidence, not GPU training or real-world accuracy evidence.

### Implementation and files

- New `python/connectors/domain_import.py`: bounded server-local COCO segmentation, WAV+transcript-manifest and strict CoNLL IOB2 converters. Native pycocotools/SciPy/IOB2 semantics. Source bytes are read once, hashed and snapshotted into CAS; canonical payload and converter/request/contract/license/source manifest are stored and atomically materialized in `domain-datasets/<manifest-sha>/`. Original files are never modified. Declared license and synthetic flag are required and explicitly user supplied, not independently verified. Integrity errors refuse altered canonical data/manifests; member traversal and escaping symlinks are refused.
- New `services/control/domain_datasets_api.py`: POST conversion (one concurrent import, stable `E_DATASET_*` failures), GET verified listing; inherits existing optional bearer token. `control/app.py` registers it. No uploads/downloads/dependency installation.
- Native loaders/provenance: vision preserves separate overlapping masks, raw category mapping, original annotation/image/license IDs, keypoint visibility and flip pairs; speech preserves normalized float32 multichannel PCM and true lengths, with no invented timing labels. NLP retains original CoNLL tokens/tags/line numbers and reconstructed-text character offsets. Source/trainer summaries and saved examples/reference/inference carry actual recorded declarations instead of hard-coded SYNTHETIC. Audio feature output now retains channel count; waveform inspection explicitly identifies channel 0, features declare channel averaging. Separate teaching sine remains labelled teaching data. Corrected the CTC inspector's fixture-only wording to explain transcript characters/space/blank generally.
- Editor `DomainDatasetImport.tsx`, workspace/App integration: required path/root/license/status, flip-pair or column policies, actual converted contract/hashes, apply to same-family draft source nodes, clear source truncation and same-family resume/fitted-tokenizer choices, update project description/declaration. No training on import; Run trains a fresh model. Old inspected runs stay explicitly tied to their immutable provenance. Registry purpose and partition labels are generic rather than incorrectly calling every imported dataset synthetic.
- New `examples/make_import_fixtures.py` and `examples/dataset_import_journey.py`: deterministic labelled SYNTHETIC COCO/CoNLL/stereo-tone WAV teaching files and real HTTP imports→native two-epoch training→saved inference. Existing model algorithms and graph primitives are retained.
- New `tests/test_domain_import.py`: 32 cases with native masks/RLE/overlap/keypoint-flip checks, Unicode entity agreement with seqeval, native unsigned/signed/float/24-bit PCM normalization, length/channel preservation, bounds/path/schema/IOB2/integrity refusals, token enforcement, source snapshots/removal and actual all-three-family worker→checkpoint→inference→production registration/reference provenance. One intentionally false flag on generated test input tests declaration transport only; public fixtures/journeys always declare SYNTHETIC. Existing tests were not weakened.
- ADR **0022**, README (only commands actually run), CAPABILITIES/ACCEPTANCE and regenerated COVERAGE updated. Source-op coverage includes the new test file. Next ADR is **0023**.

### Verification evidence

- Final `.venv/bin/pytest -q -o faulthandler_timeout=240`: **1032 passed, 1 skipped, 6 deselected, 1941 warnings, 447.24 s (7:27)**, exit 0 (`/private/tmp/void-dataset-full-final-pytest.log`). Verified after the final CTC explanation correction; no source edits afterwards.
- Final live `.venv/bin/pytest -q -m live`: **6 passed, 1033 deselected, 16.33 s**, actual local Ollama, exit 0 (`/private/tmp/void-dataset-live-final-pytest.log`). Existing Anthropic test lacks a key. No online tracker accounts/credentials supplied.
- Focused `.venv/bin/pytest -q tests/test_domain_import.py tests/test_domain_api.py tests/test_coverage_ledger.py -o faulthandler_timeout=240`: **50 passed, 108 warnings, 19.02 s**, exit 0 before the final CTC wording correction; the final full suite covers those cases again. Native pycocotools NumPy-copy deprecation warnings remain.
- Final `pnpm -C apps/editor build` / `exec tsc --noEmit`: exit 0, **250 modules**, existing large-chunk warning. `.venv/bin/python -m backends.coverage --write` / `--check`: current. `git diff --check`: clean.
- `.venv/bin/python examples/make_import_fixtures.py --out /private/tmp/void-import-fixtures`; `.venv/bin/python examples/dataset_import_journey.py --base http://127.0.0.1:8776 --fixtures /private/tmp/void-import-fixtures`: exit 0. All three final native CPU import/train/checkpoint/predict paths pass with matching dataset/source/license identities. Final CLI runs: vision **9c5b3f08e172**, NLP **493d5a4b40d8**, speech **ae3cc454484e**. Evidence `/private/tmp/void-dataset-import-cli.json`.
- Curl against backend **8776** `/api/domain/datasets` and editor **5299** `/api/registry`: exit 0/HTTP 200; import listing and actual proxied registry received. New bearer-token route coverage is in native API tests; this temporary browser backend had no token.
- Installed Chrome/Puppeteer outside the repository: actual picker→import with mandatory license/status→verify manifest/source hashes→apply→check canonical source path and clear continuation→fresh Run→source overlays/Unicode spans/waveform→saved inference with source/license provenance. Final run **all three pass with zero browser runtime and API errors**. Logs `/private/tmp/void-dataset-import-browser-final.log`, evidence JSON and screenshots `/private/tmp/void-dataset-import-*`. Earlier temporary script corrections scoped an ambiguous role=status selector and compared canonical payload hashes (CoNLL requests differ in an unused root declaration); no app controls were bypassed or tests relaxed. Final screenshots were reviewed; two-epoch models have weak actual predictions (background mask, incorrect token labels, empty CTC output), never replaced by fabricated accuracy.

| Domain | Final Chrome training run | Model prefix | Original source snapshots |
|---|---|---|---|
| Vision | 5d87104f8e57 | 6d0ee7afc53e | 9 (JSON + 8 PNGs) |
| NLP | ef818b46051d | 9fb2323db36b | 1 (CoNLL) |
| Speech | 18580303e91d | 503435a3abb2 | 9 (JSON + 8 stereo WAVs) |

Libraries used, already installed: torch **2.14.1**, torchvision **0.29.1**, torchaudio **2.11.0**, NumPy **2.5.3**, SciPy **1.18.1**, pycocotools **2.0.11**, tokenizers **0.23.2**, seqeval **1.2.2**. WAV decoding uses SciPy for explicit PCM dtype semantics (including 24-bit); torchaudio remains available and performs native mel features. No new requirements/package install.

Cleanup: own backend **PID 65777**, then final backend **PID 67880 / port 8776**, and editor **PID 65750 / port 5299** stopped with SIGTERM. Chrome closes in each script's finally. `ps` confirms only the user's historical **PID 8258 / port 8000** remains; it was never stopped/restarted. Temporary workbench, fixture files/logs/screenshots are not committed and need not survive cleanup. All delivered source/tests/docs are committed and pushed to the existing private master; use `git log -1` for the release commit.

### Bounds and next work (supersedes the completed items in §17.4)

The implemented import bounds are in ADR 0022/CAPABILITIES: 2–256 records, 64 MiB input/decoded, 8 MiB JSON/text; COCO uniform RGB 4–512 dimensions, 1–4 classes, ≤64 instances/image and shared ≤128-keypoint schema, segmentation only/no crowds; CoNLL ≤4,000 reconstructed characters/sentence and ≤32 entity types; WAV uniform 1–96 kHz rate and 1–4 channels, 512–32,000 samples/channel, ≤256 transcript characters and ≤64-character alphabet. No implicit resize/trim/resample, original whitespace reconstruction guarantee, timestamps, larger streaming conversion, upload/URL import, verified license ownership, import deletion/retention or real-world accuracy benchmark. The existing typed transforms/preprocessing stay explicitly configurable. No user source was supplied, so real-data measurement is pending.

1. **Next buildable release: bounded agent serving with real local Ollama**, preserving native graph state/context/turn provenance and isolation; ADR 0023. §17.4/§18 serving gaps still include Keras/JAX and narrower multi-input/procedure/RL/fitted-preprocessing cases.
2. Test user-supplied datasets within the above bounds when paths/license/status are supplied; expand formats/categories/streaming based on measured needs. Existing older domain models/versions require explicit fresh training/registration. Never modify old manifests to get around identity checks. A durable compatibility/migration design remains a separate future decision.
3. Repository-owned browser smoke/CI, backup/restore/upgrade/recovery evidence; wider cache/repository/editor/research features from §18; accounts/roles/security/deployment boundary before broader production use.
4. Anthropic and online MLflow/W&B remain credential-dependent; GPU/cloud/encrypted cross-machine/distributed and other infrastructure scope stays unimplemented until real infrastructure and measured evidence exist. Do not require those credentials for unrelated local work.

Continue alone, preserve user changes, read this handoff and ADR 0022, verify §4, stop only your own servers, commit/push the verified release and keep the handoff current.


## 20. Isolated native agent serving — Codex, 2026-10-05 (verified, published with this release)

User “continue work” authorizes the next scope from §19. Worked alone from clean pushed `3b639e2`, no subagents. This is a bounded single-turn production adapter; persistent conversation serving remains future work. No dependencies installed or database migrations/historical artifact edits. Libraries already installed: LangGraph **1.2.12**, checkpoint **4.2.0**, LangChain Core **1.6.6**, LangChain Ollama **1.1.0**, Ollama Python **0.6.3**, Pydantic **2.13.5**, HTTPX **0.28.1**. Domain and existing serving-family pinned files are unchanged, so no retraining is needed. Verification also exposed and fixed the existing CNN-worker cancellation race described below. Other serving-family identities remain unchanged. Never restart the user's **PID 8258 / port 8000** without their authorization.

### Implementation

- New `python/production/agent_adapter.py`: register a whole completed agent graph under `__agent_graph__`, require END rather than partial budget completion; allow native prompt/set_state/one chat node, local Ollama or model-free StateGraph only. Existing native compiler/blocks own reducers, parallel joins, routing and loops. Source/config/final-state/reference membership, package/platform/source identities and currently installed Ollama digest/runtime are pinned and verified before/after each turn. Source research runs did not capture model digests; pin time is explicitly registration. No retrospective model-identity claim.
- Shared production runtime/API integration: real isolated warmup, immutable versions/releases, route CAS and rollback, existing bounded admission, idempotency, cancellation/deadline discard and restart semantics. Agent maxBatch=1/stateless; source/reference input fields must be 1–4 declared text values. Provider HTTP timeouts narrow to remaining serving time and actual resolved settings are recorded. Fresh random execution/thread ID and temporary native store/in-memory saver per request; no research checkpoint/session/effect mutations. Full native state checked between supersteps; recursion/budget/state/context limits fail serving requests. Complete exact bounds are in ADR 0023/CAPABILITIES.
- Capture policy: successful response text, native state/event hashes, context hashes/usage/latency and execution/source/provider identities retained. Full state/events/exact sent context and source segments only under captureInputs=true; captured context bytes copied to CAS and embedded in trace. Capture off is not response redaction (output can quote inputs). Temporary execution stores are removed. Interrupted captured calls can leave unreferenced CAS bytes; broader retention/GC stays unimplemented.
- Monitoring reads recorded warmup/requests and never invokes an LLM; single-reference character-length KS is descriptive, provider usage covers successful serving calls only, and independently supplied strings measure literal exact agreement, not semantic correctness. Warmup/replay/failed-call usage and the separate Ollama process/device/resource cost are not measured here. Replay makes a new isolated native model call, may differ, never updates original state/route.
- Editor Production workspace lists native whole-agent candidates, pins/budgets, one-turn release policy, source-input reference and exact recorded contexts/state/events; new `AgentTurnInspection.tsx` provides actual call selection. React best-practices skill reviewed labels, stable call selection and read-only inspection; no placeholder controls. Real Ollama device is managed by its runtime, not falsely labelled CPU-only.
- New picker example `examples/serving_agent.project.json` / `.ui.json`: SYNTHETIC teaching prompts, real qwen3.5:2b. New `agent_serving_journey.py` performs source run→register→warmup→serve/context→idempotency→independent reference string→isolated replay→bounded real HTTP traffic→rollout/rollback. Historical `agent_*` generator prefix retained.
- New offline `tests/test_production_agent.py` has 20 native model-free cases (reducers/parallel joins, concurrent fresh-state isolation, idempotency/restart, source/config/schema refusals, capture, no-call monitoring, integrity, cancellation/deadlines, native recursion and state growth). New `test_production_agent_live.py` has 2 actual Ollama cases (exact sent context/source segments/provider token counts/CAS, original thread unchanged, fresh replay, digest mismatch and capture-off bytes absent). Existing tests unchanged. Verification also fixed a reproduced `worker/train.py` read/write cancellation race by using its existing guarded transition; new `test_worker_cancel_race.py` forces the actual control/worker SQLite interleaving and verifies one genuine partial checkpoint. ADR 0023, CAPABILITIES/README and regenerated COVERAGE updated. Next ADR **0024**.

### Final verification evidence

- Final `.venv/bin/pytest -q -o faulthandler_timeout=240`: **1053 passed, 1 skipped, 8 deselected, 1941 warnings, 444.45 s (7:24)**, exit 0 (`/private/tmp/void-agent-full-accepted-pytest.log`). Final `.venv/bin/pytest -q -m live`: **8 passed, 1054 deselected, 16.72 s**, exit 0 (`/private/tmp/void-agent-live-final-pytest.log`), after the worker fix. Existing missing-key Anthropic skip only; native deprecation warnings remain. Earlier focused agent/coverage run **25 passed, 3.72 s** (`/private/tmp/void-agent-focused.log`).
- Final build/typecheck exit 0: **251 modules**, existing large-chunk warning. Coverage write/check current; diff whitespace clean. Curl backend 8777 Production and editor 5300 registry proxy succeed.
- First real CLI run `f69a1130b0c3` served **“Blue”**, provider **48 input / 2 output tokens**, exact agreement **0** against independently supplied **“red”**. Ollama qwen3.5:2b digest `324d162be6ca5629ae4517c8710434d0bd2d665bc94dbad46e9af8fbf8a2f0df`, runtime **0.33.3**. Isolated replay, bounded traffic and rollback passed; evidence `/private/tmp/void-agent-cli.json`. Final renamed-example journey **passes**, run **260b029e6c94**, version `c9635332d12a…`, release `94354b1d9299…`, same measured “Blue” / 48 input / 2 output tokens / exact agreement 0. Two real bounded closed-loop HTTP requests succeed, zero errors; native rollback verified (`/private/tmp/void-agent-cli-final.json`).
- Initial full suite stopped with SIGINT after **357 pass / 1 failure**: existing test reserves exactly ten generated `agent_*.json` files. Corrected only the new example's name to `serving_agent`; did not weaken that test or rewrite its generator. The next full suite finished **1051 pass / 1 failure / 1 skip / 8 deselected, 446.79 s**: existing worker cancellation failed with **IllegalTransition: cancelling → cancelling**, reproduced standalone too. Read/check/write in `_cancel` races with the control writer. Fixed by calling existing `_advance`, which accepts only the already-cancelling state and retains other transition errors. Added a deterministic real-writer interleaving test. **35 focused tests pass in 14.87 s**, including all unchanged worker tests and new agent cases (`/private/tmp/void-agent-worker-focused.log`). The final acceptance run above passes, including the unchanged original process-cancellation test and the new regression. No tests weakened. This worker fix changes no pinned files of existing registered serving families.
- First temporary Chrome script selected the picker before its options loaded; wait timed out without runtime/API errors. Fixed script to wait for the existing initial CNN canvas and actual new picker option before selecting; no app workaround. A second script attempt filled the request before asynchronous reference loading completed; reference then overwrote that edit. Fixed only the temporary script to wait for the reference response and actual DOM payload before entering the new question. Final installed Chrome journey **passes, zero runtime/API errors** (`/private/tmp/void-agent-browser-source-accepted.log`, `/private/tmp/void-agent-browser-evidence.json`); source run **131f111a2494**, version `bf004693c4bd…`, release `4e751b958817…`, request `76e3a7d8-d38a-4304-a491-8d8463e84e2d`. Final UI review also connected App-selected source-run identity into AgentWorkspace: Open source run now opens that exact run and its trace instead of silently selecting the latest. Chrome created a genuinely later native source run, then confirmed the registry button opens the older pinned source. This UI-only correction is covered by final build/typecheck/Chrome; native execution-source hashes are unchanged. Actual response “Blue”, provider 49 input / 2 output tokens, exact agreement 0 against independent “red”. Actual model-call selector, exact sent segments/context, final state/control events, isolated replay and monitoring verified. Both final screenshots visually reviewed (`/private/tmp/void-agent-browser-inspection.png`, `…-monitoring.png`). Script stays outside repo and closes Chrome in finally.
- Cleanup complete: temporary backend **PID 73143**, then final **PID 77134 / port 8777**, and editor **PID 73159**, then final **PID 77152 / port 5300** stopped with SIGTERM; `ps` confirms all absent. Final curl verified `/api/serve/local/agent-cli-final/health` and editor registry proxy before shutdown. User **8258 / port 8000** still running and untouched. All Chrome sessions close in finally. Workbench `/private/tmp/void-agent-serving-smoke`, logs/screenshots temporary and uncommitted.
- Git: fetched upstream without changing local work; upstream master matched starting `3b639e2`. Source/tests/docs/this handoff are published together in the verified release; use `git log -1` for its exact commit. No generated workbench, model weights, secret, screenshot or temporary browser script is committed. Resume from master after a safe pull and read this section; do not follow the already-completed next-build entry in §19.

### Remaining work

1. Next agent scope: persistent conversations require explicit native checkpoint/turn commits atomic with successful request/cancellation handling, per-release/user/session isolation and replay/crash evidence. Pinned retrieval/index/memory dependencies and tool/interrupt side effects are separate extensions, not supported in this release.
2. Wider serving from §17/§18: Keras/JAX, multi-input/non-classifier models, procedure sequence models, continuous/image RL, fitted preprocessing before unsupervised estimators and larger domain monitoring references.
3. Repository-owned browser smoke/CI and backup/restore/upgrade/recovery evidence; compatibility/migration design for strict domain code identities; user-supplied real datasets/benchmarks within §19 bounds; broader editor/cache/repository/research scope in §18.
4. Anthropic and online tracking remain credential-dependent; roles/team security and real GPU/cloud/encrypted cross-machine/distributed infrastructure scope stay unimplemented without measured infrastructure. No credentials needed for unrelated local work.

Continue alone; preserve local work, native semantics and existing tests. Keep the handoff and completed verified changes pushed.


## 21. Persistent native agent conversations — Codex, 2026-10-05 (verified release)

The user said “continue work” after §20. Continued alone from clean pushed **bfc503a**, fetched upstream master and confirmed it matched. Scope: native persistent conversations, atomic checkpoint/request commits, isolation, replay and crash evidence. No dependencies installed; same native versions/local Ollama digest/runtime as §20. No source training algorithm or earlier pinned execution file changed. Additive `agent_sessions` table in production SQLite, automatically created at startup; no historical artifact rewrites or compatibility bypass. No retraining/re-registration needed for existing stateless agents or other families. The earlier §19 domain-retraining rule still applies to models saved before that import release. User **8258 / port 8000** remains running old code, untouched.

### Implementation and files

- New `python/production/conversation_adapter.py`: separate bounded `__agent_conversation__` candidate/manifest for completed END graphs with thread state and turn-scoped text inputs. The original ADR 0023 adapter is unchanged. Native compiler/blocks/reducers own execution; restore the actual native END checkpoint/channel versions/versions seen/metadata through the installed saver/serializer. Temporary native store/saver per candidate, stable conversation thread, fresh execution ID. Only internal SHA-verified native snapshots, no checkpoint uploads or pickle fallback. ADR 0023 bounds plus serialized snapshot ≤128 KiB. Research checkpoint/history never copied/mutated; defaults seed new conversations. Thread reducers own any declared history window.
- `production/models.py`, `runtime.py`, `store.py`: explicit conversation release mode/maxBatch=1/session required. Per-release/user/session serialization and existing admission/idempotency/route rules. Immutable candidate snapshot CAS, expected parent revision/hash, cancellation and final deadline checks, then checkpoint-head and trace pointers in one SQLite transaction. Deadline check includes snapshot/trace I/O; failed/cancelled/expired/stale/integrity/transaction failures discard candidates. Restart fails unfinished requests without resuming inference; last successful head survives. Rollback resumes the older release’s isolated head. One owning control process, no distributed claim.
- `services/control/production_api.py`, `production/monitor.py`: conversation candidates/reference/labels/health, verified read-only checkpoint state endpoint and captured replay from the selected turn’s immutable prior checkpoint, never the current live head. Warmup/replay discard their snapshots and create no conversation head. Monitoring uses immutable warmup/traces, never invokes an LLM. Existing exact-string reference-label/usage caveats stay in force; current-input length drift does not measure history or semantic answer quality. Corrected stale A50 documentation that still excluded all agent serving.
- Production editor: conversation mode follows selected/registered native version, one-record releases, explicit session/user inputs, working Inspect conversation checkpoint, parent/new checkpoint revision/hash/thread provenance, existing exact native context/segments/state/control inspection. Clear inspected checkpoint when release/user/session changes. Reviewed React labels/state handling using the already-applied React best-practices skill. Trace capture remains optional but **conversation mode necessarily persists native state/history even with capture off**; release/API/editor/docs declare this. Caller-declared identity keys are not authentication/ownership.
- New picker `examples/serving_conversation.project.json` / `.ui.json`: SYNTHETIC teaching prompts, actual qwen3.5:2b, native last-six-message history via existing prompt/set_state/chat blocks. Source run→register→isolated warmup→two turns→inspection→replay→traffic→rollout/rollback in new `examples/agent_conversation_journey.py`. Historical generated `agent_*.project.json` namespace unchanged.
- New `tests/test_production_conversation.py`: **11 native offline cases**, whole-state comparison against independent research saver, turn resets/thread reducers/history window, concurrency/scoping, rollout/rollback, idempotency/restart, missing sessions/schema, cancellation after native candidate, deadline after slow CAS, stale/cross-version/hash refusals, actual SQLite rollback via failed trace trigger and real subprocess SIGKILL after candidate/before commit. New `test_production_conversation_live.py`: actual local Ollama prior-user/assistant context/segments/CAS/usage, captured-parent replay without head mutation, fresh session and restart/history window. No model substitutes or weakened existing tests. New ADR **0024**, CAPABILITIES/README/ACCEPTANCE and regenerated COVERAGE. Next ADR **0025**.

### Final verification evidence

- `.venv/bin/pytest -q -o faulthandler_timeout=240`: **1064 passed, 1 skipped, 9 deselected, 1941 warnings, 447.84 s (7:27)**, exit 0 (`/private/tmp/void-conversation-full-pytest.log`). Full required suite ran with actual local service access; no source implementation/test edits after it started. The new HTTP example’s polling fix is verified separately below. Existing missing-key Anthropic skip and native deprecation warnings remain.
- `.venv/bin/pytest -q -m live`: **9 passed, 1065 deselected, 16.38 s**, exit 0 (`/private/tmp/void-conversation-live-final.log`). Actual Ollama, no downloads or credentials. Initial new live test incorrectly read segment kind at the top level; corrected it to the native `segment.source.kind`, without changing the runtime or an existing test. Focused live then **1 passed, 4.19 s**. Native focused conversation/stateless/production suite **62 passed, 11.81 s** (`/private/tmp/void-conversation-focused-accepted.log`). An earlier restricted-sandbox focused run had six unchanged loopback tests fail with `PermissionError` binding sockets, with 56 passing; reran with proper loopback access and all pass.
- Final `pnpm -C apps/editor build` / `exec tsc --noEmit`: exit 0, **251 modules**, existing large-chunk warning (`/private/tmp/void-conversation-build-final.log`). Coverage write/check current; `git diff --check` clean. Curl backend **8779** Production and editor **5301** registry proxy pass; post-restart `/api/serve/local/conversation-cli-final/health` ready.
- Final real CLI **passes** (`/private/tmp/void-conversation-cli-final.json`): source **8c23b96b4dda**, version `2561be013e4a…`, release `d779e83cda3f…`; second answer `You mentioned **blue**.`, real provider **89 input / 6 output tokens**, exact previous user/actual assistant/current user request verified, committed revision **2**, checkpoint `a500a62d6685…`. Exact-string agreement **0** against independently supplied **red**, not inferred ground truth. Two real closed-loop HTTP requests succeed, zero errors/timeouts, measured ~**3.53 requests/s** including drain (tiny teaching workload). Idempotency, captured-parent replay, new-release fresh state and rollback-preserved original head verified. The first CLI failed because new polling code had been accidentally dedented, leaving its loop without a job refresh; its recorded traffic actually completed two requests/zero errors in ~0.50 s. Fixed only the new example’s loop indentation, reran to completion; no timeout limits or app behavior weakened.
- Installed Chrome **passes, zero runtime/API errors** (`/private/tmp/void-conversation-browser.log`, `…-browser-evidence.json`): source **a85f9dbdc03b**, version `baeef441d859…`, release `9e676ed99e7c…`; actual source Run→register→warmup/deploy→two turns→native exact-history/context/checkpoint inspection→captured-parent replay→fresh separate session→unchanged original head→monitoring. Second answer `You mentioned **blue**.`, actual **91 input / 6 output tokens**; revision **2**, four recorded history messages and parent `551a20d49df6…` / new `ad1d5410e8c8…`. Both screenshots visually reviewed (`/private/tmp/void-conversation-browser-inspection.png`, `…-monitoring.png`). No labels supplied there, so quality remains unavailable. Three actual successful serving calls in monitor, zero errors. Existing React Flow startup console warnings about `card` fallback on the initial CNN canvas remain; they are not runtime exceptions and were not changed by this scope. Browser script stays outside repo and closes Chrome in finally.
- Real backend stop/restart against the same temporary workbench verified exact CLI/browser checkpoint preservation, then actual native third turn advances CLI revision **3** from the recorded parent and duplicate request returns identical checkpoint/trace without a second turn (`/private/tmp/void-conversation-restart-http.json`). Third call provider **120 input / 14 output tokens**. Checked **six previously registered ADR 0023 versions** from §20’s temporary workbench against the final implementation; all verify unchanged (`/private/tmp/void-conversation-old-version-compatibility.log`), including `bf004693c4bd…` and `c9635332d12a…`.
- Cleanup: temporary backend **78307 / port 8779** stopped, restarted as **80253** for persistence check, then stopped; editor **78321 / port 5301** and its pnpm parent **78308** stopped. Chrome closes in finally. Final ps check confirms all temporary services are absent and **8258 / port 8000** remains running and untouched. Workbench `/private/tmp/void-conversation-smoke`, logs/screenshots/scripts/CAS models stay temporary and uncommitted. First temporary backend shutdown reported existing loky semaphore cleanup warning; final restarted backend shutdown clean. No generated artifacts, secret or external credentials are committed.
- Git: source/tests/examples/docs and this handoff are published together on master after the checks above. Use `git log -1` for this release’s exact commit; current local/remote equality and clean tree are checked after push. Resume with a safe pull, read this section, preserve any uncommitted work and work alone.

### Remaining work

1. Conversation session reset/delete/fork/retention and deliberate checkpoint compatibility/migration; pinned retrieval/index/memory dependencies, tools/effects and approval semantics are separate extensions. No new placeholders or infrastructure claims. A CI/repository-owned browser smoke baseline and backup/restore/upgrade/recovery work are useful independent next scopes from §18.
2. Wider serving remains from §17/§18: Keras/JAX, multi-input/non-classifier model graphs, procedure sequence models, continuous/image RL, fitted preprocessing before unsupervised estimators and larger domain monitoring references. Current synchronous local conversation adapter is bounded and does not close those gaps.
3. Other editor/cache/repository/research scope and real user datasets/benchmarks remain in §18/§19. Strict domain checkpoint identity still deserves a compatibility design; do not bypass integrity or rewrite old runs. Startup React Flow fallback warnings and the existing large editor chunk remain known frontend follow-ups.
4. Anthropic/online trackers remain credential-dependent; authenticated users/roles/ownership and actual GPU/cloud/encrypted cross-machine/distributed infrastructure remain unimplemented without measured infrastructure. No credentials are required for unrelated local work.

Continue alone; keep verified work and the detailed handoff pushed. Do not stop or restart the user’s port-8000 server without authorization.


## 22. Reviewed conversation reset/fork — Codex, 2026-10-05 (verified release)

User “continue” authorizes the next bounded scope from §21. Continued alone from clean pushed **c7e1f18**, fetched upstream master and confirmed it matched. No subagents, package installations, training changes or historical artifact rewrites. Existing native adapter/compiler and other families’ pinned source files are unchanged. Additive `conversation_actions` SQLite table created automatically at startup; reset now permits a null live checkpoint in the existing session table. No retraining/re-registration required for existing versions by this release; §19’s earlier domain-identity rule remains. Libraries and local qwen3.5:2b/Ollama identities are unchanged from §20/§21. User **8258 / port 8000** remains running old code, untouched.

### Implementation and files

- New `python/production/conversations.py`: mandatory caller user/session, inspected revision/SHA, action ID and bounded reason; fork adds a different unused destination under the same release/user. Verify the registered version and actual native checkpoint before mutation. Serialize with native turns using the runtime’s existing scope locks; deterministic two-scope ordering and bounded timeout avoid opposing-fork deadlocks. Recheck the reviewed head after waiting; never apply to unseen later state. Neither action executes a model. Stateless, empty/reset, cross-user missing source, stale review, existing destination and invalid schema refuse with stable errors.
- `python/production/store.py`: immutable CAS receipt/fingerprint pointer, head change and lifecycle event commit together under BEGIN IMMEDIATE, with a second parent/destination check and a deadline check after receipt serialization. Identical user/action ID plus identical release/operation/reviewed values returns the original receipt across restart and later turns. Conflicting ID reuse refuses. No pending actions resume after restart. Failed/killed transactions roll back head, receipt pointer and event; orphan CAS bytes can remain and are not a completed action. One owning control process remains required.
- Logical reset increments the existing serving revision and clears live checkpoint/last-request pointer. GET then returns state=null, not fabricated defaults. Next successful turn uses the unchanged native adapter’s defaults/fresh thread and increments again. Old snapshots, trace bytes, original idempotent requests and captured-parent replay remain. Reset is **not physical deletion or privacy erasure**; history retention also applies with trace capture off.
- Fork verifies and clones the actual native END checkpoint via the installed saver serializer, preserving channel values, versions, versions seen, metadata and original historical call/message provenance. Rebind only the native thread identity in metadata/envelope. New destination starts serving revision 1 with no producing request yet; inherited native state may contain multiple turns. Subsequent native turns reset turn fields and preserve thread reducers independently; source head remains unchanged. Existing 128 KiB snapshot bound applies. No arbitrary historical selection, import or cross-release migration.
- `services/control/production_api.py`: bounded reset/fork models and POST routes, verified nullable checkpoint GET, existing optional shared-token middleware on both actions. Caller user remains a declared isolation key, not authenticated ownership. New `apps/editor/src/components/ConversationActions.tsx` and Production workspace integration: controls only for the matching inspected release/user/session, actual revision/SHA, reason/destination, explicit reset review checkbox, identical-attempt retry ID, immutable receipt/lineage, reread actual head after action and clear stale inspection after successful turns. React best-practices skill applied to typed props, scope/state handling, labels and busy/error behavior; no placeholder controls.
- New `examples/conversation_actions_journey.py`: actual local source Run→register→warmup/deploy→two turns→fork whole-state inspection→real inherited-history branch turn→reset→empty head→fresh turn→duplicate original action→retained captured-parent replay and old trace hash. All teaching prompts/reasons labelled SYNTHETIC; real Ollama outputs/usage are not answer-accuracy evidence.
- New `tests/test_conversation_actions.py`: **18 offline native cases**, real model-free StateGraph reducers/state, independent fork/fresh reset, monotonic revisions, restart/idempotency, schema/token/integrity/no-inference refusal, source/target scoping, active native turn interleaving, competing/opposing forks, lock/slow-CAS deadlines, real SQLite receipt-trigger rollback and two real subprocess SIGKILL cases **inside uncommitted reset/fork transactions after head/receipt/event writes**, then restart/retry once. New `tests/test_conversation_actions_live.py`: **1 actual local Ollama case** proving inherited sent history/fresh reset context/thread isolation/duplicate receipt. Existing tests unchanged. ADR **0025**, README/CAPABILITIES and coverage regeneration complete; next ADR **0026**.

### Final verification evidence

- `.venv/bin/pytest -q -o faulthandler_timeout=240`: **1082 passed, 1 skipped, 10 deselected, 1941 warnings, 457.54 s (7:37)**, exit 0 (`/private/tmp/void-session-actions-full-pytest.log`). Required suite uses actual local service access; no implementation/test/example edits after it started. Existing missing-key Anthropic skip and native deprecation warnings remain. Final `.venv/bin/pytest -q -m live`: **10 passed, 1083 deselected, 18.09 s**, exit 0 (`/private/tmp/void-session-actions-live-final.log`). No downloads/credentials/model substitutes.
- Final focused new/unchanged conversation/stateless/general production suite: **80 passed, 19.58 s** (`/private/tmp/void-session-actions-focused-accepted.log`). Focused new live test: **1 passed, 6.22 s**. An initial new slow-CAS test matched compact JSON text, but native dumps includes spaces; corrected only that new test to parse JSON and match operation semantically. Runtime unchanged, existing tests never weakened.
- Editor build and TypeScript check exit 0, **252 modules**, existing large-chunk warning (`/private/tmp/void-session-actions-build-final.log`). Coverage write/check current; whitespace check clean. Curl backend **8780** Production, editor **5302** registry proxy and post-restart serving health/proxy pass. Initial CNN React Flow `card` fallback console warnings remain the known previous frontend issue.
- Real HTTP CLI **passes on first run** (`/private/tmp/void-session-actions-cli.json`): source **4a78599ac452**, version `faed030ce0c6…`, release `f81def1325d4…`; fork receipt `bfc434eb41da…`, reset receipt `1bfdaf7fae7c…`. Fork inherits the exact four prior recorded messages, creates an independent native thread and advances destination revision 1→2. Actual branch answer **“Hello! How can I help you today? 🌟”**, provider **118 input / 13 output tokens**. Reset source 2→3/null; fresh next turn 4/new thread and exact system+current-user request, actual answer **“Hello! How can I assist you today?”**, **49 input / 10 output tokens**. Original source trace hash unchanged, duplicate reset returns original receipt without clearing fresh head, captured-parent replay does not change either head.
- Installed Chrome/Puppeteer outside the repository **passes on first run, zero runtime/API errors** (`/private/tmp/void-session-actions-browser.log`, `…-browser-evidence.json`). Actual picker→source Run→register→warmup/deploy→two turns→inspect→reason/destination→fork→inspect inherited state→real branch→source inspection→assert reset disabled until review checkbox→reset→retained old-turn replay while live head stays null→fresh turn→monitor. Source **cea1ec78cb48**, version `20adaa0f6559…`, release `5d8429315cda…`; original head revision2/hash `a7dca50b1921…`; fork receipt `3bba31f7382d…` / thread `63eae733f1d8…`; reset receipt `9b7593fd49cd…`. Actual branch **“Hello! 👋 How can I help you today?”**, **120 input / 12 output tokens**; fresh **“Hello! How can I assist you today?”**, **49 input / 10 output tokens**, revision4/hash `1e77aafbf9d6…` / new thread `4073b5255a98…`. Monitor has **4 successful serving calls, 0 errors**, provider totals **314 input / 41 output tokens**; no reference labels supplied, quality unavailable. Warmup/replay excluded from those totals. Fork/reset/monitor screenshots visually reviewed; temporary script closes Chrome in finally.
- Real backend stop/restart against the same temporary workbench preserves both CLI/browser source revision4 and branch revision2, exact original trace hashes and both action receipts. Identical reset/fork requests after restart/later turns return the original receipts without changing heads. A real CLI fork continuation then advances branch to revision3 on the same native thread, actual answer **“Hi there! Ready to learn more? 🚀”**, **131 input / 12 output tokens**; duplicate native request returns the same trace/head without another turn (`/private/tmp/void-session-actions-restart-http.json`). Initial temporary continuation assertion expected all six prior messages plus the new user. The graph’s declared keep_last_n=6 reducer drops the oldest before prompting; corrected only the temporary verification script to compare the saved parent’s last five messages plus current user against the actual sent request. No app/example/test implementation changed, no extra model call needed.
- Compatibility: verified **six earlier ADR0023 stateless agent versions** in `/private/tmp/void-agent-serving-smoke` and **three earlier ADR0024 conversation versions** in `/private/tmp/void-conversation-smoke` against final pinned sources/environment/provider identities, all unchanged. Includes `bf004693c4bd…`, `c9635332d12a…`, `2561be013e4a…`, `baeef441d859…`. This is identity verification, not a model-quality benchmark or migration of older domain weights.
- Cleanup complete: own backend **81145 / port8780** stopped and restarted as **82932** for persistence verification, then stopped cleanly; editor **81163 / port5302** and pnpm parent **81147** stopped with SIGTERM. Chrome closes in finally. Final ps confirms temporary services/Chrome absent; user **8258 / port8000** remains running and untouched. Temporary workbench `/private/tmp/void-session-actions-smoke`, CAS models, logs, screenshots and browser/restart scripts are uncommitted and need not survive cleanup.
- Git: source/tests/example/docs and this detailed handoff are published together to private master after verification. Use `git log -1` for the release commit; local HEAD/upstream/remote equality and clean tree are checked after push. No generated workbench, artifact weights, secret or temporary scripts committed. Resume with safe pull, read this section and preserve any local work.

### Remaining work (current list)

1. Useful independent next scope: repository-owned browser smoke/CI baseline, then backup/restore/upgrade/recovery evidence. Conversation management beyond this release remains explicit-ID discovery, physical deletion/privacy erasure/retention/GC, historical checkpoint selection/restore and deliberate cross-version compatibility/migration. Never bypass pins or rewrite earlier records. Session actions are one-control-process only.
2. Native agent extensions remain pinned retrieval/index/long-term-memory dependencies, tools/effects/interrupt approval semantics, structured/multiple model nodes, external-provider serving, streaming and distributed serving. Actual authenticated users/roles/ownership/encryption remain separate; shared token and caller-declared keys do not provide them.
3. Wider serving gaps from §17/§18 remain Keras/JAX, multi-input/non-classifier graph models, procedure sequence models, continuous/image RL, fitted preprocessing before unsupervised estimators and larger domain monitoring references. Other editor/cache/repository/research scope and user-supplied dataset/benchmark evidence remain in §18/§19. The original A01–A64 bounded acceptance evidence does not imply general production readiness or real-world accuracy.
4. Existing large editor chunk and initial React Flow fallback warnings remain frontend follow-ups. Anthropic and online tracking are credential-dependent; real GPU/cloud/encrypted cross-machine/distributed infrastructure and broader RL/onboarding/certification scope remain unimplemented without measured infrastructure. Unrelated local work needs no external credentials.

Continue alone, next ADR0026; verify §4 and keep completed work plus detailed handoff pushed. Do not stop/restart the user’s port-8000 server without authorization.
