# Handoff — Visual AI Workbench (project-void)

You are taking over an in-progress build. Read this whole file before doing anything.

## 1. What this project is

- **Product spec (authoritative):** `docs/VISION.md`, the same as the original `README.md` the user wrote. It covers 9 milestones (0–8) and acceptance tests A01–A64 (§24).
- **Plan and rules:** `docs/PLAN.md`.
- **What actually works:** `docs/CAPABILITIES.md`, the honest ledger. Update it with every change.
- **Design decisions:** `docs/adr/0001…0013`. Read them before changing an area.
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
| 6b Vision detection/segmentation, NLP, speech workflows | done, verified (bounded scope; see §7) | 927e9c3 |
| 7 Registry and production investigation | done, verified (bounded local tabular adapter; see §8) | 302cc50 |
| 8 Scale, integrations, community | done, verified (bounded local CPU/offline scope; see §9) | this release; `git log -1` |

After 6a: `pytest -q` → 707 passed, 1 skipped (live Anthropic test; no API key); `pytest -q -m live` → 6 passed (local Ollama).

After 6b: `pytest -q -o faulthandler_timeout=240` → **864 passed, 1 skipped, 6 deselected**; `pytest -q -m live` → **6 passed**. Editor build/type check, curl and real Chrome domain journeys passed. The skip is the existing Anthropic test without an API key. See §7 for commands, results, scope and Git details.

After 7: full suite → **894 passed, 1 skipped, 6 deselected**; live → **6 passed, 895 deselected**. Editor build/TypeScript, real HTTP CLI, curl readiness and installed Chrome train/register/serve/replay/labels/load/monitor/rollout/rollback journeys passed. See §8.

After 8: full suite → **919 passed, 1 skipped, 6 deselected**; live → **6 passed, 920 deselected**. Editor build/TypeScript, curl readiness, real native worker/tracker/connected CLI journeys, and keyboard-only installed Chrome inspection/integration journeys passed. See §9.

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

Read ADR 0013 and ACCEPTANCE before extending this milestone. Native tabular debugger interventions inside fitted sklearn operations are absent; the connected researcher evidence chain is bounded, and broad external-user onboarding/screen-reader certification remains untested. Additional registry-serving families require actually persisted model/tokenizer/transforms/optimizer/environment artifacts first (see §7/§8), not an adapter label alone. Domain checkpoints/exact resume/export, cross-host/GPU operation and online tracker sharing remain explicit future work. A09 numerical dependency caches and A44 repository source-code import remain gaps. Model inspection opened before its checkpoint appears can retain “not recorded” until the tab/context is reopened; the final keyboard proof waits for terminal training before inspecting real weights. Do not fabricate values to cover that existing refresh limitation.

Next agent: inspect `git status`, `git log -1` and `git remote -v`; preserve new user work. Select any next scope from the actual acceptance gaps and latest user direction rather than rebuilding completed milestones. Continue alone, keep this handoff current, run §4 verification for substantive changes, commit verified work and push to the private origin. Never stop the user's port 8000 service; close every temporary server/browser you start.
