# Handoff — Visual AI Workbench (project-void)

You are taking over an in-progress build. Read this whole file before doing anything.

## 1. What this project is

- **Product spec (authoritative):** `docs/VISION.md`, the same as the original `README.md` the user wrote. It covers 9 milestones (0–8) and acceptance tests A01–A64 (§24).
- **Plan and rules:** `docs/PLAN.md`.
- **What actually works:** `docs/CAPABILITIES.md`, the honest ledger. Update it with every change.
- **Design decisions:** `docs/adr/0001…0009`. Read them before changing an area.
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
| `python/tabular`, `connectors`, `studies`, `training`, `debugger`, `codeblocks`, `agent`, `rl`, `unsup` | one package per milestone area |
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
| 6a Keras/TensorFlow and JAX backends, compatibility reports, coverage ledger, benchmarks | done, verified | see git log |
| 6b Vision detection/segmentation, NLP, speech workflows | not started | none |
| 7 Registry and production investigation | not started | none |
| 8 Scale, integrations, community | not started | none |

After 6a: `pytest -q` → 707 passed, 1 skipped (live Anthropic test; no API key); `pytest -q -m live` → 6 passed (local Ollama).

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
- **Full suite:** about 3 minutes.

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

### 6b — domain workflows (VISION §9.7, §9.8, §23 Milestone 6 "Domain evidence", A56, A57, A58)

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
