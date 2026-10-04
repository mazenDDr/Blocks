# Project Void: Visual AI Workbench

Phase 0 (semantic and execution foundation), Phase 1 (visual CNN workbench: control API + editor), Milestone 2a (tabular graph kind:
data preparation, classical ML, statistics), Milestone 2b (connected data sources, versioned extraction, studies and sweeps), Milestone 3 (research-level composition) and
Milestone 4 (language-model and agent workflows on native LangGraph), Milestone 5 (reinforcement learning, unsupervised workflows), Milestone 6a (TensorFlow/Keras 3 and JAX backends for a portable model-graph subset, compatibility reports, native exports, coverage ledger, benchmarks), Milestone 6b (typed vision, NLP and speech workflows with recorded inspection), Milestone 7 (registry and production investigation through a bounded native tabular serving adapter), and bounded Milestone 8 integrations/community tooling are implemented.
See `docs/PLAN.md`, `docs/CAPABILITIES.md` and `docs/adr/` (ADR 0003: tabular graph kind; ADR 0004: connectors, snapshots, studies; ADR 0008: agent graph kind, memory, context recording; ADR 0010: additional backends, compatibility, coverage).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r python/requirements.txt
.venv/bin/pip install -e .
source .venv/bin/activate
python -m connectors.install_pgserver      # local PostgreSQL for tests/example; pgserver has no CPython 3.13 wheel (see ADR 0004); `pip install pgserver==0.1.4` suffices on <= 3.12
pnpm -C apps/editor install
```

## Run the app (two terminals)

```bash
# 1. fixture + backend (http://127.0.0.1:8000)
source .venv/bin/activate
python examples/make_shapes10.py            # writes 200 PNGs to examples/data/shapes10 (gitignored)
python examples/make_tabular_fixtures.py    # rewrites the committed SYNTHETIC CSVs in examples/fixtures/ (deterministic; byte-identical)
python -m uvicorn control.app:create_app --factory --app-dir services --host 127.0.0.1 --port 8000

# 2. editor (http://127.0.0.1:5173, proxies /api to the backend)
pnpm -C apps/editor dev
```

Open http://127.0.0.1:5173. The reference CNN loads as an unsaved draft. Use **Open** to switch project and graph kind: the examples
`tabular_regression`, `gamma_teaching` and `two_group_comparison` are tabular / statistics graphs (their data is synthetic or a teaching fixture and is
labelled as such); **New tabular graph** starts an empty one. For a tabular graph press **Run graph** in the bottom panel, then click nodes or wires:
table preview, profile, fitted state, coefficients, metrics, and the statistics view (density, shaded tail, observed statistic, p-value, alpha, decision).
For the reference CNN, click a node to edit it (change
`out_channels` and watch shapes and parameter counts propagate; lock `in_channels` to see a mismatch error), press
Run in the bottom panel (dataset folder `examples/data/shapes10` is relative to the backend's working directory),
then use the Weights / Activations tabs, click a wire, and compare runs. State lives in `.workbench/`
(`VOID_WORKBENCH` overrides it).

Production build and type check of the editor:

```bash
pnpm -C apps/editor build
pnpm -C apps/editor exec tsc --noEmit
```

## Connected data (Milestone 2b)

Open the **Data** tab in the editor to add connections (PostgreSQL, S3 or S3-compatible, DVC repository), test them and browse what they expose. Secrets are **references**
(an environment variable name, or a key in a JSON file outside the project); the value is never stored, returned or exported. Add a table, object or dataset as a source node
in a tabular graph; the PostgreSQL node has a visual query builder with the compiled parameterized SQL shown read-only (and a raw read-only SQL mode). Every run records a
snapshot of what each source returned (Source & snapshot tab); pin it to repeat from that identity. The **Experiments** tab plans and runs bounded grid/random sweeps with seeds and
folds as separate repeat dimensions, shows failed/retried/repeated trials, and compares a trial with the baseline.

The example runs on LOCAL TEST SERVICES with SYNTHETIC data (real PostgreSQL 16 via pgserver; an S3 API mock via moto, not AWS):

```bash
python examples/connected_journey.py --workbench .workbench           # seeds, saves project 'connected_journey', runs the regression and a small sweep, prints the results
python examples/connected_journey.py --workbench .workbench --serve   # ... and keeps the services + control API (127.0.0.1:8000) up; then `pnpm -C apps/editor dev` and open the project
```

## Tabular and statistics graphs from the command line

```bash
# validate + run the Gamma teaching example through the same worker path the API uses (prints the recorded p-value)
python - <<'PY'
import sys; sys.path[:0] = ["python"]
from graph_core.project_io import load_project
from tabular.engine import run_graph
from graph_core.validate import require_executable
g = load_project("examples/gamma_teaching.project.json").graph
out = run_graph(g, require_executable(g))
print(out["upper_tail"].summary["pValue"], out["test"].summary["decisionText"])
PY
```

Runs submitted through the API (`POST /api/runs` with a tabular graph and `{"config": {}}`) store events and artifacts in `.workbench/` like CNN runs.

## Research-level composition (Milestone 3)

Commands that were run for this milestone:

```bash
python examples/make_m3_examples.py        # writes the six Milestone 3 example projects (deterministic)
pytest -q                                  # unit, API and worker tests (the code-block tests start real sandbox subprocesses)
pnpm -C apps/editor build && pnpm -C apps/editor exec tsc --noEmit
```

In the editor, Open the examples `residual_cnn` (module instances, Expand / Open), `shared_encoder` (shared parameters), `masked_loss` (visual loss module as the training loss), `transformer_sequence`
(Training, Debug and Attention tabs), `code_block_demo` (Modules & code, Edit code) and `control_flow` (repeat and select). Training runs use SYNTHETIC data (labelled everywhere). See `docs/adr/0005`-`0007`.

## Language-model and agent workflows (Milestone 4)

An **agent graph** (`graphKind: "agent"`) compiles to a native LangGraph `StateGraph`: typed state with per-field reducers, nodes, conditional routes with visually built predicates,
bounded loops (route exit, step limit, budgets), SQLite checkpoints per thread, interrupts. The one tested live model path is a local **Ollama** server; an Anthropic adapter exists (key by
secret reference; its live test is skipped without a key); a scripted **FIXTURE** model exists only for control-flow tests and is labelled everywhere. See ADR 0008.

Commands that were run for this milestone:

```bash
ollama pull nomic-embed-text              # embeddings model (the qwen3.5 models answer /api/embed with "server does not support embeddings")
.venv/bin/pip install -r python/requirements.txt   # pinned: langgraph, langgraph-checkpoint-sqlite, langchain-core/-ollama/-anthropic/-text-splitters, faiss-cpu
python -m graph_core.export_schema         # the JSON Schema now includes the `agent` section
python examples/make_m4_examples.py        # writes the five agent example projects (deterministic)
pytest -q                                  # fixture-model and process-boundary tests (live Ollama tests are deselected)
pytest -q -m live                          # live tests against Ollama at http://localhost:11434 (qwen3.5:2b, nomic-embed-text)
# a second backend + editor on other ports (the editor proxy target is VOID_API; default http://127.0.0.1:8000)
VOID_WORKBENCH=/path/to/workbench python -m uvicorn control.app:create_app --factory --app-dir services --host 127.0.0.1 --port 8765
cd apps/editor && VOID_API=http://127.0.0.1:8765 ./node_modules/.bin/vite --port 5290 --host 127.0.0.1
```

In the editor open an example from **Examples - agent graphs**: `agent_retrieval_revision` (VISION 12.3, real model + embeddings over SYNTHETIC documents in `examples/agent_docs/`), `agent_memory_debugging`
(12.7: press **Seed example records** on the Memory tab, run, open **Context inspector**, search the call for "40 degrees", trace the excluded record, edit the `lab_memory` policy on the Memory tab, **Isolated preview**,
then **Rerun**), `agent_approval_tools` (tools with declared effects, approval and edit interrupts; pause, restart the service, resume), `agent_bounded_loop`, `agent_parallel_join`. The tabs of an agent graph:
Canvas (block cards, per-block forms), State & routes (schema, reducers, route predicate builder, limits, joins), Run & trace (thread selector, input, interrupt forms, state-diff timeline), Context inspector,
Memory (stores, policy editor, preview, recorded decisions), Indexes. Workbench files: `agent/checkpoints.sqlite`, `agent/memory.db`, `agent/indexes/`, `agent/outbox/`.

## Reinforcement and unsupervised research (Milestone 5)

An **rl graph** (`graphKind: "rl"`) wires reward components, a Gymnasium environment (1.3.0), a Q-network (a model graph), a replay buffer, a native-torch DQN learner and an evaluation protocol.
The editor opens it in the RL workspace (Environment, Learner, Run & curves, Rollouts, Replay buffer, Transition trace, Evaluation, Variants). The unsupervised nodes (k-means, Gaussian mixture, DBSCAN, PCA, t-SNE
projection, diagnostics) are tabular-graph nodes. See ADR 0009 and `docs/CAPABILITIES.md`.

Commands that were run for this milestone:

```bash
.venv/bin/pip install gymnasium==1.3.0 pygame==2.6.1    # also pinned in python/requirements.txt
python examples/make_tabular_fixtures.py   # adds the SYNTHETIC synthetic_cells.csv and synthetic_moons.csv
python examples/make_m5_examples.py        # writes rl_cartpole_dqn, rl_gridworld_builder, unsupervised_cells, unsupervised_moons
pytest -q                                  # includes tests/test_rl_*.py, tests/test_unsup.py, tests/test_m5_examples.py
pytest -q -m live
pnpm -C apps/editor build && pnpm -C apps/editor exec tsc --noEmit
```

In the editor open an example under "Examples - reinforcement learning" and press Run (CartPole needs about 15 s), then open Rollouts, Replay buffer and Transition trace; the Variants tab runs reward
variants x seeds as a study. Open `unsupervised_cells` / `unsupervised_moons` (SYNTHETIC data), press Run graph, and click the clustering, PCA, projection and diagnostics nodes.

## Additional backends, compatibility reports and the coverage ledger (Milestone 6a)

A model graph can target `pytorch` (default), `keras` (TensorFlow/Keras 3 on the TensorFlow backend) or `jax`. Only a portable subset (14 operations: tensor input, conv2d, relu, max_pool2d, adaptive average pooling,
flatten, linear, cross-entropy, sub/add/square/scalar_mul/sum/mean) runs on Keras and JAX; everything else, and every unsupported setting, is reported per node with a stable code **before** execution and nothing is
substituted. Keras and JAX execute forward, loss, gradients and one SGD step; **training runs, checkpoints and run history remain PyTorch-only**. See ADR 0010, `docs/CAPABILITIES.md` and the generated `docs/COVERAGE.md`.

Commands that were run for this milestone:

```bash
.venv/bin/pip install tensorflow==2.21.0 keras==3.15.1 jax==0.11.2 jaxlib==0.11.2    # CPU wheels for CPython 3.13 / macOS arm64; pinned (with their dependencies) in python/requirements.txt
python -m backends.coverage --write        # regenerates docs/COVERAGE.md from the registry, adapters and tests (tests/test_coverage_ledger.py fails when it is stale)
python benchmarks/run_benchmarks.py        # A19: graph-lowered vs handwritten native, per backend; writes benchmarks/results/m6a_raw.json and m6a_summary.md (about 3 minutes)
pytest -q                                  # includes tests/test_backend_*.py and tests/test_coverage_ledger.py (they skip with the reason when a backend is unavailable)
pytest -q -m live
pnpm -C apps/editor build && pnpm -C apps/editor exec tsc --noEmit
```

In the editor open a model graph (for example `reference_cnn`) and use the **Backend** selector in the top bar: nodes get compatibility badges (supported / converted ×n / unsupported with the code), the **Backend** tab lists
every conversion and refusal and the VISION 14.2 facets, **Export code** shows deterministic native source for PyTorch, Keras or JAX, and the **Coverage** tab shows the ledger. Try `padding_mode: reflect` on a convolution and select Keras.
The same data over HTTP: `GET /api/backends`, `POST /api/backends/compat`, `POST /api/export`, `GET /api/projects/{id}/export/{backend}`, `GET /api/coverage`.

## Run the tests

```bash
pytest -q
```

## JSON Schema export

```bash
python -m graph_core.export_schema          # writes packages/graph-schema/schema.json
```

## Train from the command line (Phase 0 CLI)

```bash
python -m worker.cli examples/reference_cnn.project.json --data examples/data/shapes10 --epochs 1
```

Prints one JSON event per line (run id, graph hash, seq, ts, type, node id, data) and a final status line.
Runs, events and checkpoints are stored in `.workbench/` (`meta.db`, `artifacts/<sha256>`). `Ctrl-C` requests a
cooperative cancel and leaves a `partial` checkpoint. A graph with an unknown op or a channel mismatch is
rejected before any training, with exit code 2 and the diagnostics on stderr.


## Vision, NLP and speech (Milestone 6b)

Commands run for this milestone:

```bash
.venv/bin/python examples/make_domain_fixtures.py  # deterministic labelled SYNTHETIC vision/audio NPZ + text JSONL; binary data is gitignored
.venv/bin/python examples/make_domain_examples.py # three valid graph + UI examples
.venv/bin/python -m backends.coverage --write
.venv/bin/python examples/domain_journey.py --workbench /private/tmp/void-m6b-cli
.venv/bin/pytest -q -o faulthandler_timeout=240
.venv/bin/pytest -q -m live
.venv/bin/pytest -q tests/test_domain_api.py -o faulthandler_timeout=240
pnpm -C apps/editor build
pnpm -C apps/editor exec tsc --noEmit
```

In **Open → Examples — vision / NLP / speech**, select `vision_segmentation_synthetic`, `nlp_token_classification` or `speech_ctc_tones`. Each opens its domain workspace. Press **Run graph**, wait for completion, then select a stage:

- Vision: **Annotated images → Geometric transforms → Train segmentation**. Inspect actual before/after annotations, the clipping/removal policy and predicted masks/derived boxes against validation ground truth. Toggle mask overlays. IoU/Dice and derived detection mAP state their reference and aggregation.
- NLP: **Character-span corpus → WordPiece tokenization → Train token classifier**. Click a subword to highlight its original character span; inspect IDs, word IDs, attention, loss labels and predictions. Label alignment and span-evaluation convention are declared, and the tokenizer is fitted offline on train sentences only.
- Speech: **Audio and teaching signal → STFT and mel features → Train CTC recognizer**. The labelled teaching signal has 32,000 samples/channel. Inspect waveform, actual frame-count formula, batch lengths/masks, mel spectrogram, selected frame time, greedy output alignment and CER/WER substitutions/deletions/insertions. The training fixture is generated tones, not recorded speech.

Use **Graph** to edit connections and inspect typed wires, or edit a stage's settings directly in the workspace. Structured transform/policy JSON is applied explicitly and validated. Old runs retain their original graph/artifact identity after edits; all values are recorded from the selected run. Models run on CPU. Native torchvision/torchaudio/tokenizers/seqeval/torchmetrics/pycocotools/jiwer versions are pinned in `python/requirements.txt` (already installed here). Domain checkpoint/resume and general dataset import are not implemented; see ADR 0011 and `docs/CAPABILITIES.md`.

## Registry and production investigation (Milestone 7)

Commands run for this milestone:

```bash
.venv/bin/python examples/make_production_example.py
VOID_WORKBENCH=/private/tmp/void-m7-smoke .venv/bin/python -m uvicorn control.app:create_app --factory --app-dir services --host 127.0.0.1 --port 8767
VOID_API=http://127.0.0.1:8767 pnpm -C apps/editor dev --host 127.0.0.1 --port 5292
.venv/bin/python examples/production_journey.py --base http://127.0.0.1:8767 --namespace m7-cli
.venv/bin/pytest -q tests/test_production.py --tb=short -o faulthandler_timeout=240
.venv/bin/python -m backends.coverage --write
.venv/bin/pytest -q -o faulthandler_timeout=240
.venv/bin/pytest -q -m live
pnpm -C apps/editor build
pnpm -C apps/editor exec tsc --noEmit
```

Open **production_sensors** under the tabular examples, then **Run graph**. This is a labelled SYNTHETIC classifier trained for real. Open **Production**:

1. **Registry:** select the completed run/estimator, specify owner/intended use/limitations and register. Inspect pinned source, fitted scaler, model artifact, raw schema, feature order, labels, native environment and evaluation. An alias is a movable reference; registered versions and existing releases stay immutable.
2. **Release:** choose that version and configure a local or staging namespace, concurrency/queue/deadline/batch limits, optional session counter and explicit input capture. **Preview release candidate** performs real compatibility/warmup; **Deploy selected release** changes that exact route. Selecting a previously deployed candidate enables rollback. Staging is a separate route in this local process.
3. **Requests:** load recorded training-reference rows or enter schema-valid JSON, then send. Inspect real timings/results and prediction → release → version → run → source → fitted state/evaluation lineage. Captured-input replay uses the pinned native version in isolation. Explicit ground-truth labels enable measured quality; loaded reference labels are in-sample evidence, not a held-out benchmark.
4. **Traffic:** configure representative batch payloads, steady/ramp/burst/response-driven arrivals, rate, generator concurrency, duration and expected statuses. Run a bounded real HTTP test. Results distinguish offered/sent/achieved rates and dropped generator arrivals, and report latency/errors/queue time, payload/release versions, CPU and sampled RSS. Stop cancels new arrivals and drains bounded outstanding requests.
5. **Monitoring:** inspect an observed request window. Input/prediction drift is separate from task quality measured only on explicitly aligned labels. Missing labels or uncaptured inputs show unavailable evidence. No automatic rollback or retraining occurs.

Serving currently supports native tabular LinearRegression/LogisticRegression with typed selection and fitted imputation/one-hot/scaling. Models and fitted state are captured during the run; old runs need a supported rerun. Remote deployment, domain/CNN serving, streaming, autoscaling and authenticated multi-user service are not implemented. Use one owning control process. User/session IDs are caller-declared isolation keys; the optional stateful application is a request counter and does not mutate the estimator. See ADR 0012 and the capabilities ledger for precise limits.

## Scale integrations and community tooling (bounded Milestone 8)

**Integrations** exposes actual worker submission/reconnect/cancel, explicit tracker export selections, inert project package export/inspection/import, native SDK conformance, recorded comparisons/conclusions and the A01–A64 acceptance checklist. Choose completed runs for sharing; numeric metrics and evaluation-summary artifacts are opt-in. Local MLflow writes native SQLite/artifacts under the workbench. W&B creates native **offline** run records and never uploads them.

The tracker environment is separate so its protobuf dependency cannot change the training backends. Commands actually run:

```bash
.venv/bin/python -m venv .venv-trackers
.venv-trackers/bin/pip install mlflow-skinny==3.16.1 wandb==0.30.0 sqlalchemy==2.0.48 alembic==1.18.4
.venv-trackers/bin/pip freeze > python/tracking/lock.txt
# Temporary verification services: fixture-only token; use your own environment reference for a new worker.
VOID_WORKER_TOKEN=void-m8-local-smoke-token-123456 VOID_WORKBENCH=/private/tmp/void-m8-smoke .venv/bin/python -m uvicorn control.app:create_app --factory --app-dir services --host 127.0.0.1 --port 8768
VOID_API=http://127.0.0.1:8768 pnpm -C apps/editor dev --host 127.0.0.1 --port 5293
VOID_WORKER_TOKEN=void-m8-local-smoke-token-123456 .venv/bin/python -m scale.worker_server --workbench /private/tmp/void-m8-worker --port 8778
.venv/bin/python examples/scale_journey.py --base http://127.0.0.1:8768 --worker http://127.0.0.1:8778 --output benchmarks/results/m8_local_worker.json
.venv/bin/python examples/connected_production_journey.py --base http://127.0.0.1:8768
.venv/bin/python -m extensions.sdk examples/plugins/offset/manifest.json
.venv/bin/pytest -q tests/test_scale.py --tb=short -o faulthandler_timeout=240
.venv/bin/python -m backends.coverage --write
.venv/bin/python -m backends.coverage --check
.venv/bin/pytest -q -o faulthandler_timeout=240
.venv/bin/pytest -q -m live
pnpm -C apps/editor build
pnpm -C apps/editor exec tsc --noEmit
```

The worker supports a native tabular CSV/preprocessing/linear/logistic/metrics subset. CSV bytes are transferred explicitly and source paths are materialized privately. The shipped adapter is **another local CPU process over authenticated loopback HTTP**; cross-host TLS, GPU and distributed execution are unavailable. Inspect retrieved values in the existing research workspace; imported model pickles remain untrusted for local serving. The three measured worker comparisons include HTTP/polling/startup/transfer and make no hardware speed claim.

Reusable packages carry graph/UI and exact operation requirements, plus optional explicitly selected CSV bytes. **Import and open package** preserves unknown operations and blocks execution for missing/different dependencies; it never installs or runs Python. Credentials and run artifacts are omitted; external sources still require configuration. For an SDK package, read its manifest/docs/source, run conformance on trusted code, then explicitly configure `VOID_PLUGIN_MANIFESTS` when starting the control service and workers. Conformance checks the declared numerical fixtures and contracts; it is not a security sandbox. The shipped `offset` package demonstrates a pure declared pandas transformation.

On **Graph**, expand **Keyboard graph tools** to select nodes/wires and connect ports using Tab and native selects. Settings forms, Run and inspection tabs use ordinary focusable controls, with visible focus. The reference journey and integration flows are recorded in HANDOFF; broad screen-reader certification is not claimed. See ADR 0013, CAPABILITIES and ACCEPTANCE for all remaining vision gaps.

## Saved domain models and local inference

New completed vision, NLP and speech runs save native model/Adam/RNG state and pinned preprocessing. In **Domain workspace**, **Saved model and local inference** downloads the checkpoint and manifest, loads a labelled SYNTHETIC held-out input and predicts with the saved model. Prediction does not train or read the original dataset.

**Prepare checkpoint continuation** edits the draft to reference the selected model and a higher total epoch count. Use **Run graph** to create a child run. The prepared data, split, fitted tokenizer, model/optimizer settings and CPU thread count must match; incompatible changes are refused. The parent run/model remains immutable. NLP continuation reloads the exact saved tokenizer rather than retraining its vocabulary.

Commands actually run for the three-domain HTTP example:

```bash
VOID_WORKBENCH=/private/tmp/void-domain-checkpoint-smoke .venv/bin/python -m uvicorn control.app:create_app --factory --app-dir services --host 127.0.0.1 --port 8769
VOID_API=http://127.0.0.1:8769 pnpm -C apps/editor dev --host 127.0.0.1 --port 5294
.venv/bin/python examples/domain_checkpoint_journey.py --base http://127.0.0.1:8769
.venv/bin/pytest -q tests/test_domain_checkpoints.py --tb=short -o faulthandler_timeout=240
```

Vision inference expects base64 RGB PNG **after explicit geometry** at the pinned size; NLP expects original text; speech expects finite normalized PCM channels with the pinned sample rate. These are bounded local CPU inference endpoints, independent of the tabular production release/monitoring adapter. Existing older domain runs have no checkpoint and require a new training run. See ADR 0014 and HANDOFF for verification and remaining limits.

## Dependency-scoped node cache (A09)

Tabular runs can reuse recorded node results. Tick **Reuse unchanged node results (cache)** in the Run panel, or submit `{"config": {"cache": "reuse"}}` to `POST /api/runs`. A node is reused only when its operation, settings, inputs, implementation files and native environment all match a recorded result. Editing a node re-executes that node and everything downstream of it; sources are always re-read and hashed. The Run record's **Node cache** table says which nodes were reused (and from which run), which ran, and what changed. The default (`"off"`) runs every node and records nothing. See ADR 0015. Cached results take disk space: the Run panel shows this project's total and offers **keep only the latest per node** or **clear**. The API is `GET /api/cache/nodes` and `POST /api/cache/nodes/prune` (dry-run unless `"dryRun": false`). Pruning never changes recorded runs.

Commands actually run (Linux x86_64 cloud container, CPython 3.13):

```bash
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r python/requirements.txt && .venv/bin/pip install -e . --no-deps
python3.13 -m venv .venv-trackers && .venv-trackers/bin/pip install -r python/tracking/requirements.txt
.venv/bin/python -m connectors.install_pgserver
.venv/bin/pytest -q tests/test_node_cache.py -o faulthandler_timeout=240
VOID_WORKBENCH=<scratch dir> .venv/bin/python -m uvicorn control.app:create_app --factory --app-dir services --host 127.0.0.1 --port 8770
VOID_API=http://127.0.0.1:8770 pnpm -C apps/editor dev --host 127.0.0.1 --port 5295
```

## Import code from a Git repository (A44)

On **Graph ▸ Modules & code**, use **Import from repository…**. Give an absolute repository path or a `file://`, `https://` or `ssh://` URL (no credentials in the URL) and a revision, then **Resolve and browse**. The revision resolves to a commit, and the content is read from a bare mirror: nothing is checked out, installed or executed. Setup scripts, hooks, LFS pointers and submodules are labelled and never run or fetched. Dependencies and the license are read statically. Open a single-file Python module, pick a function and the role of each parameter, optionally tick pinned dependencies, and **Import pinned function**. The resulting code block records its URL, commit, path and blob as its origin, runs only in the code-block sandbox, and shows whether you have modified it since import. **Compare** shows how the file changed at another revision; re-import to update deliberately. See ADR 0016.

Commands actually run:

```bash
.venv/bin/pytest -q tests/test_repos.py -o faulthandler_timeout=240
VOID_WORKBENCH=<scratch dir> .venv/bin/python -m uvicorn control.app:create_app --factory --app-dir services --host 127.0.0.1 --port 8771
VOID_API=http://127.0.0.1:8771 pnpm -C apps/editor dev --host 127.0.0.1 --port 5296
```
