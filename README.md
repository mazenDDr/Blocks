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

On **Graph ▸ Modules & code**, use **Import from repository…**. Give an absolute repository path or a `file://`, `https://` or `ssh://` URL (no credentials in the URL) and a revision, then **Resolve and browse**. The revision resolves to a commit, and the content is read from a bare mirror: nothing is checked out, installed or executed. Setup scripts, hooks, LFS pointers and submodules are labelled and never run or fetched. Dependencies and the license are read statically. Open a Python module (repository modules it imports, including package `__init__` files, are bundled from the pinned commit), pick a function and the role of each parameter, optionally tick pinned dependencies, and **Import pinned function**. The resulting code block records its URL, commit, path and blob as its origin, runs only in the code-block sandbox, and shows whether you have modified it since import. **Compare** shows how the file changed at another revision; re-import to update deliberately. See ADR 0016.

Commands actually run:

```bash
.venv/bin/pytest -q tests/test_repos.py -o faulthandler_timeout=240
VOID_WORKBENCH=<scratch dir> .venv/bin/python -m uvicorn control.app:create_app --factory --app-dir services --host 127.0.0.1 --port 8771
VOID_API=http://127.0.0.1:8771 pnpm -C apps/editor dev --host 127.0.0.1 --port 5296
```

## Serve vision, NLP and speech models (Production tab)

Train a domain example (for instance `nlp_token_classification`), then open **Production ▸ Registry**. Pick the run/node listed as an "nlp model (PyTorch)" candidate and **Register version**. Under **Release**, preview (maxBatch is capped at 4) and deploy. Under **Requests**, **Load recorded held-out example**, send it, and inspect the trace (prediction → release → run → checkpoint → source). Recording ground truth enables native quality in **Monitoring**: word accuracy/span F1, CER/WER, or pixel accuracy/IoU. See ADR 0017.

Commands actually run:

```bash
.venv/bin/pytest -q tests/test_production_domain.py -o faulthandler_timeout=240
VOID_WORKBENCH=<scratch dir> .venv/bin/python -m uvicorn control.app:create_app --factory --app-dir services --host 127.0.0.1 --port 8774
VOID_API=http://127.0.0.1:8774 pnpm -C apps/editor dev --host 127.0.0.1 --port 5299
```

The same Production flow serves **model-graph image classifiers** (for example the reference CNN after a Run on the Graph view). The candidate is listed as "image classifier graph (PyTorch)". Registration requires the training image folder to still match the dataset identity recorded at training, then freezes up to 64 held-out images as the reference; serving never reads the folder again. **Load held-out validation images** fills both the request and its labels. See ADR 0018. `.venv/bin/pytest -q tests/test_production_model.py` was run.

Completed RL runs appear as "greedy DQN policy (PyTorch)" candidates. The served policy is the final Q-network's argmax (no exploration). **Load replay-buffer observations** fills a request. Ground truth, if you have it, is your own reference actions (agreement only); environment return stays in the run's evaluation report. See ADR 0019. `.venv/bin/pytest -q tests/test_production_rl.py` was run.

Unsupervised k-means, Gaussian mixture and PCA nodes (for example in `unsupervised_cells`) are captured for serving as well. DBSCAN and t-SNE are refused because they cannot map new points. Optional external labels give ARI/NMI, never accuracy; PCA shows reconstruction error. See ADR 0020. `.venv/bin/pytest -q tests/test_production_unsup.py` was run.

## Optional access token

Mac verification of the merged cache, repository imports, production adapters and token proxy is recorded in `docs/HANDOFF.md` §18. Acceptance evidence describes the supported bounds; it does not establish broad production readiness.

Set the same `VOID_API_TOKEN` (16+ characters) for the backend and the editor dev server. The editor's proxy adds the header, so the token is not shipped to the browser. Every request without it gets 401. Scripts in `examples/` read the variable too. This is one shared secret, not user accounts, and not encryption: for accounts and TLS see "Named accounts, roles and TLS" below (ADR 0055). See ADR 0021.

```bash
VOID_API_TOKEN=<secret> VOID_WORKBENCH=<dir> .venv/bin/python -m uvicorn control.app:create_app --factory --app-dir services --host 127.0.0.1 --port 8778
VOID_API_TOKEN=<secret> VOID_API=http://127.0.0.1:8778 pnpm -C apps/editor dev --host 127.0.0.1 --port 5303
.venv/bin/pytest -q tests/test_auth.py
```


## Import local vision, text and audio datasets

Each domain workspace has an import form for backend-local files: COCO segmentation JSON + RGB images, CoNLL token/IOB2 labels, or WAV files listed in a JSON array of `{id, file, text}`. Declare synthetic status and license/permitted use explicitly. Verify the converted contract/source hashes, then apply it to the draft and run fresh training. Source declarations travel with inspections, saved-model inference and production reference provenance. See ADR 0022 and HANDOFF §19 for bounds and measured evidence.

**Compatibility:** earlier saved vision/NLP/speech models (including registered domain versions) require retraining/registration after this source-code change. No dependency reinstall; other model families are unchanged. The user's existing port-8000 server was left running.

Commands actually run on the Mac (isolated temporary workbench; all generated files below are SYNTHETIC fixtures):

```bash
PYTHONPATH=services VOID_WORKBENCH=/private/tmp/void-dataset-import-smoke .venv/bin/uvicorn control.app:create_app --factory --host 127.0.0.1 --port 8776
VOID_API=http://127.0.0.1:8776 pnpm -C apps/editor dev --host 127.0.0.1 --port 5299 --strictPort
.venv/bin/python examples/make_import_fixtures.py --out /private/tmp/void-import-fixtures
.venv/bin/python examples/dataset_import_journey.py --base http://127.0.0.1:8776 --fixtures /private/tmp/void-import-fixtures
.venv/bin/pytest -q tests/test_domain_import.py -o faulthandler_timeout=240
.venv/bin/python -m backends.coverage --write
```

Run the servers in separate terminals. The journey creates three projects, imports/verifies fixture bytes, trains native models for two epochs and predicts from their persisted checkpoints. This verifies import mechanics, not real-world model accuracy. License/status assertions are user supplied; original CoNLL whitespace and speech timestamps cannot be recovered. Larger/streamed datasets, crowd/box-only COCO and browser uploads remain outside this release.


## Serve isolated native agent turns

Open **serving_agent** in the picker, use **Run & trace** with a teaching question, then choose its whole-graph candidate in **Production**. Register, review the graph/model digest/budgets, preview a stateless single-turn release and deploy it to a local namespace. Enable capture explicitly to inspect the exact sent model context, source segments and native state/control events in Requests. Each request gets a fresh LangGraph checkpoint; research threads stay intact. Replay makes a new model call and may give a different answer. Monitoring makes no model calls; supplied reference strings measure only exact agreement. See ADR 0023 and HANDOFF §20.

Only bounded prompt/set_state/one chat-model graphs using local Ollama are accepted; persistent conversations use the separate adapter below; retrieval/memory/tools/interrupts and API providers remain outside these serving adapters. The example uses real qwen3.5:2b already installed locally, with SYNTHETIC teaching prompts. No dependency install or domain retraining is needed for these changes. Ollama chooses the model device; device utilization and cost are not measured.

Commands actually run on the Mac against an isolated temporary workbench:

```bash
VOID_WORKBENCH=/private/tmp/void-agent-serving-smoke .venv/bin/python -m uvicorn control.app:create_app --factory --app-dir services --host 127.0.0.1 --port 8777
VOID_API=http://127.0.0.1:8777 pnpm -C apps/editor dev --host 127.0.0.1 --port 5300 --strictPort
.venv/bin/python examples/agent_serving_journey.py --base http://127.0.0.1:8777 --namespace agent-cli-final
.venv/bin/pytest -q tests/test_production_agent.py tests/test_coverage_ledger.py -o faulthandler_timeout=240
.venv/bin/pytest -q -m live tests/test_production_agent_live.py
```

Run the servers in separate terminals. The journey submits a native source run, registers/warms a release, serves and inspects a real response, verifies idempotency, records an independent reference string, replays, runs bounded HTTP traffic and checks rollout/rollback. These small teaching journeys verify mechanics, not answer quality or general serving capacity.

Final verification: **1053 passed, 1 skipped**, plus **8 live Ollama tests passed**; editor build/typecheck and real Chrome serving/context/source-navigation journeys pass. The full suite exposed an existing control/worker cancellation race, fixed with its guarded transition and a deterministic native SQLite regression; existing tests were kept intact. See HANDOFF §20 for exact commands/results, compatibility, cleanup and remaining work.


## Native persistent conversations (local serving)

Open `serving_conversation`, run a source turn, and register the completed native graph in Production. Its release uses maxBatch=1 and conversation mode. Supply an explicit session in Requests; successful turns restore and advance that release/user/session’s native checkpoint. Failed/cancelled/expired turns leave it unchanged. Inspect conversation checkpoint shows actual state and hashes; captured replay restores the selected request’s prior checkpoint without advancing the live conversation. The example retains the last six native messages and uses real local Ollama with labelled SYNTHETIC teaching prompts.

Conversation mode persists native state/history **even with trace capture off**. Full trace contexts/state/events still require captureInputs. User/session values are caller-declared isolation keys; shared-token protection does not provide user accounts or ownership enforcement. Existing stateless agent and other serving-family identities are unchanged: no retraining or dependency install required. See ADR 0024 and HANDOFF §21 for bounds, crash evidence and remaining scope.

Commands run on the Mac against a temporary workbench, leaving the user’s port-8000 service untouched:

```bash
VOID_WORKBENCH=/private/tmp/void-conversation-smoke .venv/bin/python -m uvicorn control.app:create_app --factory --app-dir services --host 127.0.0.1 --port 8779
VOID_API=http://127.0.0.1:8779 pnpm -C apps/editor dev --host 127.0.0.1 --port 5301 --strictPort
.venv/bin/python examples/agent_conversation_journey.py --base http://127.0.0.1:8779 --namespace conversation-cli-final
.venv/bin/pytest -q tests/test_production_conversation.py tests/test_production_agent.py tests/test_production.py -o faulthandler_timeout=240
.venv/bin/pytest -q -o faulthandler_timeout=240
.venv/bin/pytest -q -m live
pnpm -C apps/editor build
pnpm -C apps/editor exec tsc --noEmit
.venv/bin/python -m backends.coverage --write
.venv/bin/python -m backends.coverage --check
```

Reviewed reset/fork is documented below. Physical deletion/migration, long-term memory/retrieval/tool effects, streaming, remote/distributed serving and semantic-answer benchmarks remain unimplemented. Use one owning control process.


## Review, fork or reset a serving conversation

In Production → Requests, select a conversation release, user and session, then **Inspect conversation checkpoint**. Management controls target that exact revision/SHA. Fork creates an unused destination session under the same release/user, preserves native history and gives future turns an independent thread. Reset requires the review checkbox, clears only the live head and starts its next successful turn fresh. Earlier snapshots/traces and captured replay are retained. Both actions record an immutable receipt, reject changed checkpoints and make no model call. Serving revisions remain monotonic through reset; fork begins at revision 1 with inherited native state.

Reset does not physically delete data. These are caller-declared isolation keys, with shared-token protection when configured; authenticated ownership and privacy-erasure APIs remain unimplemented. Existing registered versions keep their original identity. See ADR 0025 and HANDOFF §22 for limits, crash/restart evidence and remaining work.

Commands run on the Mac against a separate temporary workbench:

```bash
VOID_WORKBENCH=/private/tmp/void-session-actions-smoke .venv/bin/python -m uvicorn control.app:create_app --factory --app-dir services --host 127.0.0.1 --port 8780
VOID_API=http://127.0.0.1:8780 pnpm -C apps/editor dev --host 127.0.0.1 --port 5302 --strictPort
.venv/bin/python examples/conversation_actions_journey.py --base http://127.0.0.1:8780 --namespace session-actions-cli
.venv/bin/pytest -q tests/test_conversation_actions.py tests/test_production_conversation.py tests/test_production_agent.py tests/test_production.py -o faulthandler_timeout=240
.venv/bin/pytest -q -o faulthandler_timeout=240
.venv/bin/pytest -q -m live
pnpm -C apps/editor build
pnpm -C apps/editor exec tsc --noEmit
.venv/bin/python -m backends.coverage --write
.venv/bin/python -m backends.coverage --check
```


## Repeatable native editor smoke

This command starts its own backend, editor and installed Chrome, uses a new temporary workbench and ephemeral shared token, exercises the real picker/source Run/registry/release/conversation/fork/reset/replay/monitoring controls, then stops its services. It never connects to the existing port-8000 server. `serving_state` uses real native counters/reducers and declared text templates, labelled SYNTHETIC; it makes zero model calls and measures no answer accuracy.

```bash
pnpm -C apps/editor add -D --save-exact puppeteer-core@25.12.0
pnpm -C apps/editor install --frozen-lockfile
.venv/bin/python tools/editor_smoke.py --output /private/tmp/void-ci-baseline-smoke-final
.venv/bin/pytest -q tests/test_editor_smoke.py -o faulthandler_timeout=240
```

For subsequent clones, install from the lockfile; the `add` command above records the original dependency addition. Omit `--output` for an automatically chosen temporary directory. A supplied output must be new and outside the repository; it is never overwritten. Chrome is discovered on macOS/Linux, or supply `--chrome` with its executable path. No browser or LLM is downloaded. Evidence includes actual traces/receipts, screenshots, runtime/API/console messages, service logs, runner status and cleanup outcomes. Run with the project Python environment, Node/pnpm and Chrome installed. Failure returns a nonzero exit code and retains available evidence.

`.github/workflows/verify.yml` defines the offline native suite, editor build/typecheck, coverage and this Chrome baseline on GitHub. Hosted outcomes are recorded in HANDOFF §23; live Ollama remains independently verified on the Mac. This is one bounded UI flow, not all-domain/browser-platform coverage or deployment. ADR0027 extends the CI browser journey with offline recovery below. See ADR0026.

### Offline workbench backup and recovery (ADR0027)

Stop **all** control, worker, tracker, repository and other writers to the source workbench before creating a backup. `--offline` is your explicit attestation; the command cannot detect an idle server or stop it for you. Never back up the user's running port-8000 workbench as an offline source. These commands were run against a new isolated, stopped SYNTHETIC workbench:

```bash
.venv/bin/python tools/recovery_smoke.py --output /private/tmp/void-recovery-smoke-mac-final
.venv/bin/python -m workbench_backup create /private/tmp/void-recovery-smoke-mac-final/recovered /private/tmp/void-backup-cli-snapshot --offline
.venv/bin/python -m workbench_backup verify /private/tmp/void-backup-cli-snapshot
.venv/bin/python -m workbench_backup restore /private/tmp/void-backup-cli-snapshot /private/tmp/void-backup-cli-restored --trusted-local
.venv/bin/pytest -q tests/test_workbench_backup.py -o faulthandler_timeout=240
```

Use new destination paths on another run; existing directories are refused. Backup contains `manifest.json` and `data/`. All file sizes/hashes, native SQLite integrity, CAS contents and direct database references are verified. Committed WAL pages are copied through SQLite's native backup API; transient sidecars are omitted. Empty directories and ordinary bytes are retained; links/special files are refused. Output records a manifest SHA256, usable with `verify`/`restore --manifest-sha256` when separately retained. Restore requires trusted provenance because native artifacts may execute code when later loaded; checksums do not authenticate a backup. There is no web upload interface.

Absolute paths and saved native identities remain unchanged. External datasets, secret files/env values, provider runtimes, repository code and dependencies are not bundled. Root-relative CAS/store recovery works in the same pinned environment; source-dependent continuation/tracking may need original external paths or new reviewed configuration/runs. Existing checks still refuse incompatible upgrades; no migration is implemented. Backup contains unredacted data, stored with directories 0700/files 0600; executable bits are removed. No encryption, scheduling/retention or online/power-loss recovery guarantee.

The recovery smoke runs the original browser journey, closes owned services, backs up/verifies, deletes only its generated source workbench, restores, then checks the same route/version/checkpoints, independent conversation continuations, old replay and monitoring through a fresh editor/Chrome session. CI runs the same two-stage journey; it uploads only logs/JSON/screenshots/JUnit, never backup/model/workbench contents. See HANDOFF §24 for actual results and remaining scope.

### Offline W&B links and native tracker recovery (ADR0028)

Default v1 creation still refuses links. Explicit `--links internal` creates v2 with inert link metadata: relative internal ordinary-file/directory links are recreated only after restored bytes are verified. Absolute, escaping, dangling/chained links and database/CAS aliases are refused. Native W&B also links an external diagnostic log; `--omit-wandb-external-logs` explicitly omits only that link and records its target/reason in `omittedLinks`. No external target is read or bundled. These commands ran on isolated, stopped SYNTHETIC workbenches:

```bash
.venv/bin/python tools/recovery_smoke.py --trackers --output /private/tmp/void-links-recovery-smoke
.venv/bin/python -m workbench_backup create /private/tmp/void-backup-mlflow/source /private/tmp/void-links-wandb-native-backup --offline --links internal --omit-wandb-external-logs
.venv/bin/python -m workbench_backup verify /private/tmp/void-links-wandb-native-backup
.venv/bin/python -m workbench_backup restore /private/tmp/void-links-wandb-native-backup /private/tmp/void-links-wandb-native-restored --trusted-local
```

Use new output paths on another run. CI now uses the tracker recovery smoke: actual native MLflow/W&B exports, source deletion, restored conversations and confirmed export reconnect through Chrome. Native SDK bytes/internal links survive; immutable absolute tracker paths are unchanged. MLflow artifact downloads require the original absent root to be restored; relocated metadata/history remain readable. W&B confirmed-directory provenance still names its original path; this is offline export recovery, not live resume/upload. [Recovery runbook](docs/RECOVERY.md) explains external requirements and supported restore paths. No new retraining requirement; HANDOFF §25 records verification.

### Find recorded serving conversations (ADR0029)

In Production → Requests, select a conversation release and user namespace, then **Discover conversations**. A literal session prefix filters recorded session keys; next/previous pages list up to25 committed heads. **Inspect [session]** selects that Session and reads its actual native checkpoint through the existing verified inspector/reset/fork controls. Empty/reset heads show no live checkpoint. Discovery lists metadata only, makes no model calls and changes no state. User namespaces remain caller-declared isolation keys, not authenticated accounts. Pages reflect current reads rather than a frozen historical catalog.

These commands ran; the browser recovery check seeds27 actual native sessions and checks25+2 pages, back navigation, selected checkpoint and another user’s empty scope:

```bash
.venv/bin/pytest -q tests/test_conversation_discovery.py tests/test_production_conversation.py tests/test_conversation_actions.py -o faulthandler_timeout=240
.venv/bin/python tools/recovery_smoke.py --trackers --output /private/tmp/void-discovery-recovery-smoke-final
```

No model source pins or dependencies changed. Historical checkpoint restore is described below; privacy deletion, retention and authenticated ownership remain unavailable. See HANDOFF §26.

### Restore an earlier native conversation turn

In Production → Requests, inspect the current conversation, then select a successful
recorded request from that same release/user/session. Preview historical checkpoint
shows its verified state and source trace/checkpoint hashes. Supply a reason, confirm
the reviewed replacement and restore. The next turn continues from that earlier
state in a fresh native thread; serving revisions keep increasing. Earlier records
remain available, including captured replay. A session logically reset to an empty
head can also be restored. Preview/restore themselves make no model call.

Actual verification commands:

```sh
.venv/bin/pytest -q tests/test_conversation_history.py
.venv/bin/python tools/recovery_smoke.py --trackers --output /private/tmp/void-history-recovery-smoke
```

This requires the original native code/environment/provider identity and exact
scope. It does not import arbitrary checkpoints or erase historical data. See
HANDOFF §27 and ADR0030 for receipt/concurrency/recovery limits and final checks.

### Undo and redo graph drafts

Use Undo/Redo in the top bar or Cmd/Ctrl-Z and Cmd/Ctrl-Shift-Z (Ctrl-Y also works).
Graph settings and layout restore together; a node drag is one edit. Loading another
project resets history. Saving keeps it, so undoing a saved edit marks the draft dirty.
Focused text/code editors keep their own undo. Production and Integrations do not
receive draft shortcuts. Earlier runs, weights, conversations and external actions
stay recorded.

History lasts for this page session, bounded to100 prior documents and8MiB estimated
JSON history; older entries evict. Actual commands run:

```sh
node --test apps/editor/tests/documentHistory.test.mjs
.venv/bin/python tools/editor_history_smoke.py --output /private/tmp/void-undo-editor-smoke-final2
```

### Research run records

Open Records to search recorded run/project IDs, graph hashes and current notes/tags.
Filters include exact tag, project, graph family and status. Inspect a record to edit
its author label, note and tags; saving compares the reviewed revision and graph identity.
A competing edit requires reloading and reviewing the stored values. Read annotation
revisions preserves earlier notes, including when current metadata is cleared. Open
original run uses its existing inspector when that project is currently open.

These are authored observations, separate from measured results. Shared-token holders
can edit them; an author label is not an authenticated account. The additive SQLite
file is included in offline backup/restore. It does not provide privacy erasure or
automatic retention. Actual commands run:

```sh
.venv/bin/pytest -q tests/test_research_records.py
.venv/bin/python tools/recovery_smoke.py --trackers --output /private/tmp/void-records-recovery-smoke-final
```

### Copy and paste graph drafts

Open Copy and paste graph nodes beside Keyboard graph tools. Select root nodes using
checkboxes, then copy; open a compatible model/tabular/domain graph and paste. Internal
wires, configuration, relative positions and required module/code definitions are
preserved with fresh IDs. Sharing within the copied group is rebound; boundary wires
stay disconnected and native validation reports missing inputs. Paste is one Undo
edit. The panel shows the actual source graph identity and omitted boundary count.

This page-local clipboard survives project changes, clears on reload, and carries no
native weights/files. Opaque state references and conflicting definitions refuse;
agent/RL and module editor copying remain separate. Actual commands run:

```sh
node --test apps/editor/tests/*.test.mjs
.venv/bin/pytest -q tests/test_graph_clipboard.py
.venv/bin/python tools/editor_clipboard_smoke.py --output /private/tmp/void-clipboard-editor-smoke-final
```

### Structured graph outline

Open Structured graph outline beside Keyboard graph tools on model/tabular/domain
canvases or inside a module. Search nodes, operations, wires, module references or
native errors. Inspect selects the existing inspector; Center fits that node without
changing saved positions; Open module enters its existing definition editor. Native
contracts and diagnostics show the exact validation identity and disappear while a
new validation is pending. This view describes structure, not learned activations.

The list renders at most 50 nodes per page. Search/filter/navigation leave graph and
layout unchanged. Agent/RL outlines and broader editor navigation remain separate.
Actual commands run:

```sh
node --test apps/editor/tests/graphOutline.test.mjs
.venv/bin/pytest -q tests/test_graph_outline.py
.venv/bin/python tools/editor_outline_smoke.py --output /private/tmp/void-outline-editor-release
```

Controlled canvases preserve their actual DOM measurements through draft undo/redo and
validation updates. These sizes remain page-local; saved graph/layout data is unchanged.
Initial loading cancels disposed requests so a stale startup response cannot replace
a newly opened draft. Verification details and remaining work are in HANDOFF §32.

### Arrange graph nodes

Open Arrange graph nodes on model/tabular/domain canvases or inside a module. Select
2–100 cards for edge/center alignment or 3–100 for equal-gap distribution. Distribution
keeps the end cards fixed and refuses when there is insufficient space; alignment
can overlap cards. Collapse expanded module frames before arranging the root graph.
Only selected layout positions change, with one Undo/Redo; model configuration and
connections remain unchanged. Actual commands run:

```sh
.venv/bin/python tools/editor_arrangement_smoke.py --output /private/tmp/void-arrangement-all-families
```

### Node comments

Open Node comments and select a node through the outline or canvas. Enter its comment
and author label, Apply, then Save project. The comment records the reviewed native
graph/module identity; configuration changes show an earlier-identity notice until
explicit review. Comments follow renames/deletions through Undo. The catalogue finds
comments and exposes orphaned or malformed metadata for explicit removal.

These are authored observations, with a declared author and browser time. Removing
metadata does not provide privacy erasure from history/backups. Copy/paste keeps notes
with source nodes. Actual commands run:

```sh
.venv/bin/python tools/editor_comments_smoke.py --output /private/tmp/void-node-comments-editor-smoke-final
.venv/bin/python tools/recovery_smoke.py --trackers --output /private/tmp/void-node-comments-integrated-recovery-final
```


Typed wire insertion verification (owned temporary services and SYNTHETIC fixtures):

```bash
.venv/bin/pytest -q tests/test_graph_insertion.py
.venv/bin/python tools/editor_insertion_smoke.py --output /private/tmp/void-insertion-editor-scope-reentry
```

The real Chrome journey passes. Final native/environment acceptance and retained
failure diagnostics are recorded in docs/HANDOFF.md §35.


Command menu verification (owned isolated Chrome/native services):

```bash
node --test apps/editor/tests/graphCommands.test.mjs
.venv/bin/python tools/editor_commands_smoke.py --output /private/tmp/void-commands-editor-release
```

Final acceptance and retained focus-test failures are in docs/HANDOFF.md §36.


## Move selected layout cards together

In Graph, open **Move selected nodes together**, select actual layout cards and enter
horizontal/vertical offsets in canvas units. Root model/tabular/domain and shared module
definition layouts use the same offset for each selected origin. Only selected positions
change, with one Undo/Redo. Zero offset makes no edit; empty/nonfinite/out-of-bounds
requests refuse. Collapse expanded root module frames explicitly before moving. Placement
can overlap; persistent grouping and automatic layout remain separate. Mac verification
passes; hosted scope is recorded in HANDOFF§37. Actual commands run:

```bash
node --test apps/editor/tests/*.test.mjs
.venv/bin/python tools/editor_movement_smoke.py --output /private/tmp/void-movement-editor-rendered-geometry
.venv/bin/pytest -q -o faulthandler_timeout=240
.venv/bin/pytest -q -m live
pnpm -C apps/editor build
pnpm -C apps/editor exec tsc --noEmit
.venv/bin/python -m backends.coverage --write
.venv/bin/python -m backends.coverage --check
```


## Copy and paste module-internal draft nodes

In a PyTorch shared module definition, open **Copy and paste graph nodes** and select
internal nodes. The snapshot displays its actual native module hash; input/output and
generated cards are excluded. Paste into a compatible root or module: fresh identities
reserve existing orphan notes/layout, internal wires stay connected and omitted boundary
wires stay disconnected. Native missing-input errors remain visible. Target interfaces,
existing wires and original authored notes stay intact. A module paste changes the shared
definition used by all instances; one Undo/Redo restores the documents. No weights or
opaque module-local state are transferred. Mac acceptance is recorded in HANDOFF§38. Actual
commands run:

```bash
.venv/bin/pytest -q tests/test_module_clipboard.py
node --test apps/editor/tests/moduleClipboard.test.mjs
.venv/bin/python tools/editor_module_clipboard_smoke.py --output /private/tmp/void-module-clipboard-editor-ready-picker
```


## Open a native diagnostic in its module

In **Structured graph outline**, **Open diagnostic node** selects the actual stored node
and opens the correct shared module breadcrumb, including nested definitions, repeat
iterations and either select branch. Pending/unavailable native reports supply no action.
Missing/ambiguous/harness-only targets have no guessed destination. Navigation changes
no graph/UI/history; editing the shared definition affects every instance. Inspector
contracts come from actual native module validation, with no recorded execution state
selected. Mac acceptance is recorded in HANDOFF§39. Actual commands run:

```bash
.venv/bin/pytest -q tests/test_diagnostic_navigation.py
node --test apps/editor/tests/diagnosticNavigation.test.mjs
.venv/bin/python tools/editor_diagnostic_smoke.py --output /private/tmp/void-diagnostic-navigation-editor-smoke
```


## Inspect agent structure with the keyboard

In an agent **Canvas**, open **Structured agent outline**. Search declared control routes/
joins and actual native reads/writes/effects/diagnostics, filter errors, or open an actual
node inspector with keyboard controls. The current native validation hash identifies
contracts; no runtime value or taken route is inferred. Pending/unavailable analysis
withholds prior values; START/END remain terminal references. Search input is bounded
to200characters and rows to50/page. Inspect/search/paging change no graph/UI/history.
Mac acceptance is recorded in HANDOFF§40. Actual commands run:

```bash
node --test apps/editor/tests/agentOutline.test.mjs
.venv/bin/python tools/editor_agent_outline_smoke.py --output /private/tmp/void-agent-outline-editor-wire-search
```


## Inspect RL structure and native contracts

In **RL lab → Structure**, search the actual nodes/typed wires, declared configuration,
current native observation/action/reward contracts or diagnostic codes. Use **Inspect RL
node** with the keyboard to focus its exact read-only inspector. Existing Environment/
Learner controls open only when that node family is unique. Current root validation
identity labels native spaces, reward weights, network shapes/counts and DQN equations;
no trained Q-values or trajectory are inferred. Missing contracts say not returned and
pending validation withholds previous results. Search is bounded to200UTF16 units and
rows to50/page. Navigation leaves graph/UI/history unchanged. Mac acceptance is recorded in
HANDOFF§41. Actual commands run:

```bash
node --test apps/editor/tests/rlOutline.test.mjs
.venv/bin/pytest -q tests/test_rl_outline.py
.venv/bin/python tools/editor_rl_outline_smoke.py --output /private/tmp/void-rl-outline-editor-native-missing
```


## Serve a native JSON agent turn

Select **serving_json_agent** in the picker. Its labelled SYNTHETIC colour/count
teaching text is extracted by the actual installed local Ollama `qwen3.5:2b` model
using the native flat structured-output schema. Run a source turn, then register
its **schema-validated JSON turn** candidate in Production. Review the pinned
input fields/schema/provider/source evidence. Use stateless mode, maxBatch1 and a
30-second release timeout; preview invokes a real warmup before explicit deployment.

Requests return a schema-valid finite object with native provenance. Capture inputs
to inspect actual contexts/state/events and replay through a new isolated model
call. Ground truth takes one independently supplied schema-valid object per turn.
Monitoring reports canonical JSON agreement and descriptive output byte lengths;
it does not establish semantic correctness or invoke generation. Existing saved
text/conversation versions retain their native identities. Tools/retrieval/indexes,
thread/long-term memory, interrupts, remote/fixture providers and JSON conversations
remain unsupported. ADR0045/HANDOFF§42 record bounds and verification status.

Commands actually run on the Mac, using fresh owned services and generated evidence
outside Git. Both JSON journeys require the real installed model and fail without
it; default hosted Linux CI has no Ollama and does not claim provider verification.

```bash
.venv/bin/python tools/editor_json_agent_smoke.py --output /private/tmp/void-json-serving-editor-captured-replay
.venv/bin/python tools/recovery_smoke.py --trackers --json-agent --output /private/tmp/void-json-serving-ready-recovery
.venv/bin/pytest -q tests/test_production_json_agent_live.py -m live
.venv/bin/pytest -q -o faulthandler_timeout=240
.venv/bin/pytest -q -m live
pnpm -C apps/editor build
pnpm -C apps/editor exec tsc --noEmit
node --test apps/editor/tests/*.test.mjs
.venv/bin/python -m backends.coverage --write
.venv/bin/python -m backends.coverage --check
```

Reuse the commands with new evidence directory names: runners refuse overwrite.
The recovery journey retains the original native conversations/research/comments/
local MLflow/offline W&B checks, additionally verifies JSON manifests/requests,
physically deletes its disposable source workbench and makes a fresh real model
call from the restored version. It does not bundle external services or environments.


## Optional automatic cache retention

For a saved tabular project, open **Automatic cache retention** in the Run panel.
It defaults to disabled. Edit the recorded policy, choose how many latest variants
to retain per node and the minimum age/cadence, then preview actual candidates.
**Save retention policy** explicitly updates future checks for that project.
The owner service schedules native pruning and shows actual receipts/errors;
disable prevents future eligible checks. A stale policy revision requires reloading.

Checks defer during active or paused native research runs and preserve recorded
run artifacts. Removed cache results are recomputed when needed. Use one owning
control process and stop it before offline backup. The lazy policy database and
receipts survive backup/restore; enabled policies resume when that service starts.
This does not erase runs/models, add cache support for other graph families or
provide distributed/exactly-once maintenance. ADR0046/HANDOFF§43 record bounds
and final verification status. Existing saved model/JSON/legacy identities remain
unchanged. Incomplete editor project IDs do not issue background policy requests.

Commands actually run against generated private native fixtures and owned services:

```bash
.venv/bin/pytest -q tests/test_cache_retention.py tests/test_node_cache.py
.venv/bin/python tools/editor_cache_retention_smoke.py --output /private/tmp/void-cache-retention-editor-labelled-fields
.venv/bin/python tools/recovery_smoke.py --trackers --json-agent --cache-retention --output /private/tmp/void-cache-retention-integrated-recovery
.venv/bin/pytest -q -o faulthandler_timeout=240
.venv/bin/pytest -q -m live
pnpm -C apps/editor build
pnpm -C apps/editor exec tsc --noEmit
node --test apps/editor/tests/*.test.mjs
.venv/bin/python -m backends.coverage --write
.venv/bin/python -m backends.coverage --check
```

Use fresh evidence directory names. The JSON recovery option requires actual local
Ollama; omit it for provider-free native cache/conversation/tracker recovery.
Default hosted CI runs the new cache browser and source-deletion recovery alongside
every original browser stage; it does not substitute an Ollama fixture.


## Measured large-graph selection

Canvas card and wire metadata remain stable while selecting another node; graph,
layout, card-size or native report changes still update them. Actual warm Chrome
measurements on declared100/500/1000node metadata-only teaching chains are in
`benchmarks/results/editor_selection_summary.md`, with raw samples/environment/
source hashes beside it. The500-node outline-selection p95 changed181→54ms on
this Mac/dev build;1000nodes still measured149.8ms. Side panels that list every
node now reuse unchanged rows and options (ADR0048), bringing 1000-node selection
p95 to 47.1ms and 500-node to 34.2ms in the same benchmark. This does not certify all
routine editing, pan/zoom, representative models or other platforms.

Commands actually run (use a fresh evidence path):

```bash
.venv/bin/python tools/editor_performance_benchmark.py --output /private/tmp/void-editor-performance-stable-cards-wires
.venv/bin/pytest -q -o faulthandler_timeout=240
.venv/bin/pytest -q -m live
pnpm -C apps/editor build
pnpm -C apps/editor exec tsc --noEmit
node --test apps/editor/tests/*.test.mjs
.venv/bin/python tools/recovery_smoke.py --trackers --json-agent --cache-retention --output /private/tmp/void-editor-selection-integrated-recovery
.venv/bin/python -m backends.coverage --check
```

HANDOFF§44/ADR0047 record the final correctness and recovery evidence separately
from descriptive timing. No existing serving identity or dependency was changed.


## Agent graph clipboard

In an agent graph's Canvas tab, open **Copy and paste agent nodes**, tick nodes and
copy. The editor validates the graph natively and copies the nodes, transitions and
routes inside the selection (END stays END), fully selected joins, and the exact
state fields, indexes and memory policies the nodes use. Paste into the same or
another agent graph: IDs are fresh, missing definitions are added and differing ones
refuse. Pasted nodes need an entry transition before the graph validates. RL graphs
use fixed forms, so this does not apply to them. See ADR 0049.

Commands actually run (use a fresh evidence path):

```bash
node --test apps/editor/tests/*.test.mjs
.venv/bin/python tools/editor_agent_clipboard_smoke.py --output /private/tmp/void-agent-clipboard-smoke-3
```


## Reclaiming unreferenced CAS bytes

Preview what an offline collection would delete (read-only), then stop every
control/worker/tracker/repository writer and apply it. A blob is kept if its
64-hex name appears in any workbench file, SQLite cell, path, link target or kept
blob; unreferenced blobs younger than the grace period (default 24h) are kept too.
Deletion is physical; take a backup first if you may need the bytes. See ADR 0050.

```bash
.venv/bin/python -m maintenance.cas_gc --workbench .workbench
.venv/bin/python -m maintenance.cas_gc --workbench .workbench --apply --offline
```

Commands actually run:

```bash
.venv/bin/pytest -q tests/test_cas_gc.py
.venv/bin/python tools/recovery_smoke.py --trackers --cache-retention --gc --output /private/tmp/void-cas-gc-recovery-1
```

## Whole-layout auto-arrange

Open **Arrange graph nodes** and press **Auto-arrange whole layout** to place every
card in the current root or module layout left to right by its connections, using
actual card sizes without overlap. It is one layout Undo edit; wires, configuration
and the native graph hash are unchanged. See ADR 0051.

Commands actually run (use a fresh evidence path):

```bash
node --test apps/editor/tests/*.test.mjs
.venv/bin/python tools/editor_auto_layout_smoke.py --output /private/tmp/void-auto-layout-smoke-3
```

## Sealed (encrypted) backups

Seal a verified offline backup into one AES-256-GCM encrypted, authenticated file
before moving it off the machine; unseal it into a new directory, then restore as
usual. Keep the key file (or passphrase file) separate from the sealed file: a lost
key cannot be recovered. Key and passphrase files must be mode 0600. See ADR 0052.

```bash
.venv/bin/python -m workbench_backup keygen ~/void-backup.key
.venv/bin/python -m workbench_backup seal /path/to/backup /path/to/backup.sealed --key-file ~/void-backup.key
.venv/bin/python -m workbench_backup unseal /path/to/backup.sealed /path/to/unsealed --key-file ~/void-backup.key
.venv/bin/python -m workbench_backup restore /path/to/unsealed /path/to/new-workbench --trusted-local
```

Commands actually run:

```bash
.venv/bin/pytest -q tests/test_sealed_backup.py
.venv/bin/python tools/recovery_smoke.py --trackers --cache-retention --gc --sealed --output /private/tmp/void-sealed-recovery-1
```

## Streamed agent serving

Agent, conversation and JSON-agent releases can stream provider text as server-sent
events while the normal native turn runs. Events are `token` (`{"delta"}`), then
`result` (the exact recorded trace the plain predict route returns) and `end`. The
recorded trace stays authoritative; a client disconnect does not cancel the request.
In the editor, use **Stream prediction request** in Production → Requests. See ADR 0053.

```bash
curl -N -H "Authorization: Bearer $VOID_API_TOKEN" -H 'Content-Type: application/json' \
  -d '{"requestId":"r1","records":[{"question":"Name three colours."}]}' \
  http://127.0.0.1:8000/api/serve/local/<namespace>/predict/stream
```

Commands actually run:

```bash
.venv/bin/pytest -q tests/test_agent_stream.py
.venv/bin/pytest -q -m live tests/test_production_stream_live.py
.venv/bin/python tools/editor_json_agent_smoke.py --output /private/tmp/void-stream-json-agent-2
```

## Serving clustering after fitted preprocessing

k-means, Gaussian mixture and PCA estimators placed after fitted imputation, one-hot
encoding or standardization can now be registered and served. Requests send the raw
source columns; serving replays the run's own fitted steps and estimator without
refitting. See ADR 0054.

Commands actually run:

```bash
.venv/bin/pytest -q tests/test_production_unsup.py
```

## Named accounts, roles and TLS

Instead of one shared token, give each person an account. Roles: `viewer` (read
only), `operator` (changes, but serving requests/traces/conversations only in their
own user namespace), `admin` (everything, including deploy/rollback, aliases,
cache policies, connections and load tests). Mutating requests are audited in
`<workbench>/audit/requests.jsonl`. Do not also set `VOID_API_TOKEN` on the backend.
Give the editor dev server one account's token as its `VOID_API_TOKEN`. See ADR 0055.

```bash
PYTHONPATH=services .venv/bin/python -m control.accounts add --file ~/void-users.json --name alice --role operator   # prints the token once
VOID_USERS_FILE=~/void-users.json .venv/bin/python -m uvicorn control.app:create_app --factory --app-dir services \
  --host 127.0.0.1 --port 8443 --ssl-certfile cert.pem --ssl-keyfile key.pem
```

Commands actually run:

```bash
.venv/bin/pytest -q tests/test_accounts.py
```

## Erasing one serving user's data

Preview, then (with every writer stopped) erase a user's production requests,
traces, labels and conversations from the workbench: rows are deleted, the database
is VACUUMed, and content-store bytes only they referenced are physically removed.
Backups, sealed files and exports keep their own copies. See ADR 0056.

```bash
.venv/bin/python -m maintenance.erase --workbench .workbench --user alice
.venv/bin/python -m maintenance.erase --workbench .workbench --user alice --apply --offline
```

Commands actually run:

```bash
.venv/bin/pytest -q tests/test_erase_user.py
```

## Recovering runs after a crash

Workers refresh a heartbeat while they run. If a worker is killed or the control
service crashes, the restarted service marks runs with no heartbeat, event or status
change for 120 seconds as failed with `E_WORKER_LOST`, keeping everything they
recorded. Paused agent runs are left alone. Lost runs are not resumed automatically;
start a new run (or continue a model run from its last checkpoint). See ADR 0057.

Commands actually run:

```bash
.venv/bin/pytest -q tests/test_worker_recovery.py
```

## Layout groups

Select cards, open **Layout groups** above the canvas and press **Group selected
cards** to draw a named frame around them. Drag the frame's header to move the
group, click it to select its cards for the move/arrange/copy tools, or rename,
change and delete it in the panel. Groups are saved with the layout only; the graph
and its native identity do not change. See ADR 0058.

Commands actually run (use a fresh evidence path):

```bash
node --test apps/editor/tests/*.test.mjs
.venv/bin/python tools/editor_layout_groups_smoke.py --output /private/tmp/void-layout-groups-smoke-3
```

## JSON conversations

An agent graph with thread-scoped state and one structured-output node can be
served as a native conversation whose every turn returns a schema-validated JSON
object. Register its source run with node `__agent_json_conversation__` (the
Production registry lists it as a JSON conversation) and release it with session
state **conversation**. See ADR 0059.

Commands actually run:

```bash
.venv/bin/pytest -q tests/test_production_json_conversation.py
.venv/bin/pytest -q -m live tests/test_production_json_conversation_live.py
```

## Serving agents with retrieval

Agent graphs that retrieve from a declared index can be served as stateless turns.
Registering with node `__agent_retrieval_graph__` copies the exact index the source
run used into the version, so later edits to the document folder (or deleting the
research index) do not change what the release retrieves. Rebuild by running the
source again and registering a new version. See ADR 0060.

Commands actually run:

```bash
.venv/bin/pytest -q tests/test_production_retrieval.py
.venv/bin/pytest -q -m live tests/test_production_retrieval_live.py
```

## Serving a trained image classifier on Keras or JAX

A completed PyTorch image-classifier run is also listed as a Keras and a JAX
candidate. Registering one copies the checkpoint's parameters into that backend,
measures every frozen reference image against PyTorch, and refuses unless all
logits agree within the declared tolerance. Serving then runs on Keras or JAX with
the same requests and outputs. See ADR 0061.

Commands actually run:

```bash
.venv/bin/pytest -q tests/test_production_model.py tests/test_production_portable.py
```

## Pure calculator releases

Open the SYNTHETIC `serving_tools` example, enter an arithmetic expression in `question`,
run to END, then register its **pure calculator turn** candidate in Production. Preview
and deploy a stateless maxBatch=1 release. Capture enables actual native tool arguments,
results and state/events; replay executes a fresh isolated turn. Independently supplied
text labels support literal agreement; monitoring never executes the graph or model.
Graphs can also retrieve from 0–2 pinned index snapshots and call one local Ollama chat
node. File tools, approvals, conversation/memory state and model-selected tools remain
unsupported. See [ADR0062](docs/adr/0062-pure-calculator-serving.md).

Require maxToolCalls 1–8. Calculator templates read immutable input/default fields only;
rendered expressions have at most512characters/128ASTnodes/depth32, no exponentiation,
and finite literals/results of magnitude <=1e100. Native argument errors stay tool
error data; budget/bounds/cancellation/deadline failure refuses the whole result.

Compatibility: existing stateless `agent_json` versions need re-registration because
shared runtime/API/monitor files are among their pins. Existing plain agent, conversation,
JSON-conversation, retrieval and non-agent execution hashes are unchanged.

Commands actually run on this Mac (full results are in HANDOFF§60):

```sh
.venv/bin/pytest -q tests/test_production_tools.py tests/test_production_retrieval.py
.venv/bin/pytest -q -m live tests/test_production_tools_live.py
.venv/bin/pytest -q -m live
.venv/bin/python tools/editor_tools_agent_smoke.py --output /private/tmp/void-tools-editor-2
pnpm -C apps/editor build
.venv/bin/python -m backends.coverage --write
.venv/bin/pytest -q -o faulthandler_timeout=240 --junitxml=/private/tmp/void-tools-native.xml
node --test apps/editor/tests/*.test.mjs
pnpm -C apps/editor exec tsc --noEmit
.venv/bin/python tools/recovery_smoke.py --trackers --json-agent --tools-agent --cache-retention --gc --sealed --output /private/tmp/void-tools-recovery-2
```

All 18 `tools/editor_*_smoke.py` runners passed, with individual
logs and output directories under `/private/tmp/void-tools-regressions`. These paths are
local execution evidence, not committed artifacts or hosted-provider certification.

## Approval checkpoints in releases

Open the SYNTHETIC `serving_approval` example, run it, then approve its research
review to reach END. Register its **native conversation with approval checkpoints**
candidate and preview/deploy a maxBatch=1 conversation release. Warmup pauses privately.
A served turn returns HTTP202 with a committed native pending review and no prediction.
In Requests, **Inspect pending approval**, review the proposal/revision, then choose
**Approve and resume**, **Reject and resume**, or edit the proposal and resume.
The registered graph defines how each decision changes its result. See
[ADR0063](docs/adr/0063-committed-approval-checkpoints.md).

Paused state and the approval receipt persist across restart even when trace capture
is off. Resume preserves the turn inputs and requires the reviewed release, revision,
checkpoint and interrupt ID; stale or competing decisions refuse. Active execution
seconds and model/token budgets carry across the pause; human waiting is excluded.
File tools/effects, fixed parallel forks/joins, repeated interrupts, JSON output and retrieval/memory combinations
are not supported by this family. Existing stateless JSON-agent versions need
re-registration because they pin the shared integration files.

Commands actually run (full acceptance/evidence in HANDOFF§61):

```sh
.venv/bin/pytest -q tests/test_production_approval.py
.venv/bin/pytest -q -m live tests/test_production_approval_live.py
.venv/bin/python tools/editor_approval_agent_smoke.py --output /private/tmp/void-approval-editor-1
pnpm -C apps/editor exec tsc --noEmit
```

Additional commands actually run for approval verification:

```sh
.venv/bin/pytest -q -m live
.venv/bin/python -m backends.coverage --write
.venv/bin/pytest -q -o faulthandler_timeout=240 --junitxml=/private/tmp/void-approval-native-final.xml
node --test apps/editor/tests/*.test.mjs
pnpm -C apps/editor build
.venv/bin/python tools/recovery_smoke.py --trackers --json-agent --tools-agent --approval-agent --cache-retention --gc --sealed --output /private/tmp/void-approval-recovery-final
.venv/bin/python tools/editor_layout_groups_smoke.py --output /private/tmp/void-approval-layout-groups-stable-1
```

All `tools/editor_*_smoke.py` runners were executed with evidence under
`/private/tmp/void-approval-regressions-final`. The layout-groups runner now waits for
stable frame geometry before the real selection click after reload; selection,
member movement, exact saved layout and Undo checks are retained. Hosted failure
and final acceptance are recorded in HANDOFF§61.

## Serving retrieval and short-term memory conversations

Open SYNTHETIC `serving_context` for a model-free conversation, or
`serving_context_json` for actual installed local Ollama JSON extraction. Run the
source to END, register its context candidate, preview the native warmup and deploy.
Requests retain thread state and bounded short-term history. **Inspect conversation
checkpoint** shows the native state. Recorded requests show pinned chunk IDs/scores,
embedding identity and policy application stages/decisions, alongside exact sent
model context when capture is enabled. Replay uses the captured parent checkpoint
without advancing the session. Reference source inputs are not ground truth.

Four adapters support stateless/conversation and text/JSON modes, with 0–2 pinned
indexes and 0–2 bounded short-term policies; at least one retrieve or memory_select
is required. `keep_last_n<=8`, retrieve k<=8, extractive summaries only, local hash
policy embeddings are lexical hashing. No long-term memory/effect/tool/approval
combinations are served. JSON remains schema validation, not semantic correctness.
See [ADR0064](docs/adr/0064-pinned-context-conversations.md). Earlier execution
identities remain unchanged; shared integration pins require stateless agent_json
re-registration.

```sh
.venv/bin/pytest -q tests/test_production_context.py
.venv/bin/pytest -q -m live tests/test_production_context_live.py
.venv/bin/python tools/editor_context_agent_smoke.py --output /private/tmp/void-context-editor-1
.venv/bin/python tools/editor_context_agent_smoke.py --json --output /private/tmp/void-context-json-editor-2
.venv/bin/python tools/recovery_smoke.py --trackers --json-agent --tools-agent --approval-agent --context-agent --context-json --cache-retention --gc --sealed --output /private/tmp/void-context-recovery-final
```

The JSON mode requires installed `qwen3.5:2b`; no provider substitution/download.
Hosted CI runs the zero-model context journey through integrated recovery. Full
native/live/editor/recovery acceptance and exact evidence are recorded in HANDOFF§62.


Full context acceptance commands actually run (evidence in HANDOFF§62/63):

```sh
.venv/bin/pytest -q -o faulthandler_timeout=240 --junitxml=/private/tmp/void-context-native-final2.xml
.venv/bin/pytest -q -m live
node --test apps/editor/tests/*.test.mjs
pnpm -C apps/editor typecheck
pnpm -C apps/editor build
.venv/bin/python -m backends.coverage --write
.venv/bin/python -m backends.coverage --check
```

All20 editor runners passed with evidence at `/private/tmp/void-context-regressions-final`;
full native1338/1skip/25deselect, live25, sealed recovery207 files/11DBs. Retained
failed first ledger/JSON browser checks and separate hosted outcomes are in HANDOFF.

## Flat JSONL table sources

Open SYNTHETIC `jsonl_regression`, run the native graph and inspect the JSONL
source hash, row IDs, schema and original-byte provenance. Register the learned
OLS candidate and serve from pinned native transforms/model without rereading
JSONL. Local JSONL is also supported by source cache handling, bundle requirements,
explicit inert package snapshots and authenticated loopback CPU workers.

Only nonempty flat UTF-8 objects per nonblank LF line are supported. Duplicate,
nested, malformed, nonfinite and unsafe integer values refuse; no silent row skips
or date inference. Full bounded validation: 50MB / 200000 records / 256 columns /
2M cells / 1MB record. See [ADR0065](docs/adr/0065-flat-jsonl-sources.md).
Package/worker snapshots keep their smaller 8MiB bound. The existing includeCsv API
flag explicitly embeds either CSV or JSONL; the UI labels both. Examples and
in-sample labels are SYNTHETIC, with no real-world housing-quality claim.

Commands actually run (full acceptance in HANDOFF§64):

```sh
.venv/bin/python examples/make_jsonl_fixture.py
.venv/bin/pytest -q tests/test_jsonl_source.py
.venv/bin/python tools/editor_jsonl_smoke.py --output /private/tmp/void-jsonl-editor-1
```

The owned editor runner now handles a transient ConnectionResetError during its
existing bounded service-readiness poll. Hosted item3 failed before Chrome launch
at cache-seed startup; evidence and final hosted outcomes remain in HANDOFF§64.
JSONL source clicks wait for actual finished UI state after native completion.
No browser assertion or error check is removed.

Integrated JSONL/cache verification also corrects the cache-retention seed helper
to select its explicit project (`any_project=False`), retaining exact19/3/16
assertions when other projects have cached nodes. Native store/cache code is unchanged.

Additional JSONL commands actually run:

```sh
.venv/bin/pytest -q -o faulthandler_timeout=240 --junitxml=/private/tmp/void-jsonl-native-final.xml
.venv/bin/pytest -q -m live
node --test apps/editor/tests/*.test.mjs
pnpm -C apps/editor typecheck
pnpm -C apps/editor build
.venv/bin/python -m backends.coverage --write
.venv/bin/python -m backends.coverage --check
.venv/bin/python tools/recovery_smoke.py --trackers --json-agent --tools-agent --approval-agent --context-agent --context-json --jsonl --cache-retention --gc --sealed --output /private/tmp/void-jsonl-recovery-final3
.venv/bin/python tools/editor_cache_retention_smoke.py --output /private/tmp/void-jsonl-cache-helper-final
```

All21 editor_* runners passed at `/private/tmp/void-jsonl-regressions-final`;
actual live25 and integrated recovery240 files/11DBs/16 browser cases passed.
Source-readiness/cache-helper failed attempts are retained in HANDOFF§64.

Full JSONL native acceptance:1361 passed/1 skipped/25 deselected558.54s,1941
retained warnings. Hosted outcome is recorded separately in HANDOFF§64.

## SQLite schema versions

Managed stores adopt compatible legacy version0 through an explicit transactional
version1 migration, then check ownership, shape and user_version at every open.
Future versions refuse before application SQL; there is no automatic downgrade.
Nine application schemas are managed; native LangGraph and tracker/provider schemas
keep their own ownership. See [ADR0066](docs/adr/0066-sqlite-schema-migrations.md).

`python -m storage inspect --workbench <path>` is read-only.
`python -m storage migrate --workbench <path> --offline` requires all writers stopped
and upgrades existing files independently. No absent optional stores are created.
These interfaces are implemented and accepted (ADR0066, HANDOFF§67).
The CLI was exercised through subprocess tests; focused test command actually run:

```sh
.venv/bin/pytest -q tests/test_storage_schema.py tests/test_production_tools.py tests/test_cache_retention.py
```

**Compatibility:** all eleven agent families require re-registration because their
storage source pins changed. Prior versions/releases/checkpoints remain recorded;
no automatic rebinding or cross-release state migration. Non-agent prediction
source identities remain unchanged. No library-owned schema is marked version1.

Control startup preflights all existing owned schema files before mutations.
Integrated recovery explicitly constructs compatible version0 headers on stopped
SYNTHETIC native seed outputs, then verifies exact native row hashes and version1
before restored browser actions. This is not arbitrary historical-binary coverage.
SQLite failure tests use native TEMP triggers per connection, preserving real
transaction rollback without adding forbidden persisted schema objects.

Additional schema verification commands actually started (completion evidence and
pending work in HANDOFF§65):

```sh
.venv/bin/pytest -q -o faulthandler_timeout=240 --junitxml=/private/tmp/void-schema-native-final2.xml
.venv/bin/pytest -q -m live
.venv/bin/python -m backends.coverage --write
.venv/bin/python -m backends.coverage --check
node --test apps/editor/tests/*.test.mjs
pnpm -C apps/editor typecheck
pnpm -C apps/editor build
.venv/bin/python tools/recovery_smoke.py --trackers --json-agent --tools-agent --approval-agent --context-agent --context-json --jsonl --legacy-schema --cache-retention --gc --sealed --output /private/tmp/void-schema-recovery-final
```

All21 editor runners are being executed at `/private/tmp/void-schema-regressions-final`.
An interrupted native suite and failed initial focused commands are retained as
failures/interruption, with no acceptance inferred from them.

Receipt/reset/fork/history rollback fixtures likewise use real per-connection TEMP
SQLite triggers; all rollback/restart/competition assertions remain. The backup WAL
fixture records committed evidence in declared events rather than adding an
undeclared table to owned metadata; WAL presence and exact recovery checks remain.
Full native acceptance: fresh run 1386 passed, 1 skipped, 25 deselected, 1941 warnings in 516.62s (JUnit 1387 cases, 0 failures/errors; `/private/tmp/void-schema-native-final3.{log,xml}`).

Schema continuation stopped at Codex's weekly99% threshold; Claude Code completed
acceptance from the final3 run below (HANDOFF§67).
Completed targeted compatibility checks:72 pass, research20 pass; full live25,
all21 editor journeys and legacy schema recovery pass. Final3 full native then passed
(1386/1 skip/25 deselected). Earlier interrupted failures remain preserved in §65.

## Training on an NVIDIA GPU

Choose **Device: cuda** in the Train tab (or `"device": "cuda"` in the run config) when
the worker runs on a machine with a usable NVIDIA GPU; otherwise the run fails with
`E_DEVICE_UNAVAILABLE`. Checkpoints are saved as CPU tensors, so serving works on any
machine. CUDA results are not bitwise reproducible. See ADR 0067.

Commands actually run (on the GPU machine, in its `main` conda env):

```bash
PYTHONPATH=python:services:tests python -m pytest -q -m gpu tests/test_training_device.py
```

## Continuous-action RL with TD3

Open the **rl_pendulum_td3** example: reward → environment → TD3 learner →
evaluation. TD3 trains continuous-action Gymnasium environments (Pendulum-v1 in the
catalog) and evaluates the deterministic actor on declared seeds. A 20,000-step run
reached a mean task return near −172. Choose the device in the RL bar. See ADR 0068.

Commands actually run:

```bash
.venv/bin/pytest -q tests/test_rl_td3.py
.venv/bin/python tools/editor_td3_smoke.py --output /private/tmp/void-td3-smoke-2
```

## Serving a TD3 policy

A completed TD3 run is listed as a continuous-policy candidate in Production. The
served endpoint takes `{observation: [...]}` records and returns the deterministic
actor's bounded actions. Supply reference action vectors as labels to see their mean
absolute error. See ADR 0069.

Commands actually run:

```bash
.venv/bin/pytest -q tests/test_production_td3.py
.venv/bin/python tools/editor_td3_smoke.py --output /private/tmp/void-td3-smoke-4
```

## Training on Keras or JAX

In the Train tab choose **Backend: keras** or **jax** (plain SGD, momentum 0, CPU).
The run starts from the same seeded weights as PyTorch, takes each SGD step on the
chosen backend and saves PyTorch-format checkpoints. See ADR 0070.

Commands actually run:

```bash
.venv/bin/pytest -q tests/test_backend_training.py
.venv/bin/python tools/editor_backend_training_smoke.py --output /private/tmp/void-backend-training-smoke-2
```

## Running a worker on another machine

Start the worker with TLS on the other host, then submit with its certificate:

```bash
# on the worker host (token in the environment, key readable only by you)
VOID_WORKER_TOKEN=... python -m scale.worker_server --workbench ~/void-worker --host <its-address> --port 8778 \
  --ssl-certfile worker.pem --ssl-keyfile worker.key
```

In Scale → worker, use endpoint `https://<its-address>:8778` and the path of
`worker.pem` on this machine as the worker certificate file. Cleartext is accepted only
on 127.0.0.1. See ADR 0071.

Commands actually run:

```bash
.venv/bin/pytest -q tests/test_worker_tls.py
```
