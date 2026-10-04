# Project Void: Visual AI Workbench

Phase 0 (semantic and execution foundation), Phase 1 (visual CNN workbench: control API + editor), Milestone 2a (tabular graph kind:
data preparation, classical ML, statistics), Milestone 2b (connected data sources, versioned extraction, studies and sweeps), Milestone 3 (research-level composition) and
Milestone 4 (language-model and agent workflows on native LangGraph) are implemented.
See `docs/PLAN.md`, `docs/CAPABILITIES.md` and `docs/adr/` (ADR 0003: tabular graph kind; ADR 0004: connectors, snapshots, studies; ADR 0008: agent graph kind, memory, context recording).

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
