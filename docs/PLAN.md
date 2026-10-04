# Implementation Plan — Visual AI Workbench

Source spec: `docs/VISION.md` (the product README). This plan turns its §22, §23, §24 and §26 into
bounded, verifiable phases. The early sections record the original Phase 0/1 scope; later implemented
milestones and current scope are tracked in `docs/HANDOFF.md` and `docs/CAPABILITIES.md`. Unbuilt work
must not be stubbed with fake UI.

## Ground rules (from VISION §26)

- No mock canvas, no simulated training curves, no fabricated "learned" values. If something is not
  implemented, it is absent or visibly labeled "not implemented".
- The JSON graph is the single authoritative representation. Layout (positions) lives in a separate
  UI document and is excluded from the semantic hash.
- Every numeric value shown in the UI carries provenance (run id, graph revision, step, sample, node).
- Prefer native PyTorch semantics; compare against a handwritten native reference in tests.
- Never claim something works without a test or a command that actually ran.

## Stack decisions (ADR-0001, write it to `docs/adr/0001-local-execution-and-graph-format.md`)

| Area | Choice |
|---|---|
| Python | 3.13, venv at `.venv`, deps pinned in `python/requirements.txt` (torch CPU, fastapi, uvicorn, pydantic v2, pytest, httpx, numpy, pillow) |
| Graph contract | Pydantic v2 models in `python/graph_core/schema.py`; export JSON Schema to `packages/graph-schema/schema.json`; TS types in the editor generated or hand-mirrored from it |
| Execution | PyTorch CPU; graph → `nn.Module` lowering (no `eval`, no string codegen for execution) |
| Code export | Deterministic readable PyTorch source generated from the IR, with node-id comments as the source map |
| Worker | Separate OS process (`multiprocessing` spawn) per run; JSON-line events over a queue → persisted |
| Metadata | SQLite (`.workbench/meta.db`) for projects, runs, events |
| Artifacts | Content-addressed files in `.workbench/artifacts/<sha256>` |
| API | FastAPI in `services/control`; events via SSE with `Last-Event-ID` resume |
| Editor | Vite + React + TypeScript + `@xyflow/react` (React Flow) in `apps/editor`, pnpm |

## Repo layout

```
apps/editor/                 React editor
services/control/            FastAPI app (projects, validate, runs, events, artifacts)
python/graph_core/           schema, registry, shape inference, validation, hashing, lowering, codegen
python/operations/           built-in op definitions (one module per op family)
python/worker/               run process: build model, train loop, events, checkpoints, cancel
python/artifact_store/       content-addressed store + SQLite metadata
examples/                    reference_cnn.project.json, mse_teaching.project.json, tiny image fixture generator
tests/                       pytest
docs/                        VISION.md, PLAN.md, adr/, CAPABILITIES.md (coverage ledger)
```

## Phase 0 — Semantic and execution foundation (VISION Milestone 0)

Deliverables:
1. **Schema** (`schemaVersion`, `graphKind`, `backend`, nodes with `id/type/version/config/stateRef`,
   edges with `id/kind/from{node,port}/to{node,port}`) exactly in the spirit of VISION §16.2.
   Separate `ui` document `{positions: {nodeId: {x,y}}}`.
2. **Operation registry**: each op declares type id, version, typed input/output ports, config
   schema with defaults, `infer_shape(config, input_shapes) -> output_shapes | ValidationError`,
   `param_count(config, input_shapes)`, `lower(config) -> torch callable/module`, and an
   `explain(config, shapes)` returning the equation + parameter formula as structured data.
   Ops: `core.tensor_input`, `pytorch.nn.conv2d` (all fields: in/out channels, kernel, stride,
   padding, dilation, groups, bias, padding_mode), `pytorch.nn.relu`, `pytorch.nn.max_pool2d`,
   `pytorch.nn.adaptive_avg_pool2d` (global avg pool), `pytorch.nn.flatten`, `pytorch.nn.linear`,
   `pytorch.loss.cross_entropy`, `core.sub`, `core.square`, `core.mean`/`core.sum` (reduction with
   explicit divisor semantics), `core.scalar_mul`, `core.add`.
   `in_channels`/`in_features` may be `"infer"` (default) or a user-locked int.
3. **Validator**: stable error codes (e.g. `E_CHANNEL_MISMATCH`, `E_UNKNOWN_OP`, `E_CYCLE`,
   `E_MISSING_INPUT`, `E_PORT_TYPE`), with `{code, severity, nodeId, port, message, fixes[]}`.
   Symbolic batch dim `N`. Unknown ops are preserved on load and block execution (A16).
4. **Semantic hash**: canonical JSON of spec only (not UI) → sha256. Moving nodes doesn't change it (A08).
5. **Lowering** to `nn.Module` that keeps node ids (submodules named by node id) so probes/activations
   map back to nodes. Forward hooks capture activations per node on request.
6. **Codegen** export of readable PyTorch with `# node: conv_1` comments.
7. **Fixed training procedure**: CrossEntropy + SGD/Adam, seeded, on a tiny deterministic synthetic
   image fixture (generated, e.g. 10 classes of colored shapes, 64x64, ~200 images). Record
   loss per step.
8. **Run records + artifacts**: run row, events, checkpoint (state_dict + optimizer + rng + step +
   graph hash) stored content-addressed.

Exit tests (pytest, must all pass):
- A01: reference CNN infers exactly the §8.1 table shapes and **20,042** params; per-layer 896 / 18,496 / 650.
- A02: change conv_1 out_channels 32→64 → conv_2 inferred in_channels updates, new totals correct;
  if conv_2 in_channels is user-locked at 32 → `E_CHANNEL_MISMATCH` on conv_2/input.
- A03: incompatible channels report correct node + port before training.
- A06: graph model vs handwritten `nn.Sequential` reference with copied weights: outputs, loss,
  gradients and one SGD update match (atol 1e-6).
- A07: save→load round trip preserves semantic hash and meaning; unknown op preserved.
- A08: changing only positions leaves hash unchanged.
- A22: MSE graph fixture (targets [1,2,3], preds [1,2.5,2]) → loss 5/12, grads [0, 1/3, -2/3],
  with each intermediate (diff, squared, sum) addressable by node id.
- A34: SGD teaching step w=2, g=0.6, lr=0.1 → 1.94.

## Phase 1 — Visual CNN workbench (VISION Milestone 1, CPU only)

Backend:
- FastAPI endpoints: `GET /api/registry`, `POST /api/validate` (returns shapes, param counts,
  diagnostics per node), `GET/PUT /api/projects/{id}`, `POST /api/runs` (idempotency key),
  `POST /api/runs/{id}/cancel`, `GET /api/runs/{id}`, `GET /api/runs/{id}/events` (SSE, resumable),
  `GET /api/runs`, `GET /api/runs/{id}/checkpoints`, `POST /api/runs/{id}/inspect` (weights of node
  at checkpoint; activations for named sample at node; per-sample loss), `POST /api/infer`,
  `GET /api/projects/{id}/export/pytorch`.
- Image-folder dataset source (`ImageFolder`-style dir), deterministic seeded train/val split
  recorded in run, resize to 64x64 + normalize transform. Ship a script that generates the
  synthetic fixture into `examples/data/shapes10/`.
- Worker: runs in its own process, continues if UI disconnects (A10), cooperative cancel at batch
  boundary with partial checkpoint labelled `partial` (A11), checkpoints per epoch, metrics:
  train loss/step, val loss/acc per epoch, confusion matrix, per-sample val loss.
- Run lifecycle states per VISION §17.1 (subset: queued, preparing, running, cancelling, completed,
  failed, cancelled), illegal transitions rejected.

Editor (`apps/editor`):
- Three-pane layout: block library (left), React Flow canvas (center), inspector (right), run
  panel (bottom). Keyboard: add block via library + Enter, delete, connect via inspector port
  selector (basic A20 support).
- Custom node card per VISION §8.3: op name, backend badge, key config summary, input/output
  shape, param count, red error badge with message from validator. Validate on every edit
  (debounced) → shapes update immediately.
- Inspector: full config form from op config schema, "infer" vs locked toggle for in_channels;
  tabs **Architecture** (schematic channel-stack / filter-slice view, labelled "schematic"),
  **Weights** (actual values from a chosen checkpoint, heatmap grid with shared color scale and
  value range), **Activations** (feature maps for a chosen val sample at chosen checkpoint),
  **Explain** (equation + param formula from `explain`). Every value panel shows provenance line.
  Tabs with no run show "No run yet — nothing recorded" instead of fake data.
- Clickable wire: shows source node/port, shape, and real captured tensor summary (min/max/mean,
  first channels) when a run+sample is selected; otherwise "not recorded".
- Run panel: dataset path, epochs, batch size, lr, optimizer, seed → Run; live loss chart from
  SSE; Cancel; run list with status; pin a baseline and compare two runs (overlaid curves + config
  diff + final metrics); confusion matrix; worst val samples by loss.
- Inference: pick checkpoint + upload/choose image → logits/probabilities.
- Save/load project (spec + ui doc) via API; export PyTorch code view.

Phase 1 exit evidence: from a fresh clone, documented commands start backend and editor; a user
builds the reference CNN (or loads the example), trains on the fixture, sees real feature maps,
changes filter count and sees shape propagation, compares two runs, reloads, runs inference from a
checkpoint. Backend lifecycle tests (cancel, reconnect/resume events, idempotent submit, inspect)
pass. `pnpm build` and `pnpm tsc --noEmit` pass.

## Original Phase 0/1 exclusions
Milestones 2–8 were outside the initial phase. The current handoff/capabilities ledger records subsequent
implemented releases and remaining gaps. Do not add placeholder buttons for unbuilt features.


## Milestone 2b status

Connected data sources, snapshots, studies and sweeps are implemented (ADR 0004, `docs/CAPABILITIES.md`). Next bounded step: ablation builder (replace/bypass/freeze/remove with interface validation),
parallel trials with quotas, early stopping with consumed-budget display, and a research record sheet linked to immutable run/evaluation identities.


## Milestone 6a status

TensorFlow/Keras 3 and JAX portable subsets, compatibility reports, native exports, the generated coverage ledger and the A19 benchmark are implemented (ADR 0010). Domain workflows followed in 6b (ADR 0011).

## Milestone 7 status

Native tabular fitted-pipeline registry, explicit local/staging releases, traceable serving, bounded measured HTTP traffic, session counter isolation, monitoring and rollout/rollback are implemented (ADR 0012). Remote deployment and unsupported model families remain absent. Next planned scope is Milestone 8, as recorded in HANDOFF §5/§8.
