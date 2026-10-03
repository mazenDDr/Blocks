# Capabilities ledger

"Tested" means a pytest in `tests/` exercises it (`pytest -q`). Anything not listed under
Implemented is **not implemented**.

## Implemented and tested (Phase 0)

| Capability | Where | Test |
|---|---|---|
| Graph schema + separate UI doc, unknown fields preserved | `graph_core/schema.py` | `test_graph_foundation.py` (A07) |
| Operation registry: 14 ops (tensor_input, conv2d (all fields), relu, max_pool2d, adaptive_avg_pool2d, flatten, linear, cross_entropy, sub, add, square, sum, mean, scalar_mul) | `operations/` | `test_registry_has_planned_ops`, `test_every_op_declares_explain_and_param_formula` |
| Shape inference with symbolic batch `N`; `"infer"` / locked channels and features | `graph_core/validate.py` | A01, A02, `test_conv_full_config_shape_and_params_match_torch` |
| Reference CNN: VISION 8.1 shapes, 20,042 params (896 / 18,496 / 650) | `examples/reference_cnn.project.json` | A01 |
| Validator with stable codes, node/port paths, fixes | `graph_core/validate.py` | A02, A03, validator tests |
| Codes: E_CHANNEL_MISMATCH, E_FEATURE_MISMATCH, E_GROUPS_INVALID, E_SHAPE_INVALID, E_SHAPE_MISMATCH, E_RANK_MISMATCH, E_CONFIG, E_UNKNOWN_OP, E_UNSUPPORTED_VERSION/BACKEND/GRAPH_KIND/SCHEMA, E_MISSING_INPUT, E_MULTIPLE_INPUTS, E_UNKNOWN_PORT, E_PORT_TYPE, E_DANGLING_EDGE, E_DUPLICATE_ID, E_BAD_ID, E_CYCLE | | `test_graph_foundation.py` (E_GROUPS_INVALID, E_UNSUPPORTED_BACKEND/GRAPH_KIND/SCHEMA are implemented but not individually tested) |
| Semantic hash (layout excluded, order independent) | `graph_core/hashing.py` | A08, `test_hash_tracks_semantics_not_declaration_order` |
| Save/load round trip, unknown op preserved and blocks execution | `graph_core/project_io.py` | A07, A16-style tests in `test_worker.py` |
| Lowering to `nn.Module` named by node id; opt-in activation capture | `graph_core/lower.py` | A06, `test_activation_capture_is_opt_in_and_side_effect_free` |
| Graph vs native PyTorch: outputs, loss, grads, SGD step (atol 1e-6) | | A06 |
| MSE teaching fixture: loss 5/12, grads [0, 1/3, -2/3], intermediates by node id | `examples/mse_teaching.project.json` | A22 |
| SGD teaching step 2 -> 1.94 (gradient from a graph) | | A34 |
| Deterministic PyTorch code export with `# node:` source map | `graph_core/codegen.py` | codegen tests |
| Worker process: seeded CE + SGD/Adam, JSON events, cancel at batch boundary, per-epoch checkpoints, `partial` checkpoint on cancel | `worker/` | `test_worker.py` |
| Content-addressed artifact store + SQLite runs/events/artifacts; run state machine | `artifact_store/` | `test_worker.py` |
| Deterministic image-folder fixture generator (10 classes, 64x64) | `examples/make_shapes10.py` | `test_dataset_generator_is_deterministic` |
| CLI training run | `worker/cli.py` | `test_cli_*` |

## Implemented and tested (Phase 1)

| Capability | Where | Test |
|---|---|---|
| Control API: registry (config JSON Schema + defaults), validate (per-node shapes, params, diagnostics, resolved config, explain), project get/put with separate UI doc, examples | `services/control/app.py` | `test_api.py` |
| Idempotent run submit (required `Idempotency-Key`; same key+request -> same run; different request -> 409) | `app.py`, `artifact_store` | `test_idempotent_submit_returns_same_run` |
| Cancel (DB-recorded, cooperative, partial checkpoint), illegal cancel -> 409 | `app.py`, `worker/` | `test_cancel_running_run_and_illegal_cancel` |
| SSE events resumable via `Last-Event-ID` / `?after=`, no duplicates; run survives client disconnect (A10) | `app.py` | `test_sse_resume_...`, `test_run_continues_when_the_event_client_disconnects` |
| Run status/list/metrics/checkpoints; run stores its exact graph | `app.py`, `worker/process.py` | `test_run_status_split_and_metrics` |
| Image-folder split seeded and recorded in run config, `split` artifact and `run_started`; confusion matrix + per-sample val loss per epoch | `worker/dataset.py`, `worker/train.py` | `test_split_is_seeded_and_deterministic`, `test_inspect_sample_loss_and_confusion_...` |
| Inspect weights / activations (bounded slices, full-tensor stats, provenance); "not recorded" when nothing exists; inspection never trains | `services/control/inspection.py` | `test_inspect_*`, `test_inspecting_never_trains_or_adds_events` |
| Inference from a checkpoint on a val sample or uploaded image | `inspection.py` | `test_infer_from_checkpoint_...` |
| PyTorch export endpoint (blocked graphs -> 422 with diagnostics) | `app.py` | `test_export_pytorch_and_blocked_export` |
| JSON Schema export, kept current by a test | `graph_core/export_schema.py`, `packages/graph-schema/schema.json` | `test_schema_file_is_current` |
| Editor (Vite + React + TypeScript + React Flow): library, canvas with section 8.3 cards (live shapes, param counts, error badges, debounced validate), inspector (schema-driven config forms with infer/locked, Architecture schematic, Weights heatmaps, Activations feature maps, Explain), clickable wires with real tensor summary, run panel (live SSE loss chart, cancel, run list, pinned baseline + compare, confusion matrix, worst samples, inference), save/load, PyTorch export view | `apps/editor` | `pnpm -C apps/editor build` and `tsc --noEmit` (type/build only); behaviour checked by hand in a real browser, no automated UI tests |

## Not implemented

- Gradients tab and captured gradient information (VISION 8.3), saliency, activation distributions over training.
- Editor: undo/redo, copy/paste, grouping, auto-layout, comments, structured outline view, multi-select move as a unit,
  insertion into an existing connection, interactive convolution teaching mode (8.4), partial-weight transfer (8.5.6),
  resource estimates, "resume compatible checkpoint". Automated browser tests.
- Pause/resume, exact resume from a checkpoint, heartbeats and leases; a run whose worker died abnormally stays in its
  last state; cancel cannot terminate a worker that is stuck inside a batch.
- Experiment board features beyond pin + two-run compare (tags, notes, search, sweeps), smoothing, x-axes other than step.
- Auth, multi-user, remote dataset sources; datasets are server-side folders.
- Dependency lock, data manifest, project bundle export (only graph + UI files are saved).
- Milestones 2-8 (connected data, statistics, LangGraph, RL, other backends, serving, code-block editor), A05, A09,
  A12 exact resume, A13 shared parameters, A14-A15, A17-A21, A23-A64 (A04, A10, A11 partially covered above).
- Broadcasting, ops beyond the 14 listed, dtype casts, multi-input/output training graphs, losses other than
  CrossEntropy in the worker, schedulers, gradient clipping/accumulation, mixed precision, GPU.
- Model/parameter policies (initialization, freezing, sharing, regularization), conv data type/device placement.
- Data: only a local image folder; no profiling, no leakage checks.
- Loss and optimizer inspectors, response curves; optimizer state beyond checkpoints.
- Other backends (Keras, JAX, ...), LangGraph/LangChain, RL, statistics, connectors, serving, code blocks,
  registry, sweeps.
