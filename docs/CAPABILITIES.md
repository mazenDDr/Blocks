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

## Implemented and tested (Milestone 2a: tabular graph kind, classical ML, statistics)

See ADR 0003. "Tested" = `pytest -q` (`tests/test_tabular_ops.py`, `tests/test_statistics.py`, `tests/test_tabular_api.py`).

| Capability | Where | Test |
|---|---|---|
| Graph kind `tabular` with typed wires (table, fit_state, model, metrics, distribution, number, tail, test_result); model and tabular ops cannot mix (`E_OP_GRAPH_KIND`, `E_PORT_TYPE`, `E_EDGE_KIND`) | `tabular/`, `graph_core/validate.py` | `test_tensor_and_tabular_wires_are_distinct_kinds`, `test_model_ops_cannot_live_in_tabular_graphs_and_vice_versa` |
| CSV table source (local file, SHA-256 recorded in the run) | `operations/tabular_ops.py` | `test_run_records_data_hash_split_seed_and_fitted_state_artifacts` |
| Table profile (types, missing counts, exact duplicates, summary stats, suspect-column flags; exact over all rows) | `tabular.profile` | `test_profile_counts_missing_duplicates_and_flags` |
| Duplicate policy (key columns, keep first/last/none/report only, duplicate groups, conflicting-label report) | `tabular.duplicates` | `test_duplicates_report_groups_and_conflicting_labels` |
| Typed column selection with conversion-failure report; drop-missing rows | `tabular.select_columns`, `tabular.drop_missing` | `test_typed_selection_reports_and_blocks_conversion_failures`, `test_drop_missing_rows` |
| Train/validation split: seeded, recorded (row-id hashes), random / stratified / grouped | `tabular.train_validation_split` | `test_split_row_estimate_matches_actual`, `test_grouped_split_keeps_groups_together` |
| Fit/apply standardization, one-hot encoding, imputation (mean/median/constant) with fit state owned by the training partition | `tabular.fit_*`, `tabular.apply_transform` | `test_fit_state_is_owned_by_the_training_partition` |
| **A05** leakage check: `E_LEAKAGE_FIT_ON_HELDOUT`, `E_LEAKAGE_FIT_BEFORE_SPLIT` (node/port path + fix), `W_METRICS_ON_TRAIN` | `tabular/core.py::leakage_check` | `test_a05_*`, `test_validate_reports_leakage_with_path_and_fix` |
| Linear regression (OLS) and logistic regression (scikit-learn); coefficients with feature names | `sklearn.*` | `test_exported_predictions_match_a_handwritten_scikit_learn_pipeline`, `test_logistic_regression_and_classification_metrics_match_native` |
| Metrics: MSE/RMSE/MAE/R2, accuracy/log-loss/ROC-AUC/confusion matrix; predictions export (CSV artifact, source row ids) | `sklearn.metrics`, `tabular.predictions_export` | same two tests, `test_predictions_download_matches_the_recorded_table` |
| Regression journey (VISION 11) end to end on a synthetic table, predictions equal a handwritten sklearn pipeline | `examples/tabular_regression.project.json` | `test_exported_predictions_match_a_handwritten_scikit_learn_pipeline` |
| Gamma function and Gamma distribution (shape + scale or rate, location), tail probability, hypothesis-test result | `operations/stats_ops.py` | `test_gamma_*` |
| **A25** Gamma teaching example: upper-tail p = 0.0404276820 (= exp(-5)(1+5)), decision reject at alpha 0.05 | `examples/gamma_teaching.project.json` | `test_a25_gamma_teaching_example_tail_probability` |
| **A26** changing alpha changes the decision and critical value, not the p-value | | `test_a26_changing_alpha_moves_the_decision_not_the_p_value`, `test_alpha_change_via_api_keeps_p_value` |
| **A27** two-group comparison (Welch, Student, paired t; Mann-Whitney U) equals direct SciPy calls on a synthetic fixture; sample unit, pairing/independence, mean-difference CI, Cohen's d / Hedges' g shown | `examples/two_group_comparison.project.json`, `examples/fixtures/` | `test_a27_*` |
| Runs through the worker process with status, events and artifacts (data hash, split seed, fitted-state and model artifacts); failure names node + code | `worker/tabular_run.py` | `test_run_goes_through_the_worker_with_status_events_and_provenance`, `test_failed_node_is_reported_with_code_and_later_nodes_stay_absent` |
| API: validate/registry/examples for tabular graphs, run summary, bounded table preview with provenance, profile, fitted state, coefficients, metrics, test result, distribution, tail, CSV download | `services/control/` | `tests/test_tabular_api.py` |
| Deterministic synthetic fixtures (housing, churn, enzyme two-group, enzyme paired), labelled synthetic | `examples/make_tabular_fixtures.py` | `test_fixture_generator_is_deterministic_and_matches_the_committed_files` |
| Editor: project picker (saved + examples by graph kind), new model / tabular graph, tabular library groups, schema node cards (columns, types, row count, partition), table preview, profile, fitted-state, coefficient, metrics, partition, source, step views, density-with-shaded-tail statistics view, problems list with fixes, run panel (run, cancel, run record, predictions download) | `apps/editor` | `pnpm -C apps/editor build`, `tsc --noEmit`; behaviour checked in real Chrome with a throwaway puppeteer-core script (not part of the repo) |

## Implemented and tested (Milestone 2b: connected data, snapshots, studies)

See ADR 0004. Tests run against REAL local engines: PostgreSQL 16 (`pgserver`), an S3 API mock (`moto` server; it does not enforce IAM) and a Git+DVC repository with a local DVC remote
(`python/connectors/localtest.py`; the data is SYNTHETIC, `python/connectors/synthetic.py`). Tests: `tests/test_connectors.py`, `tests/test_snapshots_and_joins.py`, `tests/test_studies.py`, `tests/test_journey_script.py`.

| Capability | Where | Test |
|---|---|---|
| **A17** Connection registry (SQLite) with secrets by reference (env var or JSON secrets file outside the project/workbench); literal secrets rejected; values never in API responses, events, artifacts, DB files, bundle export or error text | `connectors/registry.py`, `secrets.py`, `connections_api.py` | `test_a17_secrets_are_references_never_values`, `test_connection_crud_validation` |
| Test connection (server version, latency, read-only enforcement, capabilities); discovery: PostgreSQL schemas/tables/columns with per-table/column SELECT permission, S3 prefixes/objects/versions (paged), DVC revisions/tree with md5 | `connectors/postgres.py`, `s3.py`, `dvc.py` | `test_connection_crud_validation`, `test_a42_dvc_revision_*`, `test_a41_s3_versioned_*` |
| **A39** Visual PostgreSQL query (table, columns, typed filters, joins, group-by/aggregates, ordering, limit) compiled to parameterized SQL (quoted identifiers, bound typed values); result and schema equal the hand-written native query; injection inert | `connectors/pg_query.py` | `test_a39_*`, `test_values_are_bound_not_interpolated_injection_is_inert`, `test_typed_comparison_mismatch_*`, `test_aggregates_group_order_and_limit` |
| In-database join report (key cardinality, duplicates, NULL keys, unmatched rows both ways) | `postgres.join_reports` | `test_in_database_join_reports_cardinality_and_unmatched_records` |
| Raw SQL mode: read-only transaction, statement timeout, typed `%(name)s` parameters, row bound | `postgres.run_query` | `test_raw_sql_is_read_only_bounded_and_parameterized` |
| **A40** Bounded previews (rows capped at the source, timeout, ranged S3 reads, paged listings); auth / permission / network / not-found / timeout / unresolved-secret failures are distinct recoverable codes with hints and leave the registry intact | `connectors/errors.py` | `test_a40_*`, `test_s3_error_classifier_*` (IAM-style S3 errors only on synthesized botocore errors) |
| **A41** Snapshot per run: PostgreSQL (query, params, server version, transaction snapshot token, result hash, materialized extract), S3 (bucket/key/versionId/ETag/hash; unversioned objects flagged "reproducibility limited" and materialized), listings; repeat from a pinned snapshot works with the connection removed or the source changed; drift check against the live source | `connectors/snapshots.py`, `operations/connector_ops.py` | `test_a41_*` |
| **A42** DVC dataset: repo, requested revision, resolved commit, DVC md5, content SHA-256; data materialized from the DVC remote; pinned repeat after the branch moved; drift check | `connectors/dvc.py` | `test_a42_*`, `test_dvc_snapshot_drift_check` |
| **A43** Cross-source join (database rows x S3 CSV x S3 object metadata): cardinality, duplicate/NULL keys, unmatched rows with samples, row multiplication, where it executed, data moved, source lineage; NULL keys never match; `expect` cardinality; key type mismatch is a static error | `tabular.join` | `test_a43_*`, `test_join_key_type_mismatch_is_a_static_error` |
| **A47** Repeat identities: seed, fold (k-fold split node), attempt are separate fields on trials, runs and events; failed vs retried vs repeated trials distinguished; metrics carry step/aggregation/partition/n semantics; mean and sample std across repeats | `studies/` | `test_a47_*`, `test_failed_retried_and_repeated_*`, `test_folds_are_a_repeat_dimension_*` |
| **A48** Study with objective + direction, baseline (group mean or pinned run), bounded grid/random plans with explicit `max_trials` (never truncated), concurrency 1, validated variants (invalid ones kept, not run), node-config and run-config targets for tabular and model graphs, results table with deltas, attribution warning, config/graph diff between runs, cancel | `studies/`, `studies_api.py` | `test_plan_*`, `test_a48_*`, `test_model_graph_sweep_*`, `test_cancel_*` |
| Connected-data journey: PostgreSQL assays + S3 spectra + S3 image metadata joined, grouped split by specimen, regression, ablation sweep on pinned sources unaffected by database changes | `connectors/journey.py`, `examples/connected_journey.py` | `test_connected_journey_ablation_*`, `test_connected_journey_script_*` |
| Project bundle export lists required connections (settings + secret field names) and pinned snapshots; no secret values or references | `GET /api/projects/{id}/export/bundle` | `test_a17_*` |
| Editor: Data workspace (add connection with secret reference, test, browse tables / objects+versions / revisions+tree, bounded preview, add as node), connector nodes with visual query builder + read-only compiled SQL + raw SQL + pin, Source & snapshot and Join report tabs, Experiments board (study list, new sweep form with plan check, groups and trials tables, retry, baseline pin, two-trial compare) | `apps/editor` | `pnpm -C apps/editor build`, `tsc --noEmit`; behaviour checked in real Chrome with a throwaway puppeteer-core script (not in the repo) |

## Implemented and tested (Milestone 3: research-level visual composition)

| Capability | Where | Test |
|---|---|---|
| Reusable modules (typed named ports, arguments), instances clone parameters by default (stated), nested identity paths (`res1/conv_a`, `block/ra/conv_a`), module versions pinned (E_MODULE_VERSION, no silent upgrade), recursion/cycle rejected | `graph_core/composite.py`, `build.py` | `test_composites.py` |
| A24: editing a module's internal op changes hash, equation (composed explain), exported source and runtime together; no cache anywhere | `services/control/views.py` | `test_a24_*` (core and HTTP) |
| Tensor primitives: constant, arange, add/sub/mul/div (declared broadcasting), matmul, reshape/permute (axis names), concat, slice, exp/log/sqrt/abs/neg/tanh/sigmoid/gelu, clamp, softmax/log_softmax, where, compare, logical, any/all, cast, layernorm, batchnorm and dropout (explicit train/eval mode), embedding, dense; each with infer_shape, explain, codegen | `operations/tensor_ops.py` | `test_tensor_ops.py` |
| A13: explicit sharing (`share` on an instance, `sharedWith` on a node): identical tensors, params counted once, gradients accumulate from all call sites, mismatches rejected | `composite.py`, `lower.py` | `test_a13_*` |
| A23: masked weighted loss from primitives packaged as a module with declared reduction (element count / valid count / weight sum) and empty-denominator policy; values and gradients equal hand-written torch; usable as the training loss | `build.masked_weighted_loss`, `training/losses.py` | `test_custom_loss.py`, `test_training_with_a_visual_masked_weighted_loss_module...` |
| Editable training procedure: ordered stages with order rules (E_PROC_ORDER/E_PROC_STAGE), accumulation (equals a large-batch step), clip norm/value with the decision recorded, SGD/momentum/Nesterov/Adam/AdamW/AMSGrad, schedulers with explicit timing, validation cadence, eval mode and no-grad separate, checkpoint cadence/retention/best, early stopping, frozen nodes | `training/` | `test_training_procedure.py` |
| Optimizer inspection: real stored moments and step counts; watched element with a hand-written mirror of the update (diff shown) | `training/optim.py`, `/api/runs/{id}/optimizer` | `test_optimizers_have_real_torch_semantics...`, `test_optimizer_state_inspector...` |
| A12: resume from a checkpoint continues bit for bit (CPU, deterministic mode) incl. dropout RNG, schedulers, accumulation; mismatching graph/procedure refused | `training/trainer.py` | `test_a12_resume_*` |
| A37: bounded repeat (static count, loop-carried state with type check, `fixed_count` only, tied or cloned parameters) and typed select (scalar bool predicate, identical branch signatures, both branches evaluated, gradient only through the selected one) | `composite.py` | `test_a37_*` |
| Debugger: probes (A35: outputs, gradients, parameters, RNG, mode, order unchanged; overhead measured), conditional breakpoints labelled execution-changing, scrubber over recorded steps, wires with real captured values (A28), `not recorded` + capture-and-rerun verified against the original (A29), gradients from a captured step, sandbox interventions (A36: original fingerprint unchanged, changes/held-fixed/outcome stored as a new run) | `debugger/`, `m3_api.py` | `test_debugger.py`, `test_m3_api.py` |
| A46: probe/histogram/timer/assert inside modules keep nested ids, respect capture scope (nodes, steps, samples, budget); observation-only blocks bit-identical, assert labelled and off by default | `graph_core/diagnostics.py`, `diag_ops.py` | `test_diagnostics.py` |
| A21/A33: transformer encoder from primitives (MHA with Q/K/V, scaled scores, padding mask, softmax, Wo, residual + post-LN, GELU MLP, shared layers); matches a hand-written reference and torch SDPA; trains on a labelled-synthetic task; attention-head inspector with real projections, scores, mask, weights, context and output, unavailable internals labelled | `build.py`, `debugger/attention.py` | `test_transformer.py` |
| A45: code blocks: typed interface (inputs, outputs, config, state, effects, randomness, differentiability), generated template, fixtures with type/shape/value/determinism/gradient checks, isolated subprocess (scrubbed env, CPU/file/wall limits), stack frames mapped to source lines, source hash and pins in the semantic hash, result cache keyed by identity, differentiable only if torch-connected, versioned immutable library | `codeblocks/`, `operations/code_ops.py` | `test_codeblocks.py` |
| Editor: composite expand/collapse in place, open module with breadcrumb, module interface/test/versions, shared marker, training procedure view, debugger (scrubber, wire/gradient/breakpoint/probe/sandbox), attention inspector, CodeMirror 6 code block editor | `apps/editor/src` | `pnpm -C apps/editor build`, `tsc --noEmit`; real Chrome check with a throwaway puppeteer-core script (27 checks, not in the repo) |
| Examples (generated by `examples/make_m3_examples.py`): residual_cnn, shared_encoder, masked_loss, transformer_sequence, code_block_demo, control_flow | `examples/` | `test_shipped_examples_*` |

Milestone 3 gaps: code-block dependencies are checked, not installed; the memory limit is enforced only where the OS allows RLIMIT_AS (not on macOS: reported as "not enforced"); the effect guards are best-effort monkey-patches, the isolation is the separate process, not a hostile-code boundary;
no breakpoints or variable inspection inside code blocks; PyTorch export of graphs containing code blocks is refused; select evaluates both branches (no lazy branch); repeat has only static `fixed_count` termination;
mixed precision, parameter groups, separate optimizers, data-loader workers and sampling policies are not in the procedure; the debugger records the first micro-batch of chosen steps only; sandbox cannot edit graph structure, only settings, node values and parameters;
accumulation of unequal micro-batches or with batch-norm is not equal to a large batch; training procedures run on CPU only; the attention inspector understands the documented inner-node contract of `multi_head_attention` (no KV cache exists);
module expand-in-place is available at project level only (inside a module use Open); the composite equation is a list of the members' own equations, not a symbolic simplification.

## Implemented and tested (Milestone 4: language-model and agent workflows)

"Fixture" below means a scripted test model (`provider: fixture`), labelled FIXTURE everywhere and never shown as a real response. "Live" means the local Ollama server (`pytest -m live`).
Design record: `docs/adr/0008-agent-graph-kind-memory-context.md`.

| Capability | Where | Test |
|---|---|---|
| `agent` graph kind compiled to a native LangGraph `StateGraph` (typed state `TypedDict` with `Annotated` reducers, `add_edge`, `add_conditional_edges`, multi-source join edges, `interrupt`, `Command(resume)`, recursion limit); equals a handwritten native reference | `agent/runtime.py`, `agent/validate.py` | `test_agent_graph.py` (`test_compiles_to_a_native_state_graph...`, `test_matches_a_handwritten_native_langgraph_reference`) |
| Typed state schema; per-field reducer chosen from replace / append / append_unique / add / max / min / merge / keep_last_n; reducer-vs-type check; turn vs thread field scope; concurrent writes to a `replace` field rejected before running | `agent/spec.py` | `test_reducer_*`, `test_parallel_branches_*`, `test_concurrent_writes_*`, `test_turn_scoped_fields_reset_*` |
| Conditional routes with typed predicates built from pickers (all/any/not, 14 operators, `len`, field-vs-field), first match wins, operand values recorded; no pasted Python | `agent/spec.py`, `agent/validate.py` | `test_predicate_evaluation`, `test_typed_predicates_are_checked_at_validation`, `test_a14_*` |
| A14: bounded cycles: route exit, step limit (`recursion_limit`, recorded `budget_exhausted`), budgets (model calls, tokens, seconds, tool calls); validation codes for dead ends, unreachable nodes, missing START, loops with no conditional exit | `agent/runtime.py` | `test_a14_cycle_routes_state_updates_and_exit`, `test_a14_terminates_at_the_configured_step_limit`, `test_budget_termination_*`, `test_validation_codes_for_broken_graphs` |
| Blocks: set state; prompt/messages (roles, templates, typed variables, ordering, memory/chunk/tool sources, rendered preview); chat model (Ollama, Anthropic, fixture; temperature, max tokens, seed where supported, timeout, think; unsupported settings recorded as ignored with the reason); structured output (visual schema, validation, retry with feedback, error routing); embeddings; retriever (k, score threshold); citation check; tools; human interrupt; memory select; memory write | `agent/blocks.py`, `agent/models.py` | `test_agent_blocks.py`, `test_agent_retrieval.py` |
| Provider paths: Ollama (live-tested, real usage and latency), Anthropic adapter (`claude-sonnet-5-5`, key by secret reference; constructed and parameter-mapped in tests, live call skipped without a key), fixture (control flow only) | `agent/models.py` | `test_anthropic_adapter_*`, `test_anthropic_live_call` (skipped: no key), `test_agent_live.py` |
| Document index: local text loader, splitter (recursive / paragraph / fixed, offsets back to the file), embeddings (Ollama `nomic-embed-text`, or `local_hash`: deterministic lexical hashing, NOT semantic), FAISS flat inner-product store persisted under the workbench, reuse of unchanged index and chunk embeddings | `agent/index.py` | `test_loader_splitter_*`, `test_index_is_persisted_reused_*`, `test_retrieval_scores_*`, live embeddings test |
| Tools with bounded capabilities: calculator (arithmetic AST only), `read_text_file` (declared directory; `..`/symlink escapes refused), `write_note` (external `file_write`: approval interrupt required, cannot be disabled, runs once) | `agent/tools.py` | `test_calculator_*`, `test_file_tools_*`, `test_tool_declarations_*` |
| Persistent checkpoints (SQLite saver), thread ids, interrupt -> `paused` run, resume in a new worker process; double resume rejected; thread busy while paused | `agent/runtime.py`, `worker/agent_run.py`, `worker/process.py` | `test_agent_persistence.py` |
| A15: paused agent survives a real SIGKILL + restart of the control service (twice), resumes with the edited value, and the protected effect happens exactly once; replay/time-travel re-run does not repeat it (effect ledger); a crash mid-effect is reported, not repeated | `agent/memory.py` (ledger) | `test_a15_paused_agent_survives_service_restarts...`, `test_replaying_a_protected_node_...`, `test_claiming_an_effect_is_atomic...` |
| Memory stores: short-term = the thread's messages in its checkpoint; long-term records (namespace, scope, kind, metadata, importance, version, generated flag, soft delete) in `memory.db`; every write audited (node, run, evidence, old/new, validation, scope); proposals can wait for accept/reject/edit; scopes do not leak between users or threads | `agent/memory.py`, `agent/blocks.py` | `test_agent_memory.py` |
| Memory policy blocks (retrieve, filter, rank with recency/relevance/importance weights, dedupe, token budget, summarize) with a per-record decision (kept / excluded / replaced), stage and specific reason; pure function of its inputs | `agent/policy.py` | `test_*_stage*`, `test_policy_is_a_pure_function...` |
| A31: context inspector: for every model call the exact message list sent, segments linked to prompt template / memory record id / retrieved chunk / tool result, provider-reported tokens beside the labelled estimate, model limit when reported, excluded items with stage and reason (disjoint from the included set), observability boundary | `agent/runtime.py:build_context` | `test_memory_debugging_journey_end_to_end`, `test_prompt_segments_*`, `test_trace_shows_passages_*` |
| A30/A32: memory debugging journey: find the omitted record, trace it to the excluding stage (rank cutoff; budget variant), edit the policy visually, isolated preview (no model call, no store/state/run change; unedited preview equals the record; predicts the rerun), rerun | `agent/preview.py` | `test_agent_journey.py`; live: `test_live_memory_debugging_journey` |
| Section 12.3 retrieval with bounded revision: transitions, query revision, N revisions then unresolved, citation validity, grader retries (fixture); real model responses and real embeddings (live) | `agent/samples.py`, `examples/agent_retrieval_revision.*` | `test_agent_retrieval.py`; live: `test_live_retrieval_with_bounded_revision_example`, `test_live_unanswerable_question_...` |
| Trace: per-step reads and state diff (reducer, before/after), route taken with predicate values, model calls (latency, provider usage or "not reported", cost unknown/0 local), tool calls, retrievals, memory events, replay markers, totals and budgets; events carry run id, graph hash, sequence | `agent/trace.py`, `/api/agent/runs/{id}/trace` | `test_run_via_api_trace_events_summary_and_idempotency` |
| API: `/api/agent/catalog`, `/api/runs` (agent graphs), `/api/runs/{id}/resume`, `/rerun`, `/api/agent/threads[/{id}/state|history|fork]`, `/api/agent/runs/{id}/trace|model-calls|final-state`, `/api/agent/memory/*` (records CRUD, history, applications, trace, seed), `/api/agent/policy/preview`, `/api/agent/indexes/*`, `/api/agent/effects` | `services/control/agent_api.py` | `test_agent_api.py` |
| Editor (agent graphs): canvas with node cards per block (reads/writes, effects, FIXTURE badge, run marks, paused marker), block library, per-block visual forms, state schema panel with reducer selection, route predicate builder, loop limits and budgets, joins, run panel with thread selector and input form, interrupt approve/reject/edit forms, trace timeline, context inspector (included vs excluded, source links, search), memory browser, policy editor, isolated preview diff, indexes tab | `apps/editor/src/components/agent/` | `pnpm -C apps/editor build` + `tsc --noEmit`; browser journeys were driven with puppeteer-core in headless Chrome (scripts are not in the repo: no automated browser tests) |
| Examples (generated by `examples/make_m4_examples.py`): agent_retrieval_revision, agent_memory_debugging, agent_approval_tools, agent_bounded_loop, agent_parallel_join; SYNTHETIC documents in `examples/agent_docs/` | `examples/` | `test_example_projects_are_listed_open_and_valid` |

Milestone 4 gaps: tools are explicit nodes with bounded capabilities, a model does not choose tools; no subgraph nodes, no LangGraph cross-thread `Store` (long-term memory is `memory.db`); no streaming of model tokens; token budgets in memory policies use the character estimate (no tokenizer is bundled);
no pricing table, so cost is unknown (0 for local inference); there is no side-by-side comparison view of chunking strategies / embedding models (indexes can be built with different specs); no retention or expiry sweeps for checkpoints and records (`expires_at` is honoured when retrieving);
document loaders read local UTF-8 text files only; the vector store is FAISS flat (no metadata filter inside the index); the Anthropic adapter has never made a live call here (no key); `local_hash` embeddings are lexical, not semantic; the preview recomputes policy -> prompt for the selected call
(other nodes between the selection and the prompt are not re-run); forking copies a checkpoint to a new thread and starts the next run from the edited state; thread-level restore of a historical checkpoint into the same thread is not offered; a run on a very large state records previews of large values (hash + first 300 characters).

## Remaining gaps (later milestone sections refine these bounds)

- Connectors/studies (2b gaps): other databases, warehouses, GCS/Azure, document/vector stores, scientific formats, repositories and model registries; a private-network connector agent; cost/transfer-size estimates; incremental reads; writes/destinations;
  schema-aware SQL completion, query history, explain/cost; windows in the query builder; DVC private-remote authentication helper; parallel trials, quotas, pruning/early stopping, conditional search spaces, an ablation builder (replace/bypass/freeze/remove),
  parallel-coordinate and parameter-importance views, smoothing/x-axis choices for metric curves, richer research record sheets/report snapshots and tags/notes/search on runs. Native local MLflow/W&B offline exports and linked conclusions now exist; see Milestone 8.
- Tabular: general estimator-level cross-validation beyond the supported study fold trials, held-out test partition and its "final evaluation" designation, sample weights, target/feature roles on the source (done on the estimator),
  Parquet/JSON sources, near-duplicate detection, stratified/temporal splits beyond stratify/group, feature construction, trees/forests/boosting/SVM/kNN (native clustering/decomposition is implemented in Milestone 5),
  broader final-test comparison workflows beyond the supported pinned baseline/variant evidence, regression plane / projection views, streaming or sampled profiles for huge tables.
- Statistics: descriptive-statistics block, parameter estimation, interval-estimation block, bootstrap/permutation tests, power analysis, multiple-comparison procedures, other distributions,
  special-function curves with singularities, generated samples, a two-sided tail rule on the generic tail block, the interactive "Explore" slider for the observed statistic, visual statistical programming (custom statistics from primitives).
- Editor: layout is a left-to-right row layout, large graphs are small at the initial fit; no automated browser tests in the repo.

- Gradients tab and captured gradient information (VISION 8.3), saliency, activation distributions over training.
- Editor: undo/redo, copy/paste, grouping, auto-layout, comments, structured outline view, multi-select move as a unit,
  insertion into an existing connection, interactive convolution teaching mode (8.4), partial-weight transfer (8.5.6),
  resource estimates, "resume compatible checkpoint". Automated browser tests.
- General pause/heartbeats/leases and orphaned-worker recovery; supported model checkpoint resume and domain completed-epoch child continuation exist. Cancellation cannot forcibly terminate a native call stuck inside a batch.
- Experiment board: tags, notes, search/filter of runs, smoothing, x-axes other than step (model graphs keep the pin + two-run compare; tabular/model sweeps have the Experiments board above).
- Auth, multi-user; image-folder datasets are server-side folders (tabular graphs can read connected sources).
- Automatic installation/migration of exported environments and portable trained-artifact import. Inert graph/UI packages with exact operation requirements and optional CSV snapshots are implemented in Milestone 8; native domain checkpoint and manifest downloads are documented below.
- Behaviors outside the documented bounds in the A01–A64 acceptance checklist; later Milestone 7/8 and domain checkpoint sections are authoritative for their implemented scope.
- Broader multi-input/output training, optimizer/scheduler/precision/device workflows outside the supported PyTorch training procedure. Native tensor broadcasting/casts and composed custom losses exist; GPU and mixed precision remain unavailable.
- Broader parameter initialization/regularization and device-placement policies beyond implemented sharing/freezing/procedure controls.
- General domain dataset importers beyond the existing labelled fixture formats. Image folders, CSV and connected PostgreSQL/S3/DVC sources are already supported.
- Loss and optimizer inspectors, response curves; optimizer state beyond checkpoints.
- Serving/registry beyond the bounded native tabular adapter below; Keras/JAX worker training runs (see Milestone 6a: only forward, loss, gradients and one SGD step exist there).

## Milestone 5: reinforcement and unsupervised research (A51-A55)

| Capability | Where | Test |
|---|---|---|
| `rl` graph kind, typed wires, six nodes, compatibility codes decided from real Gymnasium spaces (CartPole, GridWorld verified; MountainCar, Acrobot, Pendulum, FrozenLake space-contract only) | `python/rl/`, ADR 0009 | `test_rl_api.py::test_incompatible_*`, `test_rl_core.py` |
| Native torch DQN (target network, replay, epsilon schedule, Huber/MSE), update equal to a handwritten step | `rl/dqn.py` | `test_update_equals_a_handwritten_reference_step` |
| **A52** bootstrap fixture 2.98 / 1 against the learner's real target function; truncation does not cut the bootstrap | `rl/dqn.py:bootstrap_target` | `test_a52_*` |
| **A53** vector env autoreset (NextStep and SameStep, Gymnasium 1.3.0): true final next-obs stored, reset step excluded, episode-start flags | `rl/collector.py` | `test_a53_*` |
| **A51** bounded transition-to-update trace: transition, buffer slot/eviction, minibatches, TD target/loss/gradient, policy version, recomputed in tests | `rl/trace.py`, `/api/rl/runs/{id}/transitions/{tid}` | `test_a51_*` |
| Visual grid-world builder (size, walls, start, goal), separate reward components, schematic + real rendered frames | `rl/gridworld.py`, EnvPanel | `test_gridworld_*`, `test_grid_builder_preview_*` |
| **A54** reward/policy variants as study trials; multi-seed evaluation on separate envs/seeds, mean + t-CI across runs, per-component, length, termination vs truncation | `rl/compare.py`, `studies/` | `test_a54_*` |
| RL workspace: environment/spaces/reward/learner equation, live unsmoothed curves, rollout scrubber, buffer browser, trace view, eval report, variants | `components/rl/` | Chrome check (README) |
| **A55** k-means, GMM, DBSCAN, PCA, t-SNE (non-metric), diagnostics (inertia/elbow, silhouette, Davies-Bouldin, BIC/AIC, explained variance), stability (ARI seeds/resamples), external metrics only with labels | `operations/unsup_ops.py` | `tests/test_unsup.py` |

Known gaps: PPO and other algorithms, continuous actions, async vector envs, exact mid-episode resume, UMAP, offline/multi-agent RL, frames only from environment 0, prioritized replay, recurrent policies
(the reset contract is tested with a stateful stand-in, not a recurrent network), unsupervised: no self-supervised/contrastive workflows, no density-based anomaly block beyond the GMM log-density column.

## Milestone 6a: additional backends, compatibility reports, coverage ledger (A06, A18, A19)

Design record: `docs/adr/0010-additional-backends-compatibility-and-coverage.md`. The generated, per-operation ledger is `docs/COVERAGE.md` (`python -m backends.coverage --write`; also `GET /api/coverage` and the editor's Coverage tab).
Pinned and installed here (macOS arm64, CPython 3.13.12): tensorflow 2.21.0, keras 3.15.1 (TensorFlow backend), jax 0.11.2 / jaxlib 0.11.2; **all three backends are available**. Vision, NLP and speech domain workflows are documented in the Milestone 6b section below.

| Capability | Where | Test |
|---|---|---|
| Portable subset on TensorFlow/Keras 3 and JAX: tensor input, conv2d (stride, explicit/same/valid padding, dilation, groups, bias), relu, max_pool2d, adaptive (global) average pooling, flatten, linear, cross-entropy (mean/sum/none), sub, add, square, scalar_mul, sum, mean | `python/backends/` | `test_backend_conformance.py` (42 single-op cases x 3 backends vs handwritten native PyTorch: outputs, input and parameter gradients, one SGD update) |
| Reference CNN (VISION 8.1) and the A22 MSE fixture and A34 step on every backend with identical copied weights: shapes, logits, loss, gradients, one SGD update match PyTorch within the declared tolerances; residual CNN (expanded modules), conv-flatten-linear (layout), shared parameters | `backends/workloads.py` | `test_backend_workloads.py` |
| Explicit layout handling (NCHW graph vs NHWC Keras), weight layout (OIHW <-> HWIO, `(out,in)` <-> `(in,out)`), padding semantics, declared initialization differences; every conversion listed per node | `keras_spec.py`, `jax_spec.py` | `test_keras_layout_and_weight_layout_conversions_are_listed_per_node`, `test_flatten_after_spatial_maps_keeps_graph_order`, `test_init_differences_are_declared_and_real` |
| Compatibility report per graph and backend (supported / converted / unsupported with stable code and reason; facets: layout, dtype, padding, randomness, gradients, serialization, hardware, init; sharing groups; availability) | `backends/compat.py`, `POST /api/backends/compat` | `test_backend_api.py`, `test_backend_workloads.py` |
| **A18** selecting a backend with an unsupported feature fails before execution (`E_BACKEND_UNSUPPORTED_PADDING_MODE`, `_DILATION`, `_CONFIG`, `_DTYPE`, `_OP`, `E_BACKEND_OP`), in `validate`, `compile_graph`, `export_code`, `lower_graph`; the graph is unchanged | `graph_core/validate.py` | `test_a18_*`, `test_unsupported_*`, conformance refusal cases |
| Parameter sharing preserved on every backend (one set of tensors, gradients accumulated, one update moves both call sites) or the graph is refused | `plan.py`, runtimes | `test_shared_parameters_stay_shared_on_every_backend`, `test_exports_keep_the_sharing_visible` |
| Explicit backend-specific nodes: `keras.layers.separable_conv2d`, `jax.lax.cumsum`; each rejected on other backends (`E_BACKEND_OP`) | `operations/backend_ops.py` | `test_backend_specific_nodes.py` |
| Native code export (Keras `Model` class, JAX `init_params`/`apply`), deterministic, `# node:` source map, generated without the framework installed; executed in a separate Python process and matched | `keras_spec.py`, `jax_spec.py`, `POST /api/export` | `test_backend_export.py` |
| Coverage ledger generated from code; stale file fails the suite; API + editor view | `backends/coverage.py`, `docs/COVERAGE.md` | `test_coverage_ledger.py`, `test_coverage_endpoint_serves_the_generated_ledger` |
| **A19** benchmark: graph-lowered vs handwritten native per backend, forward and train step, cold vs warm, 5 fresh-process repetitions, environment captured, raw JSON committed | `benchmarks/` | `benchmarks/results/m6a_raw.json`, `m6a_summary.md` (a measurement, not a test). Measured on this M4 Pro laptop (CPU, batch 16, 5 fresh processes x 50 calls): lowered/native median ratios for the reference CNN **train step** 0.994 (PyTorch), 1.044 (Keras, tf.function; it also pays the NCHW->NHWC transposes native Keras avoids), 1.011 (JAX, jit); **forward** 1.008 / 1.077 / 1.012; spreads across processes 2-11%. The 3-element MSE graph costs +0.005 / +0.030 / +0.000 ms forward. Cold start, RSS and the method are in the summary; no GPU, no larger models, threads requested but the JAX setting unverified |
| Editor: backend selector, per-node compatibility badges (module instances show the worst child), Backend tab (report, facets, conversions, export), Export code per backend, Coverage tab, training panel labelled PyTorch-only | `apps/editor/src` | `pnpm -C apps/editor build`, `tsc --noEmit`; real Chrome check with a throwaway puppeteer-core script (not in the repo) |

Declared tolerances (`python/backends/tolerances.py`; |a - b| <= atol + rtol |b|, against PyTorch CPU with copied weights) and the worst difference measured on the reference CNN (batch 8, 5 seeds, `Keras` and `JAX` alike):

| Quantity | float32 declared (atol, rtol) Keras / JAX | measured worst | float64 declared (PyTorch / Keras) |
|---|---|---|---|
| forward (logits and every activation) | 2e-5, 1e-4 | 1.8e-6 | 1e-12, 1e-10 / 1e-10, 1e-9 |
| loss | 2e-5, 1e-4 | 2.4e-7 | same as forward |
| gradients (parameters, inputs) | 5e-5, 1e-3 | 1.4e-5 (relative up to 2.6e-2 on near-zero elements) | same as forward |
| one SGD update (lr 0.1) | 5e-6, 1e-4 | 1.5e-8 | same as forward |

PyTorch graph-lowered vs handwritten PyTorch: 1e-6 (A06 unchanged). JAX has no float64 entry: it is refused.

Known gaps (6a): only the 14-op portable subset (`tensor.*`, `diag.*`, `code.block`, transformer blocks are PyTorch-only and reported unsupported elsewhere); Keras zero padding only, no max-pool padding/dilation/ceil_mode, no dilation with stride; JAX no float64,
no max-pool dilation, adaptive pooling only when the size divides; no worker training runs, checkpoints or run history on Keras/JAX (forward, loss, gradients and one SGD step only); no cross-backend checkpoint conversion or optimizer-state transfer;
no GPU/accelerator support or measurement; multi-backend pipelines joined by artifact contracts are not implemented; max-pool gradient tie-breaking may differ between frameworks (continuous random fixtures have no ties); probes, the debugger,
recorded reruns and attention inspection are PyTorch-only (activation capture exists on all three); the coverage ledger's "tests" column counts conformance cases/workloads and test files that mention an op id literally (a static scan, not a mutation of coverage);
architecture templates and pretrained weights are tracked separately (none provided); the benchmark covers one CNN and a 3-element MSE graph on CPU only, with threads requested but the JAX thread setting unverified.

## Milestone 6b: vision, NLP and speech domain workflows (A56–A58)

Native libraries available here: torchvision 0.29.1, torchaudio 2.11.0, tokenizers 0.23.2, seqeval 1.2.2, torchmetrics 1.9.0, pycocotools 2.0.11, jiwer 4.0.0. See ADR 0011. All training data is explicitly labelled SYNTHETIC; tone recognition is not a real-speech benchmark.

| Capability | Where | Evidence |
|---|---|---|
| Separate `domain` graph kind, 11 registered operations, distinct image/token/audio/feature/report wire types; stable format/rate/policy/ignore-index errors before execution | `domain/`, `graph_core/validate.py` | `test_domain_api.py` |
| A56: declared xyxy/xywh/cxcywh, pixel/normalized coordinates; instance masks, keypoints with COCO visibility, metadata and flip pairs | `vision/contract.py` | `test_domain_vision.py` |
| Annotation-preserving resize, crop, horizontal/vertical flip, rotate-90, pad, affine; seeded random flip with actual decision; clipping/removal/refit policy removes annotation rows together | `vision/transforms.py` | native tv_tensors and exact image/mask/keypoint geometry comparisons in `test_domain_vision.py` |
| Tiny native PyTorch FCN segmentation, aggregate IoU/Dice/pixel accuracy; mAP of derived connected-component detections via torchmetrics/pycocotools | `vision/segment.py` | hand calculations/native reference metrics and actual learning in `test_domain_vision.py` |
| A58: offline WordPiece fitted on train text, original character spans/subwords/word IDs, first/all-subword labels, -100 ignore labels, truncation, padding/attention masks and IOB2 validation | `nlp/subword.py`, `domain/nlp_ops.py` | `test_domain_nlp.py`, including truncated-word/SEP boundary |
| Packed BiGRU token classifier, padding-invariant logits, word-span default and strict IOB2 evaluation with seqeval agreement | `nlp/model.py`, `nlp/iob.py` | `test_domain_nlp.py` (random valid/invalid tag sequences) |
| A57: sample rate, channels, duration and sample↔time; exactly 32,000 teaching samples/channel; native resampling | `speech/audio.py` | `test_domain_speech.py`, including stereo accounting and nondivisible resample length |
| Declared Hann/Hamming window, FFT/hop/centering/padding, torchaudio STFT/mel/log features, odd/even frame formulas, lengths/masks and downsampling lengths | `speech/features.py` | native torch.stft / torchaudio comparisons and padding invariance in `test_domain_speech.py` |
| Native CTCLoss with feasibility and reduction; tiny Conv/BiGRU tone recognizer, greedy blank/repeat decoding, onset frames; S/D/I alignment and corpus CER/WER | `speech/ctc.py` | brute-force CTC path sums, jiwer and actual learning in `test_domain_speech.py` |
| Three deterministic fixture generators and runnable JSON graph examples; source content hashes and fitted tokenizer JSON persisted | `examples/make_domain_fixtures.py`, `make_domain_examples.py` | deterministic bytes, validated examples, actual process runs in domain tests |
| Read-only recorded domain inspection with run/graph/node/source/artifact provenance; bounded source/prediction samples | control API and existing CAS worker | `test_actual_domain_worker_inspection_is_recorded_and_read_only` in `test_domain_api.py` |
| Editor domain workspaces linked from project picker; vision overlays before/after and prediction/GT, selectable source spans/subwords/labels, waveform/spectrogram/frame selection and CTC/error alignment; editable settings and typed graph canvas | `DomainWorkspace.tsx`, `DomainViews.tsx` | editor build/type check; browser verification recorded in HANDOFF |

Not implemented: domain model/optimizer checkpoints, exact resume, general dataset importers beyond fixture formats, pretrained models, dedicated detector training, speech playback/streaming/beam search/forced alignment, language-model generation and multimodal fusion. Cancellation is between domain nodes. Affine boxes enclose transformed corners unless the explicit mask-refit policy is selected; they are not claimed to be tight masks. Native tokenizer vocabulary tie-breaking is not claimed deterministic across versions.

## Milestone 7: registry and local production investigation (A50, A59–A63)

See ADR 0012. This is a **real native scikit-learn tabular adapter in one local CPU FastAPI process**. Local and staging are independent local routes. No remote deployment or simulated performance is claimed.

| Capability | Where | Verification |
|---|---|---|
| A50: immutable registered versions with owner/use/limitations; actual fitted estimator, imputer/one-hot/scaler, typed selection, ordered raw schema/features/classes, source/graph/evaluation/environment/code identities | `production/pipeline.py`, tabular worker | `test_pipeline_matches_native_training_state_and_pins_all_identities`, native regression imputation/one-hot/scaling comparison |
| Hash/trust/environment checks before loading internal native artifacts; serving does not reread source or refit; unsupported paths refused and old runs need a supported rerun | `pipeline.py`, `runtime.py` | integrity/environment refusals, source deletion and artifact immutability tests |
| Movable aliases resolve to immutable versions; real compatibility warmup; pinned local/staging release candidates and health/readiness | `store.py`, `runtime.py`, control API | registration/release API validation, local/staging route independence |
| A59: real loopback HTTP traffic builder, rotating representative batches, steady/ramp/burst/response-driven patterns, warmup, concurrency/rate/duration/request caps, measured throughput/latency/queue/errors/timeouts/payload sizes and sampled resource scope | `production/traffic.py` | real HTTP tests for all four patterns, generator saturation/drop accounting, cancellation/drain and persisted result recovery |
| A60: isolated durable per-release/user/session request counter, serialized updates, idempotent requests, bounded admission queues/timeouts; cancellation refuses state commit | `runtime.py`, `store.py` | 20 concurrent requests with separate users/sessions, active native-pass cancellation, overflow/deadline tests; restart marks uncommitted requests failed |
| A61: explicit deploy/rollback, compare-and-swap routing, append-only lifecycle; in-flight requests pin their original version | `store.py`, `runtime.py` | actual different-weight rollout/rollback, stale-route refusal, undeployed rollback refusal, alias independence |
| A62: observed numeric/categorical/prediction drift and missing/schema errors; label-based accuracy/MSE/MAE is separate/unavailable without aligned labels, with delay/window/evidence | `production/monitor.py` | SciPy/hand-calculation reference tests, changed-input evidence without quality claims, immutable aligned ground truth |
| A63: request → release → registry version → training run → source artifact → preprocessing/fitted state → evaluation is recorded; explicit captured-input replay does not update research/session state | control production API, CAS | lineage/replay/readiness/restart integration tests |
| Production workspace: registry, candidate/config review/deploy/rollback, input/trace/replay/label controls, real traffic builder/results/history and monitoring | `ProductionWorkspace.tsx` | editor build/type check and installed Chrome journey; see HANDOFF |
| Deterministic labelled SYNTHETIC sensor project and real HTTP train/register/serve/load/rollback CLI | `examples/make_production_example.py`, `production_journey.py` | native fixture tests, CLI and real-browser runs recorded in HANDOFF |

Bounds: 128 rows/256 KB per request, at most 16 active requests/64 queued per release, deadlines up to 30 s; load generation 30 s/500 arrivals/16 threads plus at most 10 warmup requests. Reference drift data uses up to 2,000 recorded training rows and reports truncation; monitoring inspects the latest 1,000 requests for a release/time window. Resources are sampled process CPU/RSS for the combined server/generator, including warmup/drain; cost is not measured. Capture defaults off; replay/input drift requires captured values. Training-reference labels are explicitly in-sample fixture evidence, not held-out production accuracy.

Not implemented: PyTorch CNN/domain/agent/RL/unsupervised/Keras/JAX registry-serving adapters, authenticated multi-user security, remote deployment, multiple control-process replicas, autoscaling, dynamic batching, streaming, asynchronous batch jobs, canary/shadow traffic splitting, online learning, automatic retraining/rollback, and retention/garbage collection. Session state is an application request counter, not an LLM memory service. Use one owning control process for this adapter; caller-declared users are isolation keys, not authenticated principals.

## Milestone 8: bounded worker/tracker integrations and community tooling

See ADR 0013 and the complete [A01–A64 acceptance checklist](ACCEPTANCE.md). These integrations execute native libraries/processes; the shipped worker is a **separate process on the same machine**, not remote infrastructure.

| Capability | Where | Verification |
|---|---|---|
| Authenticated loopback worker protocol, copied immutable CSV inputs, original/materialized graph identities, native CPU tabular execution, bounded admission, between-node cancellation | `scale/remote.py`, `worker_server.py` | real separate-process HTTP worker, native local agreement, two-active admission and cancellation tests |
| Durable submission/reconnect and atomic verified terminal artifact/event import; lost acknowledgement does not duplicate executions/imports | `scale/state.py`, `remote.py` | real lost-acknowledgement/retrieval, pending disconnection and fingerprint-conflict tests |
| A49: native local MLflow SQLite with idempotent external-run/metric/content-artifact reconciliation | `tracking/bridge.py`, `native.py` | actual native metric history/artifact records after simulated loss of local acknowledgement |
| Native W&B offline immutable export; rebuild unconfirmed snapshots and reuse confirmed mappings; **no cloud upload or live offline resume** | `tracking/native.py` | actual SDK binary history and artifact records decoded with checksum checks; one confirmed directory after reconnect |
| Explicit metadata/metric/evaluation-summary sharing selection; optional isolated tracker environment leaves training dependencies intact | `tracking/requirements.txt`, `lock.txt` | default no-metrics/no-artifacts, sharing refusals, altered-artifact and missing-runtime tests |
| Version/hash/dependency/schema/state/effect/inspection/docs/example manifest and native numerical conformance; explicit trusted startup activation, recorded identities | `extensions/sdk.py`, `examples/plugins/offset` | real native fixture/repeat/schema/immutability tests, explicit activation and integrity refusal |
| A16/A17: inert graph/UI/dependency/environment package with optional explicit CSV snapshots; unknown operations stay readable and exact mismatches block execution | `extensions/packages.py`, `graph_core/validate.py` | portable source-byte agreement, path/hash/secret refusals, available-version mismatch and unknown dependency tests |
| A20: keyboard node/wire selection and port connection through existing graph semantics; visible focus | `KeyboardGraphTools.tsx`, `App.tsx` | installed Chrome keyboard journey in HANDOFF; broad accessibility certification is not claimed |
| A64 bounded connected investigation: real local PostgreSQL visual query, pinned source variant, actual evaluation/intermediate inspection, immutable researcher conclusion, native pipeline serving and measured HTTP | `examples/connected_production_journey.py`, evidence API/editor | real HTTP integration test, CLI and installed Chrome recorded in HANDOFF |
| Working Integrations editor views: worker reconnect/cancel, tracker selection/queue/sync, package inspect/import/export, native conformance, readable evidence comparisons and A01–A64 checklist | `ScaleWorkspace.tsx` | editor build/TypeScript and Chrome flows |
| Published actual local-vs-HTTP worker measurements for seeds 71/72/73 | `examples/scale_journey.py`, `benchmarks/results/m8_local_worker.json` | exact recorded evaluation values and partition identity agreement; actual native tracker confirmation/reconnect |

Bounds: one owning control/worker process, loopback IPv4 HTTP only, 64 graph nodes/8 MiB CSV inputs/two worker jobs/12 MiB worker body/32 MiB imported artifacts. Per-export at most 32 explicitly selected numeric evaluation-summary artifacts; bridge subprocess timeout 90 s. Project packages at most 64 resources/8 MiB CSV bytes. Conformance checks declared pure-tabular fixtures in a 15 s subprocess; native Python is explicitly trusted, not sandboxed. Imported native pickles cannot be registered for serving as trusted local worker artifacts. Optional tracker dependencies use their own venv/lock. A package declares external data/native requirements and never installs them; freeform text is not automatically sanitized for arbitrary secrets.

Not implemented: cross-host encrypted/authenticated worker transport, remote GPU/cloud provisioning, distributed training/scheduling, broader worker/serving/source adapter families, multi-agent RL, online MLflow/W&B account destinations or credential configuration, streaming/live offline export resume, full third-party SDK families/visualizer plugins, team collaboration/security, external-user onboarding, or complete assistive-technology certification. **A09 numerical dependency-cache invalidation** and **A44 pinned repository source-code browse/import** remain unavailable; current graph execution reruns supported graphs. No product-finished claim follows from bounded acceptance evidence.

## Domain checkpoints and local inference (follow-on release)

See ADR 0014. New completed TinyFCN/BiGRU/CTC runs persist actual native weights, Adam state, RNG/shuffle state, epoch and curve; speech persists train-only mel mean/std and NLP pins the fitted tokenizer JSON. Older runs require rerunning training.

| Capability | Where | Verification |
|---|---|---|
| Hash-verified internal state dictionaries and pinned model/preprocessing/environment/code manifests; native checkpoint export | `domain/checkpoints.py`, `control/domain_api.py` | `test_domain_checkpoints.py`: native recorded prediction agreement, SHA/membership/environment refusals |
| Child run continuation at completed epochs with identical prepared data, tokenizer, split/config and CPU threads | native domain trainers/operations | resumed 2→4 vs uninterrupted 4: exact weights, Adam tensors, RNG and learning curves, all three families |
| Actual local inference without source files, refitting or training; read-only run evidence | `domain/inference.py`, domain API | control-service restart/source absence, native forward comparisons, batch agreement, no event/artifact changes |
| Typed RGB post-geometry image / original text / normalized audio requests, pinned tokenizer/features/normalization/labels/decoding and prediction provenance | domain API | contract/rate/batch/input refusals and real HTTP CLI |
| Working domain workspace download/manifest/continuation/input/prediction controls and readable outputs | `DomainModels.tsx` | editor build/TypeScript and installed Chrome three-domain journey in HANDOFF |

Limits: CPU and completed-epoch continuation only, local owning control process, two active inference requests and batches up to four within 1.5 MB decoded JSON. Vision expects the already geometrically prepared RGB size (max 512 each dimension); speech accepts 1–4 normalized PCM channels, exact pinned rate, ≤32,000 samples/channel and ≤4096 feature frames. These native prediction endpoints are separate from the tabular production registry/release system. Domain registry rollout/monitoring, general datasets, portable checkpoint upload/import, mid-node recovery/cancellation and GPU/cross-version exact resume remain unavailable. Historical 6b checkpoint exclusions are superseded by this section only within these bounds.
