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
- Editor: large graphs can be small at initial fit; durable real Chrome tests now exist (ADR0026 onward).

- Gradients tab and captured gradient information (VISION 8.3), saliency, activation distributions over training.
- Editor: richer grouping/auto-layout, multi-select move as a unit,
  insertion into an existing connection, interactive convolution teaching mode (8.4), partial-weight transfer (8.5.6),
  resource estimates, "resume compatible checkpoint". Bounded draft undo/redo is implemented below (ADR0031).
- General pause/heartbeats/leases and orphaned-worker recovery; supported model checkpoint resume and domain completed-epoch child continuation exist. Cancellation cannot forcibly terminate a native call stuck inside a batch.
- Experiment board: smoothing and x-axes other than step remain; authored run notes/tags and catalogue search now exist (ADR0032). Model graphs keep pin + two-run compare; tabular/model sweeps use the Experiments board.
- User accounts, roles and multi-user audit; an optional shared bearer token exists (ADR 0021). Image-folder datasets are server-side folders (tabular graphs can read connected sources).
- Automatic installation/migration of exported environments and portable trained-artifact import. Inert graph/UI packages with exact operation requirements and optional CSV snapshots are implemented in Milestone 8; native domain checkpoint and manifest downloads are documented below.
- Behaviors outside the documented bounds in the A01–A64 acceptance checklist; later Milestone 7/8 and domain checkpoint sections are authoritative for their implemented scope.
- Broader multi-input/output training, optimizer/scheduler/precision/device workflows outside the supported PyTorch training procedure. Native tensor broadcasting/casts and composed custom losses exist; GPU and mixed precision remain unavailable.
- Broader parameter initialization/regularization and device-placement policies beyond implemented sharing/freezing/procedure controls.
- Domain dataset importers outside the bounded COCO segmentation/CoNLL IOB2/WAV+transcript formats in ADR 0022. Image folders, CSV and connected PostgreSQL/S3/DVC sources are already supported.
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

Native libraries available here: torchvision 0.29.1, torchaudio 2.11.0, tokenizers 0.23.2, seqeval 1.2.2, torchmetrics 1.9.0, pycocotools 2.0.11, jiwer 4.0.0. See ADR 0011. Built-in teaching fixtures are explicitly labelled SYNTHETIC; bounded local COCO/CoNLL/WAV imports now carry user-declared source status and license (ADR 0022). Verification uses generated fixtures; no real-speech or user-dataset accuracy benchmark is claimed.

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

Not implemented (checkpoints and completed-epoch resume shipped later; see the domain-checkpoint section): dataset imports outside the bounded COCO segmentation/CoNLL IOB2/WAV+transcript formats below, pretrained models, dedicated detector training, speech playback/streaming/beam search/forced alignment, language-model generation and multimodal fusion. Cancellation is between domain nodes. Affine boxes enclose transformed corners unless the explicit mask-refit policy is selected; they are not claimed to be tight masks. Native tokenizer vocabulary tie-breaking is not claimed deterministic across versions.

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

Not implemented: Keras/JAX and non-image model-graph registry-serving adapters; agent serving outside the bounded scope in ADR 0023–0024 (domain models since ADR 0017, model-graph image classifiers since ADR 0018, greedy DQN policies since ADR 0019, k-means/GMM/PCA since ADR 0020), authenticated multi-user security, remote deployment, multiple control-process replicas, autoscaling, dynamic batching, streaming, asynchronous batch jobs, canary/shadow traffic splitting, online learning, automatic retraining/rollback, and retention/garbage collection. Session state is an application counter for non-agent families; native conversation checkpoints are supported by ADR 0024. Long-term memory/retrieval/effects serving remains unsupported. Use one owning control process for this adapter; caller-declared users are isolation keys, not authenticated principals.

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

Not implemented: cross-host encrypted/authenticated worker transport, remote GPU/cloud provisioning, distributed training/scheduling, broader worker/serving/source adapter families, multi-agent RL, online MLflow/W&B account destinations or credential configuration, streaming/live offline export resume, full third-party SDK families/visualizer plugins, team collaboration/security, external-user onboarding, or complete assistive-technology certification. A44 is now covered by bounded pinned repository browse/import (see its section below). A09 is now covered for tabular graphs by the opt-in node cache (see the section below); other graph kinds rerun every node. No product-finished claim follows from bounded acceptance evidence.

## Domain checkpoints and local inference (follow-on release)

See ADR 0014. New completed TinyFCN/BiGRU/CTC runs persist actual native weights, Adam state, RNG/shuffle state, epoch and curve; speech persists train-only mel mean/std and NLP pins the fitted tokenizer JSON. Older runs require rerunning training.

| Capability | Where | Verification |
|---|---|---|
| Hash-verified internal state dictionaries and pinned model/preprocessing/environment/code manifests; native checkpoint export | `domain/checkpoints.py`, `control/domain_api.py` | `test_domain_checkpoints.py`: native recorded prediction agreement, SHA/membership/environment refusals |
| Child run continuation at completed epochs with identical prepared data, tokenizer, split/config and CPU threads | native domain trainers/operations | resumed 2→4 vs uninterrupted 4: exact weights, Adam tensors, RNG and learning curves, all three families |
| Actual local inference without source files, refitting or training; read-only run evidence | `domain/inference.py`, domain API | control-service restart/source absence, native forward comparisons, batch agreement, no event/artifact changes |
| Typed RGB post-geometry image / original text / normalized audio requests, pinned tokenizer/features/normalization/labels/decoding and prediction provenance | domain API | contract/rate/batch/input refusals and real HTTP CLI |
| Working domain workspace download/manifest/continuation/input/prediction controls and readable outputs | `DomainModels.tsx` | editor build/TypeScript and installed Chrome three-domain journey in HANDOFF |

Limits: CPU and completed-epoch continuation only, local owning control process, two active inference requests and batches up to four within 1.5 MB decoded JSON. Vision expects the already geometrically prepared RGB size (max 512 each dimension); speech accepts 1–4 normalized PCM channels, exact pinned rate, ≤32,000 samples/channel and ≤4096 feature frames. These standalone native prediction endpoints also feed the domain production adapter added later (ADR 0017, documented below). General datasets, portable checkpoint upload/import, mid-node recovery/cancellation and GPU/cross-version exact resume remain unavailable. Historical 6b checkpoint exclusions are superseded by this section only within these bounds.

## Dependency-scoped node cache for tabular graphs (A09, ADR 0015)

| Capability | Where | Tests |
|---|---|---|
| Opt-in `cache: "reuse"` run setting; per-node keys over operation/version, node id, resolved settings (after run seed override), tabular implementation file hashes, Python/native library versions and input identities; sources hashed by content read | `tabular/cache.py`, `tabular/engine.py`, `worker/tabular_run.py`, `artifact_store/store.py` (`node_cache` index) | `test_node_cache.py` |
| Editing a node re-executes it and its dependents only; reverting reuses the earlier results; changed source bytes, run seed, implementation or environment invalidate exactly what depends on them | same | `test_node_cache.py` |
| Reused results recorded as ordinary run artifacts, byte-identical to original and uncached execution; corrupted entries are refused and recomputed | same | `test_node_cache.py` |
| Retention: per-project summary; explicit prune (dry-run default, keep latest N per node, older-than) that frees only unreferenced entry bytes and never touches run artifacts; editor size/keep-latest/clear controls | `tabular/cache.py`, `services/control/cache_api.py`, `TabularPanels.tsx` | `test_node_cache.py` |
| Recorded decision per node (`hit`/`miss`/`bypass`, reason, what changed, source run) in events, run summary and the editor Run record's node-cache table | `services/control/app.py`, `apps/editor/src/components/TabularPanels.tsx` | `test_node_cache.py` (API) |

Not implemented: caching for model/procedure/agent/RL/domain graphs, connector sources, joins, code blocks or plugin operations (they always run); sharing entries across workbenches; any import/upload path for entries. Optional automatic retention for saved tabular projects is verified below (ADR0046). Cache-off (default) runs are unchanged.

## Pinned repository browse and code import (A44, ADR 0016)

| Capability | Where | Tests |
|---|---|---|
| Fetch a Git remote (local path, file, https, ssh) into a bare mirror with hooks disabled, a protocol allowlist and no checkout; resolve branch/tag/HEAD/SHA to a commit; refuse credential-bearing URLs and transport helpers | `repos/core.py`, `services/control/repos_api.py` | `test_repos.py` |
| Browse a pinned tree with content classification (installation scripts, dependency manifests, license, datasets, weights, configuration, notebooks, LFS pointers, submodules, symlinks); raw blob reads without filters | same | `test_repos.py` |
| Static dependency parsing (requirements, pyproject, setup.cfg; setup.py reported as not run) and license SPDX guess | same | `test_repos.py` |
| ast-only Python inspection; wrap one top-level function with its pinned pure-Python package closure as a code block; immutable, integrity-checked import record; block `origin` part of the semantic hash; chosen pins verified by the sandbox | same, `apps/editor/src/components/RepoImport.tsx` | `test_repos.py` |
| Local-modification status and commit comparison for the imported file; editor import dialog and origin banner | `CodeBlockEditor.tsx`, `RepoImport.tsx` | `test_repos.py` (HTTP) |

Not implemented: notebooks or compiled extensions as entry points (pure-Python multi-file packages are bundled from pinned texts; ADR 0016 addendum), wrapping repository data/config/model files as typed sources, LFS download, submodule fetch, dependency installation, hosted-provider APIs/credential UI, history/blame, automatic mirror retention (explicit listing/removal via `/api/repos/mirrors` and `DELETE /api/repos/{id}` exists). The code-block sandbox guard is best-effort, not a hostile-code boundary.

## Domain models in the production registry (ADR 0017)

| Capability | Where | Tests |
|---|---|---|
| Register a recorded vision/NLP/speech model as a production version (pinned checkpoint, preprocessing, tokenizer/labels, environment, implementation); warmup on the recorded held-out example | `production/domain_adapter.py`, `production/runtime.py` | `test_production_domain.py` |
| Serve 1-4 records per request through the shared local/staging routes with admission, idempotency, traces, capture/replay; predictions equal `/api/domain/models/{id}/predict`; changed implementation refuses serving | same, `services/control/production_api.py` | `test_production_domain.py` |
| Per-family ground truth and native quality (NLP word accuracy + seqeval span F1, undefined without spans; speech corpus CER/WER; vision pixel accuracy/IoU/Dice); descriptive input and prediction drift against the held-out example | `production/monitor.py`, `production/domain_adapter.py` | `test_production_domain.py` |
| Editor Production workspace: domain candidates, pinned-model panel, maxBatch cap, held-out example loading, label formats, family metrics | `ProductionWorkspace.tsx` | browser journey (HANDOFF §14) |

Not implemented: Keras/JAX serving and agent serving beyond ADR 0023–0024’s bounded turns/conversations; a larger domain monitoring reference than the single recorded example. Model-graph image classifiers, greedy DQN and k-means/GMM/PCA serving were added in ADR 0018–0020, documented below. Monitoring calls the common input comparison a recorded reference, since fitted rows and replay-buffer observations are not held-out examples.

## Model-graph image classifiers in the production registry (ADR 0018)

| Capability | Where | Tests |
|---|---|---|
| Register a completed model-graph image-classification run: stored graph, latest complete checkpoint, training preprocessing, classes, environment and implementation hashes; held-out reference frozen only when the source folder matches its recorded identity | `production/model_adapter.py`, `production/runtime.py` | `test_production_model.py` |
| Serve 1-32 PNG/JPEG images through the shared routes; predictions/probabilities equal `/api/infer` on the same checkpoint; never rereads the source folder; changed implementation refuses serving | same, `services/control/production_api.py` | `test_production_model.py` |
| Accuracy from supplied class labels; image-statistic and predicted-class drift against the frozen reference; reference accuracy reported separately | `production/monitor.py` | `test_production_model.py` |
| Editor candidate/version/reference views | `ProductionWorkspace.tsx` | browser journey (HANDOFF §14) |

Not implemented: non-image or multi-input model graphs, procedure-trained sequence models, GPU/remote/multi-replica serving.

## Greedy DQN policies in the production registry (ADR 0019)

| Capability | Where | Tests |
|---|---|---|
| Register a completed DQN run's final Q-network with network graph, environment spec, full Box bounds, evaluation report, environment and implementation hashes; replay-buffer reference frozen | `production/rl_adapter.py`, `production/runtime.py` | `test_production_rl.py` |
| Serve observations (1-256) with the greedy action and Q-values, equal to the recorded network; bounds/length/type refusals | same, `services/control/production_api.py` | `test_production_rl.py` |
| Action agreement only against supplied integer actions (booleans refused); observation and action drift against the frozen reference | `production/monitor.py` | `test_production_rl.py` |
| HTTP serving refuses NaN/Infinity before admission; validation errors echoing non-finite input stay valid JSON | `services/control/production_api.py`, `services/control/app.py` | `test_production.py` |

Not implemented: continuous-action, image-observation or recurrent policies; online environment rollouts from serving.

## Unsupervised models in the production registry (ADR 0020)

| Capability | Where | Tests |
|---|---|---|
| Capture k-means/GMM/PCA servable manifests (native scaler + estimator, frozen fitted rows) or explicit refusals (DBSCAN, t-SNE, fitted preprocessing upstream) without changing the tabular adapter identity | `production/unsup_adapter.py`, `worker/tabular_run.py` | `test_production_unsup.py` |
| Serve raw feature rows (1-128): clusters/distances, responsibilities/log density, PCA scores/reconstruction error, equal to the fitted objects | same, `production/runtime.py`, `services/control/production_api.py` | `test_production_unsup.py` |
| External agreement (ARI/NMI, never accuracy) from supplied labels; PCA label-free reconstruction error; feature and assignment drift | `production/monitor.py` | `test_production_unsup.py` |

Not implemented: serving with fitted preprocessing before the estimator; out-of-sample DBSCAN/t-SNE.

## Optional bearer token (ADR 0021)

| Capability | Where | Tests |
|---|---|---|
| `VOID_API_TOKEN` enables a constant-time bearer check on every request (401 + `WWW-Authenticate` otherwise); weak tokens refused at start-up; editor proxy injects the header server-side; traffic generator and example scripts authenticate | `services/control/auth.py`, `services/control/app.py`, `apps/editor/vite.config.ts`, `production/traffic.py` | `test_auth.py` |

Not implemented: user accounts, roles, per-user audit, TLS termination (deploy behind a TLS proxy before exposing the service).


## Bounded local domain dataset imports (ADR 0022)

| Capability | Where | Evidence |
|---|---|---|
| Explicit user-declared synthetic status and license; original source byte snapshots/hashes, canonical payload and converter/manifest identity; atomic import materialization and integrity refusal | `connectors/domain_import.py`, `control/domain_datasets_api.py` | `tests/test_domain_import.py`: deterministic identities, CAS snapshots, source removal, tampered payload/manifest; token/schema enforcement |
| COCO segmentation: native polygon/RLE masks, sparse category mapping, exact xywh→xyxy semantics, overlapping instance masks, keypoint visibility/flip pairs and original annotation/license IDs | native loader + pycocotools; existing typed image source | `COCO.annToMask` comparisons, native RLE/flip checks; actual tiny FCN worker training/inference |
| CoNLL IOB2: selected whitespace columns, reconstructed text/character spans, preserved original tokens/tags/line numbers | native text source, offline WordPiece and BiGRU | Unicode spans agree with seqeval; grammar refusals; actual imported worker training/inference |
| WAV PCM + JSON transcripts: native SciPy normalization including 24-bit, preserved rate/channels/lengths, no inferred gold timing, explicit mono feature policy | native audio loader, torchaudio features and CTC | native unsigned/signed/float/stereo comparisons; actual imported CTC worker training/inference |
| Editor import→verify→apply to draft, clear incompatible continuation/tokenizer choices, update project source declaration; inspect/train/predict with recorded imported provenance | `DomainDatasetImport.tsx`, existing workspace/views/models | build/type check and installed Chrome evidence in HANDOFF §19; persisted checkpoint/production-reference provenance tests |

Bounds: 2–256 records, 64 MiB input/decoded payload, 8 MiB JSON/text, one concurrent import. COCO requires uniform RGB PNG/JPEG geometry (4–512 pixels), 1–4 categories, ≤64 instances/image, common ≤128-keypoint schema; no crowds or box-only annotations. CoNLL has ≤4,000 reconstructed characters/sentence and ≤32 entity types. WAV requires uniform 1–96 kHz rate and 1–4 channels, 512–32,000 samples/channel, ≤256 transcript characters/clip and ≤64-character alphabet. Explicit member roots reject traversal/symlink escape. Arbitrary URL downloads, streaming/large datasets, file uploads, other annotation formats, automatic geometry/rate correction, verified license ownership, import deletion/retention and real-world benchmark evidence are **not implemented**. The license/status are user declarations; fixture tests do not certify them.

Compatibility: this continuation changes pinned native domain source code, so **previous vision/NLP/speech models and registered domain versions require fresh training and registration**. Existing runs remain recorded; integrity checks are unchanged. Tabular/CNN/RL/unsupervised identities and dependencies are unchanged. No database migration or dependency reinstall.


## Bounded agent production turns (ADR 0023)

| Capability | Where | Verification |
|---|---|---|
| Register a completed whole agent graph that reached END; pin graph/config/final-state/reference membership, native source/environment and currently installed local Ollama model digest/runtime | `production/agent_adapter.py`, `runtime.py` | `test_production_agent.py`, actual installed-digest refusal in `test_production_agent_live.py` |
| Existing native LangGraph compiler/blocks, reducers, branches/joins/routes/loops; fresh checkpoint per turn; no research-thread mutation or shared conversation state | same adapter | native parallel-join/reducer, concurrent isolation, restart/idempotency and recursion/state-growth tests |
| Stable contract/scope/config errors, native budget/deadline/cancellation checks, existing bounded admission and route/commit semantics | same adapter/shared runtime | offline native refusal tests; existing serving queue/cancellation/rollout tests remain intact |
| Production exact text/state/control/context inspection with execution/thread/graph/source identities, actual message segments/settings/token counts and context CAS hashes; full values require capture enabled | `AgentTurnInspection.tsx`, shared Production workspace | live native request/CAS/context/capture tests; installed Chrome journey (HANDOFF §20) |
| Input-only source reference, real isolated candidate warmup, new-call replay with explicit nondeterminism caveat, read-only monitoring with no LLM invocation | production API/monitor | native HTTP/replay test, monitoring-execution refusal test; actual Ollama HTTP/Chrome journeys |
| Character-length KS, successful-call provider usage, exact-string agreement only against independently supplied reference strings | `production/monitor.py` | native offline hand-result and real Ollama label evidence; no semantic correctness claim |
| Runnable picker example and bounded native HTTP serving/load/rollback journey, with SYNTHETIC teaching prompts and real local model responses | `examples/serving_agent.*`, `agent_serving_journey.py` | CLI and installed Chrome (HANDOFF §20) |

Bounds: only prompt/set_state/at most one chat-model node; declared ≤25 supersteps, ≤2 calls, ≤4096 tokens, ≤30 seconds, ≤16 nodes, ≤32 state fields and ≤32 KiB graph. Defaults ≤8 KiB; templates ≤2000 characters/16 references. Exactly one record with the pinned 1–4 text fields (1–2000 characters each, total ≤8 KiB); native state ≤64 KiB after each superstep, context ≤16 KiB before a call. Local Ollama port 11434 only, think=false, max_tokens ≤128, timeout_s ≤30; no downloads. Provider identity is pinned at registration; old source runs did not capture its digest. Agent releases are stateless/maxBatch=1. Model device and cost are not measured; no CPU-only Ollama claim. Generated text remains recorded even with input capture off and can echo inputs.

Persistent native conversations were added separately in ADR 0024 (below). Not implemented: memory/retrieval/index dependencies in releases; effects/interrupt approval serving; multiple model nodes (bounded stateless structured-output serving is now in ADR0045), streaming, Anthropic/remote provider production adapters; semantic quality benchmarks; broader retention/GC or separate Ollama hardware telemetry. Active model calls may finish before cancellation is observed, but their output is refused after deadline/cancellation/budget failure. Earlier domain/CNN/tabular/RL/unsupervised identities are unchanged by this release.

Verification hardening: CNN-worker cancellation accepts a control process that wins the cancelling transition between the worker check/write, preserves the native partial checkpoint, and retains strict terminal transition errors. `test_worker_cancel_race.py` forces this interleaving with a separate real SQLite writer; existing worker/process cancellation tests remain unchanged.

Production source navigation passes the selected run identity into the agent workspace and opens its actual trace; Chrome verifies an older pinned source after a newer run exists. Final release verification: default **1053 pass / 1 skip / 8 live deselected**, live **8 pass**; final build/typecheck/curl/installed-Chrome evidence is in HANDOFF §20.


## Implemented and tested: persistent native agent conversations (ADR 0024)

| Capability | Where | Evidence |
|---|---|---|
| Separate `__agent_conversation__` registry candidate for completed END graphs with thread state / turn text inputs; original stateless adapter and other family identities unchanged | `production/conversation_adapter.py`, runtime/API | `test_declared_conversation_scope_contract`, native registration and live source runs |
| Native END checkpoint/channel-version/metadata snapshots, installed native serde/restore; exact native turn resets/reducers; source research threads untouched | conversation adapter / native compiler | `test_native_reducers_turn_reset_restart_idempotency_source_isolation` compares whole states to an independent native research saver; actual Ollama history test |
| One explicit session, one record per request; serialized per-release/user/session turns; stable native thread and fresh execution IDs; rollout isolation and rollback continuation | runtime/store | concurrent same-session, other-user/session/release and rollback test |
| Successful checkpoint revision/head and request trace commit in one SQLite transaction; cancellation/deadline/failure/stale-parent/integrity refuse state updates; restart does not resume failed turns; idempotency never double-commits | `agent_sessions` additive table; immutable CAS | cancellation and slow CAS deadline; failing SQLite trace trigger rollback; actual process SIGKILL after native candidate, restart and duplicate retry; hash/parent refusals |
| Native state/history persists even with capture off (explicit session policy); captured trace contexts/state/events remain opt-in; verified read-only checkpoint inspection | release policy, GET release conversation API / Production controls | capture-off persisted-state and HTTP inspection tests; installed Chrome |
| Captured replay restores the recorded immutable prior checkpoint, independently of the current live head, then discards its candidate; actual requests/context segments/provider usage; monitoring reads warmup/traces only | production replay / inspection / monitor | native HTTP parent replay at turn 2 after turn 3; live exact-provider-request replay; Chrome session isolation |
| SYNTHETIC teaching prompts / real local Ollama; six-message native history window; source→register→warmup→two turns→inspection→replay→traffic→rollout/rollback | `examples/serving_conversation.*`, `agent_conversation_journey.py` | actual HTTP CLI and Chrome evidence in HANDOFF §21 |

All ADR 0023 input/graph/state/context/provider/budget bounds apply; serialized checkpoints are additionally limited to 128 KiB. No checkpoint uploads, arbitrary deserialization or intermediate task resumption. Snapshot bytes are internal native products and SHA-verified. One owning control process; caller-declared user/session values are isolation keys, not authenticated ownership. Conversation mode persists state/history regardless of trace capture; response text may itself quote inputs. Native END checkpoint is restored with original channel versions/metadata rather than hand-merging JSON state. Replay may produce a different response; exact sent history is the evidence, not semantic accuracy. Input character drift measures current inputs, not conversational history.

Reviewed live-checkpoint reset/fork is implemented separately in ADR 0025 (below). Not implemented: checkpoint physical deletion/migration/retention APIs; long-term memory/retrieval/index dependencies; tools/effects/interrupt approvals; structured/multiple model nodes or external-provider serving; streaming/distributed serving; authenticated conversation ownership/encryption; semantic answer benchmarks. Failed/cancelled writes can leave unreferenced CAS bytes. Provider costs and separate Ollama hardware remain unmeasured. No new dependencies or changes to earlier pinned execution source files, so no retraining/re-registration is required for existing families/stateless agents by this release.


## Implemented and tested: reviewed native conversation reset/fork (ADR 0025)

| Capability | Where | Evidence |
|---|---|---|
| Bounded POST reset/fork with mandatory inspected revision/SHA, action ID and reason; same-release/user destination only, no overwrite | `production/conversations.py`, control production API | Native stale/empty/stateless/other-user/schema/destination refusal cases |
| Serialized with active native turns using existing scope locks; deterministic two-scope ordering and bounded wait; SQLite parent recheck | conversation actions / `ProductionStore` | Active-turn race refuses old review; competing/opposite forks; lock timeout |
| Atomic head + immutable receipt + lifecycle event; idempotent identical retries and conflicting-ID refusal across restart/later turns | additive `conversation_actions` table | SQLite receipt failure rolls back both reset/fork; actual process SIGKILL inside both uncommitted transactions, restart and retry once; slow CAS deadline |
| Logical reset increments revision and clears live checkpoint; next native turn starts at defaults with a fresh thread; old traces/snapshots/replay preserved | unchanged native adapter + nullable head | Native fresh state and monotonic revisions, original request idempotency, historical-parent HTTP replay after reset |
| Fork preserves actual native checkpoint channels/versions/metadata/history provenance in a new independent thread, then continues native reducers; source untouched | native serializer clone, no adapter changes | Whole-state equality, independent native continuation; actual Ollama inherited user/assistant messages; no-inference assertion |
| Working inspected-scope editor controls, reset review checkbox, reason/destination, retry action ID and receipt/lineage | `ConversationActions.tsx`, Production workspace | build/typecheck, installed Chrome review→fork→reset→historical replay→fresh turn/monitor; HANDOFF §22 |
| Runnable HTTP fork/reset teaching journey with real Ollama and labelled SYNTHETIC prompts | `examples/conversation_actions_journey.py` | HTTP CLI and actual backend restart; HANDOFF §22 |

Reset is logical, **not physical deletion/privacy erasure**. Conversation state/history and old CAS/trace evidence remain retained even if trace capture is off. Empty/reset heads report state=null; future defaults are not shown as measured state. Fork is from the inspected **current** successful checkpoint, under the same registered release/user; preserving a historical generated message’s original call/thread provenance is intentional. A fork revision starts at 1, while native thread state can already contain several turns. Reset increments the existing revision; the next success increments it again. Neither action invokes a model. Existing native serving-family/code/environment pins are unchanged, so no retraining or re-registration is required.

Not implemented: physical deletion, retention/GC, historical checkpoint picker/restore/import, cross-release/version migration, authenticated ownership/roles, broader session discovery, long-term memory/retrieval/tool/effect serving, streaming or distributed replicas. One owning control process. Immutable orphan bytes are possible after failed transactions.


## Reproducible native editor verification (ADR0026)

| Capability | Implementation | Evidence |
|---|---|---|
| Owned loopback backend/editor/installed Chrome, fresh workbench/output, ephemeral shared token, inherited app/provider settings excluded | `tools/editor_smoke.py` | Real accepted/failed/timeout journeys and native runner tests; HANDOFF §23 |
| Versioned browser source executed as a temporary copy, pinned Puppeteer Core, actual traces/receipts/screenshots/logs and failure status | `apps/editor/smoke/journey.mjs`, editor lockfile | Strict runtime/API/console errors, reviewed screenshots, cleanup outcomes |
| Real native model-free thread counters/history/defaults and UI source→registry→serve→fork/reset→historical replay→monitor | `examples/serving_state.*` | Independent native worker reducer test; actual Chrome baseline; zero LLM calls, SYNTHETIC text templates |
| Owned process-group teardown on pass/failure/timeout, readiness and actual child cleanup | smoke runner | Real child-group/unrelated-process test; successful run and forced browser timeout; user port8000 unchanged |
| GitHub workflow for offline suite/build/coverage/real browser with bounded evidence retention | `.github/workflows/verify.yml` | [Hosted run 37255947744](https://github.com/mazenDDr/project-void/actions/runs/37255947744) passed Linux offline 1086 tests/build/coverage/real Chrome; actual downloaded reports/screens audited, HANDOFF §23 |
| Editor favicon served without browser console 404 | `apps/editor/public/void.svg`, `index.html` | Actual Chrome zero-console-error journey and explicit curl |

This is a representative native conversation smoke, not all-domain browser coverage or model-quality testing. No provider is substituted: the graph contains no LLM node. Real Ollama tests remain separately required and run on the Mac. Services use the existing development proxy, not a deployed production bundle server. SIGTERM/deadline failures clean up owned groups with bounded force fallback; uncatchable runner/host death is not covered. Workbench contents can include synthetic native records; generated evidence stays outside Git and CI uploads only logs/JUnit/screenshots/JSON evidence, not CAS/model weights/tokens. Existing native serving identities remain unchanged. Broader browser matrix/live hosted providers, branch protections, accounts and deployment remain future work. ADR0027 below adds bounded offline recovery; cross-version migration and online backup remain unavailable.

Hosted verification is independently accepted on implementation commit 6d50d6c: offline **1086 passed, 1 skipped, 10 deselected**, actual Chrome **zero runtime/API/console messages**, native monitor four requests/zero model calls and owned-service cleanup. Live Ollama remains Mac-only evidence. A known action-internal Node 20→24 forced-runtime warning did not fail the job; project Node 25.9.0 was tested separately. Final documentation-only outcome update uses [skip ci]; the implementation/workflow/tests checked by that run remain unchanged.

### Offline whole-workbench recovery (ADR0027)

| Capability | Implementation | Evidence |
|---|---|---|
| Explicit offline trusted local directory backup, SQLite header discovery (including nested tracker DBs), simultaneous write reservations and committed WAL snapshot copies | `workbench_backup/core.py`, CLI | real WAL row recovered after physical source deletion; competing writers remain locked throughout copying; active/busy work refused |
| Full ordinary file/empty-directory inventory, size/SHA256 checks, native SQLite integrity, CAS contents and direct database references | same | all eight owned DB paths plus nested tracker fixture and indexes/library/projects/repos/imports/datasets bytes preserved; corrupted/missing/extra/escaping/linked data refused; opaque SQLite CAS remains byte-identical |
| Restore into a new root without rewriting immutable native identities or absolute paths; caught staging failures/collisions cleaned/refused | same | actual fitted scikit-learn and vision/NLP/speech predictions and lineage identical after source deletion; native conversation/fork/reset receipts, idempotent traces, research saver continuation; incompatible environment still refused |
| Real Chrome after backup/source deletion/restore; original version/release/heads, independent continuations, historical replay and monitor | `tools/recovery_smoke.py`, external executed `smoke/recovery.mjs` copy | original baseline plus restored editor journey; same CI flow and evidence-only upload; HANDOFF §24 records actual outcomes |

Stop all writers before `create --offline`; the flag is an attestation, not process detection. SQLite locks/tree checks supplement that requirement and do not prove online coherence. Restore requires `--trusted-local`; checksums do not authenticate data that native loaders may later execute. Backup includes unredacted user bytes, dirs0700/files0600, executable bits removed; symlinks/special files (including tracker latest-run links) are refused. There is no arbitrary web upload/deserialization interface. Root-relative stores/CAS recover within the same native environment; external data/secret files, services, code/dependencies and absolute references are not relocated or bundled. No cross-version migration, automatic scheduling/retention/GC, encryption/signing/compression, online snapshot, power-loss guarantee or full native tracker/FAISS/DVC recovery matrix. Earlier still-incompatible saved models remain incompatible; this release changes no pinned native implementation and causes no new retraining requirement.

Additional Mac-only manual evidence in HANDOFF §24: actual Ollama/FAISS index reuse and identical retrieval after source deletion/new-root restore; native memory versions/tombstones/audit/effect-ledger state preserved; actual MLflow metrics/history/metadata/artifact hash recovered when restored at the original root. Relocated MLflow artifact downloads fail because native absolute URLs remain unchanged. Actual offline W&B generates four symbolic links, so W&B-containing workbenches are refused by this v1 backup. These measured limits are the next recovery follow-up; they do not constitute a complete portable tracker/index recovery matrix or new committed live regression cases.

Hosted Linux recovery evidence is verified on implementation `9d3b810`, Actions [run37259943630](https://github.com/mazenDDr/project-void/actions/runs/37259943630): **1116 passed, 1 skipped, 10 deselected**, native suite **929.79 s**; build/typecheck/coverage and real Chrome seed → backup/source deletion/restore → fresh editor all pass, six requests/zero errors, owned services closed. Downloaded JUnit independently confirms zero failures/errors; restored-review/replay screenshots inspected. Live Ollama and supplemental FAISS/MLflow/W&B checks remain actual Mac evidence, not hosted claims. See HANDOFF §24 for precise environments, evidence retention and limits.

### Explicit offline links and durable native resource recovery (ADR0028)

This supersedes ADR0027's W&B-link refusal **only with explicit v2 policy**. Default v1 creation continues to refuse links; both versions are verified/restored.

| Capability | Implementation | Evidence |
|---|---|---|
| Opt-in v2 internal relative links recorded as inert metadata, exact target/kind/member validation; restored only after ordinary bytes/DB/CAS validation | `workbench_backup/core.py`, CLI `--links internal` | `test_backup_links.py`: internal file/directory restore after source loss; escaping/absolute/dangling/chained/database/CAS/sidecar refusals, manifest tampering, changed links and failed publication |
| Separate explicit omission of native external W&B debug-core.log with target/reason returned by create/verify/restore; target never followed | `--omit-wandb-external-logs`, v2 `omittedLinks` | Missing external target check and actual installed W&B export; all other external link shapes refused |
| Actual offline W&B export bytes/internal links and confirmed mappings survive new-root restore | Existing native SDK/Bridge, no pin changes | Real SDK binary SHA equality and native marker/reconnect after physical source deletion |
| Actual MLflow metrics/history/params/tags/run identity and original-root artifact download/reconciliation; relocated-path limitation verified | Existing native SDK/Bridge; `docs/RECOVERY.md` | Actual native export, source deletion, relocated download refusal, original-root exact artifact/history recovery and lost-confirmation retry |
| Native FAISS reuse/search, cache and memory versions/tombstones/audit/effect states recovered | Existing index/memory stores in fresh native subprocess | `test_backup_index.py`: separate lexical hashing offline evidence and live actual Ollama embeddings; source/corpus deletion, identical scores, uncertain effect remains uncertain |
| Real tracker recovery browser flow and CI source | `tools/recovery_smoke.py --trackers`, `tracker_recovery_seed.py`, existing restored journey | Actual native synthetic training/export and restored Chrome confirmed mappings/reconnect; HANDOFF §25 |

Limits remain offline/trusted/same-native-environment/new-destination. Link traversal/security is bounded by that administrative trust/quiescence requirement, not a hostile-filesystem guarantee. Only internal relative direct ordinary targets are admitted; omitted diagnostic logs are not restored. MLflow/W&B immutable absolute paths are not rewritten; new-root tracker relocation/live resume is not claimed. External files/services/credentials remain external. No online snapshot, general migration, encryption/signing, scheduling/retention/GC or disk crash guarantee. Broader DVC/import/repository/cache native recovery remains unverified. No pinned execution implementation or dependency changes; no new retraining.

### Read-only serving conversation discovery (ADR0029)

| Capability | Implementation | Evidence |
|---|---|---|
| Bounded per-release/caller-user session metadata, literal prefixes and keyset pages25/default,100/max; no state loading/model call | `production/discovery.py`, GET release `/conversations` | `test_conversation_discovery.py`: real native successful/reset/forked/capture-off heads, ordered pages/prefix/user/release/restart; primary-key query plan and read-only CAS/event/head evidence |
| Existing explicit checkpoint inspection from a discovered session; busy/error/scope changes guarded, wrapped labelled table | `ConversationDiscovery.tsx`, Production Requests | Actual Chrome27 native session seeds,25+2 next/previous pages, exact selected checkpoint and other-user empty list; build/typecheck |
| Optional shared-token protection and strict query/stateless/integrity refusal | Existing middleware/control API | Native HTTP token/invalid query/stateless and damaged metadata tests |

Discovery is metadata, not a checkpoint-integrity or provider-availability claim. Existing explicit inspection verifies the selected actual checkpoint; pages are current reads, not a frozen historical catalog. Caller user keys remain unauthenticated isolation scopes. No historical restore/import/migration, privacy erasure/retention/GC, roles/ownership, streaming or broader native serving dependencies. No model pin/dependency/DB migration changes. Exact Mac/hosted status is recorded in HANDOFF §26.

### Reviewed historical conversation restoration (ADR0030)

| Capability | Implementation | Evidence |
|---|---|---|
| Read-only whole native END state from a successful request in the exact release/user/session/version; trace/checkpoint/thread provenance verified | `production/history.py`, GET `/conversation/history/{request_id}` | `test_conversation_history.py`: capture-on/off, native state/reference, source scope/status/thread/refusal; no inference |
| Explicit current revision/SHA and source request/trace/checkpoint review; fresh native thread, monotonic serving revision, atomic head/receipt/audit | Existing conversation action/store transaction, POST `/conversation/restore` | Stale source/target, active/competing turns, reset/null head, rollback, actual process death/restart, deadlines and immutable retries |
| Selected request → verified preview → reason/confirmation → actual restored continuation | `ConversationHistory.tsx`, Production Requests; recovered Chrome journey | Actual earlier state/fresh thread/revision and native next turn, no fabricated data; source workbench physically removed |
| Native local Ollama continuation includes exactly the selected checkpoint's earlier messages | Existing pinned adapter/provider | Live test records provider contexts/usage and original message provenance; restore creates no prediction/model call |

Bounds: same native environment/code/provider and exact scope; no arbitrary SHA/upload,
failed or incomplete checkpoint, cross-version migration, privacy erasure/GC, authenticated
ownership or distributed replica guarantee. Earlier traces/snapshots stay retained. No
new model pins/dependencies/migrations. Final acceptance is recorded in HANDOFF §27.

### Bounded editor draft undo/redo (ADR0031)

| Capability | Implementation | Evidence |
|---|---|---|
| Whole immutable graph + UI draft undo/redo; same-event paired setters, no-op handling, project reset and count/size eviction | `documentHistory.ts`, `useDocumentHistory.ts`, App setters | Pure Node history tests compare full documents, ordered multiple undo/redo and bounded oldest eviction |
| Main/agent node drag grouped into one draft edit; backend/settings use existing history setters | Existing canvases and toolbar | Real Chrome actual API save/read: add/settings/layout undo/redo, full15-step drag, redo invalidation and project reset |
| Focus-aware keyboard shortcuts and saved dirty fingerprint retained | App Cmd/Ctrl-Z, Shift-Z/Ctrl-Y, labelled native buttons | Real focused-text shortcut leaves draft redo available; other projects start with disabled Undo/Redo |

History is session-local draft data only:100 prior documents/8MiB estimated JSON
history, with oldest eviction. Large current drafts remain editable; history may be
evicted. No run/checkpoint/external-effect undo, persistent/collaborative history or
large-graph performance claim. Broader scope transfer/navigation
work is separate. Exact verification is recorded in HANDOFF §28.


Recovery/discovery hosted verification passed on d0bdafa, [run37308500924](https://github.com/mazenDDr/project-void/actions/runs/37308500924):1152 native tests pass,1 skip, build/typecheck/coverage and recovered real Chrome discovery/tracker journey pass; downloaded JUnit has0failures/errors. Historical restoration/editor draft history have final Mac acceptance1176pass/1skip plus12live and5Node tests, actual dedicated/integrated Chrome and curl. Their next hosted run remains pending; see HANDOFF §28.

### Authored research run notes, tags and search (ADR0032)

| Capability | Implementation | Evidence |
|---|---|---|
| Per-run current note/tags/author/server time and paged earlier revisions, independent of measured execution data | `research/records.py`, additive `research.sqlite`, control research API | `test_research_records.py`: actual native run state/events/artifacts/status unchanged, absent/current/cleared/history/restart evidence |
| Reviewed graph/revision, atomic concurrent updates and durable identical mutation retries | Native SQLite revision/current/receipt transaction | Stale graph/revision, one concurrent winner, conflicting ID, rollback and malformed/missing revision refusal |
| Bounded literal Unicode-casefold search and exact tag/project/family/status filters over actual run metadata | Read-only attached execution metadata;25/default,100/max lexical run-ID pages | Unicode/percent/underscore/exact tags and keyset pages; no artifact/provider reads |
| Actual editable Records workspace, existing original inspector and restored authored records | `ResearchRecords.tsx`, source/recovered Chrome journey | Two authored revisions/search/original completed native run, physical source deletion and exact restored note/tags/history |

This is shared-token local metadata: author labels are unauthenticated; notes are not
measured results. Earlier revisions/receipts remain after clearing; no privacy erasure,
automatic retention or signed audit guarantee. Original run navigation requires its
project currently open. No semantic/full-text index, global snapshot/total count or
large-catalogue performance claim. No native pin/dependency changes. Final checks in
HANDOFF §29 supersede earlier experiment-board notes/tags/search gaps.

### Root graph draft copy/paste (ADR0033)

| Capability | Implementation | Evidence |
|---|---|---|
| Model/tabular/domain root node configuration, internal typed wires, relative layout and native source graph identity | `graphClipboard.ts`, actual validation on captured source | Node tests plus owned Chrome exact API-saved graph/config/wire/layout comparisons |
| Fresh IDs, copied-group parameter sharing, declared local draft state reference rebinding, transitive module/code definitions and package identities | Bounded pure transform with explicit conflict/opaque reference refusals | Native CNN/residual shape/count/output/loss/gradient/SGD reference equality; native shared encoder retains internal ties independently |
| Typed domain/tabular contracts and invalid partial/unknown graphs preserved for native review | Existing validators remain authoritative | Native all-four workflow validation, missing-input/unknown-op diagnostics retained |
| Accessible explicit selection/copy/paste/clear, compatible-scope reasons and one whole-document Undo/Redo | Root tools and existing history setters | Actual Chrome repeat paste uniqueness, boundary omission/refusal, kind restriction and full draft undo/redo |

Page-memory only, 100 selected nodes/100 required definitions/512 KiB estimated JSON.
No weights/files/resource relocation or automatic boundary wiring. Entire package
declarations are retained conservatively, no implicit dependency installation. Root
training/global settings are retained from the target. Agent/RL/module-editor transfer,
OS/serialized clipboard, cross-backend conversion and large-graph benchmarks remain.
Final environment-specific verification and hosted status are in HANDOFF §30.

### Structured model/tabular/domain and module graph outline (ADR0034)

| Capability | Implementation | Evidence |
|---|---|---|
| Actual nodes, typed wire endpoints, module/shared references, literal search and 50-row pages | `graphOutline.ts`, `GraphOutline.tsx`, scoped existing graph tools | Pure Node checks, real 75-node native teaching graph pages50+25/back/search reset in Chrome |
| Exact current native port contracts/counts/hash/diagnostics; stale reports withheld | Existing root/module validation responses, nearest visible diagnostic ownership | Native API→actual TS helper compares CNN/residual/tabular/NLP contracts; real nested error routing and pending-state refusal |
| Inspect/center/root-to-module navigation without graph/layout edits | Existing inspector, zero-duration fit and module editor/breadcrumbs | Actual Chrome saved byte equality, real invalid config/native codes, original Undo, exact module hash and inspection |

No native values are guessed when a contract is unavailable. Outline content describes
structure and native validation; learned activations remain in recorded inspectors.
Agent/RL outlines, deeper cross-scope navigation, custom command shortcuts,
performance benchmarks, onboarding and formal accessibility certification remain.
Synthetic description now uses tools layout flow so it cannot cover those controls.
No model pin/dependency/schema change. Final verification is recorded in HANDOFF §31.


### Controlled canvas measurements and safe initial adoption (ADR0035)

Both controlled model/module and agent canvases retain actual React Flow DOM dimension
changes in transient React state. Membership changes request native remeasurement;
removed IDs are pruned. Measurements never enter saved graph/layout or Undo/Redo.
Disposed initial-load requests are aborted and cannot adopt a stale example after a
new draft. No guessed dimensions, native model pin/dependency/schema changes.

All four final Mac actual Chrome journeys pass after this repair, retaining exact
saved graph/layout/native identity checks. Post-repair Mac native1209/1skip and live12 tests pass. Default Linux18ae842 browser
job passes all four journeys; its native job is pending at the limit handoff. HANDOFF
§32 records actual evidence and the exact run for the next agent to finish reviewing. Prior833e133
Linux native1204/build/typecheck/Node14/coverage and clipboard/recovered journeys pass,
but history failed; initial e0765ef remeasure-only diagnosis also failed. Neither is a
whole-workflow success. Large-graph/platform/browser coverage remains separate work.


Previous final18ae842 hosted verification now passes completely: native1209/1skip,
Node17/build/typecheck/coverage and all browser journeys, exact evidence in HANDOFF§33.

### Explicit measured graph alignment/distribution (ADR0036)

| Capability | Implementation | Evidence |
|---|---|---|
| Root model/tabular/domain and module-definition edge/center alignment using actual card dimensions | `graphArrangement.ts`, existing transient DOM measurements and scoped position keys | Actual Chrome all-eight-action edge/center/gap invariants; native20,042params/hash and exact graph/config/wire equality |
| Equal-gap horizontal/vertical distribution fixes endpoints; insufficient span refuses without editing | Measured rectangle geometry, stable E_LAYOUT_OVERLAP | Actual measured crowded fixture refusal and disabled Undo; no negative gaps |
| One whole-draft layout Undo/Redo and unchanged-repeat no-op | Existing history/setUi; graph never modified | Exact saved graph/UI metadata, unselected/root/module keys and native all-family/module reports retained |

2–100cards for alignment,3–100distribution. Root expanded frames must collapse first;
generated interiors are excluded, module boundary cards only affect definition layout.
Alignment can overlap. No guessed/stored DOM dimensions, native pin/schema/dependency
changes. Agent/RL arrangement, grid/guides/auto-layout and large-graph/platform benchmarks
remain. Final full/live/hosted state is recorded in HANDOFF§33; Mac1209native/12live/17Node/build/typecheck/coverage and actual all-family Chrome/curl/
recovery pass; hosted verification of this arrangement scope is pending.


### Scoped authored node comments (ADR0037; Mac accepted; Linux browser pass, native pending in HANDOFF§35)

| Capability | Implementation | Evidence |
|---|---|---|
| Root/module comments with exact current native graph/module identity reference and original target | UI-only `nodeComments`, NodeComments/NodeComment, native validation | Actual browser saved graph/hash/layout unchanged; native current/earlier review and module hash |
| Rename/delete whole-draft Undo with preserved original provenance; clipboard leaves source notes | Existing graph/UI history, metadata key conflict preflight | Exact source note bytes/times/hash through one Undo; target conflict refuses without graph mutation |
| Literal catalogue search,25row pages, explicit orphan/malformed removal and bounded metadata | Annotation helper/read/write guards and actual inspector links | Actual native/browser fault handling; no inferred node/prototype note |
| Real project UI comments recover after physical source deletion | Existing offline backup/native project files | Actual seed/restored editor comment equals original alongside native conversations/records/trackers |

4000UTF16text units/120author units,200comments/2MiB estimated metadata. Author/time/
identity references are editable authored metadata, not authenticated or signed audit.
No native model pin/dependency/schema change. Older notes can remain in draft history
and backups; no physical erasure. Agent/RL/runtime-instance/sample/plot annotations,
collaboration, append-only comment revisions and report exports remain. Mac acceptance:1209native/12live/17Node, strict real Chrome/curl and58-file source-deleted recovery pass. Linux comments/recovery browser verification passes on536dd33; its native job and separate arrangement repair remain pending; read HANDOFF§34 for evidence and limitations.


### Explicit typed-wire insertion (ADR0038; Mac and default Linux accepted, HANDOFF§35–§37)

| Capability | Implementation | Evidence |
|---|---|---|
| Explicit existing wire, installed operation, selected input/output and exact kind | GraphInsertionTools/graphInsertion, native registry defaults | Real browser static port/kind choices, fitted-state refusal without edit |
| Root model/tabular/domain and stored module input/internal edge replacement | One appended node/two fresh wires; other definitions/interfaces unchanged | Native Torch output/loss/gradient/SGD/count references; exact browser module/root graph/UI preservation |
| One whole-draft Undo/Redo, orphan note/layout ID reservation, additional inputs remain missing | Existing paired history and native validation | Real missing b/E_MISSING_INPUT; exact Undo/Redo and19Node checks |
| Native table profile and vision annotation wires | Native pandas/domain contracts with actual labelled fixture files | Actual Chrome/curl native reports; no training/quality claim |

Only new position stored at endpoint-origin midpoint; placement may overlap. Consumer
extension metadata remains on the downstream wire. Native semantic identity changes,
existing comments retain original references. No learned-weight transfer, opaque state,
backend/model-pin/dependency/schema change. Module output-interface display links,
Agent/RL/generated-instance links and structural/code operations are excluded explicitly.
Selected-group translation is recorded separately below; persistent grouping, automatic layout and drag-on-wire suggestions remain separate.
Mac full1214pass/1skip plus12live/19Node/build/typecheck/current ledger/strict browser/curl/recovery pass; read §35 for environment scope and retained failures. Arrangement also reads current DOM dimensions after native
readiness to repair the observed Linux card-resize race; full default Linux37361034202 also passes (see HANDOFF§37).


### Searchable existing editor commands (ADR0039; Mac accepted; hosted pending in HANDOFF§36)

| Capability | Implementation | Evidence |
|---|---|---|
| Literal search,25result pages, native modal/keyboard/focus/disabled reasons | GraphCommandMenu/graphCommands | Actual CmdCtrlK/Tab/arrows/Enter/Escape, input-focus shortcut exclusion, paging and disabled Undo refusal |
| Existing project-kind workspace navigation and current root/module inspectors | Shared workspace destination list and real handlers | Exact unchanged saved graph/UI/native hashes; real Agent/RL workspace destinations with generic node/create choices excluded |
| Disconnected static block creation with native registered defaults and exact whole-draft Undo/Redo | Existing node creator/history, kind/backend scope | Real native missing-input report, unchanged original wires/interfaces, exact Undo/Redo, orphan note/layout reservation and valid constructor ID |
| Module view roundtrip retains input wire identities/metadata | Existing converter endpoint-only correction | Actual bundled module deep equality plus real native/browser module creation |

200UTF16query units,25rows/page, literal simple-lowercase search, no semantic/fuzzy search
or large-graph performance claim. Native graph changes only on explicit creation/history
commands; actual earlier provenance remains untouched. No dependency/native pin/schema
or DB change. Formal accessibility certification, browser/platform matrix, user-defined
shortcuts, Agent/RL generic creation, whole-project/remote commands remain separate.
Required full1214native/12live/22Node/build/typecheck/ledger/current pin audit, strict Chrome/curl and58-file recovery pass; final dialog geometry and modal-history browser check passes. Read HANDOFF§36 for final outcomes.


### Selected-group layout movement (ADR0040; Mac accepted, hosted pending, HANDOFF§37)

Explicit horizontal/vertical layout offsets translate1–100 selected actual root/model,
tabular/domain or stored-module layout cards together. Current/result coordinates and
offsets must be finite within±1,000,000; all validation precedes the edit. Only selected
UI position keys change; graph/wires/interfaces/unknown metadata/native identity and
comments retain their original values. One whole-draft Undo/Redo; zero offset adds no
key/history. Empty inputs, ambiguous/oversized selection and bounds refuse with stable
E_MOVE_* codes. Expanded root modules require explicit collapse; generated interiors
and Agent/RL specialized canvases are excluded. Placement may overlap; no automatic
topology inference/group container/performance/platform certification claim.

Two pure Node tests/24total pass. Mac full1214pass/1skip/12deselected plus12real
Ollama tests,273module build/typecheck/coverage/pin audit and actual browser/curl/all
original journeys/58-file source-deletion recovery pass. Native graph/API coordinates
are compared exactly; browser CSS geometry within0.01 layout units accounts for actual
Chrome serialization precision. Hosted verification remains separate; see HANDOFF§37.


### Module-definition clipboard transfer (ADR0041; Mac accepted, hosted pending, HANDOFF§38)

Actual native module hash/source project/module version for stored PyTorch definition
internal nodes; compatible root/module paste preserves existing interfaces/params/$in
wires/root structure and merges exact dependency closure. Boundary wires stay omitted
and counted; real E_MISSING_INPUT remains visible. Module-local opaque state is refused.
Scoped layout/source-only authored notes/unknown UI retained; root/module orphan keys
reserve new identities, exact single Undo/Redo and repeated/cross-scope transfer.
Valid constructor topology fallback uses null-prototype dictionaries. Page-memory and
existing100node/100definition/512KiB bounds remain; generated/Agent/RL/OS clipboard/
weights/automatic boundary inference/platform/performance work stays separate.

Full1216native/1skip/12deselected plus12actualOllama/27Node/274module build/typecheck/
current ledger/pin audit pass. Focused2native cases also verify actual module harness
identity. Strict Chrome/6curlcases/alloriginal journeys/58-file recovery pass; hosted
verification remains separate.


### Native diagnostic scope navigation (ADR0042; Mac accepted, hosted pending, HANDOFF§39)

Current native outline diagnostics resolve to uniquely identified stored root/module
nodes, including composite/repeat iteration paths and exact then/otherwise branches.
Explicit keyboard-able action opens the actual shared-definition breadcrumb/inspector
without graph/UI/hash/history edits. Pending/unavailable reports withhold prior values
and controls; invalid/missing/ambiguous/recursive/harness-only targets are not guessed.
Root model/tabular/domain and local module scopes; Agent/RL and runtime checkpoint/
iteration stepping remain separate. No learned activation or platform/performance claim.
Full1219native/1skip/12deselected plus12actualOllama/29Node/275module build/typecheck/
ledger/pin audit pass. Strict current-report/held-real-request Chrome,3curlcases/6module
reports/alloriginal journeys/58-file recovery pass. No native pin/backend/schema/dependency
change; hosted scope remains separate.


### Specialized native agent outline (ADR0043; Mac verified, HANDOFF§40)

Agent Canvas has declared fixed control wires/conditional routes/joins, actual current
native reads/writes/effects/diagnostics/hash, literal200unit search/50row paging/errors
filter and keyboard selection of its existing inspector. No graph/UI/history edit,
recorded runtime/model value/taken route inferred or provider call. Current pending/
unavailable reports withhold prior native analysis; valid constructor IDs use own-property
view/position/trace lookups. START/END remain explicit terminal references. RL outline,
agent clipboard/groups and performance/platform/accessibility certification stay separate.
Full1219native/1skip/12deselected plus12actualOllama/31Node/277module build/typecheck/
ledger/pin audit pass. Strict current-report/held-real-request Chrome,4curlcases/native
serving baseline/all10original editor journeys/58-file source-deleted recovery pass.
No native pin/backend/schema/dependency change; hosted scope remains separate.


### Specialized native RL structure (ADR0044; Mac verified, HANDOFF§41)

The RL Structure tab lists declared typed wires/configuration and current native per-node
contracts, typing/parameter counts/diagnostics/root hash. Literal200unit search/50row
paging/error filtering, exact unique-node read-only inspection with keyboard heading
focus, and existing uniquely scoped Environment/Learner controls. Exact native port
contracts expose observation/action spaces, rewards, network shapes and DQN equations;
no catalog/family-wide fallback or runtime/learned values inferred. Prior report fields
withheld during pending/unavailable; duplicate IDs/families never get guessed inspectors/
controls. Saved graph/UI/history unchanged. No native identity/schema/dependency change.
Full1221native/1skip/12deselected plus12actualOllama/33Node/279module build/typecheck/
ledger/pin audit pass. Strict current-report/keyboard-focus/held-real-request Chrome,
5curlcases/native serving baseline/all11editor journeys/58-file source-deleted recovery
pass; hosted scope remains separate. RL clipboard/groups, broader algorithms,
expanded Q-network editing and performance/platform/accessibility certification remain
separate work.75-row browser fixture intentionally triggers native duplicate-family
refusal; it is pagination evidence, not learner validity/quality/performance evidence.


### Native JSON agent serving (ADR0045; Mac verified, HANDOFF§42)

| Capability | Where | Verification |
|---|---|---|
| Additive agent_json registration; native source END/schema/successful structured-event/context membership, provider digest/runtime/environment and22source files pinned | production/json_agent_adapter.py, production runtime/controlAPI |14native contract/refusal tests;2actualOllama live integration tests; declared defaults without a real call refused |
| Fresh isolated native turn, fail-closed finite object; stateless/maxBatch1; capture-on native context/state/events and new-call replay; capture-off hashes/usage | same, ProductionWorkspace.tsx |ActualChrome/nativecurl/live cancellation/bounds/privacy/idempotency/restart/replay; original source unchanged |
| Independent schema-valid object labels; sorted-key canonical JSON agreement with number representation retained; descriptive UTF8output size against recorded warmup; no model generation during monitor | production monitor/controlAPI |Actualblue9browser/blue7curl with independently supplied truth, native refusal and read-only evidence |
| Saved legacy identities preserved; new JSON version/request/source/provider evidence survives physical source deletion | Separate adapter identity; optional --json-agent native recovery |Pre-extension persisted conversation version still executes;71files/10DB/native tracker+old conversation+JSON browser recovery passes |

Bounds:16nodes/32KiBgraph,32turn-statefields/8KiBdefaults,1–4textinputs,12flat
schemafields/4KiBschema,8KiBfinite output,25steps/30s,1–2calls/4096total token
budget/128outputtokens per call. Prompt/set_state/one structured-output node only;
real installed local Ollama, think=false/on_failure=fail, retries within call budget.
Default objects cannot count as successful model evidence. Single source/warmup
reference is weak descriptive evidence; agreement is not semantic correctness.
No new dependency/database schema/weight change or legacy retraining is required.
JSON versions additionally pin shared production integration sources and therefore
require new registration if those pinned sources change.

Not implemented: JSON conversations, tools/retrieval/indexes/thread or long-term
memory/interrupts, remote/fixture/multiple providers, streaming or semantic quality
benchmarks. Default hosted Linux CI verifies native contracts/originalbrowser/
recovery but has no Ollama and does not claim JSON provider acceptance. The optional
JSON browser/recovery runners fail honestly without the actual installed model.


### Optional automatic cache retention (ADR0046; Mac verified, HANDOFF§43)

Opt-in saved-project policies, native dry-run/revision-guarded settings and actual
owner scheduling/receipts are verified on the Mac. Default disabled;
no policy database until explicit configuration. Native prune preserves recorded
artifacts and retained/shared cache bytes; queued/preparing/running/cancelling/paused
research runs defer checks under the owner submission lock. Controls use recorded
policy/revision snapshots; invalid editor IDs suppress background requests.
Native paused/resume/restart/auth/scope/artifact protection and physical source-deletion
recovery pass. Full suite1239pass/1skip/14deselected,14actualOllama live tests,
33Node tests/build/typecheck/ledger/source-pin audit pass. Real Chrome enables/prunes/
disables and preserves exact native graph/run records;13curlcases and all12original
editor regressions pass. Integrated137-file/11database recovery retains policy/receipts/
cache, old native conversations, real JSON serving and local/offline tracker evidence.
Invalid draft IDs issue no background policy requests. Default hosted CI adds the
new browser/recovery checks; its outcome remains separate from Mac acceptance.

One owning local process only; all writers must stop before offline backup. No
exactly-once or distributed guarantee, cross-workbench sharing, physical run/model
erasure or result-cache support for other graph families. No saved model/source
identity, native execution/cache semantics, dependencies or weights were changed.
The separate policy metadata transaction can lose a receipt after a process crash;
a later check recomputes actual candidates. See HANDOFF§43 for exact evidence.


### Measured large-graph selection (ADR0047; Mac verified, HANDOFF§44)

Declared native-valid100/500/1000node metadata chains, actual warm headlessChrome,
20nativeHTTPvalidation/10projectload/20outline-selection observations per size,
nearest-rank quantiles, exact graph/UI/hash/noUndo change and source/environment
provenance. Raw before/after evidence and limitations are in benchmarks/results/
editor_selection_summary.md. Stable canvas card/wire metadata changes500-node
outline-selection p95 from181→54ms;1000nodes still149.8ms. No load improvement
or full routine-editing/frame-time/heap/GPU/representative model/platform/user
certification. Native data/geometry changes still invalidate card/wire metadata.
Full1239pass/1skip/14deselected559.86s,14actualOllama live/33Node tests,280module
build/typecheck/ledger/source-pin audit, literalcurl matching native hashes/typed
ports, all12originaleditorjourneys and137file/11DB source-deletion recovery pass.
Existing native models/source pins/dependencies remain unchanged. Hosted outcomes
are recorded separately in§44; no whole-VISION completion is claimed.


### Selection-independent side panels (ADR0048; Mac verified, HANDOFF§45)

Checklists, keyboard/insertion dropdowns, inspector source options, outline rows
and the block library no longer rebuild per-node content on selection; graph,
operation or native-view changes still rebuild it. Same benchmark: outline-selection
p95 33.0/34.2/47.1ms at 100/500/1000 nodes (raw: benchmarks/results/
editor_selection_panels.json). No load, add/update, pan/zoom, frame-time, heap,
production-build or platform claim. Build/typecheck/33 Node tests and all 13 editor
smoke journeys pass; no native code changed.
