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

## Not implemented

- Connectors/studies (2b gaps): other databases, warehouses, GCS/Azure, document/vector stores, scientific formats, repositories and model registries; a private-network connector agent; cost/transfer-size estimates; incremental reads; writes/destinations;
  schema-aware SQL completion, query history, explain/cost; windows in the query builder; DVC private-remote authentication helper; parallel trials, quotas, pruning/early stopping, conditional search spaces, an ablation builder (replace/bypass/freeze/remove),
  parallel-coordinate and parameter-importance views, smoothing/x-axis choices for metric curves, a research record sheet and report snapshots, tags/notes/search on runs, MLflow/W&B adapters.
- Tabular: cross-validation (fitting inside each fold), search spaces, held-out test partition and its "final evaluation" designation, sample weights, target/feature roles on the source (done on the estimator),
  Parquet/JSON sources, near-duplicate detection, stratified/temporal splits beyond stratify/group, feature construction, trees/forests/boosting/SVM/kNN/clustering/dimensionality reduction,
  baseline-vs-alternative comparison on one partition definition, regression plane / projection views, streaming or sampled profiles for huge tables.
- Statistics: descriptive-statistics block, parameter estimation, interval-estimation block, bootstrap/permutation tests, power analysis, multiple-comparison procedures, other distributions,
  special-function curves with singularities, generated samples, a two-sided tail rule on the generic tail block, the interactive "Explore" slider for the observed statistic, visual statistical programming (custom statistics from primitives).
- Editor: layout is a left-to-right row layout, large graphs are small at the initial fit; no automated browser tests in the repo.

- Gradients tab and captured gradient information (VISION 8.3), saliency, activation distributions over training.
- Editor: undo/redo, copy/paste, grouping, auto-layout, comments, structured outline view, multi-select move as a unit,
  insertion into an existing connection, interactive convolution teaching mode (8.4), partial-weight transfer (8.5.6),
  resource estimates, "resume compatible checkpoint". Automated browser tests.
- Pause/resume, exact resume from a checkpoint, heartbeats and leases; a run whose worker died abnormally stays in its
  last state; cancel cannot terminate a worker that is stuck inside a batch.
- Experiment board: tags, notes, search/filter of runs, smoothing, x-axes other than step (model graphs keep the pin + two-run compare; tabular/model sweeps have the Experiments board above).
- Auth, multi-user; image-folder datasets are server-side folders (tabular graphs can read connected sources).
- Dependency lock, data manifest, project bundle export (only graph + UI files are saved).
- Milestones 4-8 (LangGraph, RL, other backends, serving), A09, A14-A15, A17-A20, A25-A27, A30-A32, A34, A38+ not listed above (Milestone 3 rows are above).
- Broadcasting, ops beyond the 14 listed, dtype casts, multi-input/output training graphs, losses other than
  CrossEntropy in the worker, schedulers, gradient clipping/accumulation, mixed precision, GPU.
- Model/parameter policies (initialization, freezing, sharing, regularization), conv data type/device placement.
- Data: image folders and (tabular graphs) local CSV files only.
- Loss and optimizer inspectors, response curves; optimizer state beyond checkpoints.
- Other backends (Keras, JAX, ...), LangGraph/LangChain, RL, statistics, connectors, serving, code blocks,
  registry, sweeps, connectors.
