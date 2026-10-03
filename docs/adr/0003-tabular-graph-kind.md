# ADR 0003: The `tabular` graph kind (data preparation, classical ML, statistics)

Status: accepted (Milestone 2a). Builds on ADR 0001/0002. Sources: VISION 4 ("Different graph semantics must remain explicit"),
7.1-7.4, 11, 23 (Milestone 2), 24 (A05, A25, A26, A27), 26. Connectors (PostgreSQL, S3, DVC), studies and sweeps are Milestone 2b and are not part of this.

## Decisions

1. **A new graph kind, not new ops in the model graph.** `graphKind: "tabular"`, `backend: "python"` (pandas, scikit-learn, SciPy). The model
   validator rejects tabular ops (`E_OP_GRAPH_KIND`) and vice versa; `lower_graph` and the PyTorch export refuse a tabular graph
   (`E_UNSUPPORTED_GRAPH_KIND`). `validate(graph)` dispatches on `graphKind`, so every consumer (API, editor, worker) keeps one entry point and one
   diagnostic contract `{code, severity, nodeId, port, path, message, fixes[]}`.
2. **Wires are typed values, not tensors.** Edge `kind` is one of `table`, `fit_state`, `model`, `metrics`, `distribution`, `number`, `tail`, `test_result`.
   Every port declares its kind; `E_PORT_TYPE` rejects a wire between different kinds and `E_EDGE_KIND` an edge whose declared kind is not the producer's kind.
   A fitted state therefore cannot be mistaken for a table, and a tensor wire cannot appear in this graph kind.
3. **Static types carry facts, not data.** A `table` type holds the column names/dtypes, a row count when it is knowable without data
   (a CSV source's header/rows; split sizes by `n_test = ceil(f*n)`), the **partition** (`full`, `train`, `validation`) and the id of the split node. Where the schema depends on
   fitted data (one-hot categories) the type is marked `columnsComplete: false` with a `pendingPrefixes` rule, so unknown names are still rejected but columns that will exist are accepted.
   Validation reads only the CSV (cached by path/mtime/size); it never fits or trains.
4. **Leakage policy (A05) is a validator rule.** Fit nodes (`fit_standardize`, `fit_onehot`, `fit_impute`) and estimators accept only a table whose partition is `train`.
   `E_LEAKAGE_FIT_ON_HELDOUT` (a validation-partition table) and `E_LEAKAGE_FIT_BEFORE_SPLIT` (an unpartitioned table) are errors with the node/port path
   (`/nodes/<id>/ports/train`) and a fix that names the split node's `train` output. They block execution; there is no override switch. `W_METRICS_ON_TRAIN` /
   `W_METRICS_ON_UNSPLIT` are warnings (a scientific caution, not a deterministic error). The worker repeats the check at run time as defence in depth.
5. **Fit/apply are separate nodes; fit state is owned by the training partition.** `fit_*` produce a `fit_state` (sklearn object + JSON record with `fittedOn`:
   split node, partition, row count, hash of the training row ids); `apply_transform` applies it to any partition without refitting. Models likewise record `fittedOn`.
   The recorded fit state, the split's row-id hashes and the data file hash let a run be audited for correct ownership (tested).
6. **Native library semantics.** `StandardScaler`, `OneHotEncoder`, `SimpleImputer`, `LinearRegression`, `LogisticRegression`, `train_test_split` / `GroupShuffleSplit`,
   `sklearn.metrics`, `scipy.stats` (`gamma`, `ttest_ind`, `ttest_rel`, `mannwhitneyu`, `levene`, `shapiro`), `scipy.special.gamma`. Controls mirror the algorithm
   (OLS has no learning rate or epochs). Tests compare against hand-written scikit-learn / SciPy calls.
7. **Statistics blocks keep the VISION 11 distinctions.** Gamma *function* and Gamma *distribution* are separate blocks; the distribution takes an explicit
   `shape_scale` or `shape_rate` parametrization. A tail probability is the survival function / CDF (cross-checked by numerical integration and, for integer shape,
   the closed form `exp(-t/θ) Σ (t/θ)^j/j!`), the density value is shown separately and labelled "not a probability". A hypothesis test is a structured result (null, alternative,
   statistic, p, alpha, decision, rule, critical value, assumptions). The p-value is computed upstream of alpha, so changing alpha changes only the decision and critical value (A26).
   Only one-sided tails are offered on the generic tail block (a two-sided p needs the method's own extremeness rule). The two-group block requires a declared **sample unit**,
   shows pairing/independence, refuses repeated units in an independent test (`E_REPEATED_UNITS`), and reports the mean difference with SciPy's confidence interval,
   Cohen's d / Hedges' g (pooled-SD form for Student, root-mean-square-SD form for Welch, d_z for paired) or common-language/rank-biserial effect for Mann-Whitney.
   Diagnostics (Levene, Shapiro) are shown beside the result and never choose the method.
8. **Runs reuse the existing worker/store path.** `POST /api/runs` with a tabular graph (config `{}`; `TabularRunConfig` forbids unknown keys) spawns the same
   `spawn` process; the worker (`worker/tabular_run.py`) executes nodes in topological order and appends events (`run_started` with library versions, `node_started`,
   `node_finished`, `source_recorded` with the file SHA-256, `split_recorded` with seed and row-id hashes, `node_failed`, `run_finished`). Each output port is stored as a
   content-addressed `node_output` artifact (tables as CSV indexed by source `row_id`, fit states/models/results as JSON) and each node's summary as a `node_summary`
   artifact, all with node, port, value kind and graph hash. The run record lists nodes with their status, so a failure names the node and a stable code.
9. **Inspection is read-only and bounded.** `POST /api/runs/{id}/inspect` gained kinds `table` (offset/limit, clamped to 200 rows, with provenance), `profile`,
   `fit_state`, `coefficients`, `metrics`, `test_result`, `distribution`, `tail`, `number`, `summary`. A view is rejected (422) when the node does not produce it; a node
   that did not run yields `{available: false, reason: "not_recorded"}`. `GET /api/runs/{id}/tables/{node}/{port}.csv` downloads a recorded table (e.g. exported predictions).
10. **Registry/API additions.** Registry entries carry `graphKind`, `summaryKind`, `inputKinds`, `outputKinds`; `/api/examples` and `/api/projects` return `details`
    (graph kind, description, synthetic flag). Run summaries carry `kind` (`model` or `tabular`).
11. **Editor.** The library, node cards, inspector tabs, wire inspector and run panel switch on `graphKind`. Tabular cards show the output schema (columns/types), row count,
    partition and wire kinds instead of tensor shapes. Result views render recorded values only; the density/tail plot is drawn from the arrays SciPy computed.
    Teaching and synthetic data are labelled in the project notice and in the result.

## Consequences / limits

- Row counts are claimed statically only where they are exact; after duplicate removal, row dropping or grouped splitting they are shown after a run.
- The CSV source is a local file; tables live in memory in the worker and are stored in full as artifacts (no streaming/sampling profile yet, profiles are exact).
- Cross-validation, search, baselines-vs-alternatives on one partition definition, test partition, sample weights, trees/forests/boosting, resampling and multiple-comparison blocks are not implemented.
- Fit state ownership is enforced at node level; a table's `train` partition is trusted to come from a split node in the same graph.
