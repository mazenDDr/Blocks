# ADR 0004: Connectors, source snapshots, studies and sweeps (Milestone 2b)

Status: accepted. Builds on ADR 0001 (artifact store, worker), 0002 (control service) and 0003 (tabular graph kind). VISION 7.5-7.10, 19, 21; acceptance A17, A39-A43, A47, A48.

## Decisions

1. **Connections are a registry, secrets are references.** `connections.db` (SQLite, in the workbench) stores the non-secret settings of a PostgreSQL, S3-compatible or DVC
   connection plus *references* to secrets: `{kind: env, name}` or `{kind: file, path, key}` (a JSON object in a file that must live outside the repository and the workbench).
   Settings models forbid unknown fields, so a literal password/key is rejected (422 `E_SRC_SETTINGS_INVALID`). Values are resolved in the process that needs them (control
   service or worker), are never written to the registry, API responses, events, artifacts, snapshots or exports, and error messages are redacted. The API shows only the reference
   kind, the env name or file key, and whether it currently resolves. `GET /api/projects/{id}/export/bundle` lists required connections (settings, names of the secret *fields*) and pinned
   snapshots, but no reference details. Tested by scanning every API response and every file in the workbench for the secret value.
2. **Source failures are stable, recoverable codes.** `E_SRC_AUTH`, `_PERMISSION`, `_NETWORK`, `_NOT_FOUND`, `_QUERY_TIMEOUT`, `_QUERY_INVALID`, `_TYPE_MISMATCH`, `_READONLY_VIOLATION`,
   `_SECRET_UNRESOLVED`, `_BOUNDS`, `_UNSUPPORTED`, `_SNAPSHOT_*`, `_SETTINGS_INVALID`, `_ERROR`, each with an HTTP status, a hint and `recoverable: true`. Connections, projects and runs are never
   mutated by a failure; static validation reports the same code on the node. PostgreSQL login failures carry no SQLSTATE in psycopg, so those are classified from the server's FATAL text.
3. **Previews are bounded at the source.** Rows are capped by SQL (`LIMIT`, hard cap 200) inside a `READ ONLY`, `REPEATABLE READ` transaction with a `statement_timeout` (default 5 s, max 30 s);
   S3 previews use ranged reads (<= 1 MiB) and listings are paged (<= 200 keys); DVC previews read the head of the file. The response reports the bounds that were applied.
4. **The visual query builder compiles to parameterized SQL.** `QuerySpec` (base table, columns, typed filters, joins, group-by/aggregates, ordering, limit) compiles in a pure function:
   identifiers are quoted with the PostgreSQL rule, every value is a bound parameter with an explicit cast (typed comparison), operators and aggregate functions come from whitelists, the builder
   never emits `SELECT *`. The compiled text is shown read-only and is exactly what is executed. Raw SQL is a separate mode: one statement wrapped in `SELECT * FROM (...) LIMIT n`, read-only
   transaction, statement timeout, `%(name)s` typed parameters. A leading-keyword check is *not* used as the security boundary; the transaction mode is. In-database joins get a cardinality / unmatched-key report
   computed in PostgreSQL in the same snapshot transaction.
5. **Every run resolves each connector source to a recorded snapshot.** A snapshot manifest (canonical JSON, content-addressed in the artifact store, its sha256 is the snapshot id) is attached to the run
   as a `source_snapshot` artifact, with a `source_snapshot_recorded` event:
   * PostgreSQL: query text, parameters, server version, `pg_current_snapshot()` token, isolation, extraction time, schema with type conversions, row count, SHA-256 of the result CSV, ordered/limit flags, join
     reports. The extract is stored (`source_extract`), so the result is *materialized*: repeating from the snapshot reads the stored extract. The live database is not claimed to be unchanged; a drift check
     (`POST /api/snapshots/{id}/check`) re-runs the stored query and compares hashes.
   * S3: bucket, key, **versionId** (versioned bucket), ETag, size, content SHA-256, endpoint. Pinned reads re-fetch exactly that version and verify the hash. For an unversioned bucket the manifest says
     `limited: true`, "NOT VERSIONED - REPRODUCIBILITY LIMITED", and a copy of the bytes is stored so the run can still be repeated. Prefix listings are always materialized and flagged limited.
   * DVC: repository, requested revision, **resolved Git commit**, path, DVC md5 (from DVC's own index) and content SHA-256. A branch/tag is resolved once per run; pinned reads use the commit and verify md5 and SHA-256.
   A node's `pin` (or a run's `source_pins`) makes the node read that snapshot instead of the live source; a pin recorded for a different query/object/path is refused (`E_SRC_SNAPSHOT_MISMATCH`), never substituted. Hashes verify identity; they do not preserve data the source later deletes (DVC/S3 pinned reads depend on the remote still holding the object).
   Static validation reads the schema from the pinned manifest, otherwise probes the source (`LIMIT 0`, ranged read) with a 30 s cache. Validation therefore touches the network for unpinned connector nodes.
6. **Cross-source join is an ordinary tabular node.** `tabular.join` runs in the worker on tables already moved there and says so (`execution.where`, data moved, source lineage). It reports key cardinality (from key uniqueness on both sides),
   duplicate and NULL keys, unmatched rows with samples, and row multiplication; `expect` turns a violated cardinality into `E_JOIN_CARDINALITY`. NULL keys never match (SQL semantics, unlike a raw `pandas.merge`), key dtypes must agree (`E_JOIN_KEY_TYPE`), nothing is coerced.
7. **Studies are bookkeeping on top of ordinary runs.** `studies.db` holds studies, trials and attempts. A trial = one resolved configuration (assignments to node-config or run-config fields, including dotted paths into nested config)
   x one repeat identity (**seed**, **fold**); an **attempt** is one run of a trial (retries add attempts to the same trial). They are separate fields end to end (trial table, run config `trial`, events). Plans (grid or seeded random) are expanded and validated
   before anything runs: the trial count is checked against an explicit `limits.max_trials` (hard cap 200) and rejected, never truncated; variants whose graph or run config fails validation are stored as `invalid` with diagnostics and are not scheduled.
   Concurrency is fixed at 1; trials run sequentially in the control process through the same worker-process path as manual runs. Failed trials stay in the record; retry is explicit and limited by `max_attempts`; cancel marks the rest cancelled.
   Seeds default to the run-config field `seed`: for tabular graphs a new `TabularRunConfig.seed` replaces the `seed` of every node that has one (recorded in `run_started.seedApplied`); folds are a new k-fold mode of the split node (`n_folds`, `fold`).
   Objective metrics carry semantics: tabular = one evaluation of the fitted model on the validation partition (no steps, n rows, evaluated row-id hash); model = `last|min|max` over per-epoch evaluations with the step and epoch they came from.
   Repeats of one configuration are aggregated as mean and sample standard deviation across completed repeats; the baseline is the baseline configuration's repeat mean, or one pinned run. Deltas are direction-aware; configurations that change several fields carry an attribution warning, and
   `GET /api/runs/{a}/diff/{b}` gives the structural graph diff, run-config diff and recorded source-identity diff.
8. **Local test services are real engines, labelled.** Tests and the example use PostgreSQL 16 (`pgserver`), an S3 API mock (`moto` server) and a Git+DVC repository with a local DVC remote. `pgserver` has no CPython 3.13 wheel; its wheel is pure Python plus PostgreSQL binaries, so
   `python -m connectors.install_pgserver` installs the CPython 3.12 wheel re-tagged `py3-none-<platform>`. moto does not enforce IAM, so S3 permission/credential codes are verified on the classifier with synthesized botocore errors, not against a policy.

## Consequences and limits

* No automatic retry, pruning, early stopping, resource quotas, parallel trials, parallel-coordinates/importance views, ablation builder (replace/bypass/freeze/remove) or conditional search spaces; only grid/random over explicit fields and seeds/folds.
* Cross-validation inside a model (fit per fold with nested selection) is not implemented: a fold is one validation partition per trial; training/selection/final-evaluation separation is the user's graph.
* DVC: local-path or URL repositories through `dvc.api.DVCFileSystem`; revisions can be listed only for local checkouts; no authentication helper for private Git remotes; remote credentials come from the repo's DVC config.
* PostgreSQL types: numeric becomes float64, timestamptz becomes UTC naive, other exotic types become text (all recorded per column). An empty string and NULL both read back as missing from the stored CSV extract.
* No cost/transfer-size estimates beyond rows and bytes read; no pushdown of joins across sources; no incremental reads; no private-network connector agent; the registry is single-user.
* Studies are not yet linked to a "research record" sheet (question/interpretation/report snapshots); notes and hypothesis are stored on the study only.
