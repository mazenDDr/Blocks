# ADR 0002: Control service, inspection contract and editor architecture (Phase 1)

Status: accepted (Phase 1). Builds on ADR 0001; sources: PLAN.md Phase 1, VISION 8.3, 8.5, 13.1, 16.4, 17.1, 19.2.

## Decisions

1. **One FastAPI app (`services/control`)** over the Phase 0 modules. Run it with
   `uvicorn control.app:create_app --factory --app-dir services`; storage root is `VOID_WORKBENCH` (default `.workbench`).
   Projects are stored as `<workbench>/projects/<id>.project.json` + `<id>.ui.json` (reusing `project_io`).
2. **A run executes an immutable graph.** `submit_run` stores the exact graph as a `graph` artifact of the run.
   Inspection always uses that graph, never the project's current draft, so editing the canvas can never change what an
   old run's weights/activations mean (VISION 5.5, 8.5). The editor labels runs whose `graphHash` differs from the draft
   as "older architecture" and does not map their values onto the draft.
3. **Reading never trains.** The only route that spawns a worker is `POST /api/runs`. Weights, per-sample loss and the
   confusion matrix are values recorded during the run. Activations and inference are an explicit forward pass of the run's
   stored graph with a stored checkpoint; their provenance says so. Nothing from inspection is written back.
4. **Absent data is explicit.** Inspect/infer return HTTP 200 with `{available: false, reason, message, provenance}`
   (`no_run`, `not_recorded`, `no_parameters`) instead of fabricated or empty values. Malformed requests (unknown node,
   sample out of range, no such checkpoint step) are 422. Every available response carries run id, graph hash,
   checkpoint step/status/sha256, node id and (when relevant) sample index and id, plus source and normalization.
5. **Bounded slices.** Weights and activations slice along dim 0 (filters / channels) with `offset`/`limit`; the server
   clamps `limit` so at most 40,000 values are returned and reports `{offset, limit, of, truncated}` and full-tensor
   statistics (so the UI can use one colour scale across pages).
6. **Idempotent submit.** `Idempotency-Key` is required. The key maps to a run in SQLite together with a hash of
   (semantic graph hash, resolved run config). Same key + same request returns the existing run (200,
   `idempotentReplay: true`); same key + different request is 409. Submission is serialized with a lock.
   The editor keeps one key per (graph, config) until a submission returns, so a double click cannot start two runs.
7. **Events.** The worker keeps writing events straight to SQLite (ADR 0001). SSE reads the `events` table; `seq` is the
   SSE `id`, `Last-Event-ID` (or `?after=`) resumes strictly after it. The stream ends with an `end` event once the run is
   terminal and drained. A client going away only ends its own stream.
8. **Cancel is recorded in the database.** `POST /cancel` moves the run to `cancelling`; the worker checks the
   in-process event *or* that DB state at each batch boundary, so cancellation works even if the control process was
   restarted since submit. `queued -> cancelling` was added to the state machine for this. A run cancelled before its
   first batch ends `cancelled` with a step-0 `partial` checkpoint.
9. **Dataset split.** Seeded permutation (`split_seed`, default = run seed) and `val_fraction` are part of the run
   config; the worker also stores a `split` artifact (class list, train/val relative paths, val labels, dataset sha256).
   A validation sample id is its index into that recorded list. Per epoch the worker emits `val_detail`
   (confusion matrix, per-sample loss/prediction).
10. **SQLite access is serialized per process.** With several request threads and SSE pollers opening/closing
    connections concurrently, the control process deadlocked inside libsqlite3 (macOS, SQLite 3.51.1; seen in a
    thread dump). `ArtifactStore` now takes an in-process lock around every statement. The worker process is
    unaffected (SQLite file locking between processes).
11. **Editor state model.** The graph spec (`nodes`, `edges`) is the only copy of the architecture; React Flow nodes/edges
    are derived from it on every render and positions live in the separate UI doc. Every edit triggers a debounced
    `POST /api/validate` (stale requests aborted). Config forms are generated from each op's JSON Schema; there is no
    per-op form code. The pinned baseline run id is stored in the UI document (an extra field, not part of the semantic hash).
12. **JSON Schema export.** `python -m graph_core.export_schema` writes `packages/graph-schema/schema.json` from the
    `ProjectDocument` model (graph + ui). A test fails if the committed file is stale. Editor TypeScript types are
    hand-mirrored in `apps/editor/src/types.ts`.

## Consequences / limits

- Runs started by a control process that is later killed continue (separate OS process), but this service cannot
  reattach a cancel `Event`; DB-based cancel covers that. A run whose worker died abnormally stays in its last state.
- Inspection recomputes activations per request (no cache of tensors); small CPU cost, no stale values.
- No auth, single local user; dataset paths are server-side paths supplied by the client.
