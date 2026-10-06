# ADR0075: long-term memory owned by a release and a request user

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded in HANDOFF§79.

Served agents had only short-term memory (bounded thread state, ADR0064). Long-term
memory existed in research runs (`agent/memory.db`), but no release could read or
write it: ADR0064 required a separate, durable ownership and write contract first.

Decision: a serving family `agent_memory` (`production/memory_agent_adapter.py`,
candidate node `__agent_memory_graph__`) and memory owned by the production database.

- **Ownership.** Records live in `production.sqlite` table `release_memory`, keyed by
  release, request user and record id. Memory never moves between releases; a new
  release starts empty. With accounts (ADR0055) the user is the signed-in account
  (`ensure_owner`); without accounts it is the caller-declared user label, as for
  every other per-user serving state.
- **Isolation by construction.** Each turn runs the native graph against a fresh
  temporary memory store seeded with only the requesting user's records of that
  release. Other users' records and the research `agent/memory.db` are not present,
  so no policy or template can reach them.
- **Writes commit with the turn.** The pipeline returns the turn's new records; the
  runtime commits them in the same SQLite transaction as the successful request
  trace (`finish_request`). Failed, cancelled, timed-out, invalid and idempotently
  replayed requests write nothing. Turns of one user in one release are serialized.
  The trace lists the written record ids.
- **Bounds.** At most 200 live records per release and user: a turn that would exceed
  it returns 409 `E_AGENT_MEMORY_FULL` and stores nothing. One memory_write per graph,
  long-term, direct (no approval), scope `user`, ≤1000 characters; 1–2 memory_select
  nodes whose policies retrieve `long_term` only, scopes `["user"]`, k ≤ 8, local-hash
  embeddings, extractive summaries; stateless releases (no thread-scoped fields), no
  document indexes, tools or interrupts; model-free or bounded local Ollama.
- **Inspection and deletion.** `GET /api/production/releases/{id}/memory?user=` and
  `DELETE …/memory/{record}?user=` (owner or admin). Deletion removes the row and its
  text; the lifecycle keeps only ids. Editor: a memory panel in Production → Requests.
- **Erasure.** `python -m maintenance.erase` (ADR0056) deletes the user's
  `release_memory` rows with their other serving rows; a test checks the remembered
  text is gone from every workbench byte while another user's memory keeps working.
- **Schema.** `production.sqlite` gains its first real migration (version 1 → 2,
  ADR0066 machinery). A schema definition may now be an ordered tuple of step scripts.

Compatibility: `storage/schema.py`, `production/runtime.py`, `production/store.py` and
`services/control/production_api.py` are pinned implementation files of agent serving
families. Versions registered before this change must be re-registered (declared in
`tests/fixtures/serving_sources_adr0075.json` for the ledger-checked families).

Not provided: memory shared across releases or migrated between them, memory for
conversation or tool families, approval of memory writes in serving, model-generated
memory summaries, retention/expiry policies, export, semantic embeddings, multi-process
control services (one owner process), or a quality evaluation of what is remembered.

Verification: `tests/test_production_memory.py` (11: recall and per-user isolation,
research memory untouched although it uses the same namespace and scope, idempotent
replay, duplicate skipping, restart persistence, HTTP list/delete and owner scoping,
cap/cancel/deadline/failure/invalid-input commit nothing, version 1→2 migration
preserving rows, 8 contract refusals each with its intended code); browser journey
`tools/editor_release_memory_smoke.py` (source run → register → deploy → two users →
delete from the panel → recall reflects the deletion), and `tests/test_erase_user.py`.
