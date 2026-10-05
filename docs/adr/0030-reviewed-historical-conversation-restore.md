# ADR 0030 — Reviewed historical conversation restoration

Accepted, 2026-10-05. Successful native conversation requests already retain END
checkpoints, including when input capture is off. Users can now inspect and restore
one of those checkpoints within its exact release/user/session/version. Earlier
request evidence and native model identities remain unchanged.

GET release `/conversation/history/{request_id}` verifies the successful completed
request, trace CAS, scope/version, producing request/revision, checkpoint CAS and
thread provenance, then loads the whole native state through the original pinned
adapter. It is read-only and makes no model call. No arbitrary checkpoint SHA,
upload, failed/incomplete turn, cross-session/release/user/version import or migration.
Provider/environment/code identity checks still apply even for read-only inspection.

POST release `/conversation/restore` requires an action ID, bounded reason, reviewed
current revision/checkpoint identity (explicit null after a logical reset), and the
selected historical request ID/trace SHA/checkpoint SHA. Serialize with normal turns
and reset/fork using the same session lock/deadline. Recheck the live head after
waiting. Verify and clone the selected native END checkpoint with the installed
serializer; rebind only its thread identity to a fresh UUID. Preserve channel values,
versions and historical message/context provenance. The existing 128 KiB bound
applies. Restoration itself executes no graph or model.

One SQLite transaction rechecks the current head and successful historical request
pointer, verifies source CAS bytes, advances the serving revision monotonically,
publishes the cloned checkpoint with no producing request yet, and records the
immutable action receipt/lifecycle event. The receipt's `sourceHead` is the reviewed
live head before restoration; `sourceRequestId`, `sourceTraceSha256` and
`sourceCheckpointSha256` separately identify the historical source. Existing
reset/fork receipts and policies remain unchanged. Same-ID identical retries return
the original receipt even after subsequent turns/restart; conflicting reuse refuses.
Errors, deadline expiry, transaction rollback and process death before commit leave
the live head unchanged. Unreferenced CAS bytes can remain, as with existing actions.

The next native turn starts from the restored historical state under its fresh
thread while advancing the serving revision. Research threads, old traces/checkpoints,
other sessions and recorded replay retain their earlier evidence. This is deliberate
branching of history, not erasure or rollback of immutable serving observations.

Production Requests requires an explicitly inspected current head and a selected
successful matching recorded request. Preview shows actual source state and hashes;
a reason and unchecked confirmation are required before restore. Scoped component
keys clear review on current-head, source or scope changes. Busy controls prevent
scope edits during actions; backend comparisons remain authoritative. Empty reset
heads can be restored. No guessed state, placeholders or automatic action.

Tests use actual native reducers and independent research state as reference,
capture-off/restart/idempotent lineage, corrupted/source-pointer/stale-target/scope
refusals, competing and active turns, deadlines, shared token, rollback and real
process death. Live Ollama verifies exact restored provider-request history without
asserting generated wording. Recovered Chrome exercises preview/confirmation,
restoration and a real native next turn after source deletion. See HANDOFF §27 for
actual final outcomes. No dependencies, DB schema or native identity files changed.

This is one control process, one local replica, with caller-declared isolation keys.
Authenticated ownership/roles, physical privacy deletion, retention/GC, signed or
encrypted backups, cross-version migration and general production agent effects,
retrieval, memory, interrupts or streaming remain separate scopes.
