# ADR 0032 — Revisioned user-authored research run records

Accepted, 2026-10-05. The workbench records immutable run execution evidence but
previously offered no per-run notes/tags or searchable catalogue independent of a
study. Add authored metadata in a separate additive `research.sqlite` database;
leave execution tables/artifacts, graph hashing and native model identity pins intact.

A current annotation points to an append-only revision carrying a bounded note
(max5000 characters), at most20 unique trimmed tags (each1–60 characters), author
label and actual server recording time. Absence is revision0 with no invented author
or timestamp. Notes are user statements, never measured results. Clearing current
text/tags preserves earlier revisions and receipts; this is not privacy erasure.

PUT `/api/research/runs/{run_id}/annotation` requires a mutation ID, reviewed graph
hash/current metadata revision, author, note and tags. One native SQLite transaction
checks revision/identity, appends the revision, advances the current pointer and
records an immutable mutation receipt. Same-ID identical retries return the original
receipt after later edits/restart; different reviewed values conflict. Concurrent
writers with the same revision have one winner. Missing/corrupt current revision,
bounded metadata violation and graph identity mismatches refuse rather than show an
empty annotation. Transaction failure rolls all pointer/revision/receipt writes back.
These records are local administrative metadata, not a signed/tamper-proof audit log.

GET run record/revision-history return actual execution metadata, graph provenance,
current/history annotations and explicit policy. A bounded25/default,100/max catalogue
reads native run metadata via a read-only attached `meta.db`, joining annotations in
the same connection. Search is literal Unicode-casefold substring over run/project/
graph hash and current note/tags, not SQL wildcard or semantic ranking. Exact tag,
project, graph-family and run-status filters compose. Keyset pages order by literal
run ID (not inferred chronology). No artifact decoding, model/provider verification
or execution is performed. Current pages are not a frozen historical catalogue; no
claim of indexed full-text search or million-run performance. Historical annotations
are separately paged by numeric revision. No total count is invented.

Records appears beside the workspaces for all graph families. Filters are explicit,
editing them clears stale results, and failures clear old pages. Selection shows
recorded provenance, current metadata, editable note/tags/author, explicit save/reload
and revision-history controls. Saving uses the inspected graph/revision and reuses a
mutation ID for an identical retry. Conflict leaves draft text available for review;
reload explicitly replaces it. Open original run uses the existing native inspector
when its recorded project is currently open; it is disabled for other projects, whose
notes can still be read/edited. No cross-project graph restoration or fabricated run.
Draft-history shortcuts do not affect these external metadata operations.

Shared-token protection applies when configured. Token holders can edit all records;
author labels are caller-declared, not authenticated identity, ownership or roles.
Retention/deletion/GC, hosted/team authorization, signed evidence and encrypted storage
remain separate work. Revision/receipt storage can grow across edits; bounded individual
payloads/pages do not provide automatic retention. No native compatibility bypass.

Tests create actual completed LangGraph teaching runs, preserve final state/events/
artifacts/status byte-for-byte, exercise native concurrency/optimistic review/retry/
rollback, Unicode/literal filters/pages and corruption/schema/token refusals. Offline
recovery deletes the generated source workbench and verifies current/history/old
receipts from the restored SQLite file. Actual Chrome source worker→authored revisions→
search→original inspector and backup/source deletion/restored Records compares exact
native metadata/revisions alongside existing conversations/discovery/tracker journeys.
Backup discovers the additive SQLite database automatically; older snapshots remain
valid and startup creates an empty metadata database when needed. See HANDOFF §29 for
actual environment and verification, including pending hosted checks.
