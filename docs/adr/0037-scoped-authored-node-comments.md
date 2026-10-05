# ADR 0037 — Scoped user-authored node comments with native identity references

2026-10-05. Final environment verification and acceptance state: HANDOFF §34.

VISION §5.3/§19 requests comments and research annotations. Run records exist already;
add explicit node comments to root model/tabular/domain and module-definition views.
These are project UI metadata, not graph configuration, native measured values or an
authenticated audit. No backend DB, dependency, native pin or graph schema changes.

Store `nodeComments` by root node ID or module@version/node ID. Each comment records
text, author label, browser-declared UTC time and the actual current validation hash
(graph or module) plus original node/type/version. Apply is disabled while native
identity is unavailable/pending. A saved comment remains visible after configuration
changes but is labelled as an earlier identity. Explicit review can retain its text
while recording the newly reviewed identity; layout changes do not change identity.
Save project persists metadata through the existing native UI document API.

Text is bounded to4000JavaScript UTF16 units, author120units,200comments and2MiB
estimated JSON metadata per project. A literal simple-lowercase catalogue search and
25-row pages provide actual-target inspection and explicit orphan removal. Malformed
record/collection metadata remains visible and unchanged until explicit removal;
writing refuses invalid records/oversized collections without silent eviction. Own
property checks keep valid native IDs such as `constructor` separate from JavaScript
prototype properties. React text rendering escapes comments rather than executing
HTML. Unknown record/source fields and unrelated UI metadata are preserved on update.

Rename preflights target-key collisions before editing the graph. A comment follows
its renamed key with original captured provenance intact, so identity is now earlier.
Delete removes the node's comment together with graph/layout in one existing history
edit. Undo/Redo restore exact source hashes/text/timestamps, not newly generated notes.
Malformed whole collections refuse rename/delete until reviewed rather than crash or
lose data. Explicit malformed-collection removal is reversible UI metadata only.
Clipboard transfer deliberately retains comments with source nodes rather than silently
assigning their authorship/provenance to newly generated IDs. Module comments belong
to a shared definition, not an individual runtime instance.

Actual Chrome saves/reads native graph/UI documents, compares unchanged graph hashes
and layout/configuration, escaped Unicode/HTML text, native source and module identities,
current/earlier review, rename/delete/copy/Undo, literal search, orphan-key conflicts,
malformed metadata, pagination/200-entry no-eviction refusal and valid prototype-name
IDs. Native backup/source deletion/restore preserves real UI-authored comments and
their original native references alongside the existing conversation/records/trackers.
Real curl checks exact API data and validation; required full/live checks are retained.

Author/time/reference data is editable local metadata, not signed/authenticated audit
or evidence of model quality. Older text may remain in Undo/Redo and offline backups;
remove does not promise physical privacy erasure. Agent/RL/expanded runtime-instance
comments, run/sample/plot cross-links beyond existing run records, collaboration,
authenticated roles, append-only annotation revisions and general note export/reporting
remain separate work. Pending/failure evidence is retained separately from passes.

The constructor regression additionally required own-property access for native node
reports and saved positions. Root/module validation hooks associate each response with
its exact immutable draft/request, immediately withholding older identities before
effects run and ignoring aborted success callbacks. UI-only edits retain graph identity.
Final original history/outline/clipboard/recovery journeys still pass with these guards;
no native contract changed. Environment outcomes and retained failures are in §34.
