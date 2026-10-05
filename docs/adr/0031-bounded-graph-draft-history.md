# ADR 0031 — Bounded whole-document graph draft history

Accepted, 2026-10-05. Graph edits previously had no document undo/redo, requiring
manual reconstruction of accidentally changed settings, nodes, edges or layout.
Maintain immutable in-memory history of the authoritative graph plus its separate
UI document. Keep external operations and recorded research/serving evidence outside
this history; Undo never restores trained weights, deletes runs, changes a production
conversation or undoes tool effects.

A pure reducer holds current, past and future draft documents. Functional setters
resolve against the reducer's latest document, preserving React batching semantics.
Paired graph/UI setters in the same event share a transaction ID so adding or renaming
a node is one whole-document edit. Main and agent canvas position changes carry
React Flow's dragging flag; intermediate drag frames and final position share a
transaction. Selection, dimensions, equivalent/no-op documents and polling do not
create draft history. A new actual edit invalidates redo; no-op edits preserve it.

Retain at most100 prior documents and8MiB estimated serialized JSON UTF-16 size.
Evict oldest snapshots when needed. A larger current draft remains editable but may
have no undo snapshot. This is a bounded snapshot estimate, not a measured JS heap
limit. Documents/arrays are immutable and shared where unchanged; reducer comparisons
serialize the graph/UI, which can cost time on large documents. No performance claim
for very large graphs, persistent history, collaborative editing or cross-tab merging.

Project/example load or a new graph explicitly resets history, preventing cross-project
undo. Save updates the existing saved-document fingerprint and preserves draft history:
undo after save marks the restored draft dirty when it differs from that saved state.
Existing graph hashing/validation/save remain authoritative; layout stays excluded
from the semantic hash. History includes node/edge/config/module/code-definition,
agent schema, training/backend settings and saved UI changes using existing setters.
Transient local text in an unapplied editor, project name, selection, navigation,
run inspection or unrelated forms is not part of the draft.

Toolbar controls show disabled state when history is empty. Cmd/Ctrl-Z and
Cmd/Ctrl-Shift-Z (Ctrl-Y too) operate on the draft outside input/textarea/select,
contenteditable or CodeMirror focus; text editors retain native local undo. Production
and Integrations panes do not receive draft shortcuts. Undo/redo clears stale global
selection/module navigation/code-edit/run-inspection state and keeps immutable runs.
Native toolbar buttons remain keyboard accessible with explicit labels/tooltips.

Pure Node tests verify whole-document paired transactions, immutability, ordering,
new-edit redo invalidation, no-op retention, project reset and count/size eviction.
Actual Chrome saves and reads real API documents after node add, undo/redo and grouped
15-step drag, compares complete graph/UI bytes, checks new-edit invalidation, changes
PyTorch→JAX backend then restores the original settings, resets on another project and
preserves draft redo during focused text undo. The owned browser runner provides
screenshots/errors/cleanup evidence and is included in Linux CI alongside existing
native/recovery journeys. No backend model pin, dependency or schema migration.

Further editor copying/pasting, outline/comment/navigation and multi-node workflows,
large-graph performance, collaboration and persistent history remain separate work.
See HANDOFF §28 for actual acceptance and environments.
