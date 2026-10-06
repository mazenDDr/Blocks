# ADR0049: agent graph clipboard

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded separately in HANDOFF§46.

Model, tabular and domain graphs could copy and paste nodes (ADR0033/0041); agent
graphs could not. An agent node is not self-contained: it reads and writes state
fields, leaves through fixed transitions or a conditional route, may wait in a
join, and may name an index or memory policy.

Decision: an agent clipboard in page memory, separate from the model clipboard,
selected through its own checklist (the agent canvas stays single-select).

Copy validates the exact snapshot natively and takes each node's state reads and
writes from that report, never from an editor guess; a node without native access
refuses with E_CLIPBOARD_PENDING. It copies:

- node configuration;
- fixed transitions between selected nodes or to END;
- conditional routes whose default and every case target are selected or END,
  plus their predicate fields;
- joins whose node and every awaited branch are selected;
- the exact definitions of every state field, index and memory policy used.

Anything crossing the selection (START entries, transitions, routes, joins) is
left out and counted, never retargeted, so no copied path silently changes.

Paste assigns fresh agent-valid IDs (reserving START/END, existing nodes, layout
and comment keys), rebinds transitions/routes/joins to the copies, keeps END, and
merges definitions by name: missing ones are added, identical ones reused, and a
differing one refuses with E_CLIPBOARD_DEFINITION_CONFLICT before any change. It is
one history edit. Pasted nodes have no entry, so native validation reports
E_UNREACHABLE_NODE until the user connects them; that is expected and visible.
Runs, traces, memory records and index contents are never copied. Bounds match the
model clipboard: 100 nodes, 512 KiB estimated JSON.

RL graphs are edited through fixed environment/learner forms rather than a free
node canvas, so node copy/paste does not apply; transferring RL settings between
projects remains separate work.

Verification: 5 Node unit tests (rebinding, omission counts, cross-graph merge,
conflict, refusals, repeated paste) and an owned Chrome journey on the native
serving_state example: copy two nodes with a route, same-graph paste (route
rebound, only unreachable-entry errors, natively valid once START connects, one
Undo restores the saved baseline), cross-project paste adding exactly the five
used state fields with identical definitions, and refusal on a differing field
without a draft edit. Build, typecheck, all Node tests and every existing editor
journey also pass.
