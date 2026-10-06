# ADR0048: selection-independent side panels

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded separately in HANDOFF§45.

After ADR0047, selecting an outline node on the declared 1000-node chain still
took 149.8ms at p95. A CPU profile of ten selections in owned headless Chrome
showed the canvas was no longer the main cost. Every selection re-rendered side
panels whose content grows with the graph but does not depend on selection:

- arrangement, movement and copy checklists (one checkbox per node, three lists);
- keyboard tools node/wire/port dropdowns and insertion wire dropdown;
- the inspector's "source for" dropdown (one option per compatible output port);
  the inspector remounts per node by design, so component memo state is lost;
- the outline's 50 visible rows and the block library.

Decision: keep every panel's DOM, labels and behaviour unchanged and make the
per-node work depend only on what changed.

- `NodeChecklist` renders memoized rows with a stable toggle handler and reuses
  an unchanged row's element, so a selection re-renders only rows whose checked
  state changed. Arrangement, movement and copy lists use it.
- Keyboard and insertion option lists are memoized on the graph's nodes/edges/ops.
- Inspector source options are cached per wire kind in one module-level entry
  keyed on the exact nodes/ops/native-views identities; inspecting another node
  only filters out the node itself. The no-validation views fallback is a shared
  constant so the key stays stable.
- Outline rows are memoized with stable actions and primitive selected and
  diagnostic-inspectability props; the library is memoized with stable ops and
  handler.

Any change to graph nodes, edges, operations or native views still produces new
identities and rebuilds the affected options. No custom equality comparator hides
a data change. Saved documents, history, native validation and execution are
unchanged.

Measurement uses the ADR0047 benchmark unchanged except that it now also records
source hashes of every changed component. Raw evidence is in
benchmarks/results/editor_selection_panels.json. Outline-selection p95 at
100/500/1000 nodes is 33.0/34.2/47.1ms (ADR0047: 48.1/54.0/149.8ms). Two further
runs on the same code gave 1000-node p95 of 54.1 and 48.5ms. Project-load p95 at
1000 nodes was 1517.6–1668.2ms across the three runs against 1452.6ms before;
no load claim is made either way. The same single-laptop, warm dev-mode,
synthetic-chain limits as ADR0047 apply; add/update, pan/zoom, frame time, heap
and production-build performance remain unmeasured.

Functional acceptance: editor build, typecheck, 33 Node tests and all 13 editor
smoke journeys (12 original plus cache retention) pass with empty runtime/API/
console error arrays and stopped service groups. No native/backend code changed,
so the native suite is covered by hosted CI rather than re-run locally.
