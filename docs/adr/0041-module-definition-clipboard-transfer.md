# 0041 — Explicit module-definition clipboard transfer

Extend the existing page-memory graph clipboard into stored PyTorch module
definitions. Select actual internal nodes, copy their internal wires and transitive
module/code/dependency closure, and capture the native module hash from the real
module validation endpoint. Display project, module/version and module identity
separately from root graph identity. Generated interiors and interface pseudo-nodes
are excluded. Opaque module-local state references refuse rather than guessing a rebind.

Paste into the same-kind/backend root or stored module with fresh node/wire identities.
Work directly on stored internal nodes/edges, retaining input/output/parameter lists,
original $in wires and every existing edge ID/metadata. Boundary wires are omitted and
counted, including declared module outputs; never infer new inputs or outputs. Missing
inputs remain actual native errors. A module paste intentionally edits the shared
definition used by every instance. Required same-version definitions must match exactly;
never overwrite conflicting source/dependency identities. Native weights are not copied.

Copy uses actual scoped origins or the same module topology layout as the canvas.
Only new target-scope position keys are added; unknown UI and original authored notes
stay intact. Orphan positions/note keys reserve identities for both root and module
pasting, preventing silent note reattachment. One whole-draft Undo/Redo restores exact
documents. Repeated pastes retain fresh IDs and declared offsets; scope navigation
keeps the existing history behavior. Compatible cross-scope snapshots are explicit.

Actual Node helpers are shared with Vite through .ts imports; allowImportingTsExtensions
is enabled only in the existing noEmit TypeScript configuration, with no new dependency.
Topology layout dictionaries use null prototypes for valid constructor node identities.
Native reference tests independently reconnect omitted boundaries, then compare native
parameter count, outputs, loss, gradients and SGD against Torch with independently
loaded equal reference weights. This does not claim clipboard weight transfer.

Required full/live/build/typecheck/coverage/curl/owned Chrome/recovery checks remain.
No backend/schema/native identity pins change; no saved-model retraining or reinstall.
Bounds remain100nodes/100definitions/512KiB estimated payload, page-memory only.
Agent/RL specialized transfer, generated instances, automatic boundary reconnection,
opaque state/artifact transfer, OS clipboard, remote collaboration and broad performance
or accessibility certification remain separate work.
