# ADR 0035 — Retain native measurements in controlled canvases

2026-10-05. Final evidence and environment status: HANDOFF §32.

Actual Linux Chrome undo/redo evidence showed finite geometry but all agent nodes
hidden. Installed React Flow12.12.0 adopts each rebuilt controlled node object's
`measured` dimensions; omitted dimensions clear the internal measurement. ResizeObserver
need not emit unchanged content again. Merely requesting measurement on membership
changes was insufficient because validation and selection also rebuild user nodes.

Retain only actual native dimension changes in transient React state. Model/module
and agent controlled nodes carry those dimensions; membership changes request actual
DOM/handle remeasurement and absent IDs are pruned. No guessed sizes and no dimensions
in serialized graph/UI, history, native identities or model artifacts. No dependency
upgrade or native source change is required.

A separate actual Mac browser failure exposed duplicate StrictMode asynchronous boot
requests adopting an old example after user interaction. Abort disposed requests and
check disposal before state changes/adoption; legitimate current boot remains intact.

Verification retains original actual graph/UI equality, full native drag/one-undo,
clipboard/reference identities, current module contracts/errors/pagination and native
source-deletion/recovery/tracker journeys. Failed diagnosis is retained separately
from final passes. These changes make no large-graph performance, universal browser
support, authenticated collaboration or saved-history claim.
