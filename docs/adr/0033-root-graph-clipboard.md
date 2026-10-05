# ADR 0033 — Bounded root graph draft clipboard

Accepted implementation, 2026-10-05; final release verification in HANDOFF §30.
Copy/paste operates on root model, tabular and domain draft graphs in page memory.
It does not invoke models, copy weights/files, use the OS clipboard or import an
arbitrary clipboard document. Reloading clears it; loading a different project
retains the copied snapshot. Agent/RL control state and module editor/generated
boundary nodes are excluded until their separate state/reference contracts exist.

Copy explicitly selects 1–100 actual root nodes, preserves configuration/unknown
node and wire fields, internal typed edges and relative layout, and reports omitted
boundary wires. It includes transitive versioned module/code definitions and their
origin/fixture metadata. Entire source package dependency declarations are retained
conservatively, even when an unselected node also requires them; no implicit install.
Definition closure is bounded to 100; total copied payload is at most 512 KiB estimated
UTF-16 JSON. This bounds the payload, not heap usage or large-graph performance.

The source provenance is the captured project ID and native validation identity of
that exact graph snapshot. A copy may contain unknown/invalid operations, preserved
for review; successful copying does not imply native executability. It captures no
run/weight identity. New nodes/edges receive collision-free IDs. Each paste offsets
positions by another 80 canvas units. Only the same graph kind/backend is supported;
there is no conversion or automatic external connection/placement guarantee.

Root `sharedWith` and composite `config.share` source instances must be selected and
are rebound to the fresh copied IDs, preserving sharing inside the new group while
keeping it independent of the original group. Repeat's declared tied/clone policy
and local module definition IDs stay intact. The teaching draft convention
`stateRef: model/<this-node-id>` is explicitly rebound to `model/<fresh-id>`; other
root references, and any state references inside required definitions, refuse.
These references do not transfer native saved state. No guessed arbitrary reference
or cross-group parameter tie is introduced.

Same id/version module and code definitions are reused only if their canonical JSON
content matches; mismatches refuse without altering the target. Required package
operation declarations also refuse conflicting identities. Root training/global
settings and arbitrary root metadata are not imported. Node configuration file and
connector paths remain declared references to the same resources: nothing downloads,
clones or relocates them. Native validation and the existing run-time checks decide
whether those resources/operations are available after paste.

Explicit labelled checkbox selection/copy/paste/clear controls live beside keyboard
graph tools. Incompatible paste and module scope give visible reasons. Copy awaits
native provenance and retains the captured snapshot even if the user switches
projects meanwhile. Paste updates graph and layout synchronously as one document
history transaction. Undo/redo affects the whole draft insertion only, leaving all
runs, saved weights, source project and external records unchanged.

Pure Node tests cover identity collisions, disconnected boundary edges, unknown field
preservation, sharing/refusal, recursive definitions/code provenance, package identity
conflicts, opaque state and payload limits. Native pytest invokes the actual TS helper
on bundled SYNTHETIC teaching graphs. CNN/residual graphs match native shapes/counts,
outputs/loss/gradients and SGD using equal independently loaded reference parameters;
no parameter storage is shared across original and copied models. A shared encoder
retains internal ties in a new independent group. Tabular/vision/NLP/speech native
validation retains typed contracts/configs. Partial/unknown graphs still report
native stable diagnostics. Actual owned Chrome verifies API-saved graph/layout bytes,
provenance, fresh typed edges, one undo/redo, repeated collision-free paste, declared
boundary omission/native refusal, incompatible kind and clipboard clearing. Hosted
CI records only test evidence, not workbenches, weights, CAS or provider credentials.

Module-internal copying, agent/RL state transfer, serialized/system clipboard,
collaboration, arbitrary cross-backend conversion and performance guarantees remain
separate work. This scope closes root draft duplication, not those larger features.
