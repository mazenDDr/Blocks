# ADR 0038 — Explicit insertion into an existing typed wire

2026-10-05. Final environment outcomes and acceptance: HANDOFF §35.

VISION §5.3 requests inserting a block into an existing connection. The editor now
requires an actual existing wire, installed static-port operation and explicit input
and output ports. Both ports must declare that wire's exact kind. Model choices use
the same declared backend; Python tabular/domain graphs may compose the native pandas,
scikit-learn, SciPy or domain operations belonging to that graph kind. Library backend
and registered defaults are visible. Structural/code operations with dynamic ports
remain outside this bounded picker.

Replace only the selected wire with two fresh wires and one node with cloned native
registered defaults. Other inputs remain disconnected; config/shape/dtype failures are
reported by native validation rather than fabricated auto-configuration. Stable
E_INSERT_SCOPE/OPERATION/WIRE/PORT/KIND/POSITION preflight refusals make no edit.
Refuse absent/ambiguous wires, duplicate producers or non-finite endpoint positions.
Fresh node IDs also reserve orphan note/layout keys, avoiding accidental reattachment.
Extension metadata follows the downstream wire to its original consumer only. All
other node/config/wire/definition/UI metadata bytes stay unchanged.

Layout stores only the new node at the midpoint of endpoint origins. Root fallbacks
use the existing indexed canvas positions; module fallbacks use its actual layered
layout. This is placement, not collision-free arrangement or auto-layout. Native
semantic identity changes; existing comments keep original provenance and therefore
refer to an earlier identity. No learned weights or opaque state are copied or changed.
The paired graph/UI action uses existing whole-draft history; one Undo restores exact
original documents, Redo restores the same node/edge IDs and position.

Root model/tabular/domain and stored module input/internal wires are supported.
Module input view endpoints translate back to declared $in ports; only the selected
stored edge and appended node change, preserving all other interface/definition data.
Output-interface display links are refused explicitly; changing those requires the
existing interface editor. Agent/RL/generated instance links, structural/code insertion,
sequence move/grouping, drag-on-wire suggestions and auto-layout remain separate work.

Native reference tests insert observation-only probes into CNN/root and reusable-module
input/internal paths: identical parameter counts, outputs, loss, parameter gradients
and SGD results with independently loaded equal Torch weights. Additional-input tests
require real E_MISSING_INPUT; typed fitted-state rejection uses actual native contracts.
Owned real Chrome/curl verify exact saved documents, Undo/Redo, native reports, module
scope preservation, tabular profile and labelled synthetic vision annotation wires.
Existing comments/history/clipboard/outline/arrangement/recovery regressions remain
required; no runtime/native code, dependency, DB/schema or model identity pin is edited.
