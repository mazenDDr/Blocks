# 0040 — Explicit selected-group layout movement

VISION §5.3 asks for moving selected nodes together. Provide explicit horizontal
and vertical canvas offsets for 1–100 unique actual layout cards, using the existing
shared selection and whole-draft history. Root model/tabular/domain graphs and stored
module definitions use their actual scoped positions; absent positions use the same
origins as the canvas. Only selected UI position keys change. Existing configuration,
wires, interfaces, unknown UI metadata and authored provenance remain intact.

Each selected origin receives the same declared offsets. Validate all current and
resulting coordinates and offsets as finite, bounded within ±1,000,000 layout units,
before creating any edit. Stable E_MOVE_SELECTION/E_MOVE_DELTA/E_MOVE_POSITION errors
refuse ambiguous, oversized or out-of-bounds requests without partial mutation.
Empty offset inputs are invalid. A zero offset returns no keys, materializes no
fallback position and adds no history entry. One Undo/Redo restores the exact UI.

Expanded root module interiors have generated coordinates. Require explicit collapse
before moving the root layout, including while native reports are pending. Module
definition movements affect their shared layout scope. Agent/RL specialized canvases
and generated interiors are excluded. Offsets do not infer topology or avoid overlap.
This provides translation, not persistent group containers or automatic layout.

Pure Node tests check independent expected coordinates and atomic refusals. Actual
owned Chrome checks saved API documents, real DOM origins, native reports, one-edit
Undo/Redo, zero no-op, module interfaces/root-key preservation, expanded-collapse
refusal, all native graph families and a valid constructor ID with absent saved layout.
Existing native/live/build/typecheck/coverage/browser/curl/recovery checks remain
required. No dependency/backend/schema/model-pin edits or saved-model retraining.
Broad browser/platform accessibility certification, large-graph performance, automatic
layout, group containers and Agent/RL selection integration remain separate work.
