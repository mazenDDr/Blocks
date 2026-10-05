# ADR 0036 — Explicit arrangement of actual measured graph cards

2026-10-05. Final environment evidence and acceptance state: HANDOFF §33.

VISION §5.3 calls for alignment without automatic rearrangement while working.
Add explicit root model/tabular/domain and module-definition arrangement controls.
They share current graph selection with the canvas, keyboard tools and clipboard.
Alignment accepts2–100 cards; distribution accepts3–100. No agent/RL arrangement or
automatic layout is claimed by this change.

Use actual transient DOM dimensions supplied by React Flow, and the same saved or
fallback positions used by the current canvas. Left/right/top/bottom and horizontal/
vertical centers align to the selection's bounding rectangle. Equal-gap distribution
orders current card origins on the chosen axis, keeps the two end cards fixed and
spaces interior cards using their measured widths/heights. Insufficient space refuses
with E_LAYOUT_OVERLAP rather than producing negative gaps. Alignment may intentionally
overlap cards. Missing/invalid dimensions, invalid positions, duplicate/oversized or
undersized selections and unsupported operations have explicit stable errors.

Expanded root module frames must collapse before arrangement because those frames
shift other visible origins; generated inner cards are not independently positioned.
The module-definition editor uses its existing module@version position prefix,
including actual boundary layout cards. Arrange modifies only selected position keys;
all other UI metadata and positions are preserved. Model definitions/configuration,
wires/sharing/weights and native identities are unchanged. One explicit action uses
one existing draft-history edit. An unchanged repeat creates no new edit. No automatic
graph edit, training, validation-triggered relayout, stored DOM sizes or schema/pin
migration is introduced.

Actual Chrome records native graph documents and actual card widths/heights, exercises
all eight operations with independent edge/center/gap geometry checks, compares exact
saved graph/layout metadata, native20,042-parameter identity, one Undo/Redo and repeated
no-op behavior. Actual insufficient measured span refuses with no edit. Module layout
remains scoped and native module validation is byte-identical. Native tabular/vision/
NLP/speech graphs use the same UI and retain exact semantic reports and Undo. Real curl
rechecks exact native storage/validation for every action/family. Existing native tests
and recovered-state journeys remain intact. An initial duplicate sibling React key
was caught by strict browser console checks and corrected; no assertion was weakened.

Agent/RL arrangements, grid snapping, alignment guides, general groups/automatic
layout, command menus, comments, sequence insertion/move and measured large-graph
performance remain separate work. This is current-page geometry, not a saved-width
guarantee across fonts, native validators or other platforms.
