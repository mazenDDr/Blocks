# ADR0051: whole-layout auto-arrange

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded separately in HANDOFF§48.

Alignment and distribution (ADR0036) act on up to 100 selected cards and keep
their order; nothing placed a whole graph by its connections. Imported or pasted
graphs, and graphs edited for a while, end up with wires pointing in every
direction.

Decision: one "Auto-arrange whole layout" action in the arrangement panel for the
current root or module layout scope, using a deterministic layered layout
(`src/graphLayout.ts`) on actual measured card rectangles.

- Cycles are broken at depth-first back edges in document order; self loops,
  duplicate wires and wires to cards outside the scope are ignored.
- Layers are longest paths from sources, so every remaining wire points right.
- Order within a layer comes from four down/up barycenter sweeps with document
  order as the tie-break, which removes avoidable crossings in simple cases; it
  is a heuristic, not crossing minimisation.
- Cards in a layer are stacked by their measured heights and layers are spaced by
  their widest card, so cards never overlap. The scope's current top-left corner
  is kept.
- Dimensions are read from the DOM at the explicit action, as for arrangement; a
  card without a dimension, invalid position, duplicate or more than 1000 cards
  refuses with the existing E_LAYOUT_* codes. Pending/unavailable native
  validation and expanded modules block it with the same messages as arrangement.
- The result is one layout history edit (Undo/Redo); an unchanged result makes no
  edit. Graph configuration, wires, native identity and saved graph hash are
  unchanged. The view refits afterwards; the viewport is not part of the layout.

Not provided: persistent groups, orthogonal wire routing, top-to-bottom or other
directions, layout of agent/RL canvases, or crossing-optimal ordering.

Verification: 6 Node tests (chain from the current corner, measured stacking and
spacing without overlap, cycles/self loops/duplicates/unknown endpoints and
isolated cards deterministic, barycenter crossing removal, refusals, 1000-node
chain within budget) and an owned Chrome journey on a scrambled native
reference_cnn layout: every wire points right by measured width, no overlap,
same top-left corner, graph and native hash unchanged, one Undo restores the
scrambled layout, Redo reapplies, a repeat is a no-op. Build, typecheck, all Node
tests and every existing editor journey pass.
