# Blocks — critique

One critic for the whole job (critique.md). Scores 1–5.

## Round 1 (renders: shots/r6, shots/r5-dark, shots/r6/matrix, shots/r6/film)

| # | Line | R1 | R2 |
|---|---|---|---|
| 1 | Concept on the page | 3 | 3 |
| 2 | Not the category average | 3 | 3 |
| 3 | Not a slop face | 3 | 4 |
| 4 | First screen | 2 | 3 |
| 5 | Typography | 3 | 3 |
| 6 | Colour | 3 | 3 |
| 7 | Richness | 2 | 3 |
| 8 | Structure and rhythm | 2 | 3 |
| 9 | Craft | 2 | 3 |
| 10 | Responsive | 2 | 3 |
| 11 | Everyone's computer | 3 | 3 |
| 12 | Signature | 3 | 4 |
| 13 | Screenshot test | 2 | 3 |
| 14 | Fidelity (redesign) | 4 | 4 |

Verdict: not done. The shell is a real step up and fidelity is solid; the canvas (the work) is not yet redesigned and the
concept lives only in an occasional dot.

Five asks:
1. Chrome to ≤110px at 1440: top bar as two compact rows with the tool pills in the second, one row, scrolling; Train form as
   a 4-column grid; bottom panel ≤200px with no run; no empty band above the graph at 768.
2. Draw the pieces and the track: beech-peg ports, grooved 3px wires on a 5px underlay, edge labels as surface pills above
   nodes, "pytorch" as plain ink-2 text instead of chips, fitView minZoom so node text is never under 12px.
3. Marble as a moment: constant screen size, floor shadow, accent trail on the wire, a ring on each piece it enters.
4. Phone designed, not stacked: one-row bar, sideways second row, library categories as collapsible details, panel stacking.
5. Craft and matrix: Open select overflow at 390, zoom control stretch, HC borders on zoom buttons, library bottom fade,
   placeholder ellipsis, stub wires, a minimum canvas height at Windows 150%.

Mechanical defects found by the browser journeys in the same round (not design asks): the persistent load toast moved onto
the top bar and intercepted Save clicks (agent clipboard, auto layout, clipboard journeys).


## Round 2 (renders: shots/r8, shots/r8-dark, shots/r8/matrix, shots/r8/film)

Landed: chrome down to ~180px at 1440, four-column Train form, chips gone, pegs and smoothstep track (the return loop reads as
a rail), edge labels on demand, the marble as a moment, High Contrast, the phone drawer, the dark-phone zoom control.
Declined, accepted by the critic: fitView minZoom (journeys click every node after fit), library <details> on the phone
(bounded list instead), edge-label pills over the pieces (would cover node text on short wires).

Verdict: not done (2 of 13 at 4, none below 3). The critic found what the matrix missed: at Windows 150% the canvas row grew to
the library's height and the graph was centred ~1,000px below the fold.

Five asks:
1. Windows 150%: a fixed canvas row in short viewports; fit anchored to the top when the graph is short, so spare floor sits
   below it (768, 390).
2. Semantic zoom: below 0.8 zoom each piece shows a far form (name ≥13px on screen, the Out shape at ≥12px), same box size.
3. Junctions and trail: no grey arrowheads; the trail as a separate blue overlay stroke (the colour-mix read as teal "ok").
4. The concept at rest: the marble parked in a beech cradle at the input, grey at rest, blue when a run starts.
5. Fewer boxes: the drawer's dashed box becomes an empty chart floor; library cards become borderless rows with a peg; the
   phone bar wraps to two rows (Save first) instead of hiding Undo/Redo/Commands behind an invisible scroll.

## Direction 2 (2026-10-08): Docker Desktop / Kubernetes IDE shell, Apple Liquid Glass material

User-directed change of direction (see DIRECTION.md, direction 2). Self-scored; no separate critic this round.

Landed: Docker-style sidebar (icon + label rows in Build / Ship sections, icon rail at 761–1180, scrolling row on phones);
title bar with context left, a search-field Commands (⌘K / Ctrl K by OS) centred, Save as the one primary on the right;
a window status bar (control-service light polled every 15 s, backend, runs working, graph kind, validation line);
glass navigation layer (blur + saturate, lit rim, top highlight, Chromium rim refraction through an SVG displacement
filter, plain blur elsewhere, solid under `prefers-reduced-transparency`, borders in forced colours); frosted content
panels; solid canvas; resource-card nodes; the marble as a glass bead.

Defects found by the browser journeys and fixed in the same round:
- the glass rim pseudo-element at `inset: -1px` overflowed by 1px (command dialog sideways scroll; sidebar scrollbar) → `inset: 0`;
- sticky table heads floated over rows inside the scrolling tool panel → not sticky;
- the round-2 toast was `pointer-events: none`, so clicking it never dismissed it (arrangement, comments journeys) → the
  toast now rests on the status bar's empty middle, compact, and takes its own click (phone keeps pass-through);
- dragging a layout frame below zoom 1 moved members by a long fraction that differed per member in the 13th decimal →
  the shared offset is rounded to whole flow units (App.tsx, frame branch of onNodesChange);
- a tool-row notice (synthetic data) was squeezed into a narrow column → its own full line.

Verification: all CI browser stages (base journey, every editor journey, recovery with CI flags) pass on the final tree;
editor_layout_groups timed out once while a video render shared the CPU and passed when rerun alone.
