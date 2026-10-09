# Blocks — direction 2: a desktop control plane in liquid glass

*(2026-10-08. Supersedes direction 1, the marble run, kept below as history. The user set this direction.)*

## Brief (user-directed)

- **Asked for:** the UI should look more like the Kubernetes desktop tools (Lens / Headlamp) and Docker Desktop, and use Apple's
  Liquid Glass material. The product, users, one job ("start building") and keep list are unchanged from direction 1.
- **Not explored three ways:** the user named the references and the material, so the exploration step is replaced by
  reading those three references closely. The accent stays the user's fixed VS Code blue `#007ACC`, which sits between
  Docker blue `#1D63ED` and Kubernetes blue `#326CE5`.

## What is taken from each reference

| From | Taken | Where in Blocks |
|---|---|---|
| Docker Desktop | left sidebar of icon + label rows in sections; a wide search field centred in the title bar; a footer status bar with a green "Engine running" dot and resource numbers; resource tables with status dots, quiet row actions | sidebar (Build / Ship sections), Commands as a search field, status bar ("Control service running", backend, runs, graph), tables in Data / Records / Production |
| Lens / Kubernetes IDEs | a dock at the bottom with tabs (terminal, logs); status words as small coloured pills; IDs and hashes in mono; dense tables | the bottom Train / Runs dock, run status pills, mono shapes and graph hash |
| Apple Liquid Glass (macOS 26) | controls and navigation are a separate glass layer floating over content; translucent, blurred and saturated backdrop; a specular rim that is brighter on the lit top edge; lensing (refraction) at the rim of small controls; content itself stays solid | sidebar, title bar, tool pills, status bar, zoom controls, toasts, menus and modals are glass; the canvas, library and inspector are the content layer (a denser, more opaque frosted panel) |

**Refused:** glass on content (text over blur is hard to read all day); blurred-orb "AI" wallpaper; six category hues on
nodes; KPI-card dashboards.

## Material

- **Wallpaper** (what the glass bends): the product's own pieces, three large rounded blocks in the accent family, drawn in
  SVG and softened, on a deep navy (dark) or pale steel (light) field. Fixed behind the window like a desktop picture.
- **Glass (navigation layer):** tint `--glass` (light: white 52%; dark: navy 46%), `backdrop-filter: blur(22px) saturate(1.8)`,
  a 1px rim `--glass-rim`, an inner top highlight `--glass-spec` and a lower inner shade, a soft float shadow.
  On Chromium, small controls also refract: an SVG displacement filter bends the backdrop at their rim; elsewhere the
  declaration is dropped and plain blur remains.
- **Frosted (content layer):** `--frost` (light: white 82%; dark: navy 78%) with the same rim and a lighter blur, so tables
  and forms read as solid.
- **Canvas:** solid `--canvas`, a quiet floor; no dot grid.

## Tokens

- **Type:** the platform UI face (SF Pro on macOS, Segoe UI Variable on Windows, Cantarell/Ubuntu on Linux), because both
  references and the Apple material are native-feeling desktop tools; one family, hierarchy from weight. Mono: SF Mono /
  Cascadia for shapes, IDs and hashes. Scale unchanged: 20 · 16 · 14 · 13 · 12, nothing smaller.
- **Colour roles:** unchanged names (ground, surface, sunk, ink, ink-2, rule, accent, ok/warn/bad), plus `--glass`,
  `--glass-rim`, `--glass-spec`, `--frost`, `--canvas`, `--wire`, `--wire-strong`.
- **Radius (concentric, Apple's rounder family):** window panels 20 · cards 12 · controls 10 · pills 999.
- **Layout:** sidebar 212 (icon-only 68 at 761–1180) · library 264 · canvas fluid · inspector 340 · dock 230 · status bar 30.

## Shell

Sidebar (brand on top, Build then Ship, theme at the foot) · title bar (project, Open, Backend, kind | search-field
Commands | Undo, Redo, Save as the one primary) · its second row of quiet create actions · tool pills · library | canvas |
inspector · dock · status bar across the window.

## Signature

The marble from direction 1 becomes a **glass bead**: the same path and timing, drawn as a clear sphere with a bright
specular highlight and a blue core, the only liquid-glass object on the content layer. Reduced motion: it rests at the input.

## Voice

Unchanged: plain, warm, exact. Status bar words are Docker's register: "Control service running", "1 run working".

---

# Blocks — direction

## Brief

- **What:** a redesign of the Project Void editor web app (React + Vite, `apps/editor`), renamed **Blocks**. Users build
  machine-learning model graphs, tabular pipelines, agents and RL experiments by connecting blocks, run them, inspect the
  results and ship releases.
- **Who:** people who build and run models on their own machines: from a learner wiring a first CNN to someone shipping an agent.
- **The one job:** *start building*. The user opens Blocks to put blocks together and see them work.
- **Feel:** calm, playful, friendly.
- **References outside software:** left to me (chosen per concept below).
- **What exists:** the working app and its real data (examples, measured results); no logo, no photos. New name **Blocks**, new logo
  to design. Accent fixed by the user: the VS Code logo blue `#007ACC`. Light and dark modes, both.
- **Image generation:** none connected; nothing paid. Richness is drawn in code or comes from the product itself.
- **Decisions delegated to me:** navigation structure (the user asked for "what you think is best"), logo.

**User constraint vs. process:** the skill explores three accent hues; the user fixed the accent to VS Code blue, so all three
explorations keep `#007ACC` and differ on every other column.

## Current (the "before")

Shot at `shots/before/` (390/768/1280/1440). Audit:
- Scan FAILs: sideways scroll at 390px (top bar, graph tools, inspector 640px wide); text under 11px (10–10.8px chips and hints);
  contrast 3.49:1 and 2.85:1; text clipped in node cards at 768px; the canvas background collapses on phone.
- Structure: eight collapsible tool sections (`Keyboard graph tools`, `Insert block into a wire`, `Copy and paste`, `Layout groups`,
  `Arrange`, `Move`, `Outline`, `Comments`) stacked above the canvas, so the canvas — the work — gets a third of the screen.
  The training form takes the bottom third even when nothing is training.
- Type: system-ui at 11–14px; 44 text elements at 11px. One blue accent already in use (good).
- Face: closest to an unstyled internal tool; nothing of the four slop faces, but nothing decided either.

**Keep list (contract):** every workspace and its features (Graph, Data, Training, Experiments, Debug, Attention, Backend, Coverage,
Integrations, Production, Records, Agent tabs, RL lab, Scale); the block library with search; the canvas (drag, connect, select,
auto-arrange, layout groups, module scopes, outline, comments, copy/paste, insert into wire, move); the inspector; undo/redo;
the command palette; Save; project open/new; every accessibility label the test journeys rely on.
**Lose:** the stacked collapsible tool panels as the default view (they move into a toolbar, the command palette and a tools drawer).
**Unknown:** none blocking.

## Category default refused

Node editors (n8n, ComfyUI, Node-RED, Unreal Blueprints) share: a near-black canvas with a dot grid, saturated category-coloured
node headers (six hues), a dark left icon bar, Inter or system UI at 12px, glowing selection. ML tools (W&B, MLflow, Colab) add
the KPI-card dashboard. The category colour cliché is violet for AI and cyan-on-black for "neural". The slop faces this brief
falls into naturally: **4. The Dark Dev Tool** (my first concept mockup wore it: zinc greys, Geist, coloured project dots,
initials avatars, mono labels) and **3. The shadcn Dashboard** for the home screen. Refused: dot-grid canvas, six category hues,
KPI cards, initials avatars, Geist/Inter by reflex.

## Three concepts

First associations crossed out, per directions.md:
1. **Play** — ~~a LEGO set~~ → **"Blocks is a marble run."** Pieces you snap together; drop a sample in and watch it roll through
   every piece. Calm (wood and pastel track), playful (the marble), friendly (big pieces).
2. **Place** — ~~a workshop bench~~ → **"Blocks is a planetarium at night."** You sit under a quiet dark dome; the graph is a
   constellation, and each block's tensor is a small field of stars you can read.
3. **Objects** — ~~a sticker sheet~~ → **"Blocks is a felt board."** Soft cut-out shapes pressed onto a coloured felt field,
   lifted and re-placed by hand; nothing breaks, everything can move.

| | A · marble run | B · planetarium | C · felt board |
|---|---|---|---|
| ground | light (cool white, pale track wood) | dark (deep night blue) | colour-field (sand felt) |
| type | rounded — M PLUS Rounded 1c | mono — Martian Mono + Atkinson Hyperlegible Next | grotesque — Bricolage Grotesque |
| richness | illustration (track pieces in SVG) | data (each tensor drawn as a dot field) | colour-material (felt cut-outs) |
| shell | canvas (chrome floats on the canvas) | sidebar (rail + channels) | command (top bar, ⌘K first) |
| motion | line art: a marble rolls along the wires | dot matrix: tensors shimmer into place | 3D cards: blocks lift and tilt under the pointer |
| accent | VS Code blue (user) | VS Code blue (user) | VS Code blue (user) |

## Choice

**A, the marble run, merged.** It is the most *this product*: the graph already *is* a track that data rolls through, so the
signature explains the product instead of decorating it, and it is the only one that is calm, playful and friendly at once.
- **B lost** because a dark planetarium is the category default (Dark Dev Tool) with nicer stars; kept from it: the
  Discord-style **rail** of workspaces, and its night palette as the basis of A's dark mode.
- **C lost** because the felt ground makes dense tables (Records, Production, Data) hard to read all day; kept from it:
  **⌘K first**, the existing command menu promoted to the top bar.
- History check `app|play|light|rounded|blue|illustration|canvas` → Clear.

## Tokens

All in `apps/editor/src/tokens.css`; components read only role names.
- **Colour roles (light / dark):** `--ground` the floor (#F3F7FA / deep night blue), `--surface` pieces and panels, `--sunk`
  wells, `--ink` / `--ink-2` text (14:1 and 6:1 on the ground), `--rule` / `--rule-strong` hairlines and input borders (3:1),
  `--track` / `--track-edge` the pale beech of wires and piece outlines, `--accent` #007ACC (the marble, Save-as-primary, the
  selection, the open workspace) with `--accent-text` #00639F / #5CB4F2 for blue words, `--ok` `--warn` `--bad` with tint and
  rule, `--code-ground` / `--code-ink` for code, `--scrim`, `--shadow-float` (only things that float). Dark mode follows the
  system, or the rail's switch (Auto → Light → Dark), via `data-theme`.
- **Type:** M PLUS Rounded 1c 400/500/700/800, self-hosted, with a metric-matched Arial fallback ("Blocks Fallback"). Reason:
  round terminals are friendly without being childish, it has true tabular figures for shapes and parameter counts, and it
  is not a default reach. Mono: the platform UI mono for code only.
- **Scale (app):** 20 title · 16 panel head · 14 body · 13 meta · 12 floor. Nothing below 12px. Hierarchy from weight
  (800 heads) and ink colour, not size.
- **Spacing:** 8px base; 10px gutters between floating pieces.
- **Radius family (concentric):** panel 18 · piece 14 · control 10 · pill 999 (tool pills, primary, toasts).
- **Layout:** rail 92 · library 264 · canvas fluid · inspector 340 · bottom 260. Tablet (761–1180): canvas full width, library
  and inspector side by side under it. Phone (≤760): companion; everything stacks, the rail becomes a scrolling row, the canvas
  keeps 60vh.

## Art direction (richness: illustration)

The canvas is a play-room floor in morning light: plain, no dot grid. Blocks are pieces: white with a 2px beech outline, 14px
corners, bold name. Wires are 3px beech track with round caps. The only saturated thing on the floor is the blue marble (and the
blue ring of a selected piece). In dark mode the floor is the same room after dusk; the track darkens to walnut and the marble
stays the same blue.

## Shell and density

A **canvas shell**: everything is a piece resting on the floor with a 10px gap, so the canvas reads as the room and the
chrome as furniture. Left: a **rail** of workspaces (icon plus the same text label, building workspaces first, then
Integrations · Production · Records after a divider, theme switch at the foot). Top: one floating bar with the mark and name,
the project, **Save**, Undo/Redo, **Commands ⌘K**, Open, Backend, the graph kind, quiet text actions (New…, Export code) and
the validation line. Under it the graph tools are **pills** that open into a full-width piece. Density: comfortable, 32px
controls, 36px pills, 58px rail items.

## Signature: the marble

- **Trigger:** a run of this project is working (status running or preparing); `?marble` forces it for review and screenshots.
- **Frames:** the marble (r=9 blue, small white shine) appears at the start of the first wire, rolls along every real wire in
  data-flow order (Kahn order of the graph), slows into the output, rests 0.9s, rolls again.
- **Timing and easing:** 240 flow-px/s, smoothstep ease over the whole track (gathers speed, slows at the end); the loop
  length grows with the graph, so a deeper model visibly takes longer.
- **Off-screen:** paused while the tab is hidden.
- **Reduced motion:** the marble rests at the start of the track, still saying "a run is working".
- No JavaScript, or no run: the canvas is fully usable and nothing is hidden.

## Voice

Plain, warm and exact: say what happened and what to do next, in the user's words ("Not recorded. No run selected. Press Run to
train."), never marketing, never cute about errors.

## Content

The redesign does not rewrite the product's copy; every existing label, aria-label and status line is kept (keep list above).
New or changed words:
- Brand: **Blocks** (was "Project Void"); page title "Blocks"; description "Blocks: build models, pipelines and agents by
  connecting blocks, run them, and see what each piece did."
- Theme switch: "Auto" / "Light" / "Dark"; accessible name "Theme: follows system. Switch theme" (and so on).
- Rail labels: the existing workspace names unchanged (Graph, Data, Experiments, Training, Debug, Attention, Backend, Coverage,
  Integrations, Production, Records; Agent, RL lab, Domain workspace by graph kind).

### States (app.md §3) and where to reach them
| State | What it shows | Reach it |
|---|---|---|
| Empty graph | the existing "Empty graph. Add blocks from the library…" line centred on the floor | New model graph |
| Loading | the canvas floor with the pieces arriving; tool pills present | first load |
| Error | the existing red validation line in the top bar, red wire with ✖ and its shape label | connect mismatched shapes |
| Offline / API down | the existing toast and notice line; the editor stays readable | stop the backend |
| Overflow | long tool rows wrap (desktop) or scroll sideways (phone); long node text wraps inside the piece | small widths |
| Running | the marble rolls | start a training run, or `?marble` |
