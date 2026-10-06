# ADR0072: automated accessibility-tree audit of every workspace

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded in HANDOFF§73.

Accessibility was listed as unverified: controls had been given labels case by case,
but nothing checked the editor as a whole and no human review has happened.

Decision: an automated audit journey, `tools/editor_accessibility_smoke.py`, run
locally and in CI.

- It opens a model (reference_cnn), tabular (production_sensors), agent
  (serving_state) and RL (rl_cartpole_dqn) example, clicks every workspace tab and
  reads Chrome's own computed accessibility tree (CDP `Accessibility.getFullAXTree`).
- A violation is a non-ignored node with an interactive role (button, link, textbox,
  searchbox, combobox, listbox, checkbox, radio, slider, spinbutton, switch, tab,
  menuitem, option) or an image role with an empty computed name. Violations record
  the view, role and DOM tag/class/type/placeholder.
- Self-check first: an unnamed button, an unlabeled text input and an image without
  alternative text are injected; the audit must flag all three before its counts are
  trusted, then they are removed.
- Strict mode (default, used in CI) fails on any violation.

Baseline: 27 views, 12 violations, all decorative React Flow SVGs (background grid
and control-button icons). They are now `aria-hidden` (`src/decorativeSvgs.ts`; React
Flow exposes no prop); the buttons keep their labels. After the fix: 27 views, 0
violations, self-check 3/3.

Not provided and still open: a human accessibility review, screen-reader testing,
keyboard-only task walkthroughs beyond existing journeys, colour contrast, focus
order/visibility, reduced motion, zoom/reflow, or WCAG conformance claims. Dialogs,
inner tabs and states not reached by clicking the top-level workspace tabs are not
audited.
