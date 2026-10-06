# ADR0058: persistent layout groups

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded separately in HANDOFF§55.

Modules (composite/repeat/select) group nodes semantically: they change the graph,
its native identity and execution. People also need to group cards purely visually
("stem", "classifier head") without changing anything that runs. Selections were
the only grouping and were lost on every click.

Decision: named layout groups stored in the UI document only.

- `ui.layoutGroups` holds `{id, label, scope, members}`; scope is `root` or
  `module:<id>@<version>` so a module definition's layout has its own groups. The
  UI document is already outside the semantic hash, so the graph, its hash,
  validation and execution are unchanged. Malformed stored entries are ignored.
- Frames are computed from the members' actual measured card rectangles (with the
  same display offsets the canvas uses for expanded modules) plus padding and a
  title band, and drawn behind the cards. Members without a card (deleted, or not
  yet measured) are listed as missing and excluded from the frame; a group with
  no present member draws nothing.
- Dragging a frame moves exactly its member cards by the shared offset as one
  layout history edit (drags coalesce as card drags do). Clicking a frame selects
  its members, so the existing move, arrange and copy tools apply to the group.
  Frames never enter the saved positions or the node selection.
- The "Layout groups" panel creates a group from the selected cards, renames it,
  adds/removes selected cards, selects its cards and deletes it (cards stay where
  they are). Renaming a card keeps its memberships in the same scope. Bounds:
  100 groups, 500 members, 80-character labels.

Not provided: nesting, collapsing a group to one card, colors, groups on agent/RL
canvases, copying groups with the clipboard, or automatic layout that respects
groups.

Verification: 4 Node tests (pure create/rename/update/delete keeping other UI
fields, frame geometry per scope with missing members, rename scoping, refusals
and malformed data) and an owned Chrome journey on the native reference_cnn
example: three selected cards become a saved group with an unchanged graph and
hash; the frame encloses the three cards; dragging the frame header moves exactly
those cards by one shared offset and nothing leaks into positions; one Undo
restores the layout; the group survives a reload; clicking the frame header
selects the three cards; deleting the group leaves the cards in place. The first
journey click landed on a wire crossing the frame body; it now uses the header.
