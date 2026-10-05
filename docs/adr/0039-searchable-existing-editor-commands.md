# ADR 0039 — Searchable existing editor commands

2026-10-05. Final environment acceptance and retained failures: HANDOFF §36.

VISION §5.3 requests a searchable command menu and keyboard creation/navigation.
Use a native modal dialog with explicit toolbar control and Cmd/Ctrl-K when focus
is outside input/select/textarea/contenteditable/CodeMirror editors. Escape closes;
Tab traverses controls, arrows change the current search result, Enter runs it.
Paging restores search focus when a previously focused pager becomes disabled.
Tab boundaries wrap inside the dialog; global draft-history shortcuts are withheld
while any modal dialog is open. Selected command text is announced in the status.

The catalogue is built from actual project-kind workspace destinations, current
root/module nodes and installed static operations. Workspace tabs and command routes
share one destination list, retaining their original labels/order. Agent/RL expose
existing workspaces and whole-draft history; their specialized node selectors/builders
are not pretended to use generic editor selection. Module root navigation is explicit.
A literal simple-lowercase query checks command label/detail, capped at200UTF16 input
units;25visible results per page with total/page counts. No semantic/fuzzy matching,
persisted catalogue, server recomputation, native values or performance claims.

Inspect/navigation/close/search make no graph/UI edit. Add uses the existing node
creator with cloned native registered defaults and no implicit anchor autowire;
additional inputs stay disconnected and real native validation errors remain visible.
Type/version/backend/purpose and disconnected defaults behavior are labelled before
creation. Supported model choices match the declared backend; Python tabular/domain
choices compose their own native operation libraries. Dynamic structural/code choices
remain outside this picker. Node creation reserves orphan layout/comment keys and
uses own-property position access, preserving existing provenance rather than silently
reattaching notes to a new node. One existing whole-draft Undo restores the documents;
Redo restores the same IDs/config/layout. Module creation belongs to the shared definition.

Real owned Chrome verifies dialog focus/keyboard/literal query/pages, actual inspectors
and workspace tabs, disabled history refusal, disconnected native defaults/missing-input
report, exact saved graph/UI/Undo/Redo, module interface preservation/root navigation,
and Agent/RL scope limits. Existing native/live/build/typecheck/coverage/browser/recovery
checks stay required. No dependency/backend/model identity pin/schema/DB change.
Formal accessibility certification, broad platform/browser matrix, fuzzy shortcuts,
custom key bindings, remote commands, Agent/RL node creation, whole-project search,
sequence movement/grouping and automatic layout remain separate work.

Strict module checks caught an existing view-to-definition conversion that renamed
unrelated input wire IDs. Conversion now changes only the endpoint reference and
preserves each actual wire ID/metadata. A real bundled module roundtrip and actual
command creation test require exact original interfaces/edges. This is a frontend
conversion correction, not native model code identity/schema migration. Native dialog
close events are asynchronous; ignore an older close event when the dialog has already
reopened, so rapid successive commands do not close a newer menu accidentally.
