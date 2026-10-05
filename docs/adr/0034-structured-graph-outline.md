# ADR 0034 — Structured graph outline and native error navigation

Accepted implementation, 2026-10-05; final environment evidence in HANDOFF §31.
The root model/tabular/domain canvas and reusable module editor need an alternative
to locating every node by dragging. Add a structured outline alongside existing
keyboard graph tools, with explicit inspect/center/module navigation. Agent/RL have
their own workspaces and remain outside this scope.

Rows follow the actual graph document order and list installed operation titles plus
exact operation IDs, declared incoming/outgoing endpoints and wire kinds, versioned
module references and parameter-sharing references. Search is literal case-insensitive
substring across these values and native diagnostic codes/messages. A native-errors
filter lists nodes with actual error diagnostics. At most 50 rows render per page;
filters reset pagination and changes clamp a now-invalid page. These bounds do not
claim a large-graph benchmark or full virtualized canvas.

Contracts, parameter counts, typed/unresolved status and diagnostics come only from
the current existing native validation response. Root scope identifies the native
graph hash; module scope identifies that exact module hash. While validation is
pending or unavailable, old shapes/counts/errors/identity are withheld rather than
shown as current. Unavailable node contracts remain explicitly unavailable, including
module boundary pseudo-nodes without a corresponding native view. The outline never
supplies learned activations, weights or execution results.

A nested diagnostic is attached to its nearest visible containing node while retaining
its original native node/port/code/message. Graph diagnostics with no visible owner
remain separately visible; no invented navigation target. Inspect selects the actual
node using the existing inspector; Center selects/fits it without animated movement
or changing saved layout. Composite/repeat references open the existing module editor,
which makes shared definition scope clear. Existing breadcrumbs return to the project.
Selection, filtering, pagination and viewport navigation create no graph/layout edit
or history entry. The synthetic project description is in normal tools flow so it
cannot obscure outline controls or module breadcrumbs.

Pure Node tests cover actual endpoint/reference mapping, literal search, native-error
filtering, nearest visible diagnostic routing and withholding stale validation.
Native API tests feed real CNN/residual/tabular/NLP validation responses through the
actual TS helper, compare exact contracts/hash/diagnostics, and preserve source input.
A real native invalid residual module routes its descendant errors to each affected
instance, keeping global diagnostics. Actual Chrome saves API graph/layout bytes,
searches/selects/centers a native node, introduces a real invalid config, compares
native diagnostic codes and restores the original draft with Undo. It then compares
module identity/contracts with the actual module API, and paginates a labelled
SYNTHETIC 75-input native graph 50+25, resets search to page one and confirms exact
canonical stored graph/layout bytes remain unchanged. No performance/quality claim
comes from this pagination fixture. Existing recovery journeys are retained.

Agent/RL outlines, command-menu/history navigation, comments, alignment/auto-layout,
module-internal copy/paste, deeper cross-scope error jumps, outside-user onboarding,
formal accessibility certification and large-graph performance remain separate work.
No native model pin, dependency or graph schema migration is introduced.
