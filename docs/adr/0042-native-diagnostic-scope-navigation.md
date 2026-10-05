# 0042 — Open native diagnostic nodes in their stored definition

Root outline rows already display actual nested diagnostics but only navigate to the
owning root card. Add an explicit action beside a current native diagnostic when its
node path resolves to one actual stored node. Open the correct module breadcrumb and
select its inspector, retaining shared-definition semantics. No recorded execution,
iteration state, learned values or checkpoint is selected or inferred by navigation.

Resolve declared composite references, bounded repeat iteration markers and exact
then/otherwise branch references from the current graph. Repeat defaults/max follow
local native semantics (2/64); bounds prevent unbounded traversal. Refuse missing,
ambiguous or recursive definitions/nodes, malformed paths, invalid iterations/branches,
and generated interface/harness-only targets. Current module-local diagnostics resolve
within that definition. Root model/tabular/domain nodes are supported; Agent/RL remain
outside this generic outline. A select otherwise diagnostic must open the otherwise
definition, independently of the existing manual default-then module action.

Only current native reports expose actions. Pending/unavailable prior shapes/diagnostics
stay withheld; activation checks current validation again. Navigation changes transient
scope/selection only, with no graph/UI/history entry. Existing draft Undo/Redo remains
available. The current module inspector receives real native validation/provenance and
does not pretend to show recorded activations. Editing the shared definition affects
every instance, explicitly explained in the navigation result.

Labelled SYNTHETIC intentionally disconnected graphs use actual native expansion and
E_MISSING_INPUT diagnostics. Native tests verify both repeat iterations, nested scopes
and both select branches against the actual returned paths; Node tests refuse guessed
targets. Owned Chrome verifies keyboard activation, exact inspector/breadcrumb/native
identity and unchanged saved graph/UI/hash/history. Full/live/build/typecheck/coverage,
curl, all existing browser journeys and source-deletion recovery remain required.
No native pin/backend/schema/dependency edits or saved-model retraining. Broad runtime
debugger stepping, cross-project navigation, Agent/RL specialized scopes and formal
accessibility/performance/platform certification remain separate work.
