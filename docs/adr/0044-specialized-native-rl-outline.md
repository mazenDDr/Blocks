# 0044 — RL structure and exact native contract inspection

The RL workspace exposes Environment/Learner controls but had no structured list of
the graph nodes and typed wires. Add a Structure tab with literal200UTF16 search,
50-row pages, current native errors and keyboard inspection of exact unique stored
nodes. It displays per-node native typing/parameter counts and input/output contracts,
including real Gymnasium observation/action spaces, reward components/weights, network
shapes and DQN equations. Declared configuration is labelled separately. The actual
current root RL validation hash identifies all native facts; no separate embedded
network hash, learned Q-values, trajectory or run measurements are inferred.

Use native per-node outputShapes, not family-wide aggregate/catalog fallback, so an
invalid/ambiguous graph cannot acquire another node's facts. Pending/unavailable reports
withhold prior contracts/diagnostics/hash, including an already selected inspector.
Missing native fields say not returned; no fabricated default shapes/reward/equation.
Native constructor node IDs use existing own-property outline semantics. Duplicate
identifiers have no inspector target; controls navigation requires the node's family
to be unique because the existing Environment/Learner editors select by family.

Inspect/search/filter/paging and opening those existing controls only alter transient
navigation, preserving saved graph/UI/native identity and draft history. The inspector
is read-only; existing control edits still use existing history. This introduces no
backend/schema/native identity/dependency change or retraining/reinstall requirement.

Evidence: actual native API→TS tests on both real CartPole and configured GridWorld,
missing-input refusal, exact ports/spaces/reward/equation/shapes, pending/native identity
and source preservation. Real owned Chrome verifies those same values, keyboard existing
controls, held/released real requests without substitution, constructor identity, and
75declared native rows with intentional duplicate reward-family E_RL_NODE_COUNT.
Pagination evidence is not valid learner/performance evidence. All required full/live/
build/typecheck/curl/original browser/recovery checks must finish before acceptance.

RL clipboard/grouping, arbitrary algorithm/environment coverage, expanded Q-network
graph editing, large-graph performance and formal accessibility certification remain
separate work. Existing rollout/buffer/trace/evaluation native journeys remain unchanged.
