# ADR 0005: Composite modules, sharing, structured control flow

Status: accepted (Milestone 3). Sources: VISION 6, 9.1-9.2, 15.6, A13, A24, A37, A46.

1. A module is a `ModuleDef` stored in the project (`graph.modules`): typed input ports, outputs wired from inside, arguments (`{"$param": name}` in node configs), nodes and edges. Instances are `core.composite` nodes. Definitions are embedded, so the semantic hash covers them and edits change it (A24); published versions live in an immutable library (`.workbench/library`).
2. Structural nodes (`core.composite`, `core.repeat`, `core.select`) are **expanded before validation** into a flat graph whose node ids are full paths (`res1/conv_a`). Validation, shape inference, lowering, codegen, run records, the debugger and diagnostics all see the same flat graph, so nothing can disagree and nothing is cached between definitions and runtime.
3. Parameter sharing is explicit: `share` on an instance (or `sharedWith` on a node) makes inner nodes use the very same module object (`SharedCall` wrapper per call site, so hooks and captures stay per call site). Default instantiation clones, and the UI/explain say so. A repeat ties iterations by default.
4. `core.repeat` is unrolled (static count <= 64, loop-carried type checked, only `fixed_count` termination); `core.select` evaluates both branches and selects with `tensor.where` (scalar bool predicate, identical branch signatures). Cycles in user graphs remain errors. Trade-off: no lazy branch, no data-dependent loops.
5. Rejected alternative: a nested runtime graph executor. It would need parallel shape inference, hashing and debugging paths.
