# ADR 0007: Optional Python code blocks

Status: accepted (Milestone 3). Sources: VISION 15.5, A45.

1. A code block is `CodeBlockDef` in the project (`graph.codeBlocks`): typed interface, declared effects, randomness and differentiability, pinned dependencies, source, fixtures. Nodes (`code.block`) refer to it; the interface and source hash are injected into the node's resolved config, so ports, types and the semantic hash follow the definition.
2. The control process never executes user source (a test scans for `eval`/`exec`). A long-lived child (`python -I runner.py`) with scrubbed environment, scratch cwd, CPU/file-size rlimits and a parent-enforced wall timeout runs the block; a guard layer turns undeclared network/file/subprocess use into errors (best effort, not a hostile-code boundary). Memory limits depend on the OS (not enforced on macOS; reported).
3. Differentiable blocks are an `autograd.Function` whose backward recomputes the block in the sandbox with the same seed and state; "differentiable" is checked at run time (outputs must be connected through torch ops) and in fixtures (finite difference). Others are explicit gradient boundaries with a validator warning.
4. Fixture results are cached by (source hash, pins, interface, fixture) so edits cannot be answered from a stale result. Published versions are immutable.
Gaps: dependencies are checked not installed; no source-level breakpoints; no PyTorch export for graphs with code blocks.
