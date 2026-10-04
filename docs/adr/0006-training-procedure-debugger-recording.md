# ADR 0006: Training procedure, debugger and recording

Status: accepted (Milestone 3). Sources: VISION 10, 13, A12, A28, A29, A35, A36.

1. The procedure is data (`graph.training`, a `ProcedureSpec`): an ordered stage list plus per-stage configuration. The executor interprets the list in order; order rules are checked before running (E_PROC_ORDER). Checkpoints are written only at window boundaries and hold model, optimizer, scheduler, RNG, position and early-stop state; with `deterministic` (CPU, one thread, deterministic algorithms) resume is bitwise and tested.
2. Runs of kind `procedure` use the existing worker/store/event system. Captures (activations, output gradients, parameter gradients of the first micro-batch of chosen steps) are observation-only hooks and stored as artifacts. History that was not captured is "not recorded"; the offered remedy is a **capture-and-rerun**: a new run resumed from the nearest checkpoint, verified bit for bit against the original losses.
3. Probes are forward hooks on detached copies; `verify_invariants` compares outputs, gradients, parameters, RNG, mode and node order with and without them, and `measure_overhead` reports the cost. Breakpoints and asserts are execution-changing and labelled so.
4. Sandbox interventions rebuild the state at step-1 from a checkpoint plus a verified deterministic replay, apply changes to copies and store the comparison as a new run of kind `sandbox`. The original run (row, events, artifact bytes) is hashed before and after and the equality is part of the result.
Limits: CPU only; the sandbox runs in the control process (bounded steps), not a separate process.
