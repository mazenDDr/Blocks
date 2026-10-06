# ADR0069: serving continuous TD3 policies

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded in HANDOFF§70.

TD3 runs (ADR0068) produced deterministic actors that could not be deployed: the RL
serving adapter (ADR0019) serves DQN greedy policies only and pins its own files.

Decision: an `rl_td3` serving family in `production/td3_adapter.py`.

- TD3 runs now record up to 1024 observations they actually collected
  (`rl_td3_observations`, oldest first); registration freezes them as the reference.
- A version pins the run's final actor checkpoint (CPU state, sizes, action bounds),
  environment spec and real Gymnasium spaces, observation bounds and the evaluation
  report. Identity: environment versions plus hashes of the adapter, `rl/td3.py`
  and `rl/envs.py`.
- Requests are observation vectors validated against shape, finiteness and the
  environment's observation bounds. Responses are the deterministic actor's action
  vectors (no exploration noise), always inside the action bounds.
- Labels are reference action vectors inside the bounds; monitoring reports their
  mean absolute error (overall and per dimension), never accuracy, plus per-dimension
  observation and action drift against the reference and the usual health metrics.
- Candidates, registration, release warmup, reference input, labels, monitoring and
  the editor's Production panel include the family.

Not provided: environment-return measurement in production, stochastic policies,
vector actions beyond tests on 1-D Pendulum, batched GPU serving, or SAC/PPO policies.

Verification: 2 pytest cases on a real 600-step TD3 run (candidate listing,
registration, deployment, served actions equal to the run's own actor within 1e-6
and inside bounds; out-of-bounds and wrong-length observations refused; out-of-bound
labels refused; MAE equals |action − label|; monitoring family/drift/schema errors)
and the TD3 browser journey extended to register the trained actor and show its
pinned version in the Production workspace.
