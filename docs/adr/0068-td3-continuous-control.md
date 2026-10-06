# ADR0068: TD3 for continuous-action reinforcement learning

Status: accepted (2026-10-06); Mac arm64 and the user's GPU machine (HANDOFF§69).

RL graphs supported only DQN, which needs discrete actions. Pendulum-v1 was in the
environment catalog as a space contract only, and every continuous-control
environment was refused (E_RL_ALGO_ACTION_SPACE).

Decision: a TD3 learner node, `rl.td3_learner`, for 1-D Box action spaces, with its
own module `rl/td3.py`. The DQN path (buffer, collector, evaluation, trace) assumes
integer actions throughout and is not changed.

- Graph shape: reward → environment → td3_learner → evaluation. The learner owns its
  actor and twin-critic MLPs (configurable hidden sizes) and a float-action replay;
  DQN-only nodes in a TD3 graph are refused. Validation refuses non-Box actions,
  non-flat observations and budgets without gradient steps.
- Algorithm (Fujimoto et al., 2018): random uniform actions for `learning_starts`
  steps, then actor + Gaussian noise; clipped double-Q targets with target policy
  smoothing; actor updated every `policy_delay` critic updates; Polyak targets.
  Termination stops bootstrapping, the time limit does not. Actions are always
  inside the declared bounds (tanh scaling).
- Evaluation: deterministic actor, one fresh environment per declared seed, the DQN
  report fields (return, task return, components, termination/truncation, success).
- Events use the DQN event names and fields (`episode_end`, `train_update` with
  `loss` = critic loss and `epsilon` = exploration noise fraction, `eval`), plus TD3
  extras, so the RL API, curves, comparisons and the editor read TD3 runs unchanged.
  Checkpoints hold the CPU actor state, bounds and sizes.
- Runs accept `device: cpu|cuda` (TD3 only; ADR0067 rules: explicit, refused when
  unavailable).
- Editor: TD3 learner panel (defaults shown for omitted fields), algorithm badge,
  device select; example project `rl_pendulum_td3`.

Measured (`benchmarks/results/td3_pendulum.json`, gpu-box, torch 2.10+cu128, seed 0,
20,000 steps, mean task return over 10 evaluation seeds): CUDA −984.8 → −171.0 →
−171.8 → −172.3 at 5k/10k/15k/20k steps (95.0 s); CPU −995.5 → −170.4 → −168.6 →
−172.7 (82.6 s). One seed per device; descriptive, not a benchmark claim. For these
small networks CUDA is not faster. That run used the event name `evaluation`, later
renamed to `eval` to match DQN; numbers are unaffected.

Not provided: TD3 serving adapter (the RL serving adapter is DQN-only), trace/frame
capture and replay-buffer browsing for TD3, vector environments, SAC/PPO,
multi-dimensional action-space tests beyond 1-D, or Run-tab wording specific to TD3
(it still describes ε-greedy). The Pendulum catalog status text is unchanged because
`rl/envs.py` is pinned by RL serving identities.

Verification: 4 pytest cases (validation and refusals; a real 600-step worker run
with evaluations, checkpoints, bounded actions and exact update counts; CUDA refused
without CUDA; RL curves API on a TD3 run) and an owned Chrome journey on the example
(learner edits saved, native worker run completes, curves and evaluation render).
