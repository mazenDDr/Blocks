# ADR 0009: The `rl` graph kind, a native DQN learner, bounded transition tracing, and unsupervised workflows

Status: accepted (Milestone 5). Builds on ADR 0003/0004/0008. Sources: VISION 9.5, 9.6, 9.9, 19.3, 23 (Milestone 5), 24 (A51-A55), 26.

## Decisions

1. **`graphKind: "rl"`, backend `gymnasium`, typed wires** (`reward_spec`, `env`, `network`, `buffer`, `learner`, `eval_report`). It reuses the typed-wire validator of the tabular kind
   (`validate_tabular(graph, kind, backend)`), so diagnostics, fixes and node views share one contract. RL agents are a separate kind from `agent` (LangGraph) graphs.
2. **Environments are native Gymnasium (pinned 1.3.0).** A tested catalog declares facts per environment; spaces are read from the real environment. A custom `Void/GridWorld-v0` is
   defined by data (size, walls, start, goal) and reports raw reward components (goal, step, bump) separately; an `RewardComponents` wrapper re-weights/disables them. Only
   `verified` entries have a tested learner pair (CartPole, grid world); others are validated on their spaces only and say so.
3. **Compatibility is decided before execution** from the real spaces, with stable codes: `E_RL_ALGO_ACTION_SPACE`, `E_RL_OBS_SPACE`, `E_RL_ENV_VERSION`, `E_RL_ENV_NOT_CATALOGED`,
   `E_RL_NETWORK_INPUT/OUTPUT/GRAPH`, `E_RL_REWARD_COMPONENT`, `E_RL_BUFFER_CAPACITY`, `E_RL_BUDGET`, `E_RL_NODE_COUNT`, `E_RL_ENV_CONFIG`.
4. **The Q-network is an ordinary model graph** embedded in `rl.q_network` and lowered by `graph_core.lower` (shape-checked against the spaces).
5. **DQN is native torch** (`rl/dqn.py`): `bootstrap_target = r + gamma (1 - terminated) max Q_target(s')` is the single target definition (A52: 2.98 / 1); only `terminated` removes the bootstrap,
   `truncated` does not. A test compares updates with a handwritten reference step.
6. **Autoreset follows the declared Gymnasium mode.** `Collector` reads `metadata["autoreset_mode"]`. NextStep (default in 1.3.0): the ending step returns the true final observation, the next
   call is a reset step (action ignored, reward 0) which is NOT stored; SameStep: the true final observation comes from `info["final_obs"]`. Tests compare vector collection with independent single
   environments in both modes and check `episode_start` flags (recurrent-reset contract). Reward components are read from the sub-environment wrappers, not from merged vector infos. SyncVectorEnv only; Disabled mode is not offered.
7. **Tracing is bounded and declared** (`TraceConfig`): transitions of a few captured training episodes of environment 0 (with real `env.render()` frames) are tracked with insertion, eviction, every
   minibatch use (TD target, mask, loss contribution, dL/dQ, versions) and parameter hashes. Untracked transitions are reported as not captured, never invented.
8. **Evaluation is separate**: fresh environments seeded by declared seeds, greedy policy; reported both under the training reward and the environment default reward (task return), with
   per-component returns, termination/truncation counts and the catalog's success rule. Variants are study trials (`rl` run config, `eval_*` objectives); the comparison aggregates across runs
   (unit of replication = run) with Student t intervals.
9. **Unsupervised workflows live in the tabular kind** (`sklearn.kmeans|gaussian_mixture|dbscan|pca|projection|cluster_diagnostics`) with new value kinds `unsup_model`, `cluster_report`. Diagnostics are
   method-specific; stability is ARI across seeds/resamples; external metrics require a declared label column and are labelled external; t-SNE is labelled a non-metric projection that cannot transform new points.
   k-means iteration changes are a replay of the winning initialisation with growing `max_iter` (scikit-learn exposes no per-iteration state).

## Consequences / limits
CPU only, synchronous vector envs, one algorithm (DQN, discrete actions), no exact mid-episode resume. UMAP and PPO are not implemented.
