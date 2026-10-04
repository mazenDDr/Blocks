# ADR 0019 — Greedy DQN policies in the local production registry

Status: implemented within the bounds below, 2026-10-04. Extends ADR 0012/0017/0018 to the RL workbench (ADR 0009).

- **What is pinned.** A completed DQN run is listed with its `rl.q_network` node. Registration pins:
  - the FINAL Q-network checkpoint (native state dict, `weights_only`, strict) with its policy version and parameter hash;
  - the network model graph and environment spec, and the observation/action spaces with full Box bounds read from the real Gymnasium environment;
  - the evaluation report;
  - the Python/torch/gymnasium/numpy versions and the implementation hashes of the adapter, `rl/networks.py`, `rl/envs.py`, lowering/validation and layer code.
  Only Box observations with Discrete actions are served.
- **Policy semantics.** Serving returns argmax_a Q(s, a), the evaluation policy with epsilon 0, plus the Q-values. It is explicitly not the epsilon-greedy behaviour policy used for exploration. Tests compare the served actions and Q-values with the recorded final network loaded independently.
- **Input contract.** Records are `{observation: [obs_dim finite numbers]}` within the environment's declared bounds, 1–256 per request. Out-of-bounds, wrong-length and non-numeric observations become `E_REQUEST_SCHEMA` traces.
- **Reference and quality.** Up to 256 replay-buffer observations (oldest first) are frozen at registration. Their recorded actions came from the behaviour policy, so they are not offered as ground truth. The only label-based metric is action agreement with actions a person supplies, labelled as such and distinct from environment return, which stays in the run's evaluation report shown with the reference. Booleans are refused as action labels. Monitoring reports per-dimension observation drift (KS) and greedy-action frequencies (total variation) against the frozen reference.
- **Shared fix found while testing.** Python's JSON parser accepts `NaN`/`Infinity`. A serving request containing them used to fail while echoing them in the JSON trace. The HTTP serving route now refuses non-finite numbers anywhere in `records` before admission (`E_REQUEST_SCHEMA`, 422). In-process callers keep the recorded-trace behaviour, and an existing test relies on it. A global request-validation handler makes echoed non-finite inputs JSON-safe.

Remaining: no continuous-action, image-observation or recurrent policies; no online environment rollouts from the serving route; agent, unsupervised and Keras/JAX serving stay unimplemented.
