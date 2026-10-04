"""Environment/algorithm compatibility, decided from the environment's real Gymnasium spaces before any execution (VISION 9.9)."""
from __future__ import annotations

from .envs import CATALOG, EnvSpec, env_spaces, env_version

import gymnasium as gym


def env_problems(env: EnvSpec) -> list[tuple[str, str]]:
    """Problems with the environment itself: catalog membership and version."""
    if env.env_id in CATALOG:
        return []
    base = env.env_id.rsplit("-v", 1)[0]
    bases = {k.rsplit("-v", 1)[0] for k in gym.registry if isinstance(k, str)}
    if env.env_id not in gym.registry and env_version(env.env_id) is not None and base in bases:
        return [("E_RL_ENV_VERSION", f"'{env.env_id}' is not a registered version of {base} (registered: {sorted(k for k in gym.registry if isinstance(k, str) and k.rsplit('-v', 1)[0] == base)}).")]
    return [("E_RL_ENV_NOT_CATALOGED", f"'{env.env_id}' is {'a registered Gymnasium environment' if env.env_id in gym.registry else 'not a registered Gymnasium environment'} but not in the tested catalog "
            f"{sorted(CATALOG)}; an adapter must be added and verified before it can be used.")]


def dqn_compatibility(env: EnvSpec) -> list[tuple[str, str]]:
    probs = env_problems(env)
    if probs:
        return probs
    obs, act, _ = env_spaces(env)
    out = []
    if act["type"] != "Discrete":
        out.append(("E_RL_ALGO_ACTION_SPACE", f"DQN needs a Discrete action space (it takes max over actions of Q(s, a)); {env.env_id} has a {act['type']} action space. "
                    "Use a policy-gradient or actor-critic method for continuous actions, or discretize the actions in a wrapper."))
    if obs["type"] != "Box" or len(obs.get("shape", [])) != 1:
        out.append(("E_RL_OBS_SPACE", f"The MLP Q-network takes a flat float vector (Box with a 1-D shape); {env.env_id} has a {obs['type']} observation space"
                    f"{' of shape ' + str(obs['shape']) if 'shape' in obs else ''}. Add an observation encoding (for example one-hot for Discrete) before the network."))
    return out
