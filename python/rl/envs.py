"""Environment configuration, catalog, construction and the reward-component wrapper (VISION 9.9).

Environments are native Gymnasium environments. This module only (1) declares the tested catalog with its facts, (2) builds an environment from a
declarative `EnvSpec` + `RewardSpec`, (3) describes the spaces and the wrapper chain so wrappers (which change the learned task) are visible."""
from __future__ import annotations

import functools
import json
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pydantic import BaseModel, ConfigDict, Field

from . import gridworld as gw

GRID_ID = "Void/GridWorld-v0"
GYMNASIUM_PINNED = "1.3.0"
AUTORESET_MODES = ("NextStep", "SameStep")


def register_envs() -> None:
    if GRID_ID not in gym.registry:
        gym.register(id=GRID_ID, entry_point="rl.gridworld:GridWorldEnv", max_episode_steps=100)


register_envs()


class _S(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WrapperSpec(_S):
    type: Literal["ClipReward"] = "ClipReward"
    min: float = -1.0
    max: float = 1.0


class EnvSpec(_S):
    env_id: str = "CartPole-v1"
    max_episode_steps: int | None = Field(None, ge=1, le=100000, description="TimeLimit applied by gym.make; null = the registered default")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="constructor arguments of the environment (grid world: width, height, walls, start, goal, random_start, observation)")
    wrappers: list[WrapperSpec] = Field(default_factory=list)
    seed: int = 0
    autoreset_mode: Literal["NextStep", "SameStep"] = "NextStep"


class RewardComponent(_S):
    name: str
    weight: float = 1.0
    enabled: bool = True


class RewardSpec(_S):
    components: list[RewardComponent] = Field(default_factory=list, description="overrides on top of the environment's default weights; a component not listed keeps its default weight")


# ------------------------------------------------------------------------------------------------ catalog
CATALOG: dict[str, dict[str, Any]] = {
    "CartPole-v1": {
        "title": "CartPole", "family": "classic control", "status": "verified",
        "summary": "Balance a pole on a cart by pushing left or right.",
        "rewardComponents": ["env_reward"], "defaultWeights": {"env_reward": 1.0},
        "rewardDefinition": "+1 for every step taken, including the terminating step.",
        "termination": "pole angle beyond +-12 degrees, or cart position beyond +-2.4 (terminated).",
        "truncation": "500 steps (TimeLimit, registered max_episode_steps). Reaching it is NOT a failure state: the bootstrap continues.",
        "successRule": "truncated", "successText": "episode reaches the time limit without terminating",
        "renderModes": ["rgb_array", "human"], "license": "Gymnasium: MIT. Rendering needs pygame (LGPL).", "dependencies": ["gymnasium==1.3.0", "pygame==2.6.1 (rendering only)"],
        "resources": "CPU, a few thousand steps per second", "seeding": "env.reset(seed=...) seeds the initial-state RNG (np_random)",
        "snapshot": "no native snapshot API; the physical state is env.unwrapped.state (4 floats) and is not restored by this workbench: exact mid-episode resume is NOT supported",
    },
    GRID_ID: {
        "title": "Grid world (visual builder)", "family": "grid / tabular", "status": "verified",
        "summary": "Navigate a configurable grid with walls to a goal cell. Defined by data (see the environment builder).",
        "rewardComponents": list(gw.COMPONENTS), "defaultWeights": dict(gw.DEFAULT_WEIGHTS),
        "rewardDefinition": "three separate raw components per step: goal (+1 on reaching the goal), step (-1 every step), bump (-1 when the move is blocked by a wall or the border); scalar reward = sum of weight x component.",
        "termination": "the goal cell is reached (terminated).", "truncation": "100 steps unless max_episode_steps is set (TimeLimit).",
        "successRule": "terminated", "successText": "the goal is reached",
        "renderModes": ["rgb_array"], "license": "Project Void (this repository), MIT-compatible.", "dependencies": ["gymnasium==1.3.0", "numpy"],
        "resources": "CPU, tens of thousands of steps per second", "seeding": "reset(seed) seeds np_random; the start cell is random only when random_start is true",
        "snapshot": "the full state is the agent position (x, y); snapshotting is possible but not exposed in this milestone",
    },
    "MountainCar-v0": {
        "title": "MountainCar", "family": "classic control", "status": "space contract only",
        "summary": "Drive an underpowered car up a hill (sparse reward).", "rewardComponents": ["env_reward"], "defaultWeights": {"env_reward": 1.0},
        "rewardDefinition": "-1 per step until the goal.", "termination": "position >= 0.5 (terminated).", "truncation": "200 steps (TimeLimit).",
        "successRule": "terminated", "successText": "the flag is reached", "renderModes": ["rgb_array", "human"], "license": "Gymnasium: MIT. Rendering needs pygame.",
        "dependencies": ["gymnasium==1.3.0"], "resources": "CPU", "seeding": "env.reset(seed=...)", "snapshot": "no native snapshot API",
    },
    "Acrobot-v1": {
        "title": "Acrobot", "family": "classic control", "status": "space contract only",
        "summary": "Swing a two-link arm above a line.", "rewardComponents": ["env_reward"], "defaultWeights": {"env_reward": 1.0},
        "rewardDefinition": "-1 per step, 0 on reaching the goal.", "termination": "free end above the target height (terminated).", "truncation": "500 steps (TimeLimit).",
        "successRule": "terminated", "successText": "the target height is reached", "renderModes": ["rgb_array", "human"], "license": "Gymnasium: MIT. Rendering needs pygame.",
        "dependencies": ["gymnasium==1.3.0"], "resources": "CPU", "seeding": "env.reset(seed=...)", "snapshot": "no native snapshot API",
    },
    "Pendulum-v1": {
        "title": "Pendulum", "family": "classic control", "status": "space contract only",
        "summary": "Swing up and balance a pendulum with a continuous torque.", "rewardComponents": ["env_reward"], "defaultWeights": {"env_reward": 1.0},
        "rewardDefinition": "-(theta^2 + 0.1 theta_dot^2 + 0.001 torque^2) per step.", "termination": "none (never terminates).", "truncation": "200 steps (TimeLimit).",
        "successRule": None, "successText": "no success definition", "renderModes": ["rgb_array", "human"], "license": "Gymnasium: MIT. Rendering needs pygame.",
        "dependencies": ["gymnasium==1.3.0"], "resources": "CPU", "seeding": "env.reset(seed=...)", "snapshot": "no native snapshot API",
    },
    "FrozenLake-v1": {
        "title": "FrozenLake", "family": "grid / tabular", "status": "space contract only",
        "summary": "Cross a slippery frozen lake (Discrete observation).", "rewardComponents": ["env_reward"], "defaultWeights": {"env_reward": 1.0},
        "rewardDefinition": "+1 on reaching the goal, 0 otherwise.", "termination": "goal or a hole (terminated).", "truncation": "100 steps (TimeLimit).",
        "successRule": "terminated", "successText": "the goal is reached (reward 1)", "renderModes": ["rgb_array", "ansi", "human"], "license": "Gymnasium: MIT. Rendering needs pygame.",
        "dependencies": ["gymnasium==1.3.0"], "resources": "CPU", "seeding": "env.reset(seed=...)", "snapshot": "no native snapshot API",
    },
}


def env_version(env_id: str) -> int | None:
    tail = env_id.rsplit("-v", 1)
    return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else None


# ------------------------------------------------------------------------------------------------ spaces
def _num(x: float) -> float | str:
    f = float(x)
    if not np.isfinite(f) or abs(f) > 1e30:
        return "unbounded" if f > 0 else "-unbounded"
    return f


def describe_space(sp: spaces.Space) -> dict[str, Any]:
    if isinstance(sp, spaces.Discrete):
        return {"type": "Discrete", "n": int(sp.n), "start": int(sp.start), "dtype": str(sp.dtype)}
    if isinstance(sp, spaces.Box):
        return {"type": "Box", "shape": list(sp.shape), "dtype": str(sp.dtype), "low": [_num(x) for x in np.ravel(sp.low)[:16]], "high": [_num(x) for x in np.ravel(sp.high)[:16]],
                "bounded": bool(np.all(np.isfinite(sp.low)) and np.all(np.isfinite(sp.high)))}
    if isinstance(sp, spaces.MultiDiscrete):
        return {"type": "MultiDiscrete", "nvec": [int(v) for v in sp.nvec.ravel()], "dtype": str(sp.dtype)}
    if isinstance(sp, spaces.MultiBinary):
        return {"type": "MultiBinary", "shape": list(sp.shape), "dtype": str(sp.dtype)}
    return {"type": type(sp).__name__, "repr": str(sp)}


def wrapper_chain(env: gym.Env) -> list[dict[str, Any]]:
    """Outermost to innermost, with the settings that change the learned task."""
    out, e = [], env
    while True:
        name = type(e).__name__
        d: dict[str, Any] = {"name": name}
        if isinstance(e, gym.wrappers.TimeLimit):
            d["maxEpisodeSteps"] = e._max_episode_steps
            d["effect"] = "truncates the episode (truncated=True) after this many steps; not a terminal state"
        elif isinstance(e, RewardComponents):
            d["weights"] = e.effective_weights()
            d["effect"] = "reward = sum of weight x raw component over enabled components; raw components stay in info"
        elif isinstance(e, gym.wrappers.ClipReward):
            d["range"] = list(getattr(e, "_void_range", []))
            d["effect"] = "clips the scalar reward the learner sees"
        elif name == "OrderEnforcing":
            d["effect"] = "raises if step() is called before reset()"
        elif name == "PassiveEnvChecker":
            d["effect"] = "checks the environment API on the first calls"
        out.append(d)
        if isinstance(e, gym.Wrapper):
            e = e.env
        else:
            d["base"] = True
            break
    return out


# ------------------------------------------------------------------------------------------------ reward components
class RewardComponents(gym.Wrapper):
    """Re-weights the environment's raw reward components. Environments without declared components expose one component, `env_reward`."""

    def __init__(self, env: gym.Env, spec: RewardSpec, declared: list[str], defaults: dict[str, float]):
        super().__init__(env)
        self.declared, self.defaults = list(declared), dict(defaults)
        weights = dict(defaults)
        if spec.components:
            for c in spec.components:
                if c.name not in declared:
                    raise ValueError(f"reward component '{c.name}' is not declared by the environment (it declares {declared})")
                weights[c.name] = float(c.weight) if c.enabled else 0.0
        self.weights = weights
        # the last step's components, read by the in-process collector (vector-env info merging would hide them for autoreset steps)
        self.last: dict[str, Any] = {"raw": {k: 0.0 for k in declared}, "weighted": {k: 0.0 for k in declared}, "native": 0.0}

    def effective_weights(self) -> dict[str, float]:
        return dict(self.weights)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        raw = info.get("reward_components")
        raw = {k: float(v) for k, v in raw.items()} if raw is not None else {"env_reward": float(r)}
        info = dict(info)
        weighted = {k: self.weights[k] * raw[k] for k in self.declared}
        info["reward_components_raw"] = raw
        info["reward_components"] = weighted
        info["native_reward"] = float(r)
        self.last = {"raw": raw, "weighted": weighted, "native": float(r)}
        return obs, float(sum(weighted.values())), term, trunc, info


# ------------------------------------------------------------------------------------------------ construction
def make_env(spec: EnvSpec, reward: RewardSpec | None = None, render: bool = False) -> gym.Env:
    cat = CATALOG[spec.env_id]
    kwargs = dict(spec.kwargs)
    if render:
        kwargs["render_mode"] = "rgb_array"
    if spec.max_episode_steps is not None:
        kwargs["max_episode_steps"] = spec.max_episode_steps
    env = gym.make(spec.env_id, **kwargs)
    env = RewardComponents(env, reward or RewardSpec(), cat["rewardComponents"], cat["defaultWeights"])
    for w in spec.wrappers:
        env = gym.wrappers.ClipReward(env, w.min, w.max)
        env._void_range = (w.min, w.max)  # the wrapper keeps only a closure; record the bounds so the chain can show them
    return env


def env_fn(spec: EnvSpec, reward: RewardSpec | None = None, render: bool = False):
    """A picklable-free thunk for vector environments."""
    return lambda: make_env(spec, reward, render)


@functools.lru_cache(maxsize=64)
def _spaces_cached(key: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec = EnvSpec.model_validate(json.loads(key))
    env = make_env(spec)
    try:
        te = env
        limit = None
        while isinstance(te, gym.Wrapper):
            if isinstance(te, gym.wrappers.TimeLimit):
                limit = te._max_episode_steps
            te = te.env
        return describe_space(env.observation_space), describe_space(env.action_space), {"timeLimit": limit, "chain": wrapper_chain(env)}
    finally:
        env.close()


def env_spaces(spec: EnvSpec) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """(observation space, action space, extras) of the real environment built from `spec` (cached by the spec's JSON)."""
    return _spaces_cached(json.dumps(spec.model_dump(mode="json"), sort_keys=True))


def catalog_entry(env_id: str) -> dict[str, Any]:
    e = dict(CATALOG[env_id])
    spec = EnvSpec(env_id=env_id)
    obs, act, extra = env_spaces(spec)
    e.update({"id": env_id, "version": env_version(env_id), "observationSpace": obs, "actionSpace": act, "timeLimit": extra["timeLimit"], "wrapperChain": extra["chain"],
              "gymnasium": gym.__version__, "autoresetModes": list(AUTORESET_MODES)})
    return e
