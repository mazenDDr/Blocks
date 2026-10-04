"""Vector-environment collection that follows Gymnasium's DECLARED autoreset mode (VISION 9.9 'Termination, truncation, and autoreset must be correct').

Gymnasium 1.x (pinned 1.3.0) vector environments declare `metadata["autoreset_mode"]`:

* NextStep (the default): the step that ends an episode returns the TRUE final observation with terminated/truncated set. The following
  `step()` call does not apply the action: it resets that sub-environment and returns the first observation of the new episode with
  reward 0 and both flags False. That reset step is NOT a transition of the task and must not enter the replay buffer.
* SameStep: the ending step is followed by an immediate reset inside the same call. The returned observation is the NEW episode's first
  observation; the true final observation is in `info["final_obs"][i]`.

Either way the stored `next_obs` of an ending transition is the real final observation, never the next episode's first observation.
`episode_start[i]` is True for the first transition of a new episode: a recurrent policy must reset its hidden state for exactly those
sub-environments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from .envs import EnvSpec, RewardSpec, env_fn

MODES = {"NextStep": AutoresetMode.NEXT_STEP, "SameStep": AutoresetMode.SAME_STEP}


@dataclass
class Transition:
    env_index: int
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray  # the TRUE next observation (final observation when the episode ended here)
    terminated: bool
    truncated: bool
    components: dict[str, float]  # weighted (sum = reward before any ClipReward)
    raw_components: dict[str, float]
    native_reward: float
    episode_id: int
    episode_step: int  # 0-based index of this transition within its episode
    episode_start: bool  # first transition of an episode (recurrent state must be reset before acting on `obs`)


@dataclass
class EpisodeSummary:
    env_index: int
    episode_id: int
    length: int
    ret: float
    task_return: float
    components: dict[str, float]
    raw_components: dict[str, float]
    terminated: bool
    truncated: bool


@dataclass
class StepOutcome:
    transitions: list[Transition] = field(default_factory=list)
    episodes: list[EpisodeSummary] = field(default_factory=list)
    reset_steps: list[int] = field(default_factory=list)  # envs whose NextStep reset step happened in this call (no transition produced)


class Collector:
    def __init__(self, spec: EnvSpec, reward: RewardSpec, num_envs: int, seed: int, render_env0: bool = False):
        self.spec, self.n = spec, num_envs
        self.envs = SyncVectorEnv([env_fn(spec, reward, render=(render_env0 and i == 0)) for i in range(num_envs)], autoreset_mode=MODES[spec.autoreset_mode])
        declared = self.envs.metadata.get("autoreset_mode")
        self.mode = declared if isinstance(declared, AutoresetMode) else AutoresetMode(declared)
        self.component_names = list(self.envs.envs[0].get_wrapper_attr("declared"))
        self.obs, _ = self.envs.reset(seed=seed)
        self.pending_reset = np.zeros(num_envs, bool)
        self.episode_start = np.ones(num_envs, bool)
        self.episode_id = np.arange(num_envs, dtype=np.int64)
        self.next_episode_id = num_envs
        self.ep_step = np.zeros(num_envs, np.int64)
        self._zero()

    def _zero(self) -> None:
        z = lambda: np.zeros(self.n)  # noqa: E731
        self.ret, self.task_ret = z(), z()
        self.comp = np.zeros((self.n, len(self.component_names)))
        self.raw = np.zeros((self.n, len(self.component_names)))

    def render(self, i: int = 0):
        return self.envs.envs[i].render()

    def step(self, actions: np.ndarray) -> StepOutcome:
        prev_obs = self.obs
        nxt, rew, term, trunc, info = self.envs.step(np.asarray(actions))
        out = StepOutcome()
        next_pending = np.zeros(self.n, bool)
        for i in range(self.n):
            if self.mode == AutoresetMode.NEXT_STEP and self.pending_reset[i]:
                # the reset step: `nxt[i]` is the first observation of a new episode; the action was ignored; reward 0
                out.reset_steps.append(i)
                self.episode_id[i], self.next_episode_id = self.next_episode_id, self.next_episode_id + 1
                self.ep_step[i] = 0
                self.episode_start[i] = True
                continue
            w = self.envs.envs[i].get_wrapper_attr("last")
            true_next = nxt[i]
            if self.mode == AutoresetMode.SAME_STEP and (term[i] or trunc[i]):
                true_next = info["final_obs"][i]
            comps = {k: float(w["weighted"][k]) for k in self.component_names}
            raws = {k: float(w["raw"][k]) for k in self.component_names}
            out.transitions.append(Transition(i, np.array(prev_obs[i]), int(actions[i]), float(rew[i]), np.array(true_next), bool(term[i]), bool(trunc[i]), comps, raws, float(w["native"]),
                                              int(self.episode_id[i]), int(self.ep_step[i]), bool(self.episode_start[i])))
            self.episode_start[i] = False
            self.ep_step[i] += 1
            self.ret[i] += rew[i]
            self.task_ret[i] += w["native"]
            self.comp[i] += [comps[k] for k in self.component_names]
            self.raw[i] += [raws[k] for k in self.component_names]
            if term[i] or trunc[i]:
                out.episodes.append(EpisodeSummary(i, int(self.episode_id[i]), int(self.ep_step[i]), float(self.ret[i]), float(self.task_ret[i]),
                                                   dict(zip(self.component_names, self.comp[i].tolist())), dict(zip(self.component_names, self.raw[i].tolist())), bool(term[i]), bool(trunc[i])))
                self.ret[i] = self.task_ret[i] = 0.0
                self.comp[i] = self.raw[i] = 0.0
                if self.mode == AutoresetMode.NEXT_STEP:
                    next_pending[i] = True
                else:  # SameStep: the new episode begins now
                    self.episode_id[i], self.next_episode_id = self.next_episode_id, self.next_episode_id + 1
                    self.ep_step[i] = 0
                    self.episode_start[i] = True
        self.pending_reset = next_pending
        self.obs = nxt
        return out

    def close(self) -> None:
        self.envs.close()
