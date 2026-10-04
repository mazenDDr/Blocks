"""Uniform replay buffer with per-transition provenance (VISION 9.9 'Replay/buffer': selected transitions, age, policy version, sampling weights).

Every transition has a global, monotonically increasing `tid` (transition id) assigned at insertion, so a transition can be followed from the
environment step to the buffer slot, to every minibatch that sampled it and to the learner update that used it."""
from __future__ import annotations

from typing import Any

import numpy as np

FIELDS = ("tid", "env_index", "episode_id", "episode_step", "policy_version", "insert_tick", "eps", "action_source", "action", "reward", "terminated", "truncated",
          "sample_count", "last_sampled_update")


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], n_components: int, seed: int = 0):
        self.capacity, self.n = int(capacity), 0
        self.pos = 0
        self.next_tid = 0
        self.obs = np.zeros((capacity, *obs_shape), np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), np.float32)
        self.action = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.terminated = np.zeros(capacity, bool)
        self.truncated = np.zeros(capacity, bool)
        self.tid = np.full(capacity, -1, np.int64)
        self.env_index = np.zeros(capacity, np.int16)
        self.episode_id = np.zeros(capacity, np.int64)
        self.episode_step = np.zeros(capacity, np.int32)
        self.policy_version = np.zeros(capacity, np.int32)
        self.insert_tick = np.zeros(capacity, np.int64)
        self.eps = np.zeros(capacity, np.float32)
        self.action_source = np.zeros(capacity, np.int8)  # 0 greedy, 1 random (epsilon exploration)
        self.components = np.zeros((capacity, n_components), np.float32)  # weighted components that sum to `reward`
        self.raw_components = np.zeros((capacity, n_components), np.float32)
        self.sample_count = np.zeros(capacity, np.int32)
        self.last_sampled_update = np.full(capacity, -1, np.int32)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n

    def add(self, *, obs, action, reward, next_obs, terminated, truncated, env_index, episode_id, episode_step, policy_version, tick, eps, action_source,
            components, raw_components) -> tuple[int, int, int]:
        """Insert one transition. Returns (tid, slot, overwritten_tid or -1). `next_obs` must be the TRUE next observation of this step."""
        i = self.pos
        evicted = int(self.tid[i]) if self.n == self.capacity else -1
        self.obs[i], self.next_obs[i], self.action[i], self.reward[i] = obs, next_obs, action, reward
        self.terminated[i], self.truncated[i] = terminated, truncated
        self.tid[i] = self.next_tid
        self.env_index[i], self.episode_id[i], self.episode_step[i] = env_index, episode_id, episode_step
        self.policy_version[i], self.insert_tick[i], self.eps[i], self.action_source[i] = policy_version, tick, eps, action_source
        self.components[i], self.raw_components[i] = components, raw_components
        self.sample_count[i], self.last_sampled_update[i] = 0, -1
        tid = self.next_tid
        self.next_tid += 1
        self.pos = (self.pos + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)
        return tid, i, evicted

    def sample_slots(self, batch_size: int, update_index: int) -> np.ndarray:
        """Uniform sampling WITHOUT replacement within one minibatch (probability batch/size each); records the use of every sampled slot."""
        slots = self.rng.choice(self.n, size=min(batch_size, self.n), replace=False)
        self.sample_count[slots] += 1
        self.last_sampled_update[slots] = update_index
        return slots

    def batch(self, slots: np.ndarray) -> dict[str, np.ndarray]:
        return {"obs": self.obs[slots], "action": self.action[slots], "reward": self.reward[slots], "next_obs": self.next_obs[slots],
                "terminated": self.terminated[slots], "truncated": self.truncated[slots], "tid": self.tid[slots]}

    def slot_of(self, tid: int) -> int | None:
        """The slot currently holding `tid`, or None if never inserted / already overwritten."""
        if tid < 0 or tid >= self.next_tid:
            return None
        s = np.nonzero(self.tid == tid)[0]
        return int(s[0]) if len(s) else None

    def record(self, slot: int, component_names: list[str]) -> dict[str, Any]:
        i = int(slot)
        return {"tid": int(self.tid[i]), "slot": i, "envIndex": int(self.env_index[i]), "episodeId": int(self.episode_id[i]), "episodeStep": int(self.episode_step[i]),
                "policyVersion": int(self.policy_version[i]), "insertTick": int(self.insert_tick[i]), "epsilon": float(self.eps[i]),
                "actionSource": "random" if self.action_source[i] else "greedy", "obs": self.obs[i].tolist(), "action": int(self.action[i]), "reward": float(self.reward[i]),
                "components": {k: float(v) for k, v in zip(component_names, self.components[i])}, "rawComponents": {k: float(v) for k, v in zip(component_names, self.raw_components[i])},
                "nextObs": self.next_obs[i].tolist(), "terminated": bool(self.terminated[i]), "truncated": bool(self.truncated[i]),
                "sampleCount": int(self.sample_count[i]), "lastSampledUpdate": int(self.last_sampled_update[i])}

    def to_npz(self, max_rows: int = 50000) -> dict[str, np.ndarray]:
        """The stored transitions in insertion order (oldest first), newest `max_rows` at most."""
        order = np.arange(self.n) if self.n < self.capacity else (np.arange(self.capacity) + self.pos) % self.capacity
        order = order[-max_rows:]
        return {"obs": self.obs[order], "next_obs": self.next_obs[order], "action": self.action[order], "reward": self.reward[order], "terminated": self.terminated[order],
                "truncated": self.truncated[order], "tid": self.tid[order], "env_index": self.env_index[order], "episode_id": self.episode_id[order],
                "episode_step": self.episode_step[order], "policy_version": self.policy_version[order], "insert_tick": self.insert_tick[order], "eps": self.eps[order],
                "action_source": self.action_source[order], "components": self.components[order], "raw_components": self.raw_components[order],
                "sample_count": self.sample_count[order], "last_sampled_update": self.last_sampled_update[order]}
