"""A small grid-world defined by data (VISION 9.9 environment builder): grid size, walls, start, goal.

The environment reports its reward as SEPARATE RAW COMPONENTS in `info["reward_components"]` (goal, step, bump); the scalar reward is
their weighted sum under the environment's DEFAULT weights. The reward node of an rl graph can re-weight or disable components
(`rl.envs.RewardComponents`). Rendering is a separate operation from stepping: `render()` never advances the state.

Actions: 0 up (y-1), 1 down (y+1), 2 left (x-1), 3 right (x+1). State: (x, y); cell index = y * width + x.
Termination: the goal cell is reached. Truncation is a time limit applied by Gymnasium's TimeLimit wrapper (registered max_episode_steps)."""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

COMPONENTS = ("goal", "step", "bump")
DEFAULT_WEIGHTS = {"goal": 1.0, "step": 0.01, "bump": 0.0}
ACTIONS = ("up", "down", "left", "right")
_DELTA = ((0, -1), (0, 1), (-1, 0), (1, 0))
CELL = 32


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, width: int = 5, height: int = 5, walls: list[list[int]] | None = None, start: list[int] | None = None, goal: list[int] | None = None,
                 random_start: bool = False, observation: str = "onehot", render_mode: str | None = None):
        super().__init__()
        self.width, self.height = int(width), int(height)
        self.walls = {(int(x), int(y)) for x, y in (walls or [])}
        self.start = tuple(start) if start is not None else (0, 0)
        self.goal = tuple(goal) if goal is not None else (self.width - 1, self.height - 1)
        self.random_start = bool(random_start)
        self.observation_kind = observation
        self.render_mode = render_mode
        for name, c in (("start", self.start), ("goal", self.goal)):
            if not self._inside(c) or c in self.walls:
                raise ValueError(f"{name} {c} must be inside the {self.width}x{self.height} grid and not a wall")
        if observation not in ("onehot", "xy"):
            raise ValueError("observation must be 'onehot' or 'xy'")
        self.observation_space = (spaces.Box(0.0, 1.0, (self.width * self.height,), np.float32) if observation == "onehot"
                                  else spaces.Box(0.0, 1.0, (2,), np.float32))
        self.action_space = spaces.Discrete(4)
        self.pos = self.start

    def _inside(self, c) -> bool:
        return 0 <= c[0] < self.width and 0 <= c[1] < self.height

    def _obs(self) -> np.ndarray:
        if self.observation_kind == "onehot":
            o = np.zeros(self.width * self.height, np.float32)
            o[self.pos[1] * self.width + self.pos[0]] = 1.0
            return o
        return np.array([self.pos[0] / max(1, self.width - 1), self.pos[1] / max(1, self.height - 1)], np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.random_start:
            free = [(x, y) for y in range(self.height) for x in range(self.width) if (x, y) not in self.walls and (x, y) != self.goal]
            self.pos = free[int(self.np_random.integers(len(free)))]
        else:
            self.pos = self.start
        return self._obs(), {"position": list(self.pos)}

    def step(self, action):
        dx, dy = _DELTA[int(action)]
        nxt = (self.pos[0] + dx, self.pos[1] + dy)
        bump = (not self._inside(nxt)) or nxt in self.walls
        if not bump:
            self.pos = nxt
        reached = self.pos == self.goal
        raw = {"goal": 1.0 if reached else 0.0, "step": -1.0, "bump": -1.0 if bump else 0.0}
        reward = float(sum(DEFAULT_WEIGHTS[k] * v for k, v in raw.items()))
        return self._obs(), reward, bool(reached), False, {"reward_components": raw, "position": list(self.pos), "bumped": bool(bump)}

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        img = np.full((self.height * CELL, self.width * CELL, 3), 235, np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                if (x + y) % 2:
                    img[y * CELL:(y + 1) * CELL, x * CELL:(x + 1) * CELL] = 225
        for (x, y) in self.walls:
            img[y * CELL:(y + 1) * CELL, x * CELL:(x + 1) * CELL] = (60, 60, 70)
        gx, gy = self.goal
        img[gy * CELL + 4:(gy + 1) * CELL - 4, gx * CELL + 4:(gx + 1) * CELL - 4] = (46, 160, 90)
        ax, ay = self.pos
        yy, xx = np.mgrid[0:CELL, 0:CELL]
        disk = (yy - CELL / 2) ** 2 + (xx - CELL / 2) ** 2 <= (CELL * 0.32) ** 2
        cell = img[ay * CELL:(ay + 1) * CELL, ax * CELL:(ax + 1) * CELL]
        cell[disk] = (40, 90, 200)
        return img


def schematic(spec: dict[str, Any]) -> list[list[str]]:
    """Text schematic of the configured grid (one string per cell: '#' wall, 'S' start, 'G' goal, '.' free), straight from the spec."""
    w, h = int(spec.get("width", 5)), int(spec.get("height", 5))
    walls = {tuple(c) for c in spec.get("walls", [])}
    start = tuple(spec.get("start") or (0, 0))
    goal = tuple(spec.get("goal") or (w - 1, h - 1))
    return [["#" if (x, y) in walls else "G" if (x, y) == goal else "S" if (x, y) == start else "." for x in range(w)] for y in range(h)]
