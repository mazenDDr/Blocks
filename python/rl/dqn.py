"""DQN with exact, inspectable update math (native torch).

    target   y_i = r_i + gamma * (1 - terminated_i) * max_a' Q_target(s'_i, a')                (bootstrap_target)
    loss     L   = (1/B) sum_i huber_delta(Q(s_i, a_i) - y_i)     (or mean squared error)
    update   theta <- Adam(theta, grad L);  every `target_update_interval` updates  theta_target <- theta

`terminated` is the ONLY flag that removes the bootstrap. `truncated` (a time limit or collection boundary) does not: the state is not terminal, so
the target still bootstraps from Q_target of the TRUE next observation (the final observation of the episode, not the first of the next)."""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field


class DQNConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_steps: int = Field(20000, ge=1, description="environment interactions (transitions) collected by the training collector")
    num_envs: int = Field(1, ge=1, le=16)
    gamma: float = Field(0.99, ge=0.0, le=1.0)
    lr: float = Field(5e-4, gt=0)
    batch_size: int = Field(64, ge=1)
    learning_starts: int = Field(500, ge=1, description="no gradient step before the buffer holds this many transitions")
    train_freq: int = Field(1, ge=1, description="collection (vector) steps between learning phases")
    gradient_steps: int = Field(1, ge=1, description="gradient steps per learning phase")
    target_update_interval: int = Field(500, ge=1, description="hard target-network copy every this many gradient steps")
    loss: Literal["huber", "mse"] = "huber"
    huber_delta: float = Field(1.0, gt=0)
    max_grad_norm: float | None = Field(10.0, gt=0)
    eps_start: float = Field(1.0, ge=0, le=1)
    eps_end: float = Field(0.05, ge=0, le=1)
    eps_decay_steps: int = Field(6000, ge=1, description="environment steps over which epsilon decays linearly")
    log_interval: int = Field(100, ge=1, description="a train_update event every this many gradient steps (unsmoothed)")


def bootstrap_target(reward, terminated, next_value, gamma):
    """r + gamma * (1 - terminated) * next_value, for python numbers or tensors. This is the single definition of the DQN target."""
    if isinstance(terminated, torch.Tensor):
        mask = 1.0 - terminated.to(next_value.dtype if isinstance(next_value, torch.Tensor) else torch.float32)
    else:
        mask = 1.0 - float(terminated)
    return reward + gamma * mask * next_value


def td_targets(target_net: nn.Module, next_obs: torch.Tensor, reward: torch.Tensor, terminated: torch.Tensor, gamma: float):
    """(targets, next_value, bootstrap_mask) for a batch. No gradient flows into the target."""
    with torch.no_grad():
        next_q = target_net(next_obs)
        next_value = next_q.max(dim=1).values
        y = bootstrap_target(reward, terminated, next_value, gamma)
    return y, next_value, 1.0 - terminated.to(next_value.dtype)


def epsilon_at(cfg: DQNConfig, env_steps: int) -> float:
    frac = min(1.0, env_steps / cfg.eps_decay_steps)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def param_sha256(net: nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(net.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@dataclass
class UpdateResult:
    update_index: int  # 1-based index of this gradient step
    loss: float
    grad_norm: float
    q_sa: np.ndarray  # Q(s,a) before the step
    target: np.ndarray
    next_value: np.ndarray
    mask: np.ndarray
    td_error: np.ndarray  # Q(s,a) - y
    loss_contribution: np.ndarray  # per-sample loss / B (sums to `loss`)
    mean_q: float
    version_before: int
    version_after: int
    target_version: int  # update index at which the target network held the weights used for these targets
    target_synced: bool


class DQNLearner:
    def __init__(self, q_net: nn.Module, cfg: DQNConfig):
        self.cfg = cfg
        self.q = q_net
        self.q_target = copy.deepcopy(q_net)
        self.q_target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.updates = 0  # gradient steps done == the policy (online network) version
        self.target_version = 0
        self.target_syncs = 0

    @property
    def version(self) -> int:
        return self.updates

    def act_values(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.q(torch.as_tensor(obs, dtype=torch.float32)).numpy()

    def update(self, batch: dict[str, np.ndarray]) -> UpdateResult:
        c = self.cfg
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        act = torch.as_tensor(batch["action"], dtype=torch.int64)
        rew = torch.as_tensor(batch["reward"], dtype=torch.float32)
        nxt = torch.as_tensor(batch["next_obs"], dtype=torch.float32)
        term = torch.as_tensor(batch["terminated"], dtype=torch.bool)
        y, next_value, mask = td_targets(self.q_target, nxt, rew, term, c.gamma)
        q_sa = self.q(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        per = (F.huber_loss(q_sa, y, reduction="none", delta=c.huber_delta) if c.loss == "huber" else F.mse_loss(q_sa, y, reduction="none"))
        loss = per.mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(self.q.parameters(), c.max_grad_norm if c.max_grad_norm is not None else float("inf")))
        self.opt.step()
        before = self.updates
        tv = self.target_version
        self.updates += 1
        synced = self.updates % c.target_update_interval == 0
        if synced:
            self.q_target.load_state_dict(self.q.state_dict())
            self.target_version = self.updates
            self.target_syncs += 1
        return UpdateResult(self.updates, float(loss.detach()), gn, q_sa.detach().numpy().copy(), y.numpy().copy(), next_value.numpy().copy(), mask.numpy().copy(),
                            (q_sa.detach() - y).numpy().copy(), (per.detach() / per.shape[0]).numpy().copy(), float(q_sa.detach().mean()), before, self.updates, tv, synced)


def explain(cfg: DQNConfig) -> dict[str, Any]:
    """The update equation with this configuration's actual values (VISION: 'Expanded formula'); only DQN terms appear."""
    loss_eq = (f"L = (1/B) sum_i huber(Q(s_i,a_i) - y_i), delta = {cfg.huber_delta}  [0.5 e^2 if |e| <= delta else delta (|e| - 0.5 delta)]" if cfg.loss == "huber"
               else "L = (1/B) sum_i (Q(s_i,a_i) - y_i)^2")
    return {
        "algorithm": "DQN (Mnih et al. 2015 form): online network, hard-updated target network, uniform replay, epsilon-greedy",
        "targetEquation": f"y_i = r_i + {cfg.gamma} * (1 - terminated_i) * max_a' Q_target(s'_i, a')",
        "lossEquation": loss_eq,
        "update": f"theta <- Adam(lr={cfg.lr}) step on grad L" + (f" after clipping the gradient norm to {cfg.max_grad_norm}" if cfg.max_grad_norm else " (no clipping)"),
        "targetUpdate": f"theta_target <- theta every {cfg.target_update_interval} gradient steps (hard copy)",
        "exploration": f"epsilon-greedy; epsilon decays linearly {cfg.eps_start} -> {cfg.eps_end} over {cfg.eps_decay_steps} environment steps (greedy at evaluation)",
        "boundaries": "terminated removes the bootstrap (mask 0). truncated does NOT: the target bootstraps from Q_target of the true final observation.",
        "updateToDataRatio": cfg.gradient_steps / (cfg.train_freq * cfg.num_envs),
        "learningStarts": cfg.learning_starts,
    }
