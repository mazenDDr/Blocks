"""TD3 for continuous (Box) action spaces (ADR 0068): twin critics, clipped double-Q targets, target policy smoothing and delayed actor updates.

The DQN path (dqn.py/train.py) assumes discrete actions throughout (integer actions, argmax, epsilon); TD3 has its own float-action replay
buffer, training loop and deterministic evaluation, and reports through the same Sink and the same evaluation report fields, so run
storage, evaluation reports and comparison stay uniform. Equations follow Fujimoto et al. (2018) as implemented in common references.
"""
from __future__ import annotations

import copy
import hashlib
import io
import time
from typing import Any, Callable

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.nn import functional as F

from .envs import CATALOG, EnvSpec, RewardSpec, env_spaces, make_env
from .evaluate import mean_ci, success_of


class TD3Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_steps: int = Field(20_000, ge=100, le=1_000_000)
    learning_starts: int = Field(1_000, ge=1)
    buffer_capacity: int = Field(100_000, ge=1_000, le=1_000_000)
    batch_size: int = Field(256, ge=8, le=4096)
    gamma: float = Field(0.99, gt=0, le=1)
    tau: float = Field(0.005, gt=0, le=1)
    actor_lr: float = Field(3e-4, gt=0)
    critic_lr: float = Field(3e-4, gt=0)
    hidden: list[int] = Field(default_factory=lambda: [256, 256], min_length=1, max_length=4)
    exploration_noise: float = Field(0.1, ge=0, le=1, description="std of Gaussian action noise while collecting, as a fraction of the action half-range")
    policy_noise: float = Field(0.2, ge=0, le=1, description="target policy smoothing noise std (fraction of the half-range)")
    noise_clip: float = Field(0.5, ge=0, le=1)
    policy_delay: int = Field(2, ge=1, le=10)
    eval_every: int = Field(5_000, ge=100)


def td3_compatibility(env: EnvSpec) -> list[tuple[str, str]]:
    obs, act, _ = env_spaces(env)
    out = []
    if act["type"] != "Box" or len(act.get("shape", [])) != 1:
        out.append(("E_RL_ALGO_ACTION_SPACE", f"TD3 needs a 1-D continuous (Box) action space; {env.env_id} has a {act['type']} action space. Use DQN for discrete actions."))
    if obs["type"] != "Box" or len(obs.get("shape", [])) != 1:
        out.append(("E_RL_OBS_SPACE", f"TD3 takes a flat float observation vector (Box with a 1-D shape); {env.env_id} has {obs['type']}."))
    return out


def _mlp(inp: int, hidden: list[int], out: int, tanh: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out))
    if tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """a = tanh(net(s)) * scale + bias, so every action is inside the declared bounds."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int], low: np.ndarray, high: np.ndarray):
        super().__init__()
        self.net = _mlp(obs_dim, hidden, act_dim, tanh=True)
        self.register_buffer("scale", torch.as_tensor((high - low) / 2, dtype=torch.float32))
        self.register_buffer("bias", torch.as_tensor((high + low) / 2, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs) * self.scale + self.bias


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int]):
        super().__init__()
        self.q1, self.q2 = _mlp(obs_dim + act_dim, hidden, 1), _mlp(obs_dim + act_dim, hidden, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], 1)
        return self.q1(x), self.q2(x)


def param_sha256(module: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _bytes(obj) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def evaluate_actor(actor: Actor, spec: EnvSpec, reward: RewardSpec, seeds: list[int]) -> dict[str, Any]:
    """Deterministic actor (no exploration noise), one fresh environment per declared seed; the DQN report fields."""
    cat = CATALOG[spec.env_id]
    comps = cat["rewardComponents"]
    rows = []
    for s in seeds:
        env = make_env(spec, reward)
        try:
            obs, _ = env.reset(seed=int(s))
            steps, ret, task, term, trunc = 0, 0.0, 0.0, False, False
            wsum, rsum = np.zeros(len(comps)), np.zeros(len(comps))
            while not (term or trunc):
                with torch.no_grad():
                    a = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].numpy()
                obs, r, term, trunc, info = env.step(a)
                steps += 1
                ret += r
                task += info["native_reward"]
                wsum += [info["reward_components"][k] for k in comps]
                rsum += [info["reward_components_raw"][k] for k in comps]
            rows.append({"seed": int(s), "length": steps, "return": float(ret), "taskReturn": float(task), "terminated": bool(term), "truncated": bool(trunc),
                         "success": success_of(spec.env_id, term, trunc), "components": dict(zip(comps, wsum.tolist())), "rawComponents": dict(zip(comps, rsum.tolist()))})
        finally:
            env.close()
    succ = [r["success"] for r in rows if r["success"] is not None]
    return {"seeds": [int(s) for s in seeds], "episodes": rows, "epsilon": 0.0, "policy": "deterministic actor (no exploration noise)",
            "return": mean_ci([r["return"] for r in rows]), "taskReturn": mean_ci([r["taskReturn"] for r in rows]), "length": mean_ci([r["length"] for r in rows]),
            "terminatedCount": sum(r["terminated"] for r in rows), "truncatedCount": sum(r["truncated"] for r in rows),
            "successRate": (sum(succ) / len(succ)) if succ else None, "successRule": cat["successText"],
            "componentReturns": {k: mean_ci([r["components"][k] for r in rows]) for k in comps},
            "rawComponentReturns": {k: mean_ci([r["rawComponents"][k] for r in rows]) for k in comps},
            "rewardBasis": "return = sum of the reward the learner trained on (weights as configured); taskReturn = sum of the environment's DEFAULT reward"}


class Cancelled(Exception):
    pass


def train_td3(env_spec: EnvSpec, reward: RewardSpec, cfg: TD3Config, eval_seeds: list[int], seed: int, sink,
              should_cancel: Callable[[], bool] = lambda: False, device: str = "cpu") -> dict[str, Any]:
    """Collect with exploration noise, learn from uniform replay, evaluate the deterministic actor every `eval_every` steps and at the end."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env(env_spec, reward)
    act_space = env.action_space
    low, high = act_space.low.astype(np.float32), act_space.high.astype(np.float32)
    obs_dim, act_dim = int(np.prod(env.observation_space.shape)), int(np.prod(act_space.shape))
    half = (high - low) / 2
    actor = Actor(obs_dim, act_dim, cfg.hidden, low, high).to(device)
    critic = Critic(obs_dim, act_dim, cfg.hidden).to(device)
    actor_t, critic_t = copy.deepcopy(actor), copy.deepcopy(critic)
    a_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    cap = min(cfg.buffer_capacity, cfg.total_steps)
    B = {"obs": np.zeros((cap, obs_dim), np.float32), "act": np.zeros((cap, act_dim), np.float32), "rew": np.zeros(cap, np.float32),
         "next": np.zeros((cap, obs_dim), np.float32), "term": np.zeros(cap, np.float32)}
    size = ptr = updates = actor_updates = episodes = 0
    evals, t0 = [], time.time()
    lo_t, hi_t, half_t = (torch.as_tensor(x, device=device) for x in (low, high, half))

    def evaluate(tick: int, final: bool):
        actor_cpu = copy.deepcopy(actor).cpu().eval()
        rep = evaluate_actor(actor_cpu, env_spec, reward, eval_seeds)
        sha = sink.put_artifact("rl_checkpoint", _bytes({"actor": actor_cpu.state_dict(), "algorithm": "TD3", "obsDim": obs_dim, "actDim": act_dim, "hidden": cfg.hidden,
                                                          "low": low.tolist(), "high": high.tolist()}),
                                {"policyVersion": actor_updates, "tick": tick, "paramSha256": param_sha256(actor_cpu), "purpose": "final" if final else "evaluation", "algorithm": "TD3"})
        entry = {"tick": tick, "update": updates, "policyVersion": actor_updates, "elapsedSec": time.time() - t0, "final": final, "checkpointSha256": sha, **rep}
        # Same event shape as DQN evaluations, so curves, comparisons and the editor read TD3 runs unchanged.
        sink.emit("eval", **{k: v for k, v in entry.items() if k != "episodes"}, episodeReturns=[e["taskReturn"] for e in rep["episodes"]], episodes=rep["episodes"])
        evals.append(entry)

    comps = CATALOG[env_spec.env_id]["rewardComponents"]
    obs, _ = env.reset(seed=seed)
    ep_ret, ep_task, ep_len = 0.0, 0.0, 0
    ep_w, ep_raw = np.zeros(len(comps)), np.zeros(len(comps))
    for tick in range(1, cfg.total_steps + 1):
        if should_cancel():
            raise Cancelled()
        if tick <= cfg.learning_starts:
            act = rng.uniform(low, high).astype(np.float32)
            source = "random"
        else:
            with torch.no_grad():
                act = actor(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))[0].cpu().numpy()
            act = np.clip(act + rng.normal(0, cfg.exploration_noise * half), low, high).astype(np.float32)
            source = "policy+noise"
        nxt, r, term, trunc, info = env.step(act)
        B["obs"][ptr], B["act"][ptr], B["rew"][ptr], B["next"][ptr], B["term"][ptr] = obs, act, r, nxt, float(term)
        ptr, size = (ptr + 1) % cap, min(size + 1, cap)
        ep_ret, ep_task, ep_len = ep_ret + float(r), ep_task + float(info["native_reward"]), ep_len + 1
        ep_w += [info["reward_components"][k] for k in comps]
        ep_raw += [info["reward_components_raw"][k] for k in comps]
        obs = nxt
        if term or trunc:  # the true next observation was stored; truncation does not cut the bootstrap
            sink.emit("episode_end", tick=tick, envIndex=0, episodeId=episodes, length=ep_len, **{"return": ep_ret}, taskReturn=ep_task,
                      components=dict(zip(comps, ep_w.tolist())), rawComponents=dict(zip(comps, ep_raw.tolist())), terminated=bool(term), truncated=bool(trunc),
                      policyVersion=actor_updates, epsilon=cfg.exploration_noise, actionSource=source)
            episodes += 1
            obs, _ = env.reset()
            ep_ret, ep_task, ep_len = 0.0, 0.0, 0
            ep_w, ep_raw = np.zeros(len(comps)), np.zeros(len(comps))
        if tick > cfg.learning_starts and size >= cfg.batch_size:
            idx = rng.integers(0, size, cfg.batch_size)
            o, a, rw, n2, tm = (torch.as_tensor(B[k][idx], device=device) for k in ("obs", "act", "rew", "next", "term"))
            with torch.no_grad():
                noise = (torch.randn_like(a) * cfg.policy_noise * half_t).clamp(-cfg.noise_clip * half_t, cfg.noise_clip * half_t)
                a2 = torch.max(torch.min(actor_t(n2) + noise, hi_t), lo_t)
                q1t, q2t = critic_t(n2, a2)
                target = rw.unsqueeze(1) + cfg.gamma * (1 - tm.unsqueeze(1)) * torch.min(q1t, q2t)
            q1, q2 = critic(o, a)
            c_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            c_opt.zero_grad()
            c_loss.backward()
            c_opt.step()
            updates += 1
            a_loss = None
            if updates % cfg.policy_delay == 0:
                a_loss = -critic(o, actor(o))[0].mean()
                a_opt.zero_grad()
                a_loss.backward()
                a_opt.step()
                actor_updates += 1
                with torch.no_grad():
                    for net, tgt in ((actor, actor_t), (critic, critic_t)):
                        for p, pt in zip(net.parameters(), tgt.parameters()):
                            pt.mul_(1 - cfg.tau).add_(cfg.tau * p)
            if updates % 250 == 0:
                # DQN field names (loss = critic loss; epsilon = exploration noise fraction; targetVersion = actor updates) plus TD3 extras.
                sink.emit("train_update", update=updates, tick=tick, loss=float(c_loss.detach()), gradNorm=None, meanQ=float(q1.detach().mean()), epsilon=cfg.exploration_noise,
                          bufferSize=size, policyVersion=actor_updates, targetVersion=actor_updates, meanTarget=float(target.mean()), elapsedSec=time.time() - t0,
                          criticLoss=float(c_loss.detach()), actorLoss=None if a_loss is None else float(a_loss.detach()), algorithm="TD3")
        if tick % cfg.eval_every == 0 and tick < cfg.total_steps:
            evaluate(tick, False)
    env.close()
    # Observations the run actually collected (oldest first): the frozen serving reference for continuous policies (ADR 0069).
    obs_buf = io.BytesIO()
    np.savez_compressed(obs_buf, obs=B["obs"][:min(size, 1024)])
    sink.put_artifact("rl_td3_observations", obs_buf.getvalue(), {"rows": min(size, 1024), "bufferSize": size, "order": "oldest first"})
    evaluate(cfg.total_steps, True)
    elapsed = time.time() - t0
    summary = {"algorithm": "TD3", "envSteps": cfg.total_steps, "gradientUpdates": updates, "actorUpdates": actor_updates, "episodes": episodes,
               "elapsedSec": elapsed, "stepsPerSec": cfg.total_steps / elapsed if elapsed else None, "bufferSize": size, "seed": seed, "device": device,
               "finalPolicyVersion": actor_updates}
    return {"evals": evals, "final": evals[-1], "summary": summary}
