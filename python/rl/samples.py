"""Builders of the rl example graphs (the same ones the tests and the example generator use)."""
from __future__ import annotations

from typing import Any

from .envs import GRID_ID
from .networks import mlp_graph

PRESETS = {"cartpole": {"obs": 4, "actions": 2}}


def _edge(i: int, a: tuple[str, str], b: tuple[str, str], kind: str) -> dict[str, Any]:
    return {"id": f"e{i}", "kind": kind, "from": {"node": a[0], "port": a[1]}, "to": {"node": b[0], "port": b[1]}}


def rl_graph(*, env: dict[str, Any], obs_dim: int, n_actions: int, reward: list[dict[str, Any]] | None = None, hidden: tuple[int, ...] = (64, 64), dqn: dict[str, Any] | None = None,
             buffer: dict[str, Any] | None = None, evaluation: dict[str, Any] | None = None, trace: dict[str, Any] | None = None) -> dict[str, Any]:
    learner = {**(dqn or {})}
    if trace is not None:
        learner["trace"] = trace
    nodes = [
        {"id": "reward", "type": "rl.reward", "version": "1.0.0", "config": {"components": reward or []}},
        {"id": "env", "type": "rl.environment", "version": "1.0.0", "config": env},
        {"id": "qnet", "type": "rl.q_network", "version": "1.0.0", "config": {"network": mlp_graph(obs_dim, hidden, n_actions)}},
        {"id": "replay", "type": "rl.replay_buffer", "version": "1.0.0", "config": buffer or {}},
        {"id": "learner", "type": "rl.dqn_learner", "version": "1.0.0", "config": learner},
        {"id": "evaluation", "type": "rl.evaluation", "version": "1.0.0", "config": evaluation or {}},
    ]
    edges = [_edge(0, ("reward", "reward"), ("env", "reward"), "reward_spec"), _edge(1, ("env", "env"), ("qnet", "env"), "env"), _edge(2, ("env", "env"), ("replay", "env"), "env"),
             _edge(3, ("qnet", "network"), ("learner", "network"), "network"), _edge(4, ("replay", "buffer"), ("learner", "buffer"), "buffer"),
             _edge(5, ("learner", "learner"), ("evaluation", "learner"), "learner")]
    return {"schemaVersion": "1.0.0", "graphKind": "rl", "backend": "gymnasium", "nodes": nodes, "edges": edges}


def cartpole_graph(total_steps: int = 20000, **kw: Any) -> dict[str, Any]:
    dqn = {"total_steps": total_steps, **kw.pop("dqn", {})}
    return rl_graph(env={"env_id": "CartPole-v1", "seed": 0}, obs_dim=4, n_actions=2, dqn=dqn, evaluation=kw.pop("evaluation", None), **kw)


GRID_SPEC = {"width": 6, "height": 5, "walls": [[2, 0], [2, 1], [2, 2], [4, 2], [4, 3], [4, 4]], "start": [0, 0], "goal": [5, 4], "random_start": False, "observation": "onehot"}


def gridworld_graph(total_steps: int = 6000, grid: dict[str, Any] | None = None, reward: list[dict[str, Any]] | None = None, **kw: Any) -> dict[str, Any]:
    g = {**GRID_SPEC, **(grid or {})}
    obs = g["width"] * g["height"] if g.get("observation", "onehot") == "onehot" else 2
    dqn = {"total_steps": total_steps, "learning_starts": 200, "eps_decay_steps": max(1, total_steps // 2), "target_update_interval": 100, "lr": 1e-3, "batch_size": 64, **kw.pop("dqn", {})}
    return rl_graph(env={"env_id": GRID_ID, "kwargs": g, "max_episode_steps": 60, "seed": 0}, obs_dim=obs, n_actions=4, reward=reward, hidden=(64,), dqn=dqn,
                    buffer={"capacity": 5000}, evaluation=kw.pop("evaluation", {"seeds": list(range(1000, 1008))}), **kw)
