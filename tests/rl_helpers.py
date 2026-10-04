from __future__ import annotations

from graph_core.schema import Graph
from rl import samples


def tiny_cartpole(total_steps: int = 600, **kw) -> Graph:
    dqn = {"learning_starts": 100, "eps_decay_steps": 300, "batch_size": 32, **kw.pop("dqn", {})}
    return Graph.model_validate(samples.cartpole_graph(total_steps, dqn=dqn, evaluation={"seeds": [1000, 1001, 1002]}, **kw))


def tiny_grid(total_steps: int = 800, **kw) -> Graph:
    return Graph.model_validate(samples.gridworld_graph(total_steps, **kw))
