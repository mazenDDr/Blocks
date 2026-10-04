"""Write the Milestone 5 example projects (graph JSON + separate UI JSON) into examples/. Deterministic.

    python examples/make_m5_examples.py

- rl_cartpole_dqn         DQN on Gymnasium CartPole-v1 (a simulator: nothing here is measured data)
- rl_gridworld_builder    a grid world defined by data (size, walls, goal) with separate reward components; also a DQN problem
- unsupervised_cells      k-means, Gaussian mixture, PCA and a t-SNE view on SYNTHETIC cell measurements (examples/fixtures/synthetic_cells.csv)
- unsupervised_moons      k-means vs DBSCAN on SYNTHETIC non-convex data (examples/fixtures/synthetic_moons.csv)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "python")]

from graph_core.hashing import semantic_hash  # noqa: E402
from graph_core.schema import Graph  # noqa: E402
from graph_core.validate import validate  # noqa: E402
from rl import samples as rs  # noqa: E402
from unsup import samples as us  # noqa: E402


def layout(graph: Graph, dx: int = 300, dy: int = 190) -> dict:
    depth: dict[str, int] = {n.id: 0 for n in graph.nodes}
    for _ in range(len(graph.nodes)):
        for e in graph.edges:
            depth[e.to.node] = max(depth[e.to.node], depth[e.from_.node] + 1)
    rows: dict[int, int] = {}
    pos = {}
    for n in graph.nodes:
        d = depth[n.id]
        pos[n.id] = {"x": d * dx, "y": rows.get(d, 0) * dy}
        rows[d] = rows.get(d, 0) + 1
    return {"schemaVersion": "1.0.0", "positions": pos}


EXAMPLES = {
    "rl_cartpole_dqn": (rs.cartpole_graph(20000), False,
                        "DQN on Gymnasium CartPole-v1 (installed Gymnasium 1.3.0): a simulator, not measured data. Press Run in the RL workspace; an untrained greedy policy balances about 10 steps, and the evaluation "
                        "(10 fixed seeds, separate environments) shows the learner's return, length and termination vs truncation. Open the Transition trace tab after the run."),
    "rl_gridworld_builder": (rs.gridworld_graph(6000), False,
                             "A grid world defined visually (size, walls, start, goal) with the reward declared as three separate components (goal, step, bump) that can be re-weighted or switched off. "
                             "The environment is a custom Gymnasium environment (Void/GridWorld-v0); frames are real renders of rollouts."),
    "unsupervised_cells": (us.cells_graph(), True,
                           "SYNTHETIC cell measurements with three generating groups (true_group). k-means, a Gaussian mixture, PCA and a t-SNE view with method-appropriate diagnostics and stability. "
                           "The t-SNE embedding is a non-metric projection: its separation is not evidence of clusters. true_group is an EXTERNAL label, not used for fitting."),
    "unsupervised_moons": (us.moons_graph(), True,
                           "SYNTHETIC two-moons data. k-means gets a respectable silhouette on the wrong partition; DBSCAN recovers the shapes. Internal metrics assume convex clusters; "
                           "the external agreement with true_shape (which was not used for fitting) shows the difference."),
}


def main(out: Path = HERE) -> None:
    for name, (gj, synthetic, desc) in EXAMPLES.items():
        g = Graph.model_validate(gj)
        rep = validate(g)
        assert rep.ok, (name, [(d.code, d.message) for d in rep.errors])
        (out / f"{name}.project.json").write_text(json.dumps(g.to_json(), indent=2, ensure_ascii=False) + "\n")
        ui = layout(g)
        ui["description"], ui["synthetic"] = desc, synthetic
        (out / f"{name}.ui.json").write_text(json.dumps(ui, indent=2, ensure_ascii=False) + "\n")
        print("wrote", name, semantic_hash(g)[:12])


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
