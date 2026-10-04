"""Policy/value networks are ordinary MODEL graphs (VISION 9.9 'Value/Q model'): the same nodes, shape inference and lowering as everywhere else."""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from graph_core.lower import lower_graph
from graph_core.schema import Graph
from graph_core.validate import ExecutionBlocked, validate


class NetworkError(Exception):
    def __init__(self, code: str, message: str, port: str | None = None):
        super().__init__(message)
        self.code, self.message, self.port = code, message, port


def mlp_graph(obs_dim: int, hidden: tuple[int, ...] = (64, 64), n_actions: int = 2) -> dict[str, Any]:
    """A model graph: tensor_input [N, obs_dim] -> (Linear -> ReLU)* -> Linear(n_actions). Q-values are the final linear outputs (no activation)."""
    nodes = [{"id": "obs", "type": "core.tensor_input", "version": "1.0.0", "config": {"shape": ["N", obs_dim], "dtype": "float32", "layout": "NF"}}]
    edges = []
    prev, k = ("obs", "value"), 0
    for h in hidden:
        k += 1
        nodes.append({"id": f"fc{k}", "type": "pytorch.nn.linear", "version": "1.0.0", "config": {"in_features": "infer", "out_features": h, "bias": True}})
        nodes.append({"id": f"act{k}", "type": "pytorch.nn.relu", "version": "1.0.0", "config": {}})
        edges += [_edge(f"e{len(edges)}", prev, (f"fc{k}", "input")), _edge(f"e{len(edges) + 1}", (f"fc{k}", "output"), (f"act{k}", "input"))]
        prev = (f"act{k}", "output")
    nodes.append({"id": "q", "type": "pytorch.nn.linear", "version": "1.0.0", "config": {"in_features": "infer", "out_features": n_actions, "bias": True}})
    edges.append(_edge(f"e{len(edges)}", prev, ("q", "input")))
    return {"schemaVersion": "1.0.0", "graphKind": "model", "backend": "pytorch", "nodes": nodes, "edges": edges}


def _edge(eid: str, a: tuple[str, str], b: tuple[str, str]) -> dict[str, Any]:
    return {"id": eid, "kind": "tensor", "from": {"node": a[0], "port": a[1]}, "to": {"node": b[0], "port": b[1]}}


def check_network(network: dict[str, Any], obs_dim: int, n_actions: int) -> dict[str, Any]:
    """Validate the embedded model graph against the environment's spaces. Raises NetworkError with a stable code; returns facts on success."""
    try:
        g = Graph.model_validate(network)
    except Exception as e:  # noqa: BLE001
        raise NetworkError("E_RL_NETWORK_GRAPH", f"The network is not a valid model graph: {e}") from e
    if g.graphKind != "model":
        raise NetworkError("E_RL_NETWORK_GRAPH", f"The network must be a 'model' graph, got '{g.graphKind}'.")
    r = validate(g)
    if not r.ok:
        d = r.errors[0] if r.errors else None
        raise NetworkError("E_RL_NETWORK_GRAPH", f"The network graph has errors: {d.code + ' at ' + str(d.nodeId or d.path) + ': ' + d.message if d else 'unresolved nodes'}")
    inputs = [n for n in g.nodes if n.type == "core.tensor_input"]
    if len(inputs) != 1:
        raise NetworkError("E_RL_NETWORK_INPUT", f"The network needs exactly one core.tensor_input (the observation), found {len(inputs)}.")
    shape = list(inputs[0].config.get("shape", []))
    if shape != ["N", obs_dim]:
        raise NetworkError("E_RL_NETWORK_INPUT", f"The network input shape is {shape} but the environment observation is a vector of {obs_dim} features: expected [\"N\", {obs_dim}].")
    mod = lower_graph(g, r)
    out_id = mod.output_ids[0]
    out = r.output_types[out_id]
    t = next(iter(out.values()))
    if len(mod.output_ids) != 1 or list(t.shape) != ["N", n_actions]:
        raise NetworkError("E_RL_NETWORK_OUTPUT", f"The network must output one Q-value per action, shape [\"N\", {n_actions}]; its output '{out_id}' has shape {list(t.shape)}"
                           f"{'' if len(mod.output_ids) == 1 else f' and there are {len(mod.output_ids)} outputs'}.")
    return {"outputNode": out_id, "params": r.total_params, "shapes": {n: {p: t.to_json() for p, t in ports.items()} for n, ports in r.output_types.items()}}


def build_network(network: dict[str, Any], obs_dim: int, n_actions: int) -> nn.Module:
    check_network(network, obs_dim, n_actions)
    g = Graph.model_validate(network)
    try:
        return lower_graph(g)
    except ExecutionBlocked as e:  # pragma: no cover (check_network already validated)
        raise NetworkError("E_RL_NETWORK_GRAPH", str(e)) from e
