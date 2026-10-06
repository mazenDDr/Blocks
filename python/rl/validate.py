"""Validation of `rl` graphs: typed wires (shared engine with tabular graphs), then graph-level rules and the translation to a runnable RLSpec."""
from __future__ import annotations

from graph_core.schema import Graph
from graph_core.types import Diagnostic, Fix
from graph_core.validate import Report
from tabular.validate import validate_tabular

from .train import RLSpec

BACKEND = "gymnasium"
REQUIRED = ("rl.reward", "rl.environment", "rl.q_network", "rl.replay_buffer", "rl.dqn_learner", "rl.evaluation")
TD3_REQUIRED = ("rl.reward", "rl.environment", "rl.td3_learner", "rl.evaluation")


def is_td3(graph: Graph) -> bool:
    return any(n.type == "rl.td3_learner" for n in graph.nodes)


def validate_rl(graph: Graph) -> Report:
    r = validate_tabular(graph, "rl", BACKEND)
    required = TD3_REQUIRED if is_td3(graph) else REQUIRED
    for extra in (set(REQUIRED) - set(TD3_REQUIRED) if is_td3(graph) else ()):
        if any(x.type == extra for x in graph.nodes):
            r.diagnostics.append(Diagnostic("E_RL_NODE_COUNT", f"A TD3 graph owns its networks and replay; remove '{extra}'.", "error", None, None, "/nodes", [Fix(f"Remove {extra}")]))
    for t in required:
        n = sum(1 for x in graph.nodes if x.type == t)
        if n != 1:
            r.diagnostics.append(Diagnostic("E_RL_NODE_COUNT", f"An rl graph needs exactly one '{t}' node; found {n}.", "error", None, None, "/nodes",
                                            [Fix(f"{'Add' if n == 0 else 'Remove the extra'} {t}")]))
    return r


def td3_spec_from_graph(graph: Graph):
    """(env, reward, TD3 config, evaluation config) of a valid TD3 graph."""
    from graph_core import registry

    from .envs import EnvSpec, RewardSpec
    from .td3 import TD3Config
    from .train import EvalConfig

    cfg = {t: registry.get_op(t).Config.model_validate(next(x for x in graph.nodes if x.type == t).config).model_dump(mode="json") for t in TD3_REQUIRED}
    return (EnvSpec.model_validate(cfg["rl.environment"]), RewardSpec.model_validate(cfg["rl.reward"]),
            TD3Config.model_validate(cfg["rl.td3_learner"]), EvalConfig.model_validate(cfg["rl.evaluation"]))


def spec_from_graph(graph: Graph) -> RLSpec:
    """The runnable spec of a VALID rl graph (node configs parsed through each operation's Config, so defaults apply)."""
    from graph_core import registry

    cfg = {}
    for t in REQUIRED:
        n = next(x for x in graph.nodes if x.type == t)
        cfg[t] = registry.get_op(t).Config.model_validate(n.config).model_dump(mode="json")
    learner = dict(cfg["rl.dqn_learner"])
    trace = learner.pop("trace")
    return RLSpec.model_validate({"env": cfg["rl.environment"], "reward": cfg["rl.reward"], "network": cfg["rl.q_network"]["network"], "buffer": cfg["rl.replay_buffer"],
                                  "dqn": learner, "eval": cfg["rl.evaluation"], "trace": trace})
