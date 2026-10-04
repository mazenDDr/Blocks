"""Run one module of a project on concrete tensors (the "tiny numerical example" of VISION 9.2) through the normal lowering path."""
from __future__ import annotations

from typing import Any

import torch

from .composite import module_harness
from .lower import lower_graph
from .schema import Graph
from .validate import ExecutionBlocked, validate


def _dims(spec_shape, t: torch.Tensor):
    return [("N" if (spec_shape and i < len(spec_shape) and spec_shape[i] == "N") else d) for i, d in enumerate(t.shape)]


def run_module(graph: Graph, module_id: str, inputs: dict[str, torch.Tensor], version: str | None = None, args: dict[str, Any] | None = None,
               grads: bool = False) -> dict[str, Any]:
    """Returns {"outputs": {name: tensor}, "grads": {input name: tensor} (when grads and an output is scalar), "report": validation report}."""
    mod = graph.module(module_id, version)
    if mod is None:
        raise KeyError(module_id)
    shapes = {p.name: _dims(p.shape, inputs[p.name]) for p in mod.inputs if p.name in inputs}
    h, inst = module_harness(graph, module_id, version, shapes)
    if args:
        for n in h.nodes:
            if n.id == inst:
                n.config["args"] = args
    r = validate(h)
    if not r.ok:
        raise ExecutionBlocked(r.errors)
    model = lower_graph(h, r)
    leaf = {k: v.detach().clone().requires_grad_(grads and v.dtype.is_floating_point) for k, v in inputs.items()}
    values = model.evaluate({f"in_{k}": v for k, v in leaf.items()})
    info = r.expansion.instances[inst]
    outs = {name: model.value_of(values, *ep) for name, ep in info.out_map.items()}
    result: dict[str, Any] = {"outputs": outs, "report": r, "instance": info}
    if grads:
        scalar = [o for o in outs.values() if o.dim() == 0 and o.requires_grad]
        if scalar:
            gs = torch.autograd.grad(scalar[0], [v for v in leaf.values() if v.requires_grad], allow_unused=True)
            result["grads"] = dict(zip([k for k, v in leaf.items() if v.requires_grad], gs))
    return result
