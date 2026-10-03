"""Lower a validated graph to a torch.nn.Module whose submodules are named by node id."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator

import torch
import torch.nn as nn

from . import registry
from .schema import Graph
from .validate import ExecutionBlocked, Report, require_executable


class GraphModule(nn.Module):
    """forward(*inputs) takes one tensor per tensor_input node (sorted by node id) and returns
    the terminal node values (sorted by node id): a tensor if there is one, else a tuple."""

    def __init__(self, graph: Graph, report: Report):
        super().__init__()
        self._plan: list[tuple[str, list[str]]] = []  # (node, source node per input port, in port order)
        src_of: dict[tuple[str, str], str] = {(e.to.node, e.to.port): e.from_.node for e in graph.edges}
        used: set[str] = set()
        types = {n.id: n.type for n in graph.nodes}
        for nid in report.order:
            op = registry.get_op(types[nid])
            self.add_module(nid, op.lower(report.resolved[nid]))
            srcs = [src_of[(nid, p)] for p in op.inputs]
            used.update(srcs)
            self._plan.append((nid, srcs))
        self.input_ids = sorted(n for n in report.order if types[n] == "core.tensor_input")
        self.output_ids = sorted(n for n in report.order if n not in used)

    def forward(self, *args: torch.Tensor):
        if len(args) != len(self.input_ids):
            raise TypeError(f"expected {len(self.input_ids)} inputs {self.input_ids}, got {len(args)}")
        given = dict(zip(self.input_ids, args))
        values: dict[str, torch.Tensor] = {}
        for nid, srcs in self._plan:
            values[nid] = self._modules[nid](given[nid]) if nid in given else self._modules[nid](*[values[s] for s in srcs])
        outs = tuple(values[o] for o in self.output_ids)
        return outs[0] if len(outs) == 1 else outs


def lower_graph(graph: Graph, report: Report | None = None) -> GraphModule:
    """Raises ExecutionBlocked if the graph is not executable (unknown op, mismatch, ...)."""
    if graph.graphKind != "model":
        from .types import Diagnostic

        raise ExecutionBlocked([Diagnostic("E_UNSUPPORTED_GRAPH_KIND", f"Graph kind '{graph.graphKind}' is not lowered to PyTorch.", path="/graphKind")])
    if report is None or not report.ok:
        report = require_executable(graph)
    return GraphModule(graph, report)


@contextmanager
def capture_activations(model: GraphModule, nodes: Iterable[str] | None = None) -> Iterator[dict[str, torch.Tensor]]:
    """Opt-in forward hooks recording each (selected) node's output, keyed by node id."""
    store: dict[str, torch.Tensor] = {}
    handles = []
    wanted = set(nodes) if nodes is not None else None
    for nid, mod in model.named_children():
        if wanted is None or nid in wanted:
            handles.append(mod.register_forward_hook(lambda m, i, o, nid=nid: store.__setitem__(nid, o.detach().clone())))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()
