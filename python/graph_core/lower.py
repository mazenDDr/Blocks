"""Lower a validated graph to a torch.nn.Module whose submodules are named by node id."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator

import torch
import torch.nn as nn

from . import registry
from .schema import Graph
from .validate import ExecutionBlocked, Report, require_executable


class SharedCall(nn.Module):
    """A call site that uses another node's module (and so its parameter tensors). Distinct wrapper per call site, so hooks and
    captures see each call separately while `parameters()` still lists every tensor once."""

    def __init__(self, inner: nn.Module, target: str):
        super().__init__()
        self.inner, self.target = inner, target

    def forward(self, *args):
        return self.inner(*args)


class GraphModule(nn.Module):
    """forward(*inputs) takes one tensor per tensor_input node (sorted by node id) and returns
    the terminal node values (sorted by node id): a tensor if there is one, else a tuple.

    Node ids may be paths (res1/conv_a). Nodes with several output ports return a tuple in port order."""

    def __init__(self, graph: Graph, report: Report):
        super().__init__()
        self._plan: list[tuple[str, list[tuple[str, str]]]] = []  # (node, (source node, source port) per input port, in port order)
        src_of: dict[tuple[str, str], tuple[str, str]] = {(e.to.node, e.to.port): (e.from_.node, e.from_.port) for e in graph.edges}
        consumers: dict[str, set[str]] = {}
        types = {n.id: n.type for n in graph.nodes}
        self.node_types = types
        self.shared = dict(report.shared)  # call site -> node that owns the parameters
        self._out_index: dict[tuple[str, str], int | None] = {}
        built: dict[str, nn.Module] = {nid: registry.get_op(types[nid]).lower(report.resolved[nid]) for nid in report.order if nid not in report.shared}
        for nid in report.order:
            op = registry.get_op(types[nid])
            cfg = report.resolved[nid]
            outs = op.output_ports(cfg)
            for i, p in enumerate(outs):
                self._out_index[(nid, p)] = i if len(outs) > 1 else None
            if nid in report.shared:
                self.add_module(nid, SharedCall(built[report.shared[nid]], report.shared[nid]))
            else:
                if hasattr(built[nid], "bind_node"):
                    built[nid].bind_node(nid)  # diagnostic blocks know their full path
                self.add_module(nid, built[nid])
            srcs = [src_of[(nid, p)] for p in op.input_ports(cfg)]
            for s_ in srcs:
                consumers.setdefault(s_[0], set()).add(nid)
            self._plan.append((nid, srcs))
        self.input_ids = sorted(n for n in report.order if types[n] == "core.tensor_input")
        # a diagnostic block whose output nobody (alive) consumes is a tap on a wire, not an output of the model
        dead: set[str] = set()
        changed = True
        while changed:
            changed = False
            for n in report.order:
                if n not in dead and types[n].startswith("diag.") and consumers.get(n, set()) <= dead:
                    dead.add(n)
                    changed = True
        self.output_ids = sorted(n for n in report.order if n not in dead and consumers.get(n, set()) <= dead)
        self.sources = {nid: [s[0] for s in srcs] for nid, srcs in self._plan}

    def _fetch(self, values: dict, node: str, port: str):
        v = values[node]
        i = self._out_index.get((node, port))
        return v if i is None else v[i]

    def evaluate(self, given: dict[str, torch.Tensor]) -> dict[str, object]:
        """Run every node once, in order, and return every node's output (a tensor, or a tuple for multi-output nodes). Used by forward
        and by the debugger, so both execute exactly the same operations in the same order."""
        values: dict[str, object] = {}
        for nid, srcs in self._plan:
            values[nid] = self._modules[nid](given[nid]) if nid in given else self._modules[nid](*[self._fetch(values, s, p) for s, p in srcs])
        return values

    def forward(self, *args: torch.Tensor):
        if len(args) != len(self.input_ids):
            raise TypeError(f"expected {len(self.input_ids)} inputs {self.input_ids}, got {len(args)}")
        values = self.evaluate(dict(zip(self.input_ids, args)))
        outs = tuple(values[o] for o in self.output_ids)
        return outs[0] if len(outs) == 1 else outs

    def value_of(self, values: dict, node: str, port: str):
        return self._fetch(values, node, port)


def lower_graph(graph: Graph, report: Report | None = None) -> GraphModule:
    """Raises ExecutionBlocked if the graph is not executable (unknown op, mismatch, ...)."""
    if graph.graphKind != "model":
        from .types import Diagnostic

        raise ExecutionBlocked([Diagnostic("E_UNSUPPORTED_GRAPH_KIND", f"Graph kind '{graph.graphKind}' is not lowered to PyTorch.", path="/graphKind")])
    if graph.backend != "pytorch":
        from .types import Diagnostic

        raise ExecutionBlocked([Diagnostic("E_BACKEND_MISMATCH", f"This graph targets backend '{graph.backend}'; lower_graph builds PyTorch modules. "
                                           f"Use backends.compile_graph(graph, '{graph.backend}'), or set the graph's backend to 'pytorch'.", path="/backend")])
    if report is None or not report.ok:
        report = require_executable(graph)
    return GraphModule(report.graph or graph, report)


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
