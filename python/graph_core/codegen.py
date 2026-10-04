"""Deterministic, readable PyTorch source generated from the validated graph.

The text is for reading and exporting; it is never executed by this project. Every statement
carries a `# node: <id>` comment, which is the source map back to the graph."""
from __future__ import annotations

import re

from . import registry
from .hashing import semantic_hash
from .schema import Graph
from .validate import require_executable

_NODE_RE = re.compile(r"#\s*node:\s*([A-Za-z0-9_/]+)")


def _py(nid: str) -> str:
    """Node ids that are paths (res1/conv_a) become identifiers (res1__conv_a); the `# node:` comment keeps the real path."""
    return nid.replace("/", "__")


def generate_pytorch(graph: Graph) -> str:
    if graph.graphKind != "model":
        from .types import Diagnostic
        from .validate import ExecutionBlocked

        raise ExecutionBlocked([Diagnostic("E_UNSUPPORTED_GRAPH_KIND", f"Graph kind '{graph.graphKind}' has no PyTorch export.", path="/graphKind")])
    report = require_executable(graph)
    orig, graph = graph, (report.graph or graph)
    types = {n.id: n.type for n in graph.nodes}
    src_of = {(e.to.node, e.to.port): e.from_.node for e in graph.edges}
    used = {e.from_.node for e in graph.edges}
    inputs = sorted(n for n in report.order if types[n] == "core.tensor_input")
    outputs = sorted(n for n in report.order if n not in used)

    init, body = [], []
    for nid in report.order:
        op = registry.get_op(types[nid])
        cfg = report.resolved[nid]
        ctor, fwd = op.codegen(cfg)
        py = _py(nid)
        if nid in report.shared:
            ctor = f"self.{_py(report.shared[nid])}"  # the same module object: the parameters are shared
            fwd = fwd if "{m}" in fwd else "{m}(" + ", ".join("{%d}" % i for i in range(len(op.input_ports(cfg)))) + ")"
        if ctor:
            note = f" (shares parameters with {report.shared[nid]})" if nid in report.shared else ""
            init.append(f"        self.{py} = {ctor}  # node: {nid}{note}")
        if nid in inputs:
            body.append(f"        # node: {nid} (forward argument)")
        else:
            args = [_py(src_of[(nid, p)]) for p in op.input_ports(cfg)]
            expr = fwd.format(*args, m=f"self.{py}")
            body.append(f"        {py} = {expr}  # node: {nid}")

    lines = [
        f"# Generated from graph {semantic_hash(orig)}. Regenerate rather than edit.",
        "import torch",
        "import torch.nn as nn",
        "",
        "",
        "class Model(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
        *init,
        "",
        f"    def forward(self, {', '.join(_py(i) for i in inputs)}):",
        *body,
        f"        return {', '.join(_py(o) for o in outputs)}",
        "",
    ]
    return "\n".join(lines)


def source_map(code: str) -> dict[str, list[int]]:
    """node id -> 1-based line numbers carrying its `# node:` comment."""
    out: dict[str, list[int]] = {}
    for i, line in enumerate(code.splitlines(), 1):
        m = _NODE_RE.search(line)
        if m:
            out.setdefault(m.group(1), []).append(i)
    return out
