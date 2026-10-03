"""Deterministic, readable PyTorch source generated from the validated graph.

The text is for reading and exporting; it is never executed by this project. Every statement
carries a `# node: <id>` comment, which is the source map back to the graph."""
from __future__ import annotations

import re

from . import registry
from .hashing import semantic_hash
from .schema import Graph
from .validate import require_executable

_NODE_RE = re.compile(r"#\s*node:\s*([A-Za-z0-9_]+)")


def generate_pytorch(graph: Graph) -> str:
    report = require_executable(graph)
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
        if ctor:
            init.append(f"        self.{nid} = {ctor}  # node: {nid}")
        if nid in inputs:
            body.append(f"        # node: {nid} (forward argument)")
        else:
            args = [src_of[(nid, p)] for p in op.inputs]
            expr = fwd.format(*args, m=f"self.{nid}")
            body.append(f"        {nid} = {expr}  # node: {nid}")

    lines = [
        f"# Generated from graph {semantic_hash(graph)}. Regenerate rather than edit.",
        "import torch",
        "import torch.nn as nn",
        "",
        "",
        "class Model(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
        *init,
        "",
        f"    def forward(self, {', '.join(inputs)}):",
        *body,
        f"        return {', '.join(outputs)}",
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
