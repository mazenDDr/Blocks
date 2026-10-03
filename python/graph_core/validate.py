"""Validation and shape inference. One pass produces diagnostics, shapes, and parameter counts."""
from __future__ import annotations

import heapq
import keyword
import re
from dataclasses import dataclass, field

import torch.nn as nn
from pydantic import BaseModel, ValidationError

from . import registry
from .schema import SCHEMA_VERSION, Graph
from .types import Diagnostic, Fix, OpError, TensorType

ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
RESERVED_IDS = {"torch", "nn", "self", "training"}


@dataclass
class Report:
    diagnostics: list[Diagnostic] = field(default_factory=list)
    order: list[str] = field(default_factory=list)  # deterministic topological order (acyclic part)
    output_types: dict[str, dict[str, TensorType]] = field(default_factory=dict)
    input_types: dict[str, dict[str, TensorType]] = field(default_factory=dict)
    params: dict[str, int] = field(default_factory=dict)
    resolved: dict[str, BaseModel] = field(default_factory=dict)  # configs with "infer" resolved
    unresolved_nodes: list[str] = field(default_factory=list)  # nodes that could not be typed

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors and not self.unresolved_nodes

    @property
    def total_params(self) -> int:
        return sum(self.params.values())

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "diagnostics": [d.to_json() for d in self.diagnostics],
            "shapes": {n: {p: t.to_json() for p, t in ports.items()} for n, ports in self.output_types.items()},
            "paramCounts": self.params,
            "totalParams": self.total_params,
        }


class ExecutionBlocked(Exception):
    def __init__(self, diagnostics: list[Diagnostic]):
        self.diagnostics = diagnostics
        lines = [f"{d.code} at {d.path or d.nodeId}: {d.message}" for d in diagnostics]
        super().__init__("Execution blocked:\n" + "\n".join(lines))


def _config_message(err: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(x) for x in e['loc']) or 'config'}: {e['msg']}" for e in err.errors())


def validate(graph: Graph) -> Report:
    if graph.graphKind == "tabular":
        from tabular.validate import validate_tabular

        return validate_tabular(graph)
    r = Report()
    diag = r.diagnostics

    def add(code, msg, node=None, port=None, path="", fixes=None, severity="error"):
        diag.append(Diagnostic(code, msg, severity, node, port, path, fixes or []))

    if graph.backend != "pytorch":
        add("E_UNSUPPORTED_BACKEND", f"Backend '{graph.backend}' is not implemented; only 'pytorch' is.", path="/backend")
    if graph.graphKind != "model":
        add("E_UNSUPPORTED_GRAPH_KIND", f"Graph kind '{graph.graphKind}' is not implemented; only 'model' and 'tabular' are.", path="/graphKind")
    if graph.schemaVersion.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        add("E_UNSUPPORTED_SCHEMA", f"Schema version {graph.schemaVersion} is not supported (supported major: {SCHEMA_VERSION}).", path="/schemaVersion")

    # ---- nodes
    ops: dict[str, registry.Operation] = {}  # nodes with a known op and parsed config
    cfgs: dict[str, BaseModel] = {}
    seen: set[str] = set()
    for n in graph.nodes:
        npath = f"/nodes/{n.id}"
        if n.id in seen:
            add("E_DUPLICATE_ID", f"Node id '{n.id}' is used more than once.", n.id, path=npath)
            continue
        seen.add(n.id)
        if not ID_RE.match(n.id) or keyword.iskeyword(n.id) or n.id in RESERVED_IDS or hasattr(nn.Module, n.id):
            add("E_BAD_ID", f"Node id '{n.id}' is not usable as a Python/PyTorch module name.", n.id, path=npath,
                fixes=[Fix("Rename the node to a letters/digits/underscore identifier that is not a reserved word")])
            continue
        op = registry.get_op(n.type)
        if op is None:
            add("E_UNKNOWN_OP", f"Operation '{n.type}' is not available. The node is preserved but cannot execute until it is resolved.",
                n.id, path=npath, fixes=[Fix("Install the plugin that provides this operation"), Fix("Replace the node with a supported operation")])
            continue
        if op.graph_kind != "model":
            add("E_OP_GRAPH_KIND", f"'{n.type}' belongs to a '{op.graph_kind}' graph and cannot appear in a 'model' graph.", n.id, path=npath)
            continue
        if n.version != op.version:
            add("E_UNSUPPORTED_VERSION", f"'{n.type}' version {n.version} is not available (have {op.version}). No silent upgrade is performed.",
                n.id, path=npath)
            continue
        try:
            cfgs[n.id] = op.Config.model_validate(n.config)
        except ValidationError as e:
            add("E_CONFIG", f"Invalid config for {n.type}: {_config_message(e)}", n.id, path=f"{npath}/config")
            continue
        ops[n.id] = op

    nodes_by_id = {n.id: n for n in graph.nodes if n.id in seen}

    # ---- edges
    incoming: dict[tuple[str, str], tuple[str, str]] = {}  # (node, in_port) -> (src node, src port)
    edge_ids: set[str] = set()
    for e in graph.edges:
        epath = f"/edges/{e.id}"
        if e.id in edge_ids:
            add("E_DUPLICATE_ID", f"Edge id '{e.id}' is used more than once.", path=epath)
            continue
        edge_ids.add(e.id)
        if e.kind != "tensor":
            add("E_PORT_TYPE", f"Edge kind '{e.kind}' is not supported; only 'tensor' is.", e.to.node, e.to.port, epath)
            continue
        bad = False
        for end, side in ((e.from_, "from"), (e.to, "to")):
            if end.node not in nodes_by_id:
                add("E_DANGLING_EDGE", f"Edge '{e.id}' {side} refers to missing node '{end.node}'.", path=f"{epath}/{side}")
                bad = True
        if bad:
            continue
        src, dst = ops.get(e.from_.node), ops.get(e.to.node)
        if src and e.from_.port not in src.outputs:
            add("E_UNKNOWN_PORT", f"Node '{e.from_.node}' has no output port '{e.from_.port}' (has {list(src.outputs)}).",
                e.from_.node, e.from_.port, f"{epath}/from")
            continue
        if dst and e.to.port not in dst.inputs:
            add("E_UNKNOWN_PORT", f"Node '{e.to.node}' has no input port '{e.to.port}' (has {list(dst.inputs)}).",
                e.to.node, e.to.port, f"{epath}/to")
            continue
        key = (e.to.node, e.to.port)
        if key in incoming:
            add("E_MULTIPLE_INPUTS", f"Input port '{e.to.port}' of '{e.to.node}' has more than one incoming edge.",
                e.to.node, e.to.port, f"/nodes/{e.to.node}/ports/{e.to.port}",
                [Fix(f"Delete edge '{e.id}' or the other edge into this port")])
            continue
        incoming[key] = (e.from_.node, e.from_.port)

    for nid, op in ops.items():
        for p in op.inputs:
            if (nid, p) not in incoming:
                add("E_MISSING_INPUT", f"Input '{p}' of '{nid}' is not connected.", nid, p, f"/nodes/{nid}/ports/{p}",
                    [Fix(f"Connect an output to {nid}.{p}")])

    # ---- deterministic topological order (ties broken by id); leftovers are on or after a cycle
    adj: dict[str, list[str]] = {i: [] for i in nodes_by_id}
    indeg = {i: 0 for i in nodes_by_id}
    for (dst, _), (src, _) in incoming.items():
        adj[src].append(dst)
        indeg[dst] += 1
    heap = [i for i, d in indeg.items() if d == 0]
    heapq.heapify(heap)
    while heap:
        i = heapq.heappop(heap)
        r.order.append(i)
        for j in sorted(adj[i]):
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(heap, j)
    cyc = sorted(set(nodes_by_id) - set(r.order))
    if cyc:
        add("E_CYCLE", f"The graph contains a cycle involving: {', '.join(cyc)}.", cyc[0], path=f"/nodes/{cyc[0]}",
            fixes=[Fix("Remove one edge on the cycle")])

    # ---- shape inference; nodes downstream of a failure are skipped, not re-reported
    for nid in r.order:
        op = ops.get(nid)
        if op is None:
            r.unresolved_nodes.append(nid)
            continue
        ins: dict[str, TensorType] = {}
        ready = True
        for p in op.inputs:
            src = incoming.get((nid, p))
            if src is None or src[0] not in r.output_types:
                ready = False
                break
            ins[p] = r.output_types[src[0]][src[1]]
        if not ready:
            r.unresolved_nodes.append(nid)
            continue
        try:
            cfg = op.resolve(cfgs[nid], ins)
            outs = op.infer_shape(cfg, ins)
            r.params[nid] = op.param_count(cfg, ins)
        except OpError as e:
            fixes = [Fix(f.label, f.node or nid, f.key, f.value) for f in e.fixes]
            add(e.code, e.message, nid, e.port, f"/nodes/{nid}/ports/{e.port}" if e.port else f"/nodes/{nid}", fixes)
            r.unresolved_nodes.append(nid)
            continue
        r.resolved[nid], r.input_types[nid], r.output_types[nid] = cfg, ins, outs
    r.unresolved_nodes += [i for i in cyc if i not in r.unresolved_nodes]
    return r


def require_executable(graph: Graph) -> Report:
    """Validate and raise ExecutionBlocked unless the graph can run."""
    r = validate(graph)
    if not r.ok:
        raise ExecutionBlocked(r.errors)
    return r
