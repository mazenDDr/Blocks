"""Validation and shape inference. One pass produces diagnostics, shapes, and parameter counts."""
from __future__ import annotations

import heapq
import keyword
import re
from dataclasses import dataclass, field

import torch.nn as nn
from pydantic import BaseModel, ValidationError

from . import composite, registry
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
    graph: Graph | None = None  # the flat graph that was validated (structural nodes expanded; node ids are full paths)
    expansion: composite.Expansion | None = None
    shared: dict[str, str] = field(default_factory=dict)  # flat node -> flat node whose parameters it uses

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
    if graph.graphKind == "domain":
        from tabular.validate import validate_tabular

        return validate_tabular(graph, "domain", "python")
    if graph.graphKind == "rl":
        from rl.validate import validate_rl

        return validate_rl(graph)
    if graph.graphKind == "agent":
        from agent.validate import validate_agent

        return validate_agent(graph)
    ex = composite.expand(graph)
    graph = ex.graph
    r = Report()
    r.graph, r.expansion = graph, ex
    diag = r.diagnostics
    diag.extend(ex.diagnostics)

    def add(code, msg, node=None, port=None, path="", fixes=None, severity="error"):
        diag.append(Diagnostic(code, msg, severity, node, port, path, fixes or []))

    from backends.registry import BACKEND_IDS

    if graph.backend not in BACKEND_IDS:
        add("E_UNSUPPORTED_BACKEND", f"Backend '{graph.backend}' is not implemented; model graphs support {list(BACKEND_IDS)}.", path="/backend")
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
        parts = n.id.split("/")
        top_only = len(parts) == 1
        if (not all(ID_RE.match(x) for x in parts) or (top_only and (keyword.iskeyword(n.id) or n.id in RESERVED_IDS or hasattr(nn.Module, n.id)))
                or n.id.startswith(".")):
            add("E_BAD_ID", f"Node id '{n.id}' is not usable as a Python/PyTorch module name.", n.id, path=npath,
                fixes=[Fix("Rename the node to a letters/digits/underscore identifier that is not a reserved word")])
            continue
        op = registry.get_op(n.type)
        if op is None and n.type.startswith("unresolved."):
            continue  # a structural node that could not be expanded: the expansion already reported why
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
        if op.backend in ("keras", "jax") and graph.backend in BACKEND_IDS and op.backend != graph.backend:
            add("E_BACKEND_OP", f"'{n.type}' is a {op.backend}-specific node and the graph targets '{graph.backend}'. It is not replaced by a similar operation; "
                f"switch the graph's backend to '{op.backend}' or remove the node.", n.id, path=f"/nodes/{n.id}",
                fixes=[Fix(f"Set backend to '{op.backend}'", None, "backend", op.backend)])
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
        src_out = src.output_ports(cfgs[e.from_.node]) if src else ()
        dst_in = dst.input_ports(cfgs[e.to.node]) if dst else ()
        if src and e.from_.port not in src_out:
            add("E_UNKNOWN_PORT", f"Node '{e.from_.node}' has no output port '{e.from_.port}' (has {list(src_out)}).",
                e.from_.node, e.from_.port, f"{epath}/from")
            continue
        if dst and e.to.port not in dst_in:
            add("E_UNKNOWN_PORT", f"Node '{e.to.node}' has no input port '{e.to.port}' (has {list(dst_in)}).",
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
        for p in op.input_ports(cfgs[nid]):
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
        for p in op.input_ports(cfgs[nid]):
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
    _check_sharing(r, graph, add)
    if graph.backend in ("keras", "jax"):
        _check_backend(r, graph, add)
    _check_signatures(r, ex, add)
    _check_boundaries(r, graph, incoming, add)
    return r


def _check_backend(r: Report, graph: Graph, add) -> None:
    """A18: a node the selected backend cannot run is an error NOW, with the adapter's stable code. (Backend-specific nodes on the wrong backend
    were already reported by the node loop.)"""
    from backends.compat import node_checks

    for nid, chk in node_checks(r, graph.backend).items():
        if chk.status == "unsupported" and chk.code != "E_BACKEND_OP":
            add(chk.code, f"{chk.reason} (backend '{graph.backend}')", nid, path=f"/nodes/{nid}",
                fixes=[Fix("Set backend to 'pytorch', which supports every operation of the model graph kind", None, "backend", "pytorch")])


def _check_boundaries(r: Report, graph: Graph, incoming: dict, add) -> None:
    """A non-differentiable code block is an explicit gradient boundary: say so when parameters sit upstream of it."""
    parents: dict[str, set[str]] = {}
    for (dst, _), (src, _p) in incoming.items():
        parents.setdefault(dst, set()).add(src)
    for n in graph.nodes:
        cfg = r.resolved.get(n.id)
        if n.type != "code.block" or cfg is None or cfg.interface.get("differentiable"):
            continue
        seen, stack = set(), list(parents.get(n.id, ()))
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            stack.extend(parents.get(p, ()))
        trained = sorted(p for p in seen if r.params.get(p, 0) > 0)
        if trained:
            add("W_NONDIFF_BOUNDARY", f"Code block '{n.id}' is not differentiable: no gradient flows through it, so the parameters of {', '.join(trained[:4])}"
                f"{'...' if len(trained) > 4 else ''} upstream of it receive no gradient from losses downstream of it.", n.id, path=f"/nodes/{n.id}", severity="warning")


def _check_sharing(r: Report, graph: Graph, add) -> None:
    """A13: a node with sharedWith uses the target's parameter tensors. The target must be the same operation with the same
    resolved config (so the tensors have the same shapes) and must not share itself."""
    by_id = {n.id: n for n in graph.nodes}
    for n in graph.nodes:
        if not n.sharedWith:
            continue
        t = by_id.get(n.sharedWith)
        if t is None:
            add("E_SHARE_TARGET", f"'{n.id}' shares parameters with '{n.sharedWith}', which does not exist.", n.id, path=f"/nodes/{n.id}")
            continue
        if t.sharedWith:
            add("E_SHARE_CHAIN", f"'{n.id}' shares with '{t.id}', which itself shares with '{t.sharedWith}'. Share with the original.", n.id, path=f"/nodes/{n.id}")
            continue
        if t.type != n.type:
            add("E_SHARE_MISMATCH", f"'{n.id}' ({n.type}) cannot share parameters with '{t.id}' ({t.type}).", n.id, path=f"/nodes/{n.id}")
            continue
        if n.id in r.resolved and t.id in r.resolved:
            a, b = r.resolved[n.id].model_dump(mode="json"), r.resolved[t.id].model_dump(mode="json")
            if a != b:
                diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
                add("E_SHARE_MISMATCH", f"'{n.id}' and '{t.id}' differ in {diff}, so their parameters have different shapes and cannot be shared.", n.id, path=f"/nodes/{n.id}")
                continue
            op = registry.get_op(t.type)
            if op is not None and op.has_state(r.resolved[t.id], r.params.get(t.id, 0)):
                r.shared[n.id] = t.id
                r.params[n.id] = 0  # counted once, at the original


def _spec_ok(spec_shape, spec_dtype, t: TensorType, binding: dict, what: str) -> str | None:
    if spec_dtype and t.dtype != spec_dtype:
        return f"{what} expects dtype {spec_dtype}, got {t.dtype}"
    if spec_shape is None:
        return None
    if len(spec_shape) != len(t.shape):
        return f"{what} expects rank {len(spec_shape)} {spec_shape}, got rank {len(t.shape)} {list(t.shape)}"
    for i, (want, got) in enumerate(zip(spec_shape, t.shape)):
        if want is None:
            continue
        if isinstance(want, str) and want != "N":
            if binding.setdefault(want, got) != got:
                return f"{what} dimension {i} is named '{want}' and was {binding[want]} elsewhere, but is {got} here"
        elif want != got:
            return f"{what} expects dimension {i} = {want}, got {got}"
    return None


def _check_signatures(r: Report, ex: composite.Expansion, add) -> None:
    """Typed signatures of composite instances, loop-carried state and select branches (checked after inference)."""
    def typ(ep):
        return r.output_types.get(ep[0], {}).get(ep[1]) if ep else None

    bindings: dict[str, dict] = {}
    for path, name, src, spec in ex.sig_in:
        t = typ(src)
        if t is None:
            continue
        msg = _spec_ok(spec.shape, spec.dtype, t, bindings.setdefault(path, {}), f"input '{name}' of '{path}'")
        if msg:
            add("E_PORT_TYPE", msg + ".", path, name, f"/nodes/{path}/ports/{name}")
    for path, name, src, spec in ex.sig_out:
        t = typ(src)
        if t is None:
            continue
        msg = _spec_ok(spec.shape, spec.dtype, t, bindings.setdefault(path, {}), f"output '{name}' of '{path}'")
        if msg:
            add("E_PORT_TYPE", msg + ".", path, name, f"/nodes/{path}/ports/{name}")
    for path, ep in ex.pred_checks:
        t = typ(ep)
        if t is not None and (t.dtype != "bool" or len(t.shape) != 0):
            add("E_BRANCH_PRED", f"The predicate of '{path}' must be a scalar bool tensor (shape [], dtype bool), got {list(t.shape)} {t.dtype}. "
                "Reduce it with tensor.any_all.", path, "pred", f"/nodes/{path}/ports/pred")
    for path, name, a, b in ex.branch_checks:
        ta, tb = typ(a), typ(b)
        if ta is not None and tb is not None and ta != tb:
            add("E_BRANCH_TYPE", f"Output '{name}' of '{path}' has type {list(ta.shape)} {ta.dtype} in the then-branch but {list(tb.shape)} {tb.dtype} "
                "in the otherwise-branch; branches must return the same type.", path, name, f"/nodes/{path}/ports/{name}")
    for path, k, o, oep, p, init in ex.carry_checks:
        tn, ti = typ(oep), typ(init)
        if tn is not None and ti is not None and tn != ti:
            add("E_CARRY_TYPE", f"Loop-carried state '{p}' of '{path}' is {list(ti.shape)} {ti.dtype} on entry but iteration {k - 1} produces "
                f"{list(tn.shape)} {tn.dtype}; the body must preserve the type of carried state.", path, p, f"/nodes/{path}/ports/{p}")


def require_executable(graph: Graph) -> Report:
    """Validate and raise ExecutionBlocked unless the graph can run."""
    r = validate(graph)
    if not r.ok:
        raise ExecutionBlocked(r.errors)
    return r
