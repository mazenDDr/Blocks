"""Validation of `tabular` graphs: typed wires, schema inference, partition tracking and the leakage policy.

Same diagnostic contract as model graphs ({code, severity, nodeId, port, path, message, fixes[]}); the value flowing on an
edge is a typed object (table, fit_state, model, ...) instead of a tensor, and edge.kind must equal that kind."""
from __future__ import annotations

import heapq

from pydantic import BaseModel, ValidationError

from graph_core import registry
from graph_core.schema import SCHEMA_VERSION, Graph
from graph_core.types import Diagnostic, Fix, OpError
from graph_core.validate import ID_RE, Report, _config_message

from .core import TabularOperation, VType

BACKEND = "python"


def validate_tabular(graph: Graph) -> Report:
    r = Report()
    diag = r.diagnostics

    def add(code, msg, node=None, port=None, path="", fixes=None, severity="error"):
        diag.append(Diagnostic(code, msg, severity, node, port, path, fixes or []))

    if graph.backend != BACKEND:
        add("E_UNSUPPORTED_BACKEND", f"Tabular graphs use backend '{BACKEND}' (pandas, scikit-learn, SciPy), got '{graph.backend}'.", path="/backend")
    if graph.schemaVersion.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        add("E_UNSUPPORTED_SCHEMA", f"Schema version {graph.schemaVersion} is not supported.", path="/schemaVersion")

    ops: dict[str, TabularOperation] = {}
    cfgs: dict[str, BaseModel] = {}
    seen: set[str] = set()
    for n in graph.nodes:
        npath = f"/nodes/{n.id}"
        if n.id in seen:
            add("E_DUPLICATE_ID", f"Node id '{n.id}' is used more than once.", n.id, path=npath)
            continue
        seen.add(n.id)
        if not ID_RE.match(n.id):
            add("E_BAD_ID", f"Node id '{n.id}' must be letters, digits and underscores, starting with a letter.", n.id, path=npath)
            continue
        op = registry.get_op(n.type)
        if op is None:
            add("E_UNKNOWN_OP", f"Operation '{n.type}' is not available. The node is preserved but cannot execute until it is resolved.",
                n.id, path=npath, fixes=[Fix("Replace the node with a supported operation")])
            continue
        if op.graph_kind != "tabular":
            add("E_OP_GRAPH_KIND", f"'{n.type}' belongs to a '{op.graph_kind}' graph and cannot appear in a 'tabular' graph.", n.id, path=npath)
            continue
        if n.version != op.version:
            add("E_UNSUPPORTED_VERSION", f"'{n.type}' version {n.version} is not available (have {op.version}).", n.id, path=npath)
            continue
        try:
            cfgs[n.id] = op.Config.model_validate(n.config)
        except ValidationError as e:
            add("E_CONFIG", f"Invalid config for {n.type}: {_config_message(e)}", n.id, path=f"{npath}/config")
            continue
        ops[n.id] = op  # type: ignore[assignment]

    node_ids = {n.id for n in graph.nodes if n.id in seen}

    incoming: dict[tuple[str, str], tuple[str, str]] = {}
    edge_ids: set[str] = set()
    for e in graph.edges:
        epath = f"/edges/{e.id}"
        if e.id in edge_ids:
            add("E_DUPLICATE_ID", f"Edge id '{e.id}' is used more than once.", path=epath)
            continue
        edge_ids.add(e.id)
        bad = False
        for end, side in ((e.from_, "from"), (e.to, "to")):
            if end.node not in node_ids:
                add("E_DANGLING_EDGE", f"Edge '{e.id}' {side} refers to missing node '{end.node}'.", path=f"{epath}/{side}")
                bad = True
        if bad:
            continue
        src, dst = ops.get(e.from_.node), ops.get(e.to.node)
        if src and e.from_.port not in src.outputs:
            add("E_UNKNOWN_PORT", f"Node '{e.from_.node}' has no output port '{e.from_.port}' (has {list(src.outputs)}).", e.from_.node, e.from_.port, f"{epath}/from")
            continue
        if dst and e.to.port not in dst.inputs:
            add("E_UNKNOWN_PORT", f"Node '{e.to.node}' has no input port '{e.to.port}' (has {list(dst.inputs)}).", e.to.node, e.to.port, f"{epath}/to")
            continue
        if src and dst:
            sk, dk = src.out_kinds[e.from_.port], dst.in_kinds[e.to.port]
            if sk != dk:
                add("E_PORT_TYPE", f"{e.from_.node}.{e.from_.port} produces a '{sk}' but {e.to.node}.{e.to.port} needs a '{dk}'. "
                    "Wires of different kinds never mean the same thing.", e.to.node, e.to.port, f"/nodes/{e.to.node}/ports/{e.to.port}",
                    [Fix(f"Connect a '{dk}' output to {e.to.node}.{e.to.port}")])
                continue
            if e.kind != sk:
                add("E_EDGE_KIND", f"Edge '{e.id}' has kind '{e.kind}' but carries a '{sk}'.", e.to.node, e.to.port, epath,
                    [Fix(f"Set the edge kind to '{sk}'")])
                continue
        key = (e.to.node, e.to.port)
        if key in incoming:
            add("E_MULTIPLE_INPUTS", f"Input port '{e.to.port}' of '{e.to.node}' has more than one incoming edge.", e.to.node, e.to.port,
                f"/nodes/{e.to.node}/ports/{e.to.port}", [Fix(f"Delete edge '{e.id}' or the other edge into this port")])
            continue
        incoming[key] = (e.from_.node, e.from_.port)

    for nid, op in ops.items():
        for p in op.inputs:
            if (nid, p) not in incoming:
                add("E_MISSING_INPUT", f"Input '{p}' of '{nid}' is not connected.", nid, p, f"/nodes/{nid}/ports/{p}", [Fix(f"Connect an output to {nid}.{p}")])

    adj: dict[str, list[str]] = {i: [] for i in node_ids}
    indeg = {i: 0 for i in node_ids}
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
    cyc = sorted(node_ids - set(r.order))
    if cyc:
        add("E_CYCLE", f"The graph contains a cycle involving: {', '.join(cyc)}.", cyc[0], path=f"/nodes/{cyc[0]}", fixes=[Fix("Remove one edge on the cycle")])

    types: dict[str, dict[str, VType]] = r.output_types  # type: ignore[assignment]
    for nid in r.order:
        op = ops.get(nid)
        if op is None:
            r.unresolved_nodes.append(nid)
            continue
        ins: dict[str, VType] = {}
        ready = True
        for p in op.inputs:
            src = incoming.get((nid, p))
            if src is None or src[0] not in types:
                ready = False
                break
            ins[p] = types[src[0]][src[1]]
        if not ready:
            r.unresolved_nodes.append(nid)
            continue
        try:
            cfg = cfgs[nid]
            outs = op.infer(cfg, ins, nid)
            r.params[nid] = op.param_count(cfg, ins)  # type: ignore[arg-type]
            for code, msg, port in op.warnings(cfg, ins, nid):
                add(code, msg, nid, port, f"/nodes/{nid}/ports/{port}" if port else f"/nodes/{nid}", severity="warning")
        except OpError as e:
            fixes = [Fix(f.label, f.node or nid, f.key, f.value) for f in e.fixes]
            add(e.code, e.message, nid, e.port, f"/nodes/{nid}/ports/{e.port}" if e.port else f"/nodes/{nid}", fixes)
            r.unresolved_nodes.append(nid)
            continue
        r.resolved[nid], r.input_types[nid], types[nid] = cfg, ins, outs  # type: ignore[assignment]
    r.unresolved_nodes += [i for i in cyc if i not in r.unresolved_nodes]
    return r
