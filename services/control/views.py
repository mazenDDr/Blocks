"""Per-node views for the editor: shapes, parameter counts, diagnostics, resolved config, explain data.

The validated graph is the flat graph (structural nodes expanded, every inner node addressed by its full path). The editor works with the top-level
graph, so top-level entries are returned under `nodes` (a composite instance gets a synthesized entry: its ports, the types crossing them, the sum of
its members' parameters, their diagnostics and an equation composed from its members' own explain data) and everything nested under `flat`."""
from __future__ import annotations

import json
from typing import Any

from fastapi.encoders import jsonable_encoder

from graph_core import registry
from graph_core.composite import Expansion, Instance
from graph_core.schema import Graph
from graph_core.validate import Report


def _explain(op, report: Report, nid: str) -> dict[str, Any]:
    try:
        return jsonable_encoder(op.explain(report.resolved[nid], report.input_types[nid], report.output_types[nid]))
    except Exception as e:  # noqa: BLE001  (an explain bug must not break validation)
        return {"error": str(e)}


def flat_entry(graph: Graph, report: Report, n, diags: dict[str, list]) -> dict[str, Any]:
    op = registry.get_op(n.type)
    entry: dict[str, Any] = {"known": op is not None, "diagnostics": diags.get(n.id, []), "typed": n.id in report.output_types, "type": n.type, "nodeVersion": n.version}
    if n.sharedWith:
        entry["sharedWith"] = n.sharedWith
    if n.id in report.output_types and op is not None:
        cfg = report.resolved[n.id]
        entry["inputShapes"] = {p: t.to_json() for p, t in report.input_types[n.id].items()}
        entry["outputShapes"] = {p: t.to_json() for p, t in report.output_types[n.id].items()}
        entry["params"] = report.params[n.id]
        entry["resolvedConfig"] = _config(cfg)
        entry["explain"] = _explain(op, report, n.id)
        entry["inputPorts"], entry["outputPorts"] = list(op.input_ports(cfg)), list(op.output_ports(cfg))
        entry["sharesParameters"] = n.id in report.shared
        if hasattr(op, "observation_only"):
            entry["observationOnly"] = bool(op.observation_only)
    elif op is not None:
        entry["inputPorts"], entry["outputPorts"] = list(op.inputs), list(op.outputs)
    return entry


def _non_default(op, cfg) -> dict[str, Any]:
    """Settings of a node that differ from the operation's defaults (compact summary for composed equations)."""
    try:
        base = op.Config().model_dump(mode="json")
    except Exception:  # noqa: BLE001
        base = {}
    cur = cfg.model_dump(mode="json")
    return {k: v for k, v in cur.items() if k != "interface" and base.get(k) != v}


def _config(cfg) -> dict[str, Any]:
    d = cfg.model_dump(mode="json")
    if isinstance(d.get("interface"), dict):  # the source is shown in the code editor, not in the config panel
        d["interface"] = {k: v for k, v in d["interface"].items() if k != "source"}
    return d


def instance_entry(report: Report, inst: Instance, diags: dict[str, list], graph_node=None) -> dict[str, Any]:
    flat = report.graph
    members = [m for m in inst.members if m in report.params or m in report.output_types]
    direct = [m for m in inst.members if m.rsplit("/", 1)[0] == inst.path]
    own_diags = list(diags.get(inst.path, []))
    for m in inst.members:
        own_diags += [d for d in diags.get(m, [])]
    entry: dict[str, Any] = {"known": True, "diagnostics": own_diags, "structural": True, "kind": inst.kind, "module": inst.module, "version": inst.version,
                             "inputPorts": inst.inputs, "outputPorts": inst.outputs, "members": inst.members, "children": inst.children, "note": inst.note,
                             "sharedWith": inst.shared_with, "iterations": inst.iterations, "termination": inst.termination}
    outs = {p: report.output_types[ep[0]][ep[1]].to_json() for p, ep in inst.out_map.items() if ep[0] in report.output_types and ep[1] in report.output_types[ep[0]]}
    ins = {}
    for p, ep in inst.in_src.items():
        if ep and ep[0] in report.output_types and ep[1] in report.output_types[ep[0]]:
            ins[p] = report.output_types[ep[0]][ep[1]].to_json()
    entry["typed"] = len(outs) == len(inst.outputs) and bool(inst.outputs)
    entry["inputShapes"], entry["outputShapes"] = ins, outs
    entry["params"] = sum(report.params.get(m, 0) for m in inst.members)
    steps = []
    for m in direct:
        n = flat.node(m)
        op = registry.get_op(n.type)
        if op is None or m not in report.output_types:
            continue
        ex = _explain(op, report, m)
        out_t = next(iter(report.output_types[m].values()))
        steps.append({"id": m, "local": m.rsplit("/", 1)[-1], "type": n.type, "equation": ex.get("equation", ""), "params": report.params.get(m, 0),
                      "config": _non_default(op, report.resolved[m]), "outputShape": out_t.to_json()["shape"], "sharesParameters": m in report.shared})
    for c in inst.children:
        ci = report.expansion.instances.get(c)
        if ci is not None:
            steps.append({"id": c, "local": c.rsplit("/", 1)[-1], "type": ci.kind, "equation": f"module {ci.module} {ci.version}: {', '.join(ci.inputs)} -> {', '.join(ci.outputs)}",
                          "params": sum(report.params.get(m, 0) for m in ci.members), "sharesParameters": bool(ci.shared_with)})
    entry["explain"] = {"equation": f"{inst.kind.split('.')[-1]} of module {inst.module} {inst.version}: " + "; ".join(
                            f"{s['local']} [{s['type']}{(' ' + json.dumps(s['config'], sort_keys=True)) if s.get('config') else ''} -> {s.get('outputShape', '')}]: {s['equation']}" for s in steps),
                        "steps": steps, "parameters": {"formula": "sum of the members' parameters", "terms": [{"name": s["local"], "shape": [], "count": s["params"]} for s in steps if s["params"]],
                                                       "total": entry["params"]},
                        "note": inst.note, "composite": {"instances": inst.children, "iterations": inst.iterations}}
    return entry


def boundary_edges(flat: Graph, inst: Instance) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Wires inside an instance (both ends are members) and the wires that cross its border, labelled with the instance port they belong to.
    The editor uses them to draw an expanded instance with its inner nodes."""
    members = set(inst.members)
    inner, boundary = [], []
    for e in flat.edges:
        a, b = e.from_.node in members, e.to.node in members
        if a and b:
            inner.append({"id": e.id, "from": {"node": e.from_.node, "port": e.from_.port}, "to": {"node": e.to.node, "port": e.to.port}})
        elif b:
            port = next((p for p, ep in inst.in_src.items() if ep and tuple(ep) == (e.from_.node, e.from_.port)), None)
            if port is not None:
                boundary.append({"kind": "in", "port": port, "node": e.to.node, "nodePort": e.to.port})
    for p, ep in inst.out_map.items():
        boundary.append({"kind": "out", "port": p, "node": ep[0], "nodePort": ep[1]})
    return inner, boundary


def node_view(graph: Graph, report: Report) -> dict[str, Any]:
    """Backward compatible with the Phase 1 shape: {node id: entry} for the nodes of `graph`. Nested nodes and structural instances: see `views`."""
    return views(graph, report)["nodes"]


def views(graph: Graph, report: Report) -> dict[str, Any]:
    flat = report.graph or graph
    ex: Expansion | None = report.expansion
    diags: dict[str, list] = {}
    for d in report.diagnostics:
        if d.nodeId:
            diags.setdefault(d.nodeId, []).append(d.to_json())
    nodes: dict[str, Any] = {}
    nested: dict[str, Any] = {}
    instances = ex.instances if ex is not None else {}
    for n in flat.nodes:
        entry = flat_entry(graph, report, n, diags)
        (nested if "/" in n.id else nodes)[n.id] = entry
    for n in graph.nodes:
        if n.id in instances:
            nodes[n.id] = instance_entry(report, instances[n.id], diags)
        elif n.id not in nodes:
            op = registry.get_op(n.type)
            nodes[n.id] = {"known": op is not None, "diagnostics": diags.get(n.id, []), "typed": False,
                           "inputPorts": list(op.inputs) if op else [], "outputPorts": list(op.outputs) if op else []}
    inst_json = {}
    for path, inst in instances.items():
        e = instance_entry(report, inst, diags)
        inner, boundary = boundary_edges(flat, inst)
        inst_json[path] = {**inst.to_json(), "typed": e["typed"], "params": e["params"], "inputShapes": e["inputShapes"], "outputShapes": e["outputShapes"],
                           "diagnostics": e["diagnostics"], "explain": e["explain"], "innerEdges": inner, "boundary": boundary}
    return {"nodes": nodes, "flat": nested, "instances": inst_json}
