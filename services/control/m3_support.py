"""Small helpers shared by the validation response and the Milestone 3 routes."""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from graph_core.schema import Graph
from graph_core.validate import Report
from libstore.store import content_hash
from training.spec import DEFAULT_STAGES, ProcedureSpec, check_procedure


def module_summaries(graph: Graph, report: Report) -> list[dict[str, Any]]:
    ex = report.expansion
    uses: dict[tuple[str, str], list[str]] = {}
    if ex is not None:
        for path, inst in ex.instances.items():
            if inst.kind == "core.composite" and inst.module and "/" not in path:
                uses.setdefault((inst.module, inst.version or ""), []).append(path)
            elif inst.kind in ("core.repeat", "core.select") and "/" not in path and inst.module:
                for mid, ver in zip(inst.module.split("|"), (inst.version or "").split("|")):
                    uses.setdefault((mid, ver), []).append(path)
    out = []
    for m in graph.modules:
        out.append({"id": m.id, "version": m.version, "description": m.description, "inputs": [p.model_dump(mode="json") for p in m.inputs],
                    "outputs": [o.model_dump(mode="json", by_alias=True) for o in m.outputs], "params": [p.model_dump(mode="json") for p in m.params],
                    "nodes": len(m.nodes), "contentHash": content_hash(m.model_dump(mode="json", by_alias=True)), "usedBy": uses.get((m.id, m.version), []),
                    "reduction": m.reduction})
    return out


def procedure_check(raw: dict[str, Any]) -> dict[str, Any]:
    try:
        spec = ProcedureSpec.model_validate(raw)
    except ValidationError as e:
        return {"valid": False, "diagnostics": [{"code": "E_PROC_CONFIG", "severity": "error", "nodeId": None, "port": None, "path": "/training",
                                                 "message": "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()), "fixes": []}], "stages": raw.get("stages", DEFAULT_STAGES)}
    ds = [d.to_json() for d in check_procedure(spec)]
    return {"valid": not any(d["severity"] == "error" for d in ds), "diagnostics": ds, "stages": spec.stages}
