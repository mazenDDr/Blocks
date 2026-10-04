"""Graph / run-config diff between two runs (structural and parameter diff, VISION 19)."""
from __future__ import annotations

import json
from typing import Any

from artifact_store import ArtifactStore

IDENTITY = {"trial", "project_id", "source_pins"}


def _flat(prefix: str, v: Any, out: dict[str, Any]) -> None:
    if isinstance(v, dict):
        for k, x in v.items():
            _flat(f"{prefix}.{k}" if prefix else k, x, out)
        if not v:
            out[prefix] = {}
    elif isinstance(v, list) and any(isinstance(x, (dict, list)) for x in v):
        for i, x in enumerate(v):
            _flat(f"{prefix}.{i}", x, out)
        if not v:
            out[prefix] = []
    else:
        out[prefix] = v


def graph_of(store: ArtifactStore, run_id: str) -> dict[str, Any] | None:
    a = store.artifacts(run_id, "graph")
    return json.loads(store.read_artifact(a[0]["sha256"])) if a else None


def diff_graphs(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    if a is None or b is None:
        return {"available": False, "reason": "a run has no stored graph"}
    na, nb = {n["id"]: n for n in a["nodes"]}, {n["id"]: n for n in b["nodes"]}
    changes = []
    for nid in sorted(set(na) | set(nb)):
        if nid not in na:
            changes.append({"kind": "node_added", "node": nid, "type": nb[nid]["type"]})
        elif nid not in nb:
            changes.append({"kind": "node_removed", "node": nid, "type": na[nid]["type"]})
        else:
            if na[nid]["type"] != nb[nid]["type"]:
                changes.append({"kind": "type_changed", "node": nid, "from": na[nid]["type"], "to": nb[nid]["type"]})
            fa, fb = {}, {}
            _flat("", na[nid].get("config", {}), fa)
            _flat("", nb[nid].get("config", {}), fb)
            for k in sorted(set(fa) | set(fb)):
                if fa.get(k) != fb.get(k):
                    changes.append({"kind": "config", "node": nid, "type": nb[nid]["type"], "field": k, "from": fa.get(k), "to": fb.get(k)})
    ea = {(e["from"]["node"], e["from"]["port"], e["to"]["node"], e["to"]["port"]) for e in a["edges"]}
    eb = {(e["from"]["node"], e["from"]["port"], e["to"]["node"], e["to"]["port"]) for e in b["edges"]}
    for e in sorted(ea - eb):
        changes.append({"kind": "edge_removed", "edge": f"{e[0]}.{e[1]} -> {e[2]}.{e[3]}"})
    for e in sorted(eb - ea):
        changes.append({"kind": "edge_added", "edge": f"{e[0]}.{e[1]} -> {e[2]}.{e[3]}"})
    return {"available": True, "changes": changes}


def diff_runs(store: ArtifactStore, a: str, b: str) -> dict[str, Any]:
    ra, rb = store.get_run(a), store.get_run(b)
    g = diff_graphs(graph_of(store, a), graph_of(store, b))
    ca, cb = {k: v for k, v in ra["config"].items() if k not in IDENTITY}, {k: v for k, v in rb["config"].items() if k not in IDENTITY}
    fa, fb = {}, {}
    _flat("", ca, fa)
    _flat("", cb, fb)
    run_changes = [{"kind": "run_config", "field": k, "from": fa.get(k), "to": fb.get(k)} for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
    # data identity: recorded source snapshots / hashes
    def srcs(rid):
        return {e["node_id"]: e["data"].get("snapshotId") or e["data"].get("sha256") for e in store.events(rid, -1, ("source_snapshot_recorded", "source_recorded"))}

    sa, sb = srcs(a), srcs(b)
    data_changes = [{"kind": "source_identity", "node": n, "from": sa.get(n), "to": sb.get(n)} for n in sorted(set(sa) | set(sb)) if sa.get(n) != sb.get(n)]
    n_changes = len(run_changes) + len(data_changes) + (len(g["changes"]) if g.get("available") else 0)
    ident = {"a": ra["config"].get("trial"), "b": rb["config"].get("trial")}
    return {"a": a, "b": b, "graph": g, "runConfig": run_changes, "data": data_changes, "identity": ident, "changeCount": n_changes,
            "warning": (f"{n_changes} things differ between these runs; an outcome difference cannot be attributed to any single one of them." if n_changes > 1 else None)}
