"""Read-only inspection of tabular runs: bounded table slices, profiles, fitted state, coefficients, metrics, test results.

Everything returned was recorded by the worker during the run (artifacts `node_output` / `node_summary`); nothing is recomputed and
nothing is written. Each response carries provenance: run id, graph hash, node, port, partition, source data hash and split seed."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException

from artifact_store import ArtifactStore
from graph_core import registry
from graph_core.schema import Graph

from tabular.core import clean

MAX_ROWS = 200
DEFAULT_ROWS = 50

KIND_OF_VIEW = {"profile": "profile", "fit_state": "fit_state", "coefficients": "coefficients", "metrics": "metrics", "test_result": "test_result",
                "distribution": "distribution", "tail": "tail", "number": "number"}


def _bad(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, {"code": code, "message": message})


class TabularRun:
    def __init__(self, store: ArtifactStore, run_id: str):
        row = store.get_run(run_id)
        if row is None:
            raise _bad(404, "not_found", f"unknown run '{run_id}'")
        self.store, self.run_id, self.row = store, run_id, row
        g = store.artifacts(run_id, "graph")
        self.graph = Graph.model_validate(json.loads(store.read_artifact(g[0]["sha256"]))) if g else None
        self.outputs = {(a["meta"]["node"], a["meta"]["port"]): a for a in store.artifacts(run_id, "node_output")}
        self.summaries = {a["meta"]["node"]: a for a in store.artifacts(run_id, "node_summary")}
        self.types = {n.id: n.type for n in self.graph.nodes} if self.graph else {}

    @property
    def graph_hash(self) -> str:
        return self.row["graph_hash"]

    def prov(self, node: str | None = None, port: str | None = None, **extra: Any) -> dict[str, Any]:
        p: dict[str, Any] = {"runId": self.run_id, "graphHash": self.graph_hash, "nodeId": node, "port": port, "source": "recorded by the worker during the run"}
        p.update(extra)
        return p

    def need_node(self, node: str | None) -> str:
        if not node:
            raise _bad(422, "inspect_invalid", "'node' is required")
        if node not in self.types:
            raise _bad(422, "inspect_invalid", f"run {self.run_id} has no node '{node}'")
        return node


def table_artifact_path(store: ArtifactStore, run_id: str, node: str, port: str) -> Path:
    tr = TabularRun(store, run_id)
    a = tr.outputs.get((node, port))
    if a is None or a["meta"].get("valueKind") != "table":
        raise _bad(404, "not_found", f"run {run_id} recorded no table at {node}.{port}")
    return store.path_of(a["sha256"])


def _unavailable(tr: TabularRun, kind: str, message: str, node: str | None, port: str | None = None, reason: str = "not_recorded") -> dict[str, Any]:
    return {"available": False, "kind": kind, "reason": reason, "message": message, "provenance": tr.prov(node, port)}


def inspect_tabular(store: ArtifactStore, run_id: str, req) -> dict[str, Any]:
    tr = TabularRun(store, run_id)
    kind = req.kind
    if kind in ("weights", "activations", "sample_loss", "confusion"):
        raise _bad(422, "inspect_invalid", f"'{kind}' inspection applies to model graphs")
    node = tr.need_node(req.node)
    if kind == "table":
        return _table(tr, node, req)
    op = registry.get_op(tr.types[node])
    want = getattr(op, "summary_kind", "step")
    if kind != "summary" and KIND_OF_VIEW.get(kind) != want:
        raise _bad(422, "inspect_invalid", f"node '{node}' ({tr.types[node]}) produces a '{want}' view, not '{kind}'")
    a = tr.summaries.get(node)
    if a is None:
        return _unavailable(tr, kind, f"Node '{node}' has no recorded result in run {run_id} (it did not run; see the run's failure).", node)
    data = json.loads(store.read_artifact(a["sha256"]))
    prov = tr.prov(node, None, summaryKind=want, summarySha256=a["sha256"], nodeType=tr.types[node])
    fo = data.get("fittedOn") if isinstance(data, dict) else None
    if fo:
        prov["fittedOn"] = fo
    if isinstance(data, dict) and data.get("path") and data.get("sha256"):
        prov["sourceSha256"] = data["sha256"]
    return {"available": True, "kind": kind, "node": node, "nodeType": tr.types[node], "summaryKind": want, "data": data, "provenance": prov}


def _table(tr: TabularRun, node: str, req) -> dict[str, Any]:
    ports = [p for (n, p), a in tr.outputs.items() if n == node and a["meta"].get("valueKind") == "table"]
    if not ports:
        return _unavailable(tr, "table", f"Node '{node}' recorded no table in this run.", node)
    port = req.port or ports[0]
    a = tr.outputs.get((node, port))
    if a is None or a["meta"].get("valueKind") != "table":
        raise _bad(422, "inspect_invalid", f"node '{node}' has no table port '{port}' (has {ports})")
    meta = a["meta"]
    total = int(meta["rows"])
    limit = max(1, min(req.limit or DEFAULT_ROWS, MAX_ROWS))
    offset = max(0, min(req.offset, max(0, total - 1)))
    path = tr.store.path_of(a["sha256"])
    df = pd.read_csv(path, skiprows=range(1, offset + 1), nrows=limit)
    ids = [int(i) for i in df.pop("row_id")]
    lineage = meta.get("lineage") or {}
    prov = tr.prov(node, port, partition=meta.get("partition"), artifactSha256=a["sha256"], rowsTotal=total)
    if lineage.get("source"):
        prov["source"] = {"node": lineage["source"].get("node"), "path": lineage["source"].get("path"), "sha256": lineage["source"].get("sha256")}
        prov["sourceSha256"] = lineage["source"].get("sha256")
    if lineage.get("split"):
        sp = lineage["split"]
        prov["split"] = {k: sp.get(k) for k in ("node", "seed", "validationFraction", "nTrain", "nValidation")}
    return {"available": True, "kind": "table", "node": node, "port": port, "columns": meta["columns"], "partition": meta.get("partition"),
            "rowIds": ids, "rows": [[clean(v) for v in r] for r in df.itertuples(index=False, name=None)],
            "total": total, "offset": offset, "limit": len(df), "truncated": offset + len(df) < total, "provenance": prov}
