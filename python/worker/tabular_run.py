"""Run a `tabular` graph: validate, execute node by node in this (worker) process, record events and artifacts with provenance.

Events: run_queued, run_preparing, run_started (source data hashes, graph hash), node_started, node_finished (summary), node_failed,
validation_error, cancel_acknowledged, error, run_finished. Artifacts (kind `node_output`, one per output port) carry node, port,
value kind, graph hash and, for tables, partition and split lineage; the graph itself is stored as the `graph` artifact by the submitter."""
from __future__ import annotations

import traceback
from typing import Callable

from pydantic import BaseModel

from artifact_store import ArtifactStore, IllegalTransition
from graph_core.hashing import semantic_hash
from graph_core.schema import Graph
from graph_core.validate import ExecutionBlocked, require_executable
from tabular.core import Table, clean, dumps
from tabular.engine import Cancelled, NodeFailed, NodeOutcome, artifact_bytes, describe_value, run_graph

from .events import Emitter


class TabularRunConfig(BaseModel):
    model_config = {"extra": "forbid"}
    kind: str = "tabular"
    project_id: str | None = None


def _advance(store: ArtifactStore, run_id: str, new: str) -> None:
    try:
        store.set_status(run_id, new)
    except IllegalTransition:
        if store.get_run(run_id)["status"] != "cancelling":
            raise


def run_tabular(graph: Graph, cfg: TabularRunConfig, store: ArtifactStore, run_id: str, should_cancel: Callable[[], bool] = lambda: False) -> str:
    graph_hash = semantic_hash(graph)
    em = Emitter(store, run_id, graph_hash)
    em.emit("run_queued", config=cfg.model_dump())

    def finish(status: str, error: str | None = None, **data) -> str:
        store.set_status(run_id, status, error)
        em.emit("run_finished", status=status, error=error, **data)
        return status

    try:
        _advance(store, run_id, "preparing")
        em.emit("run_preparing")
        try:
            report = require_executable(graph)
        except ExecutionBlocked as e:
            for d in e.diagnostics:
                em.emit("validation_error", d.nodeId, **d.to_json())
            return finish("failed", str(e))
        import numpy, pandas, scipy, sklearn  # noqa: E401

        _advance(store, run_id, "running")
        em.emit("run_started", kind="tabular", order=report.order, libraries={"pandas": pandas.__version__, "scikit-learn": sklearn.__version__,
                                                                              "scipy": scipy.__version__, "numpy": numpy.__version__})

        def on_start(nid: str, typ: str) -> None:
            em.emit("node_started", nid, type=typ)

        def on_finish(o: NodeOutcome) -> None:
            arts = []
            for port, v in o.outs.items():
                meta = {"node": o.node, "port": port, "graph_hash": graph_hash, **describe_value(v)}
                if isinstance(v, Table):
                    meta["lineage"] = clean({k: x for k, x in v.lineage.items() if k in ("source", "split")})
                a = store.add_artifact(run_id, "node_output", artifact_bytes(v), "complete", None, meta)
                arts.append({"port": port, "sha256": a["sha256"], "size": a["size"], **{k: meta[k] for k in ("valueKind",)}})
            # the summary (profile, fitted state, metrics, ...) is its own artifact so large ones stay out of the event stream
            s = store.add_artifact(run_id, "node_summary", dumps(o.summary).encode(), "complete", None, {"node": o.node, "type": o.type, "graph_hash": graph_hash})
            em.emit("node_finished", o.node, type=o.type, outputs=arts, summarySha256=s["sha256"], rows={p: len(v.df) for p, v in o.outs.items() if isinstance(v, Table)})
            if o.type == "tabular.csv_source":
                em.emit("source_recorded", o.node, path=o.summary["path"], sha256=o.summary["sha256"], bytes=o.summary["bytes"], rows=o.summary["rows"])
            if o.type == "tabular.train_validation_split":
                em.emit("split_recorded", o.node, **{k: o.summary[k] for k in ("seed", "validationFraction", "stratifyBy", "groupBy", "nTrain", "nValidation",
                                                                              "trainRowIdsSha256", "validationRowIdsSha256")})

        try:
            run_graph(graph, report, run_id=run_id, graph_hash=graph_hash, on_start=on_start, on_finish=on_finish, should_cancel=should_cancel)
        except Cancelled:
            if store.get_run(run_id)["status"] != "cancelling":
                store.set_status(run_id, "cancelling")
            em.emit("cancel_acknowledged")
            store.set_status(run_id, "cancelled")
            em.emit("run_finished", status="cancelled", error=None)
            return "cancelled"
        except NodeFailed as e:
            em.emit("node_failed", e.node_id, code=e.code, message=e.message)
            return finish("failed", f"{e.code} at {e.node_id}: {e.message}")
        return finish("completed")
    except Exception as e:  # noqa: BLE001
        em.emit("error", message=str(e), traceback=traceback.format_exc())
        return finish("failed", f"{type(e).__name__}: {e}")
