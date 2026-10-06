"""Run a `tabular` graph: validate, execute node by node in this (worker) process, record events and artifacts with provenance.

Events: run_queued, run_preparing, run_started (source data hashes, graph hash), node_started, node_finished (summary), node_failed,
validation_error, cancel_acknowledged, error, run_finished. Artifacts (kind `node_output`, one per output port) carry node, port,
value kind, graph hash and, for tables, partition and split lineage; the graph itself is stored as the `graph` artifact by the submitter."""
from __future__ import annotations

import traceback
from typing import Callable, Literal

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
    # Replaces the `seed` config of every node that has one (e.g. the split); the nodes and the original value are recorded in `run_started`.
    seed: int | None = None
    # node id -> snapshot id: read these recorded source snapshots instead of the live sources (repeat from a pinned source identity).
    source_pins: dict[str, str] = {}
    # Study identity (study, trial, attempt, seed, fold) when the run is a sweep trial; informational, never changes computation.
    trial: dict | None = None
    # "reuse": look up and record dependency-scoped node results (tabular.cache, A09). "off" (default): every node executes, nothing is recorded.
    cache: Literal["off", "reuse"] = "off"


def apply_run_seed(graph: Graph, seed: int | None) -> tuple[Graph, list[dict]]:
    """Return a copy of `graph` with `seed` set on every node whose config has a `seed` field, and what was replaced."""
    if seed is None:
        return graph, []
    from graph_core import registry

    g = graph.model_copy(deep=True)
    applied = []
    for n in g.nodes:
        op = registry.get_op(n.type)
        if op is not None and "seed" in op.Config.model_fields:
            applied.append({"node": n.id, "was": n.config.get("seed", op.Config.model_fields["seed"].default), "now": seed})
            n.config["seed"] = seed
    return g, applied


def _advance(store: ArtifactStore, run_id: str, new: str) -> None:
    try:
        store.set_status(run_id, new)
    except IllegalTransition:
        if store.get_run(run_id)["status"] != "cancelling":
            raise


def run_tabular(graph: Graph, cfg: TabularRunConfig, store: ArtifactStore, run_id: str, should_cancel: Callable[[], bool] = lambda: False) -> str:
    graph_hash = semantic_hash(graph)  # identity of the stored graph; a run-level seed override is recorded separately
    em = Emitter(store, run_id, graph_hash)
    from connectors import context as _ctx

    _ctx.set_workbench(store.root)  # connector nodes find the connection registry next to the artifact store
    em.emit("run_queued", config=cfg.model_dump())

    def finish(status: str, error: str | None = None, **data) -> str:
        store.set_status(run_id, status, error)
        em.emit("run_finished", status=status, error=error, **data)
        return status

    try:
        _advance(store, run_id, "preparing")
        em.emit("run_preparing")
        exec_graph, seed_applied = apply_run_seed(graph, cfg.seed)
        try:
            report = require_executable(exec_graph)
        except ExecutionBlocked as e:
            for d in e.diagnostics:
                em.emit("validation_error", d.nodeId, **d.to_json())
            return finish("failed", str(e))
        import numpy, pandas, scipy, sklearn  # noqa: E401
        from extensions.sdk import LOADED

        _advance(store, run_id, "running")
        libraries = {"pandas": pandas.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "numpy": numpy.__version__}
        if graph.graphKind == "domain":
            import torch
            torch.set_num_threads(1)  # these tiny CPU examples are bounded to one intra-op thread
            from importlib.metadata import version
            libraries.update({name: version(name) for name in ("torch", "torchvision", "torchaudio", "tokenizers", "seqeval", "torchmetrics", "pycocotools")})
        node_cache = None
        if cfg.cache == "reuse":
            from tabular.cache import NodeCache
            node_cache = NodeCache(store, cfg.project_id, libraries, graph.graphKind)
        em.emit("run_started", kind=graph.graphKind, order=report.order, libraries=libraries, cache=node_cache.describe() if node_cache else {"mode": "off"},
                seed=cfg.seed, seedApplied=seed_applied, sourcePins=cfg.source_pins, trial=cfg.trial,
                plugins={k: {f: v[f] for f in ("name", "version", "sha256")} for k, v in LOADED.items() if any(n.type == k for n in graph.nodes)})

        def on_start(nid: str, typ: str) -> None:
            em.emit("node_started", nid, type=typ)

        def on_finish(o: NodeOutcome) -> None:
            arts = []
            for port, v in o.outs.items():
                meta = {"node": o.node, "port": port, "graph_hash": graph_hash, **describe_value(v)}
                if isinstance(v, Table):
                    meta["lineage"] = clean({k: x for k, x in v.lineage.items() if k in ("source", "sources", "join", "split")})
                a = store.add_artifact(run_id, "node_output", artifact_bytes(v), "complete", None, meta)
                arts.append({"port": port, "sha256": a["sha256"], "size": a["size"], **{k: meta[k] for k in ("valueKind",)}})
            # the summary (profile, fitted state, metrics, ...) is its own artifact so large ones stay out of the event stream
            s = store.add_artifact(run_id, "node_summary", dumps(o.summary).encode(), "complete", None, {"node": o.node, "type": o.type, "graph_hash": graph_hash})
            extra = {"cache": o.cache} if o.cache is not None else {}
            em.emit("node_finished", o.node, type=o.type, outputs=arts, summarySha256=s["sha256"], rows={p: len(v.df) for p, v in o.outs.items() if isinstance(v, Table)}, **extra)
            if o.summary.get("snapshotId"):
                sn = o.summary["snapshot"]
                em.emit("source_snapshot_recorded", o.node, connector=o.summary["connector"], mode=o.summary["mode"], snapshotId=o.summary["snapshotId"],
                        kind=sn.get("kind"), contentSha256=o.summary.get("sha256"), rows=o.summary["rows"], reproducibility=sn.get("reproducibility"))
            if o.type in ("tabular.csv_source", "tabular.jsonl_source"):
                em.emit("source_recorded", o.node, path=o.summary["path"], sha256=o.summary["sha256"], bytes=o.summary["bytes"], rows=o.summary["rows"])
            if o.type == "tabular.train_validation_split":
                em.emit("split_recorded", o.node, **{k: o.summary[k] for k in ("seed", "validationFraction", "stratifyBy", "groupBy", "nFolds", "fold", "nTrain", "nValidation",
                                                                              "trainRowIdsSha256", "validationRowIdsSha256")})

        try:
            outcomes = run_graph(exec_graph, report, run_id=run_id, graph_hash=graph_hash, on_start=on_start, on_finish=on_finish, should_cancel=should_cancel,
                                 store=store, pins=cfg.source_pins, cache=node_cache)
            if graph.graphKind == "tabular":
                from production.pipeline import capture_pipelines
                capture_pipelines(store, exec_graph, report, outcomes, run_id)
                from production.unsup_adapter import capture as capture_unsupervised
                capture_unsupervised(store, exec_graph, outcomes, run_id)
                from production.unsup_fitted import capture as capture_unsupervised_fitted
                capture_unsupervised_fitted(store, exec_graph, outcomes, run_id)
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
