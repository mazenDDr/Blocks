"""Execute a validated `tabular` graph node by node in topological order on native pandas / scikit-learn / SciPy objects."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from graph_core import registry
from graph_core.schema import Graph
from graph_core.validate import Report

from .core import ExecCtx, ExecutionError, FitState, FittedModel, Plain, Table, UnsupModel, clean, dumps, schema_of


class NodeFailed(Exception):
    def __init__(self, node_id: str, code: str, message: str):
        super().__init__(f"{code} at {node_id}: {message}")
        self.node_id, self.code, self.message = node_id, code, message


class Cancelled(Exception):
    pass


@dataclass
class NodeOutcome:
    node: str
    type: str
    outs: dict[str, Any]
    summary: dict[str, Any]


def run_graph(graph: Graph, report: Report, *, run_id: str | None = None, graph_hash: str | None = None,
              on_start: Callable[[str, str], None] | None = None, on_finish: Callable[[NodeOutcome], None] | None = None,
              should_cancel: Callable[[], bool] = lambda: False, store: Any = None, pins: dict[str, str] | None = None) -> dict[str, NodeOutcome]:
    types = {n.id: n.type for n in graph.nodes}
    src = {(e.to.node, e.to.port): (e.from_.node, e.from_.port) for e in graph.edges}
    done: dict[str, NodeOutcome] = {}
    for nid in report.order:
        if should_cancel():
            raise Cancelled()
        op = registry.get_op(types[nid])
        cfg = report.resolved[nid]
        if on_start:
            on_start(nid, types[nid])
        ins = {p: done[src[(nid, p)][0]].outs[src[(nid, p)][1]] for p in op.inputs}
        try:
            outs, summary = op.execute(cfg, ins, ExecCtx(nid, run_id, graph_hash, store, dict(pins or {})))
        except ExecutionError as e:
            raise NodeFailed(nid, e.code, e.message) from e
        except Exception as e:  # noqa: BLE001  (library errors are reported against the node, not swallowed)
            raise NodeFailed(nid, "E_RUNTIME", f"{type(e).__name__}: {e}") from e
        done[nid] = NodeOutcome(nid, types[nid], outs, summary)
        if on_finish:
            on_finish(done[nid])
    return done


def describe_value(v: Any) -> dict[str, Any]:
    """Small JSON description of a runtime value (used in events and artifact metadata)."""
    if isinstance(v, Table):
        return {"valueKind": "table", "rows": len(v.df), "columns": schema_of(v.df), "partition": v.partition}
    if isinstance(v, FitState):
        return {"valueKind": "fit_state", "transform": v.transform, "columns": v.columns}
    if isinstance(v, FittedModel):
        return {"valueKind": "model", "task": v.task, "features": v.features, "target": v.target}
    if isinstance(v, UnsupModel):
        return {"valueKind": "unsup_model", "method": v.method, "features": v.features}
    if isinstance(v, Plain):
        return {"valueKind": v.kind}
    return {"valueKind": type(v).__name__}


def artifact_bytes(v: Any) -> bytes:
    if isinstance(v, Table):
        return v.df.to_csv(index=True, index_label="row_id").encode()
    if isinstance(v, (FitState, FittedModel, UnsupModel)):
        return dumps(v.details).encode()
    if isinstance(v, Plain):
        return dumps(v.data).encode()
    raise TypeError(type(v))
