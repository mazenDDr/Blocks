"""Objective metric extraction with explicit step / aggregation semantics (A47)."""
from __future__ import annotations

import json
from typing import Any

from artifact_store import ArtifactStore

from .models import MetricSpec

MODEL_METRICS = {"val_loss": "validation cross-entropy (mean over validation samples)", "val_acc": "validation accuracy (fraction of validation samples)",
                 "train_loss": "training loss (mean over the epoch's training samples)"}


def check_spec(kind: str, spec: MetricSpec, report, graph) -> str | None:
    """Return an error message or None."""
    if kind == "model":
        if spec.name not in MODEL_METRICS:
            return f"Model graphs report {sorted(MODEL_METRICS)} per epoch; '{spec.name}' is not one of them."
        return None
    if not spec.node:
        return "A tabular objective needs 'node': the sklearn.metrics node."
    try:
        n = graph.node(spec.node)
    except KeyError:
        return f"Node '{spec.node}' does not exist."
    if n.type != "sklearn.metrics":
        return f"Node '{spec.node}' is a {n.type}, not an sklearn.metrics node."
    t = report.output_types.get(spec.node, {}).get("metrics")
    if t is not None and spec.name not in t.info.get("metrics", []):
        return f"'{spec.name}' is not computed by '{spec.node}' (it computes {t.info.get('metrics')})."
    return None


def extract(store: ArtifactStore, run_id: str, kind: str, spec: MetricSpec) -> dict[str, Any]:
    """The objective value of one run with its provenance and semantics. `available: False` says why not."""
    base = {"metric": spec.name, "runId": run_id}
    if kind == "model":
        evs = [e for e in store.events(run_id, -1, ("epoch_end",))]
        pts = [(e["data"]["step"], e["data"]["epoch"], e["data"].get(spec.name)) for e in evs if e["data"].get(spec.name) is not None]
        if not pts:
            return {**base, "available": False, "reason": "no epoch_end evaluation was recorded"}
        pick = {"last": pts[-1], "min": min(pts, key=lambda p: p[2]), "max": max(pts, key=lambda p: p[2])}[spec.select]
        return {**base, "available": True, "value": float(pick[2]), "step": pick[0], "epoch": pick[1], "stepUnit": "optimizer steps",
                "aggregation": f"{spec.select} over {len(pts)} epoch-end evaluations", "definition": MODEL_METRICS[spec.name],
                "partition": "train" if spec.name.startswith("train") else "validation", "provenance": {"runId": run_id, "source": "epoch_end events"}}
    summ = [a for a in store.artifacts(run_id, "node_summary") if a["meta"]["node"] == spec.node]
    if not summ:
        return {**base, "available": False, "reason": f"node '{spec.node}' has no recorded result in this run"}
    data = json.loads(store.read_artifact(summ[-1]["sha256"]))
    v = (data.get("values") or {}).get(spec.name)
    if v is None:
        return {**base, "available": False, "reason": (data.get("unavailable") or {}).get(spec.name, "metric not computed")}
    return {**base, "available": True, "value": float(v), "step": None, "epoch": None, "stepUnit": None,
            "aggregation": "single evaluation of the fitted model (no training steps); computed over all rows of the evaluated partition",
            "definition": f"{spec.name} from {spec.node}", "partition": data.get("evaluatedPartition"), "n": data.get("n"),
            "provenance": {"runId": run_id, "node": spec.node, "summarySha256": summ[-1]["sha256"], "evaluatedRowIdsSha256": data.get("evaluatedRowIdsSha256")}}
