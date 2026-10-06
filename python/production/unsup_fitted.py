"""Served unsupervised estimators whose recorded table path applies fitted preprocessing (ADR 0054).

The original unsupervised adapter (ADR 0020) serves raw numeric features and refuses fitted preprocessing upstream of the
estimator. Its source file is part of every registered unsupervised version's identity, so this path lives in its own module.
Capture walks the recorded table path exactly like the supervised tabular adapter: fitted `apply_transform` steps keep their
native FitState objects, `select_columns` is replayed, row filters/split/profile are training-only, and the path must end at a
source. Requests carry the raw source columns; inference replays the pinned steps through the same native operations and then
uses the run's own scaler/estimator. Nothing is refitted. Unsupported operations still refuse.
"""
from __future__ import annotations

import hashlib
import io
import json
import pickle
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from graph_core import registry
from tabular.core import ExecCtx, FitState, Table, UnsupModel, dumps, schema_of
from . import unsup_adapter
from .pipeline import Pipeline, ProductionError, read_verified

TRAINING_ONLY = ("tabular.train_validation_split", "tabular.profile", "tabular.duplicates", "tabular.drop_missing")
_PY = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FILES = ("production/unsup_fitted.py", "operations/tabular_ops.py", *unsup_adapter.IMPLEMENTATION_FILES)


def implementation() -> dict[str, str]:
    return {f: hashlib.sha256((_PY / f).read_bytes()).hexdigest() for f in IMPLEMENTATION_FILES}


def _walk(graph, done, edges, node):
    """Return (steps, states, trainingOnly, source, source_frame) or None when no fitted step is involved."""
    parent, steps, states, training_only = edges[(node, "table")], [], {}, []
    while True:
        nid, port = parent
        n = graph.node(nid)
        op = registry.get_op(n.type)
        cfg = op.Config.model_validate(n.config).model_dump() if op.Config else {}
        if n.type == "tabular.apply_transform":
            fn, fp = edges[(nid, "fit")]
            fs = done[fn].outs[fp]
            if not isinstance(fs, FitState):
                raise ProductionError("E_PIPELINE_UNSUPPORTED", "apply_transform without a recorded native fit state is not served.")
            states[fn] = fs
            steps.append({"node": nid, "type": n.type, "config": cfg, "fitNode": fn, "fitDetails": fs.details})
        elif n.type == "tabular.select_columns":
            steps.append({"node": nid, "type": n.type, "config": cfg})
        elif n.type in TRAINING_ONLY:
            training_only.append({"node": nid, "type": n.type, "config": cfg, "policy": "training row selection/inspection; never applied to requests"})
        elif "table" in done[nid].outs and not op.inputs:
            source = {"node": nid, "type": n.type, "summary": done[nid].summary}
            return list(reversed(steps)), states, list(reversed(training_only)), source, done[nid].outs[port].df
        else:
            raise ProductionError("E_PIPELINE_UNSUPPORTED", f"{n.type} upstream of the estimator is not served; path refused.")
        parent = edges[(nid, "table")]


def capture(store, graph, done, run_id) -> None:
    """Record a fitted-preprocessing manifest for each servable unsupervised model whose path applies a fitted transform."""
    edges = {(e.to.node, e.to.port): (e.from_.node, e.from_.port) for e in graph.edges}
    for node, outcome in done.items():
        for model in outcome.outs.values():
            if not isinstance(model, UnsupModel) or model.method not in unsup_adapter.SERVED:
                continue
            try:
                walked = _walk(graph, done, edges, node)
                if walked[1]:
                    _capture_one(store, graph, done, edges, run_id, node, model, *walked)
            except ProductionError as e:
                store.add_artifact(run_id, "unsup_fitted_refusal", dumps({"node": node, "code": e.code, "message": e.message}).encode(), "complete", None, {"node": node})


def _capture_one(store, graph, done, edges, run_id, node, model, steps, states, training_only, source, source_frame) -> None:
    # Walk back from the estimator's features. A column selection keeps only columns that something downstream needs, so
    # requests never carry columns the served path does not use (e.g. a supervised target selected for another model).
    needed = set(model.features)
    for s in reversed(steps):
        if "fitNode" in s:
            fs = states[s["fitNode"]]
            if fs.transform == "onehot":
                needed -= set(fs.transformer.get_feature_names_out())
            needed.update(fs.columns)
        else:
            kept = [c for c in s["config"]["columns"] if c["name"] in needed]
            if {c["name"] for c in kept} != needed:
                raise ProductionError("E_PIPELINE_SCHEMA", "A column needed downstream is not produced by the recorded column selection.")
            s["config"] = {**s["config"], "columns": kept}
    cols = [c for c in schema_of(source_frame) if c["name"] in needed]
    if {c["name"] for c in cols} != needed:
        raise ProductionError("E_PIPELINE_SCHEMA", "Raw input dependencies could not be resolved.")
    imputed = {c for fs in states.values() if fs.transform == "impute" for c in fs.columns}
    for c in cols:
        c["nullable"] = c["name"] in imputed
    fitted_rows = done[edges[(node, "table")][0]].outs[edges[(node, "table")][1]].df
    raw = source_frame.loc[fitted_rows.index, [c["name"] for c in cols]]
    reference = store.add_artifact(run_id, "unsup_fitted_reference", raw.iloc[:2000].to_csv(index=False).encode(), "complete", None, {"node": node})
    fit_refs = {fn: store.add_artifact(run_id, "unsup_fitted_fit", pickle.dumps(fs, protocol=5), "complete", None,
                                       {"node": node, "fitNode": fn, "trustedWorkerArtifact": True})["sha256"] for fn, fs in states.items()}
    blob = store.add_artifact(run_id, "unsup_fitted_model", pickle.dumps(model, protocol=5), "complete", None, {"node": node, "trustedWorkerArtifact": True})
    source["outputArtifacts"] = [a["sha256"] for a in store.artifacts(run_id, "node_output") if a["meta"].get("node") == source["node"]]
    manifest = {"format": 2, "adapter": "native-sklearn-unsupervised-fitted-local", "runId": run_id, "node": node, "method": model.method,
                "graphHash": store.get_run(run_id)["graph_hash"], "graphSha256": store.artifacts(run_id, "graph")[0]["sha256"],
                "modelSha256": blob["sha256"], "referenceSha256": reference["sha256"], "referenceRows": min(len(raw), 2000), "referenceTotalRows": len(raw),
                "features": [c["name"] for c in cols], "inputSchema": cols, "modelFeatures": model.features, "steps": steps, "trainingOnly": training_only,
                "scaled": model.scaler is not None, "params": model.details.get("params"), "fittedOn": model.fitted_on,
                "environment": unsup_adapter.environment(), "implementation": implementation(), "source": source, "fitArtifacts": fit_refs,
                "outputSchema": {"task": model.method, "classes": None, "target": None},
                "evaluationArtifacts": [a["sha256"] for a in store.artifacts(run_id, "node_summary")
                                        if a["meta"].get("node") == node or edges.get((a["meta"].get("node"), "model"), (None,))[0] == node]}
    store.add_artifact(run_id, "unsup_fitted_pipeline", dumps(manifest).encode(), "complete", None, {"node": node})


def manifest_for(store, run_id: str, node: str) -> tuple[str, dict[str, Any]] | None:
    arts = [a for a in store.artifacts(run_id, "unsup_fitted_pipeline") if a["meta"].get("node") == node]
    return (arts[-1]["sha256"], json.loads(read_verified(store, arts[-1]["sha256"]))) if arts else None


def is_fitted(manifest) -> bool:
    return manifest.get("format") == 2 and "steps" in manifest


def verify(store, manifest) -> None:
    if manifest["environment"] != unsup_adapter.environment() or manifest["implementation"] != implementation():
        raise ProductionError("E_SERVING_ENVIRONMENT", "Pinned native environment or implementation differs; serving is refused.", 409)
    for sha in (manifest["modelSha256"], manifest["referenceSha256"], manifest["graphSha256"], *manifest["fitArtifacts"].values()):
        read_verified(store, sha)


class FittedUnsupPipeline:
    """Raw request columns → pinned native steps → the run's scaler and estimator (same outputs as UnsupPipeline)."""

    def __init__(self, store, manifest):
        verify(store, manifest)
        arts = store.artifacts(manifest["runId"])

        def load(sha, kind):
            if not any(a["kind"] == kind and a["sha256"] == sha and a["meta"].get("trustedWorkerArtifact") for a in arts):
                raise ProductionError("E_ARTIFACT_TRUST", "Only recorded native worker artifacts can be loaded.", 409)
            return pickle.loads(read_verified(store, sha))  # noqa: S301  (internal, hash-verified worker artifact)

        self.store, self.manifest = store, manifest
        self.model: UnsupModel = load(manifest["modelSha256"], "unsup_fitted_model")
        self.states = {n: load(sha, "unsup_fitted_fit") for n, sha in manifest["fitArtifacts"].items()}
        self.lock = threading.Lock()

    def validate_records(self, records, max_batch=unsup_adapter.MAX_BATCH):
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise ProductionError("E_REQUEST_SCHEMA", "Records must be a list of objects.")
        Pipeline.validate_records(self, records, min(max_batch, unsup_adapter.MAX_BATCH))

    def _features(self, records):
        table = Table(pd.DataFrame(records, columns=self.manifest["features"]), "request")
        for s in self.manifest["steps"]:
            op = registry.get_op(s["type"])
            ins = {"table": table, **({"fit": self.states[s["fitNode"]]} if "fitNode" in s else {})}
            table = op.execute(op.Config.model_validate(s["config"]), ins, ExecCtx(s["node"]))[0]["table"]
        X = table.df[self.manifest["modelFeatures"]].to_numpy(dtype="float64")
        if not np.isfinite(X).all():
            raise ProductionError("E_REQUEST_FEATURES", "Nonfinite values remain after pinned preprocessing.")
        return X

    def _scaled(self, records) -> np.ndarray:
        X = self._features(records)
        return self.model.scaler.transform(X) if self.model.scaler is not None else X

    def predict(self, records):
        t0 = time.perf_counter()
        result, timing = unsup_adapter.UnsupPipeline.predict(self, records)
        timing["pipelineMs"] = (time.perf_counter() - t0) * 1000
        return {**result, "meaning": result["meaning"] + "; raw inputs pass the run's pinned fitted preprocessing first"}, timing

    def reference_frame(self) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(read_verified(self.store, self.manifest["referenceSha256"])), keep_default_na=True)

    def reference_records(self, n: int | None = None) -> list[dict[str, Any]]:
        df = self.reference_frame()
        return json.loads((df.head(n) if n else df).to_json(orient="records"))
