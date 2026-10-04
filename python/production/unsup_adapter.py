"""Unsupervised (k-means / Gaussian mixture / PCA) family for the local production registry (ADR 0020).

Captured in the tabular worker next to the supervised pipelines, but in its own module, so the tabular adapter identity (ADR 0012)
does not change. Only estimators that can map NEW points are served: k-means (nearest centroid), Gaussian mixture (responsibilities)
and PCA (projection). DBSCAN and t-SNE record an explicit refusal. Requests carry the raw numeric feature columns; a fitted
preprocessing step upstream of the estimator is refused rather than approximated. The fitted scaler and estimator are the native
objects from the run (internal, hash-verified pickles). Clusters are not classes: label-based quality is external agreement (ARI/NMI)
with labels a person supplies, never accuracy. PCA has no labels; monitoring reports its reconstruction error instead."""
from __future__ import annotations

import hashlib
import io
import json
import math
import pickle
import platform
import threading
import time
from importlib.metadata import version as dist_version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tabular.core import UnsupModel, dumps
from .pipeline import ProductionError, read_verified

SERVED = ("kmeans", "gmm", "pca")
PASS_THROUGH = {"tabular.select_columns", "tabular.profile", "tabular.duplicates", "tabular.drop_missing", "tabular.train_validation_split"}
MAX_BATCH = 128
_PY = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FILES = ("production/unsup_adapter.py", "operations/unsup_ops.py", "unsup/methods.py", "tabular/core.py")


def environment() -> dict[str, str]:
    return {"python": platform.python_version(), **{n: dist_version(n) for n in ("scikit-learn", "numpy", "pandas")}}


def implementation() -> dict[str, str]:
    return {f: hashlib.sha256((_PY / f).read_bytes()).hexdigest() for f in IMPLEMENTATION_FILES}


def capture(store, graph, done, run_id) -> None:
    """Record a servable manifest (or an explicit refusal) for every unsupervised model output of a completed tabular run."""
    edges = {(e.to.node, e.to.port): (e.from_.node, e.from_.port) for e in graph.edges}
    for node, outcome in done.items():
        model = outcome.outs.get("model")
        if not isinstance(model, UnsupModel):
            continue
        try:
            _capture_one(store, graph, done, edges, run_id, node, model)
        except ProductionError as e:
            store.add_artifact(run_id, "unsup_refusal", dumps({"node": node, "code": e.code, "message": e.message}).encode(), "complete", None, {"node": node})


def _capture_one(store, graph, done, edges, run_id, node, model: UnsupModel) -> None:
    if model.method not in SERVED:
        raise ProductionError("E_PIPELINE_UNSUPPORTED", f"{model.method} cannot map new points (no native predict/transform); it is not served.")
    parent, path = edges[(node, "table")], []
    while True:
        n = graph.node(parent[0])
        if n.type in PASS_THROUGH:
            path.append(n.type)
            parent = edges[(n.id, "table")]
        elif not [p for p in edges if p[0] == n.id]:  # a source: no inputs
            source = {"node": n.id, "type": n.type, "summary": done[n.id].summary,
                      "outputArtifacts": [a["sha256"] for a in store.artifacts(run_id, "node_output") if a["meta"].get("node") == n.id]}
            break
        else:
            raise ProductionError("E_PIPELINE_UNSUPPORTED", f"{n.type} upstream of the estimator is not served; requests must carry the raw feature columns.")
    training = done[edges[(node, "table")][0]].outs[edges[(node, "table")][1]].df
    ref = training[model.features].iloc[:2000]
    reference = store.add_artifact(run_id, "unsup_reference", ref.to_csv(index=False).encode(), "complete", None, {"node": node})
    blob = store.add_artifact(run_id, "unsup_model", pickle.dumps(model, protocol=5), "complete", None, {"node": node, "trustedWorkerArtifact": True})
    manifest = {"format": 1, "adapter": "native-sklearn-unsupervised-local", "runId": run_id, "node": node, "method": model.method,
                "graphHash": store.get_run(run_id)["graph_hash"], "graphSha256": store.artifacts(run_id, "graph")[0]["sha256"],
                "modelSha256": blob["sha256"], "referenceSha256": reference["sha256"], "referenceRows": len(ref), "referenceTotalRows": len(training),
                "features": model.features, "scaled": model.scaler is not None, "upstream": list(reversed(path)), "params": model.details.get("params"),
                "fittedOn": model.fitted_on, "environment": environment(), "implementation": implementation(), "source": source, "fitArtifacts": {},
                "outputSchema": {"task": model.method, "classes": None, "target": None},
                "evaluationArtifacts": [a["sha256"] for a in store.artifacts(run_id, "node_summary")
                                        if a["meta"].get("node") == node or edges.get((a["meta"].get("node"), "model"), (None,))[0] == node]}
    store.add_artifact(run_id, "unsup_pipeline", dumps(manifest).encode(), "complete", None, {"node": node})


def manifest_for(store, run_id: str, node: str) -> tuple[str, dict[str, Any]] | None:
    arts = [a for a in store.artifacts(run_id, "unsup_pipeline") if a["meta"].get("node") == node]
    return (arts[-1]["sha256"], json.loads(read_verified(store, arts[-1]["sha256"]))) if arts else None


def verify(store, manifest) -> None:
    if manifest["environment"] != environment() or manifest["implementation"] != implementation():
        raise ProductionError("E_SERVING_ENVIRONMENT", "Pinned native environment or implementation differs; serving is refused.", 409)
    for sha in (manifest["modelSha256"], manifest["referenceSha256"], manifest["graphSha256"]):
        read_verified(store, sha)


class UnsupPipeline:
    def __init__(self, store, manifest):
        verify(store, manifest)
        if not any(a["kind"] == "unsup_model" and a["sha256"] == manifest["modelSha256"] and a["meta"].get("trustedWorkerArtifact")
                   for a in store.artifacts(manifest["runId"])):
            raise ProductionError("E_ARTIFACT_TRUST", "Only recorded native worker artifacts can be loaded.", 409)
        self.store, self.manifest = store, manifest
        self.model: UnsupModel = pickle.loads(read_verified(store, manifest["modelSha256"]))  # noqa: S301  (internal, hash-verified worker artifact)
        self.lock = threading.Lock()

    def validate_records(self, records, max_batch=MAX_BATCH):
        if not records or len(records) > min(max_batch, MAX_BATCH):
            raise ProductionError("E_REQUEST_BOUNDS", f"Requests take 1-{min(max_batch, MAX_BATCH)} records.", 413)
        feats = self.manifest["features"]
        for r in records:
            if not isinstance(r, dict) or set(r) != set(feats):
                raise ProductionError("E_REQUEST_SCHEMA", f"Expected exactly the feature fields {sorted(feats)}.")
            for f in feats:
                v = r[f]
                if type(v) not in (int, float) or not math.isfinite(v):
                    raise ProductionError("E_REQUEST_SCHEMA", f"Field {f} requires a finite number.")

    def _scaled(self, records) -> np.ndarray:
        X = np.array([[r[f] for f in self.manifest["features"]] for r in records], dtype="float64")
        return self.model.scaler.transform(X) if self.model.scaler is not None else X

    def predict(self, records):
        t0 = time.perf_counter()
        Xs = self._scaled(records)
        t1 = time.perf_counter()
        est, method = self.model.estimator, self.manifest["method"]
        with self.lock:
            if method == "kmeans":
                d = est.transform(Xs)
                out = {"predictions": est.predict(Xs).tolist(), "distances": d.tolist()}
            elif method == "gmm":
                out = {"predictions": est.predict(Xs).tolist(), "responsibilities": est.predict_proba(Xs).tolist(), "logDensity": est.score_samples(Xs).tolist()}
            else:
                Z = est.transform(Xs)
                recon = est.inverse_transform(Z)
                out = {"predictions": Z.tolist(), "reconstructionError": ((Xs - recon) ** 2).sum(1).tolist()}
        t2 = time.perf_counter()
        meaning = {"kmeans": "cluster = nearest centroid in scaled units; clusters are not classes",
                   "gmm": "cluster = most responsible component; responsibilities sum to 1",
                   "pca": "predictions = principal-component scores; reconstructionError in scaled units"}[method]
        return {**out, "method": method, "meaning": meaning}, {"preprocessingMs": (t1 - t0) * 1000, "inferenceMs": (t2 - t1) * 1000, "postprocessingMs": None}

    def reference_frame(self) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(read_verified(self.store, self.manifest["referenceSha256"])))

    def reference_records(self, n: int | None = None) -> list[dict[str, Any]]:
        df = self.reference_frame()
        return json.loads((df.head(n) if n else df).to_json(orient="records"))  # plain JSON numbers, exactly as a client would send them
