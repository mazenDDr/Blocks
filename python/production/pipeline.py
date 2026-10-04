"""Capture native fitted objects during training; inference never rereads sources or refits.

Pickles are internal worker artifacts only. No API accepts uploaded pickles or arbitrary
artifact paths. Verify membership, hashes and the dependency manifest before loading.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from graph_core import registry
from tabular.core import ExecCtx, FitState, FittedModel, Table, clean, dumps, schema_of


class ProductionError(Exception):
    def __init__(self, code, message, status=422):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def environment():
    return {"python": platform.python_version(), **{n: version(n) for n in ("scikit-learn", "numpy", "pandas")}}


def adapter_hash():
    return hashlib.sha256(dumps(implementation_sources()).encode()).hexdigest()


def implementation_sources():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(), root / "operations/tabular_ops.py", root / "tabular/core.py")}


def capture_pipelines(store, graph, report, done, run_id):
    """Bounded compatibility: known table path, native estimator and fitted transforms.

    Training row filters/split/profile are recorded as training-only. Unsupported paths
    produce an explicit refusal artifact rather than changing the training result.
    """
    if not store.artifacts(run_id, "graph"):
        store.add_artifact(run_id, "graph", graph.model_dump_json(by_alias=True).encode(), "complete", None, {})
    edges = {(e.to.node, e.to.port): (e.from_.node, e.from_.port) for e in graph.edges}
    for node, outcome in done.items():
        for model in outcome.outs.values():
            if not isinstance(model, FittedModel):
                continue
            try:
                capture_one(store, graph, report, done, edges, run_id, node, model)
            except ProductionError as e:
                store.add_artifact(run_id, "inference_refusal", dumps({"node": node, "code": e.code, "message": e.message}).encode(),
                                   "complete", None, {"node": node})


def capture_one(store, graph, report, done, edges, rid, node, model):
    parent = edges[(node, "train")]
    training_frame = done[parent[0]].outs[parent[1]].df
    steps, training_only, states, source = [], [], {}, None
    baseline = None
    while True:
        nid, port = parent
        n = graph.node(nid)
        cfg = report.resolved[nid]
        out = done[nid].outs[port]
        if n.type == "tabular.apply_transform":
            fn, fp = edges[(nid, "fit")]
            fs = done[fn].outs[fp]
            if not isinstance(fs, FitState) or model.target in fs.columns:
                raise ProductionError("E_PIPELINE_TARGET_TRANSFORM", "Inference cannot require a target-fitted transform.")
            states[fn] = fs
            steps.append({"node": nid, "type": n.type, "config": cfg.model_dump(), "fitNode": fn, "fitDetails": fs.details})
        elif n.type == "tabular.select_columns":
            config = cfg.model_dump()
            config["columns"] = [c for c in config["columns"] if c["name"] != model.target]
            steps.append({"node": nid, "type": n.type, "config": config})
        elif n.type in ("tabular.train_validation_split", "tabular.profile", "tabular.duplicates", "tabular.drop_missing"):
            training_only.append({"node": nid, "type": n.type, "config": cfg.model_dump(), "policy": "training row selection/inspection; never applied to requests"})
            if n.type == "tabular.train_validation_split":
                baseline = out.df
        elif "table" in done[nid].outs and not registry.get_op(n.type).inputs:
            source = {"node": nid, "type": n.type, "summary": done[nid].summary}
            source["outputArtifacts"] = [a["sha256"] for a in store.artifacts(rid,"node_output") if a["meta"].get("node") == nid]
            source_frame = out.df
            break
        else:
            raise ProductionError("E_PIPELINE_UNSUPPORTED", f"Serving cannot preserve table operation {n.type}; path refused.")
        parent = edges[(nid, "table")]
    steps.reverse()
    needed = set(model.features)
    for s in reversed(steps):
        if "fitNode" in s:
            fs = states[s["fitNode"]]
            if fs.transform == "onehot":
                needed -= set(fs.transformer.get_feature_names_out())
            needed.update(fs.columns)
        else:
            needed.update(c["name"] for c in s["config"]["columns"])
    cols = [c for c in schema_of(source_frame) if c["name"] in needed]
    if {c["name"] for c in cols} != needed:
        raise ProductionError("E_PIPELINE_SCHEMA", "Raw input dependencies could not be resolved.")
    null_cols = {c for fs in states.values() if fs.transform == "impute" for c in fs.columns}
    for c in cols:
        c["nullable"] = c["name"] in null_cols
    raw_baseline = source_frame.loc[training_frame.index]
    raw_baseline = raw_baseline[[c["name"] for c in cols]]
    reference_total = len(raw_baseline)
    raw_baseline = raw_baseline.iloc[:2000]
    # CSV preserves complete baseline rows for measured reference distributions.
    b = store.add_artifact(rid, "inference_reference", raw_baseline.to_csv(index=False).encode(), "complete", None, {"node": node, "partition": "train"})
    labels = store.add_artifact(rid,"inference_reference_labels",dumps(training_frame.loc[raw_baseline.index,model.target].tolist()).encode(),"complete",None,{"node":node,"partition":"train"})
    state_refs = {}
    for fn, fs in states.items():
        a = store.add_artifact(rid, "inference_fit", pickle.dumps(fs, protocol=5), "complete", None, {"node": node, "fitNode": fn, "trustedWorkerArtifact": True})
        state_refs[fn] = a["sha256"]
    ma = store.add_artifact(rid, "inference_model", pickle.dumps(model, protocol=5), "complete", None, {"node": node, "trustedWorkerArtifact": True})
    evaluations = [a for a in store.artifacts(rid, "node_summary") if graph.node(a["meta"]["node"]).type == "sklearn.metrics"
                   and edges.get((a["meta"]["node"], "model"), (None,))[0] == node]
    manifest = {"format": 1, "adapter": "native-sklearn-local", "adapterSha256": adapter_hash(), "runId": rid, "node": node,
                "graphHash": store.get_run(rid)["graph_hash"], "graphSha256": store.artifacts(rid, "graph")[0]["sha256"],
                "modelSha256": ma["sha256"], "fitArtifacts": state_refs, "referenceSha256": b["sha256"], "referencePartition": "train",
                "referenceLabelsSha256": labels["sha256"],
                "referenceSampling": {"rows": len(raw_baseline), "totalTrainRows": reference_total, "policy": "first 2000 rows in recorded training order; truncated when larger"},
                "inputSchema": cols, "outputSchema": {"task": model.task, "classes": model.details.get("classes"), "target": model.target},
                "featureOrder": model.features, "steps": steps, "trainingOnly": list(reversed(training_only)), "source": source,
                "environment": environment(), "implementationSources": implementation_sources(), "evaluationArtifacts": [a["sha256"] for a in evaluations],
                "postprocessing": "native predict; native predict_proba for classification", "tokenizer": None,
                "customCode": "not supported by this adapter"}
    store.add_artifact(rid, "inference_pipeline", dumps(manifest).encode(), "complete", None, {"node": node})


def read_verified(store, sha):
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha) or not store.verify(sha):
        raise ProductionError("E_ARTIFACT_INTEGRITY", "Pinned artifact is missing or fails its SHA-256 check.", 409)
    return store.read_artifact(sha)


class Pipeline:
    def __init__(self, store, manifest):
        self.manifest = manifest
        if manifest["environment"] != environment() or manifest["adapterSha256"] != adapter_hash():
            raise ProductionError("E_SERVING_ENVIRONMENT", "Pinned adapter/dependency environment differs; rebuild and register a compatible version.", 409)
        arts = store.artifacts(manifest["runId"])
        def load(sha, kind):
            if not any(a["kind"] == kind and a["sha256"] == sha and a["meta"].get("trustedWorkerArtifact") for a in arts):
                raise ProductionError("E_ARTIFACT_TRUST", "Only recorded native worker artifacts can be loaded.", 409)
            return pickle.loads(read_verified(store, sha))
        self.model = load(manifest["modelSha256"], "inference_model")
        self.states = {n: load(sha, "inference_fit") for n, sha in manifest["fitArtifacts"].items()}

    def validate_records(self, records, max_batch=128):
        if not records or len(records) > max_batch or len(dumps(records).encode()) > 256_000:
            raise ProductionError("E_REQUEST_BOUNDS", "Batch or payload exceeds the pinned serving limits.", 413)
        schema = self.manifest["inputSchema"]
        keys = {c["name"] for c in schema}
        for row in records:
            if set(row) != keys:
                raise ProductionError("E_REQUEST_SCHEMA", f"Expected exactly fields {sorted(keys)}; received {sorted(row)}.")
            for c in schema:
                v = row[c["name"]]
                if v is None and c["nullable"]:
                    continue
                numeric = c["dtype"] in ("int", "float")
                valid = (isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)) if numeric else isinstance(v, bool if c["dtype"] == "bool" else str)
                if not valid or (c["dtype"] == "int" and int(v) != v):
                    raise ProductionError("E_REQUEST_SCHEMA", f"Field {c['name']} requires {c['dtype']} (nullable={c['nullable']}).")

    def predict(self, records):
        t0 = time.perf_counter()
        table = Table(pd.DataFrame(records), "request")
        stages = []
        for s in self.manifest["steps"]:
            op = registry.get_op(s["type"])
            ins = {"table": table}
            if "fitNode" in s:
                ins["fit"] = self.states[s["fitNode"]]
            table = op.execute(op.Config.model_validate(s["config"]), ins, ExecCtx(s["node"]))[0]["table"]
            stages.append({"node": s["node"], "type": s["type"], "columns": list(table.df.columns)})
        pre_ms = (time.perf_counter() - t0) * 1000
        X = table.df[self.model.features].to_numpy(dtype="float64")
        if not np.isfinite(X).all():
            raise ProductionError("E_REQUEST_FEATURES", "Nonfinite values remain after pinned preprocessing.")
        t1 = time.perf_counter()
        predicted = self.model.estimator.predict(X)
        probabilities = self.model.estimator.predict_proba(X) if self.model.task == "classification" else None
        infer_ms = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        result = clean({"predictions": predicted, "probabilities": probabilities, "classes": self.manifest["outputSchema"]["classes"]})
        return result, {"preprocessingMs": pre_ms, "inferenceMs": infer_ms, "postprocessingMs": (time.perf_counter()-t2)*1000,
                        "stages": stages, "transformedFeatures": clean(X[:3]),
                        "featureOrder": self.model.features}
