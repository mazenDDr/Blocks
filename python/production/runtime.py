from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import nullcontext

from tabular.core import dumps
from .pipeline import Pipeline, ProductionError, adapter_hash, environment, read_verified
from .store import ProductionStore


class Admission:
    def __init__(self, config):
        self.config = config
        self.slots = threading.BoundedSemaphore(config["concurrency"])
        self.lock = threading.Lock()
        self.waiting = 0

    def enter(self, deadline):
        if self.slots.acquire(blocking=False):
            return
        with self.lock:
            if self.waiting >= self.config["queueLimit"]:
                raise ProductionError("E_QUEUE_FULL", "Pinned concurrency/queue limit reached.", 429)
            self.waiting += 1
        try:
            if not self.slots.acquire(timeout=max(0, deadline-time.perf_counter())):
                raise ProductionError("E_REQUEST_TIMEOUT", "Request timed out in the admission queue.", 504)
        finally:
            with self.lock:
                self.waiting -= 1


class ProductionRuntime:
    def __init__(self, store):
        self.store = store
        self.ps = ProductionStore(store)
        self.ps.recover_incomplete()
        self.lock = threading.RLock()
        self.pipelines, self.admissions, self.session_locks = {}, {}, {}

    def pipeline(self, version_id):
        version = self.ps.get("version", version_id)
        if version.get("adapter") in ("model", "rl", "unsup"):
            from . import model_adapter, rl_adapter, unsup_adapter
            mod, cls = {"model": (model_adapter, model_adapter.ModelGraphPipeline), "rl": (rl_adapter, rl_adapter.PolicyPipeline),
                        "unsup": (unsup_adapter, unsup_adapter.UnsupPipeline)}[version["adapter"]]
            mod.verify(self.store, version["manifest"])  # environment, implementation and pinned hashes on every use
            with self.lock:
                if version_id not in self.pipelines:
                    self.pipelines[version_id] = cls(self.store, version["manifest"])
                return self.pipelines[version_id]
        if version.get("adapter") == "domain":
            from .domain_adapter import DomainPipeline, manifest_of
            manifest_of(self.store, version["modelId"])  # re-verify manifest, checkpoint, environment and implementation on every use
            with self.lock:
                if version_id not in self.pipelines:
                    self.pipelines[version_id] = DomainPipeline(self.store, version["modelId"])
                return self.pipelines[version_id]
        # Recheck immutable bytes even when native objects are cached.
        manifest = json.loads(read_verified(self.store, version["pipelineSha256"]))
        if manifest["environment"] != environment() or manifest["adapterSha256"] != adapter_hash():
            raise ProductionError("E_SERVING_ENVIRONMENT", "Pinned adapter/dependency environment differs; serving is refused.",409)
        for sha in [manifest["modelSha256"], manifest["referenceSha256"], manifest["referenceLabelsSha256"], manifest["graphSha256"], *manifest["fitArtifacts"].values(), *manifest["evaluationArtifacts"], *manifest["source"]["outputArtifacts"]]:
            read_verified(self.store, sha)
        with self.lock:
            if version_id not in self.pipelines:
                self.pipelines[version_id] = Pipeline(self.store, manifest)
            return self.pipelines[version_id]

    def register_version(self, req):
        row = self.store.get_run(req.runId)
        if row is None or row["status"] != "completed":
            raise ProductionError("E_REGISTER_RUN", "Registration requires a completed recorded run.")
        arts = [a for a in self.store.artifacts(req.runId, "inference_pipeline") if a["meta"]["node"] == req.node]
        domain = [a for a in self.store.artifacts(req.runId, "domain_model") if a["meta"].get("node") == req.node and a["meta"].get("internalDomainModel")]
        from . import unsup_adapter
        unsup = unsup_adapter.manifest_for(self.store, req.runId, req.node)
        if not arts and unsup:
            sha, manifest = unsup
            unsup_adapter.UnsupPipeline(self.store, manifest)
            return self.ps.save("version", {**req.model_dump(), "adapter": "unsup", "family": manifest["method"], "pipelineSha256": sha, "manifest": manifest})
        refused = [json.loads(self.store.read_artifact(a["sha256"])) for a in self.store.artifacts(req.runId, "unsup_refusal") if a["meta"]["node"] == req.node]
        if not arts and refused:
            raise ProductionError(refused[-1]["code"], refused[-1]["message"])
        if not arts and not domain:
            from .model_adapter import build_manifest, is_candidate, ModelGraphPipeline
            if is_candidate(self.store, row) == req.node:
                manifest = build_manifest(self.store, req.runId, req.node)
                ModelGraphPipeline(self.store, manifest)  # loads the native model: registration fails if it cannot be served
                return self.ps.save("version", {**req.model_dump(), "adapter": "model", "family": "image_classifier",
                                                "pipelineSha256": manifest["checkpointSha256"], "manifest": manifest})
            from . import rl_adapter
            if rl_adapter.is_candidate(self.store, row) == req.node:
                manifest = rl_adapter.build_manifest(self.store, req.runId, req.node)
                rl_adapter.PolicyPipeline(self.store, manifest)
                return self.ps.save("version", {**req.model_dump(), "adapter": "rl", "family": "rl_policy",
                                                "pipelineSha256": manifest["checkpointSha256"], "manifest": manifest})
        if not arts and domain:
            from .domain_adapter import DomainPipeline
            p = DomainPipeline(self.store, domain[-1]["sha256"])  # loads the native model: registration fails if it cannot be served
            p.reference_records()
            return self.ps.save("version", {**req.model_dump(), "adapter": "domain", "family": p.m["family"], "modelId": domain[-1]["sha256"],
                                            "pipelineSha256": domain[-1]["sha256"], "manifest": p.manifest})
        if not arts:
            reasons = [json.loads(self.store.read_artifact(a["sha256"])) for a in self.store.artifacts(req.runId, "inference_refusal") if a["meta"]["node"] == req.node]
            raise ProductionError("E_PIPELINE_NOT_RECORDED", "No compatible inference pipeline recorded; rerun a supported tabular estimator. " + str(reasons))
        manifest = json.loads(read_verified(self.store, arts[-1]["sha256"]))
        Pipeline(self.store, manifest)
        return self.ps.save("version", {**req.model_dump(), "pipelineSha256": arts[-1]["sha256"], "manifest": manifest})

    def create_release(self, req):
        v = self.ps.resolve_version(req.versionId)
        p = self.pipeline(v["id"])
        domain = v.get("adapter") == "domain"
        if v.get("adapter") in ("model", "rl", "unsup"):
            records = p.reference_records(1)
        elif domain:
            if req.config.maxBatch > 4:
                raise ProductionError("E_RELEASE_CONFIG", "Domain releases accept at most 4 records per request (maxBatch <= 4).")
            records = p.reference_records()
        else:
            refs = self.store.read_artifact(p.manifest["referenceSha256"])
            import io
            import pandas as pd
            records = pd.read_csv(io.BytesIO(refs)).astype(object).where(lambda x: x.notna(), None).head(1).to_dict("records")
        p.validate_records(records, req.config.maxBatch)
        result, timing = p.predict(records)
        r = self.ps.save("release", {"versionId": v["id"], "pipelineSha256": v["pipelineSha256"], "config": req.config.model_dump(),
                                     "adapter": f"local FastAPI / native PyTorch {v['family']} CPU" if v.get("adapter") in ("domain", "model", "rl") else "local FastAPI / native scikit-learn CPU",
                                     "replicas": 1, "mode": "real local endpoint",
                                     "compatibility": {"ok": True, "warmupResultSha256": self.store.put_bytes(dumps(result).encode()), "timings": timing},
                                     "resources": {"placement": "same local control process, CPU", "autoscaling": "not implemented", "cost": "not measured"}})
        return r

    def activate(self, rid, expected, rollback=False):
        release = self.ps.get("release", rid)
        self.pipeline(release["versionId"])
        return self.ps.activate(rid, expected, rollback)

    def predict(self, target, namespace, req):
        # Pin route at admission. A concurrent rollout never changes this request's version.
        release = self.ps.route(target, namespace)
        if req.expectedRelease and req.expectedRelease != release["id"]:
            raise ProductionError("E_RELEASE_CHANGED", "Target no longer routes the selected release; request not sent.", 409)
        rid, cfg = release["id"], release["config"]
        fingerprint = hashlib.sha256(json.dumps({"release": rid, **req.model_dump()}, sort_keys=True).encode()).hexdigest()
        replay = self.ps.begin_request(req.user, req.requestId, fingerprint, rid)
        if replay is not None:
            return {**self.ps.trace(req.user, req.requestId), "idempotentReplay": True}
        start, admitted, locked = time.perf_counter(), False, False
        deadline = start + cfg["timeoutSeconds"]
        scope = dumps([rid, req.user, req.session]) if cfg["sessionMode"] == "counter" and req.session else None
        trace = {"requestId": req.requestId, "user": req.user, "session": req.session, "releaseId": rid, "versionId": release["versionId"],
                 "target": target, "namespace": namespace, "receivedAt": time.time(), "inputSha256": hashlib.sha256(json.dumps(req.records,sort_keys=True).encode()).hexdigest(),
                 "records": req.records if cfg["captureInputs"] else None, "capturePolicy": "inputs captured explicitly" if cfg["captureInputs"] else "input hash only; replay unavailable",
                 "batchSize": len(req.records), "status": 200, "error": None, "result": None, "queueMs": None,
                 "totalMsDefinition": "admission through inference/error handling, excluding trace/session persistence and HTTP encoding; traffic client measures end-to-end latency"}
        with self.lock:
            admission = self.admissions.setdefault(rid, Admission(cfg))
            slock = self.session_locks.setdefault(scope, threading.Lock()) if scope else None
        try:
            admission.enter(deadline)
            admitted = True
            if slock:
                locked = slock.acquire(timeout=max(0, deadline-time.perf_counter()))
                if not locked:
                    raise ProductionError("E_REQUEST_TIMEOUT", "Timed out waiting for this session's serialized update.", 504)
            trace["queueMs"] = (time.perf_counter()-start)*1000
            self.ps.query("UPDATE requests SET state='running' WHERE user=? AND id=?",(req.user,req.requestId))
            if self.ps.cancelled(req.user, req.requestId):
                raise ProductionError("E_REQUEST_CANCELLED", "Cancelled before native inference.", 409)
            if cfg["sessionMode"] == "counter" and req.session is None:
                raise ProductionError("E_SESSION_REQUIRED", "This release requires an explicit session identity.")
            p = self.pipeline(release["versionId"])
            t0 = time.perf_counter()
            p.validate_records(req.records, cfg["maxBatch"])
            trace["validationMs"] = (time.perf_counter()-t0)*1000
            trace["lineage"] = {"runId": p.manifest["runId"], "node": p.manifest["node"], "graphHash": p.manifest["graphHash"],
                                "graphSha256": p.manifest["graphSha256"], "pipelineSha256": release["pipelineSha256"], "modelSha256": p.manifest["modelSha256"],
                                "source": p.manifest["source"], "fitArtifacts": p.manifest["fitArtifacts"], "evaluationArtifacts": p.manifest["evaluationArtifacts"]}
            result, timing = p.predict(req.records)
            if not cfg["captureInputs"]:
                timing.pop("transformedFeatures", None)
            trace.update(result=result, timings=timing)
            if time.perf_counter() > deadline:
                raise ProductionError("E_REQUEST_TIMEOUT", "Native inference finished after the deadline; session state not committed.", 504)
        except Exception as e:
            code = e.code if isinstance(e, ProductionError) else "E_INFERENCE"
            trace.update(status=e.status if isinstance(e, ProductionError) else 422, error={"code": code, "message": str(e)}, result=None)
        finally:
            trace["totalMs"] = (time.perf_counter()-start)*1000
            try:
                finished = self.ps.finish_request(req.user, req.requestId, trace, scope)
            finally:
                if locked:
                    slock.release()
                if admitted:
                    admission.slots.release()
        return finished
