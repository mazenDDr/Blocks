from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import nullcontext

from tabular.core import dumps
from .pipeline import Pipeline, ProductionError, adapter_hash, environment, read_verified
from .store import ProductionStore



CONTEXT_ADAPTERS = ("agent_context", "conversation_context", "agent_context_json", "conversation_context_json")
MEMORY_ADAPTERS = ("agent_memory",)
MEMORY_MAX_RECORDS = 200  # live records per release and user (memory_agent_adapter.MAX_RECORDS)  # long-term memory per release and request user (ADR0075)
AGENT_ADAPTERS = (*CONTEXT_ADAPTERS, "agent", "conversation", "agent_json", "conversation_json", "agent_retrieval", "agent_tools", "conversation_approval", *MEMORY_ADAPTERS, "agent_tool_choice")
CONVERSATION_ADAPTERS = ("conversation_context", "conversation_context_json", "conversation", "conversation_json", "conversation_approval")  # native checkpoints per release/user/session
JSON_ADAPTERS = ("agent_context_json", "conversation_context_json", "agent_json", "conversation_json")  # validated JSON object per turn
PORTABLE_ADAPTERS = ("model_keras", "model_jax")  # PyTorch-trained image classifiers served on Keras/JAX

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
        if version.get("adapter") in AGENT_ADAPTERS:
            from . import agent_adapter, conversation_adapter, json_agent_adapter, json_conversation_adapter, retrieval_agent_adapter, tools_agent_adapter, approval_adapter, context_agent_adapter, memory_agent_adapter, tool_choice_adapter
            adapter = {**{name: context_agent_adapter for name in CONTEXT_ADAPTERS}, "agent_memory": memory_agent_adapter, "agent_tool_choice": tool_choice_adapter, "agent": agent_adapter, "conversation": conversation_adapter, "agent_json": json_agent_adapter,
                       "conversation_json": json_conversation_adapter, "agent_retrieval": retrieval_agent_adapter, "agent_tools": tools_agent_adapter, "conversation_approval": approval_adapter}[version["adapter"]]
            AgentPipeline = (json_agent_adapter.JsonAgentPipeline if adapter is json_agent_adapter
                             else json_conversation_adapter.JsonConversationPipeline if adapter is json_conversation_adapter
                             else tools_agent_adapter.ToolsPipeline if adapter is tools_agent_adapter
                             else retrieval_agent_adapter.RetrievalPipeline if adapter is retrieval_agent_adapter else adapter.AgentPipeline)
            verify = adapter.verify
            if json.loads(read_verified(self.store, version["pipelineSha256"])) != version["manifest"]:
                raise ProductionError("E_AGENT_SOURCE", "Agent version differs from its pinned manifest.", 409)
            verify(self.store, version["manifest"])
            with self.lock:
                if version_id not in self.pipelines:
                    self.pipelines[version_id] = AgentPipeline(self.store, version["manifest"])
                return self.pipelines[version_id]
        if version.get("adapter") == "rl_td3":
            from . import td3_adapter
            td3_adapter.verify(self.store, version["manifest"])
            with self.lock:
                if version_id not in self.pipelines:
                    self.pipelines[version_id] = td3_adapter.TD3PolicyPipeline(self.store, version["manifest"])
                return self.pipelines[version_id]
        if version.get("adapter") in ("model", "rl", "unsup", *PORTABLE_ADAPTERS):
            from . import model_adapter, rl_adapter, unsup_adapter
            mod, cls = {"model": (model_adapter, model_adapter.ModelGraphPipeline), "rl": (rl_adapter, rl_adapter.PolicyPipeline),
                        "unsup": (unsup_adapter, unsup_adapter.UnsupPipeline)}.get(version["adapter"], (None, None))
            if version["adapter"] in PORTABLE_ADAPTERS:
                from . import portable_model_adapter
                mod, cls = portable_model_adapter, portable_model_adapter.PortablePipeline
            if version["adapter"] == "unsup":
                from . import unsup_fitted
                if unsup_fitted.is_fitted(version["manifest"]):
                    mod, cls = unsup_fitted, unsup_fitted.FittedUnsupPipeline
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
        if row["config"].get("kind") == "agent":
            from . import agent_adapter, conversation_adapter, json_agent_adapter, json_conversation_adapter, retrieval_agent_adapter, tools_agent_adapter, approval_adapter, context_agent_adapter, memory_agent_adapter, tool_choice_adapter
            adapter, name = {**{node: (context_agent_adapter, name) for name, node in context_agent_adapter.NODES.items()}, memory_agent_adapter.NODE: (memory_agent_adapter, "agent_memory"),
                             tool_choice_adapter.NODE: (tool_choice_adapter, "agent_tool_choice"), conversation_adapter.NODE: (conversation_adapter, "conversation"), json_agent_adapter.NODE: (json_agent_adapter, "agent_json"),
                             json_conversation_adapter.NODE: (json_conversation_adapter, "conversation_json"),
                             retrieval_agent_adapter.NODE: (retrieval_agent_adapter, "agent_retrieval"),
                             tools_agent_adapter.NODE: (tools_agent_adapter, "agent_tools"),
                             approval_adapter.NODE: (approval_adapter, "conversation_approval")}.get(req.node, (agent_adapter, "agent"))
            manifest = adapter.build_manifest(self.store, req.runId, req.node)
            sha = self.store.put_bytes(dumps(manifest).encode())
            return self.ps.save("version", {**req.model_dump(), "adapter": name, "family": manifest["family"],
                                            "pipelineSha256": sha, "manifest": manifest})
        arts = [a for a in self.store.artifacts(req.runId, "inference_pipeline") if a["meta"]["node"] == req.node]
        domain = [a for a in self.store.artifacts(req.runId, "domain_model") if a["meta"].get("node") == req.node and a["meta"].get("internalDomainModel")]
        from . import unsup_adapter
        unsup = unsup_adapter.manifest_for(self.store, req.runId, req.node)
        if not arts and unsup:
            sha, manifest = unsup
            unsup_adapter.UnsupPipeline(self.store, manifest)
            return self.ps.save("version", {**req.model_dump(), "adapter": "unsup", "family": manifest["method"], "pipelineSha256": sha, "manifest": manifest})
        from . import unsup_fitted
        fitted = unsup_fitted.manifest_for(self.store, req.runId, req.node)
        if not arts and fitted:
            sha, manifest = fitted
            unsup_fitted.FittedUnsupPipeline(self.store, manifest)
            return self.ps.save("version", {**req.model_dump(), "adapter": "unsup", "family": manifest["method"], "pipelineSha256": sha, "manifest": manifest})
        refused = [json.loads(self.store.read_artifact(a["sha256"])) for a in self.store.artifacts(req.runId, "unsup_fitted_refusal") if a["meta"]["node"] == req.node]
        refused = refused or [json.loads(self.store.read_artifact(a["sha256"])) for a in self.store.artifacts(req.runId, "unsup_refusal") if a["meta"]["node"] == req.node]
        if not arts and refused:
            raise ProductionError(refused[-1]["code"], refused[-1]["message"])
        if not arts and not domain and row["config"].get("kind") == "rl":
            from . import td3_adapter
            if td3_adapter.is_candidate(self.store, row) == req.node:
                manifest = td3_adapter.build_manifest(self.store, req.runId, req.node)
                td3_adapter.TD3PolicyPipeline(self.store, manifest)
                return self.ps.save("version", {**req.model_dump(), "adapter": "rl_td3", "family": "rl_continuous_policy",
                                                "pipelineSha256": manifest["checkpointSha256"], "manifest": manifest})
        if not arts and not domain and req.node.split(":", 1)[0] in ("keras", "jax"):
            from . import portable_model_adapter
            manifest = portable_model_adapter.build_manifest(self.store, req.runId, req.node)
            portable_model_adapter.PortablePipeline(self.store, manifest)
            return self.ps.save("version", {**req.model_dump(), "adapter": f"model_{manifest['portable']['backend']}", "family": "image_classifier",
                                            "pipelineSha256": self.store.put_bytes(dumps(manifest).encode()), "manifest": manifest})
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
        agent = v.get("adapter") in AGENT_ADAPTERS
        conversation = v.get("adapter") in CONVERSATION_ADAPTERS
        if agent:
            if req.config.maxBatch != 1 or req.config.sessionMode != ("conversation" if conversation else "stateless"):
                raise ProductionError("E_RELEASE_CONFIG", "Agent releases require maxBatch=1 and their declared stateless/conversation mode.")
            records = p.reference_records()
        elif req.config.sessionMode == "conversation":
            raise ProductionError("E_RELEASE_CONFIG", "Native conversations require a registered conversation graph.")
        elif v.get("adapter") in ("model", "rl", "unsup", "rl_td3", *PORTABLE_ADAPTERS):
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
        result, timing = p.predict(records, deadline=time.perf_counter()+req.config.timeoutSeconds) if agent else p.predict(records)
        timing.pop("_checkpoint", None)  # isolated warmup never commits conversation state
        timing.pop("_memory", None)  # nor long-term memory
        timing.pop("_status", None)
        r = self.ps.save("release", {"versionId": v["id"], "pipelineSha256": v["pipelineSha256"], "config": req.config.model_dump(),
                                     "adapter": "local FastAPI / native LangGraph / local Ollama" if agent else f"local FastAPI / PyTorch-trained {v['family']} on {v['adapter'][6:]} CPU" if v.get("adapter") in PORTABLE_ADAPTERS else f"local FastAPI / native PyTorch {v['family']} CPU" if v.get("adapter") in ("domain", "model", "rl") else "local FastAPI / native scikit-learn CPU",
                                     "replicas": 1, "mode": "real local endpoint",
                                     "memoryPolicy": "Long-term records per release and request user; each turn sees only its user's records; writes commit with the successful request trace; research memory never copied." if v.get("adapter") in MEMORY_ADAPTERS else None,
                                     "conversationPolicy": "Native paused/END checkpoints persist per release/user/session; reviewed approve/reject/edit resumes; file effects unsupported." if v.get("adapter") == "conversation_approval" else "Native state/history persists per release/user/session even when trace capture is off; only successful END turns commit; research history never copied." if conversation else None,
                                     "compatibility": {"ok": True, "warmupResultSha256": self.store.put_bytes(dumps(result).encode()), "timings": timing},
                                     "resources": {"placement": "local control process; model device managed by Ollama, not measured" if agent else "same local control process, CPU", "autoscaling": "not implemented", "cost": "not measured"}})
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
        approval_family = self.ps.get("version", release["versionId"]).get("adapter") == "conversation_approval"
        memory_family = self.ps.get("version", release["versionId"]).get("adapter") in MEMORY_ADAPTERS
        if getattr(req, "approval", None) is not None and not approval_family:
            raise ProductionError("E_RELEASE_CONFIG", "This release does not support approval resumes.")
        fingerprint = hashlib.sha256(json.dumps({"release": rid, **req.model_dump()}, sort_keys=True).encode()).hexdigest()
        replay = self.ps.begin_request(req.user, req.requestId, fingerprint, rid)
        if replay is not None:
            return {**self.ps.trace(req.user, req.requestId), "idempotentReplay": True}
        start, admitted, locked = time.perf_counter(), False, False
        deadline = start + cfg["timeoutSeconds"]
        scope = dumps([rid, req.user, req.session]) if cfg["sessionMode"] in ("counter", "conversation") and req.session else None
        candidate, previous, memory_writes = None, None, None
        lock_scope = dumps([rid, req.user, "memory"]) if memory_family else scope  # one memory turn at a time per release and user
        trace = {"requestId": req.requestId, "user": req.user, "session": req.session, "releaseId": rid, "versionId": release["versionId"],
                 "target": target, "namespace": namespace, "receivedAt": time.time(), "inputSha256": hashlib.sha256(json.dumps(req.records,sort_keys=True).encode()).hexdigest(),
                 "records": req.records if cfg["captureInputs"] else None, "capturePolicy": "inputs captured explicitly" if cfg["captureInputs"] else "input hash only; replay unavailable",
                 "batchSize": len(req.records), "status": 200, "error": None, "result": None, "queueMs": None,
                 "totalMsDefinition": "admission through inference/error handling, excluding trace/session persistence and HTTP encoding; traffic client measures end-to-end latency"}
        if cfg["sessionMode"] == "conversation":
            trace["capturePolicy"] += "; native conversation checkpoint/history persists independently of trace capture"
        if approval_family:
            trace["capturePolicy"] += "; pending review payload and submitted approval decision also persist independently of trace capture"
        with self.lock:
            admission = self.admissions.setdefault(rid, Admission(cfg))
            slock = self.session_locks.setdefault(lock_scope, threading.Lock()) if lock_scope else None
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
            if cfg["sessionMode"] in ("counter", "conversation") and req.session is None:
                raise ProductionError("E_SESSION_REQUIRED", "This release requires an explicit session identity.")
            p = self.pipeline(release["versionId"])
            t0 = time.perf_counter()
            p.validate_records(req.records, cfg["maxBatch"])
            trace["validationMs"] = (time.perf_counter()-t0)*1000
            trace["lineage"] = {"runId": p.manifest["runId"], "node": p.manifest["node"], "graphHash": p.manifest["graphHash"],
                                "graphSha256": p.manifest["graphSha256"], "pipelineSha256": release["pipelineSha256"], "modelSha256": p.manifest["modelSha256"],
                                "source": p.manifest["source"], "fitArtifacts": p.manifest["fitArtifacts"], "evaluationArtifacts": p.manifest["evaluationArtifacts"]}
            if self.ps.get("version", release["versionId"]).get("adapter") in AGENT_ADAPTERS:
                kwargs = {}
                if memory_family:
                    kwargs["memory"] = self.ps.release_memory(rid, req.user)
                if cfg["sessionMode"] == "conversation":
                    previous = self.ps.conversation(scope)
                    parent = previous["checkpointSha256"] if previous else None
                    trace["conversationParent"] = {"revision": previous["revision"] if previous else 0, "checkpointSha256": parent}
                    kwargs["checkpoint"] = parent
                    if approval_family:
                        from .approval_requests import reviewed_parent
                        reviewed_parent(previous, getattr(req, "approval", None))
                        kwargs["approval"] = getattr(req, "approval", None)
                        trace["approval"] = req.approval.model_dump() if getattr(req, "approval", None) else None
                result, timing = p.predict(req.records, capture=cfg["captureInputs"], deadline=deadline,
                                          cancelled=lambda: self.ps.cancelled(req.user, req.requestId), **kwargs)
                candidate = timing.pop("_checkpoint", None)
                memory_writes = timing.pop("_memory", None)
                if approval_family:
                    trace["status"] = timing.pop("_status")
            else:
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
                from .approval_store import finish_request as finish_approval
                finish = (lambda *a, **kw: finish_approval(self.ps, *a, **kw)) if approval_family else self.ps.finish_request
                finished = finish(req.user, req.requestId, trace,
                                                  scope if cfg["sessionMode"] == "counter" else None,
                                                  conversation=(scope, previous, candidate) if cfg["sessionMode"] == "conversation" and scope else None,
                                                  deadline=deadline, **({"memory": (rid, req.user, memory_writes, MEMORY_MAX_RECORDS)} if memory_family else {}))
            finally:
                if locked:
                    slock.release()
                if admitted:
                    admission.slots.release()
        return finished
