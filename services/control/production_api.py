"""Registry, release preview/actions, real local serving and measured investigation."""
from __future__ import annotations

import json
import math
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import Field

from production.models import RegisterVersion, ReleaseCreate, PredictRequest, Strict, TrafficSpec
from production.pipeline import ProductionError
from production.runtime import ProductionRuntime
from production.traffic import TrafficRunner
from production.monitor import monitoring


class Activate(Strict):
    expectedCurrent: str | None = None


class Alias(Strict):
    versionId: str


class User(Strict):
    user: str = Field("local-user", pattern=r"^[A-Za-z0-9_-]{1,64}$")


class Labels(User):
    labels: list[Any] = Field(min_length=1, max_length=128)


class ConversationReset(User):
    actionId: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    session: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    expectedRevision: int = Field(ge=1)
    expectedCheckpointSha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str = Field(min_length=1, max_length=500)


class ConversationFork(ConversationReset):
    destinationSession: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")


def register(app: FastAPI, sv):
    rt = ProductionRuntime(sv.store)
    traffic = TrafficRunner(rt, getattr(sv, "api_token", None))
    sv.production, sv.traffic = rt, traffic
    ps = rt.ps

    @app.exception_handler(ProductionError)
    async def production_error(_: Request, e: ProductionError):
        return JSONResponse({"detail": {"code": e.code, "message": e.message}}, status_code=e.status)

    @app.get("/api/production")
    def overview():
        candidates = []
        for row in sv.store.list_runs()[-100:]:
            if row["status"] == "completed":
                for a in sv.store.artifacts(row["id"], "inference_pipeline"):
                    candidates.append({"runId": row["id"], "node": a["meta"]["node"], "pipelineSha256": a["sha256"], "graphHash": row["graph_hash"], "adapter": "tabular"})
                from production.model_adapter import is_candidate
                out_node = is_candidate(sv.store, row)
                if out_node:
                    candidates.append({"runId": row["id"], "node": out_node, "pipelineSha256": None, "graphHash": row["graph_hash"], "adapter": "model", "family": "image_classifier"})
                for a in sv.store.artifacts(row["id"], "unsup_pipeline"):
                    candidates.append({"runId": row["id"], "node": a["meta"]["node"], "pipelineSha256": a["sha256"], "graphHash": row["graph_hash"], "adapter": "unsup",
                                       "family": json.loads(sv.store.read_artifact(a["sha256"]))["method"]})
                from production import rl_adapter
                q_node = rl_adapter.is_candidate(sv.store, row)
                if q_node:
                    candidates.append({"runId": row["id"], "node": q_node, "pipelineSha256": None, "graphHash": row["graph_hash"], "adapter": "rl", "family": "rl_policy"})
                from production import agent_adapter, conversation_adapter
                conversation_node = conversation_adapter.is_candidate(sv.store, row)
                if conversation_node:
                    candidates.append({"runId": row["id"], "node": conversation_node, "pipelineSha256": None,
                                       "graphHash": row["graph_hash"], "adapter": "conversation", "family": "agent_turn"})
                agent_node = agent_adapter.is_candidate(sv.store, row)
                if agent_node:
                    candidates.append({"runId": row["id"], "node": agent_node, "pipelineSha256": None,
                                       "graphHash": row["graph_hash"], "adapter": "agent", "family": "agent_turn"})
                for a in sv.store.artifacts(row["id"], "domain_model"):
                    if a["meta"].get("internalDomainModel"):
                        candidates.append({"runId": row["id"], "node": a["meta"]["node"], "pipelineSha256": a["sha256"], "graphHash": row["graph_hash"],
                                           "adapter": "domain", "family": a["meta"].get("family")})
        events = ps.query("SELECT * FROM lifecycle ORDER BY seq DESC LIMIT 100")
        return {"candidates": candidates, "versions": ps.list("version"), "releases": ps.list("release"),
                "routes": ps.query("SELECT * FROM routes"), "aliases": ps.query("SELECT * FROM aliases"),
                "lifecycle": [{**r, "data": json.loads(r["data"])} for r in events], "traffic": ps.list("traffic"),
                "capabilities": {"adapter": "native scikit-learn tabular pipeline; native PyTorch vision/NLP/speech domain models (1-4 records per request); PyTorch model-graph image classifiers (1-32 images); greedy DQN policies (1-256 observations); k-means / Gaussian mixture / PCA (1-128 rows)", "targets": ["local", "staging"], "replicas": 1,
                                 "mode": "real native serving in this local FastAPI process; Ollama model device managed by its runtime; staging is a separate local route",
                                 "remoteDeployment": "not implemented: no infrastructure configured", "batch": "bounded synchronous batch",
                                 "streaming": "not implemented", "agent": "isolated native LangGraph turns; local Ollama pinned by installed digest/runtime; prompt/set_state/one chat node only; bounded native persistent conversations; no tool/memory effects",
                                 "session": "optional durable per-release/user/session counter for other families; conversation graphs require explicit sessions; stateless graphs keep stateless mode",
                                 "authentication": "single-user local workbench; user/session fields are caller-declared identities, not authentication",
                                 "limits": "latest 100 records per family / 100 lifecycle events; full source lineage stays in CAS"}}

    @app.post("/api/production/versions", status_code=201)
    def register_version(req: RegisterVersion):
        return rt.register_version(req)

    @app.get("/api/production/versions/{vid}")
    def version(vid: str):
        return ps.get("version", vid)

    @app.get("/api/production/versions/{vid}/reference-input")
    def reference_input(vid: str):
        import io
        import pandas as pd
        p = rt.pipeline(vid)
        if ps.get("version", vid).get("adapter") in ("agent", "conversation"):
            return {"records": p.reference_records(), "observedLabels": None, "family": "agent_turn", "inputContract": p.manifest["inputContract"],
                    "labelNote": "Source run input only. No ground truth is inferred from its response. Supply reference text for exact string agreement; this is not semantic accuracy.",
                    "provenance": {"versionId": vid, "runId": p.manifest["runId"], "referenceSha256": p.manifest["referenceSha256"],
                                   "partition": p.manifest["referencePartition"], "provider": p.manifest["provider"]}}
        if ps.get("version", vid).get("adapter") == "unsup":
            return {"records": p.reference_records(3), "observedLabels": None, "family": p.manifest["method"],
                    "inputContract": f"records: [{{{', '.join(p.manifest['features'])}: finite numbers}}]",
                    "labelNote": "Rows the estimator was fitted on (in-sample). Clusters have no ground truth; supply external labels only if you have them.",
                    "provenance": {"versionId": vid, "runId": p.manifest["runId"], "referenceSha256": p.manifest["referenceSha256"], "partition": "fitted rows (in-sample)"}}
        if ps.get("version", vid).get("adapter") == "rl":
            return {"records": p.reference_records(3), "observedLabels": None, "family": "rl_policy", "inputContract": p.manifest["inputContract"],
                    "labelNote": "Replay-buffer observations of the source run. Their recorded actions came from the epsilon-greedy behaviour policy, so they are not offered as ground truth.",
                    "provenance": {"versionId": vid, "runId": p.manifest["runId"], "referenceSha256": p.manifest["referenceSha256"], "partition": "replay buffer (training experience)",
                                   "source": p.manifest["source"]}}
        if ps.get("version", vid).get("adapter") == "model":
            refs = p.reference()[:3]
            return {"records": p.reference_records(3), "observedLabels": [r["label"] for r in refs], "family": "image_classifier", "inputContract": p.manifest["inputContract"],
                    "labelNote": "Held-out validation images of the source run with their recorded labels (frozen at registration).",
                    "provenance": {"versionId": vid, "runId": p.manifest["runId"], "referenceSha256": p.manifest["referenceSha256"], "files": [r["file"] for r in refs],
                                   "partition": "held-out validation split", "source": p.manifest["source"]}}
        if ps.get("version", vid).get("adapter") == "domain":
            return {"records": p.reference_records(), "observedLabels": None, "family": p.manifest["family"], "inputContract": p.manifest["inputContract"],
                    "labelNote": "Recorded held-out example of the source run; synthetic status and license declaration are recorded in source provenance. Its ground truth is not exposed here.",
                    "provenance": {"versionId": vid, "runId": p.manifest["runId"], "modelId": p.manifest["modelId"], "exampleSha256": p.manifest["exampleSha256"],
                                   "partition": "recorded held-out example", "source": p.manifest["source"]}}
        df = pd.read_csv(io.BytesIO(sv.store.read_artifact(p.manifest["referenceSha256"]))).head(3)
        records = df.astype(object).where(df.notna(),None).to_dict("records")
        import json
        labels = json.loads(sv.store.read_artifact(p.manifest["referenceLabelsSha256"]))[:len(records)]
        return {"records": records, "observedLabels": labels, "labelNote": "Recorded training-reference labels; quality on these inputs is in-sample evidence, not a held-out production benchmark.",
                "provenance": {"versionId": vid, "runId": p.manifest["runId"], "referenceSha256": p.manifest["referenceSha256"], "referenceLabelsSha256": p.manifest["referenceLabelsSha256"],
                                                   "partition": "recorded training reference", "source": p.manifest["source"]}}

    @app.put("/api/production/aliases/{name}")
    def alias(name: str, req: Alias):
        if not 1 <= len(name) <= 100:
            raise ProductionError("E_ALIAS_NAME", "Alias length must be 1..100 characters.")
        ps.alias(name, req.versionId)
        return {"name": name, "versionId": req.versionId}

    @app.post("/api/production/releases", status_code=201)
    def candidate(req: ReleaseCreate):
        return rt.create_release(req)

    @app.get("/api/production/releases/{rid}")
    def release(rid: str):
        return ps.get("release", rid)

    @app.post("/api/production/releases/{rid}/deploy")
    def deploy(rid: str, req: Activate):
        return rt.activate(rid, req.expectedCurrent)

    @app.post("/api/production/releases/{rid}/rollback")
    def rollback(rid: str, req: Activate):
        return rt.activate(rid, req.expectedCurrent, True)

    @app.get("/api/serve/{target}/{namespace}/health")
    def health(target: str, namespace: str):
        r = ps.route(target, namespace)
        rt.pipeline(r["versionId"])
        agent = ps.get("version", r["versionId"]).get("adapter") in ("agent", "conversation")
        return {"ready": True, "releaseId": r["id"], "versionId": r["versionId"], "replicas": 1, "placement": "local control process; Ollama model device not measured" if agent else "local CPU",
                "limits": r["config"]}

    def first_non_finite(v, path):
        if isinstance(v, float) and not math.isfinite(v):
            return path
        if isinstance(v, dict):
            return next((p for k, x in v.items() if (p := first_non_finite(x, f"{path}.{k}"))), None)
        if isinstance(v, list):
            return next((p for i, x in enumerate(v) if (p := first_non_finite(x, f"{path}[{i}]"))), None)
        return None

    @app.post("/api/serve/{target}/{namespace}/predict")
    def predict(target: str, namespace: str, req: PredictRequest):
        # Python's JSON parser accepts NaN/Infinity. They are never valid inputs, and a JSON response could not echo them in a trace,
        # so the HTTP route refuses them before admission (in-process callers still get a recorded E_REQUEST_SCHEMA trace).
        bad = first_non_finite(req.records, "records")
        if bad:
            raise ProductionError("E_REQUEST_SCHEMA", f"Non-finite number at {bad}; NaN and Infinity are not valid inputs.", 422)
        result = rt.predict(target, namespace, req)
        return JSONResponse(result, status_code=result["status"])

    @app.get("/api/production/requests")
    def requests(release: str | None = None):
        return {"requests": ps.traces(release), "max": 1000}

    @app.get("/api/production/requests/{id_}")
    def request_trace(id_: str, user: str = "local-user"):
        return ps.trace(user, id_)

    @app.post("/api/production/requests/{id_}/cancel")
    def cancel_request(id_: str, req: User):
        return ps.cancel(req.user, id_)

    @app.post("/api/production/requests/{id_}/labels")
    def label_request(id_: str, req: Labels):
        trace = ps.trace(req.user, id_)
        if trace.get("status") != 200:
            raise ProductionError("E_LABEL_ALIGNMENT", "Only completed predictions can receive labels.")
        p = rt.pipeline(trace["versionId"])
        classes = p.manifest["outputSchema"]["classes"]
        if ps.get("version", trace["versionId"]).get("adapter") == "domain":
            from production.domain_adapter import validate_labels
            validate_labels(p.manifest["family"], trace["result"]["predictions"], req.labels, classes)
            ps.add_labels(req.user, id_, req.labels)
            return {"recorded": True, "requestId": id_, "rows": len(req.labels)}
        adapter = ps.get("version", trace["versionId"]).get("adapter")
        if adapter in ("agent", "conversation"):
            if not all(isinstance(v, str) and len(v) <= 32_768 for v in req.labels):
                raise ProductionError("E_LABEL_SCHEMA", "Supply one bounded reference string per agent turn (exact string agreement only).")
            ps.add_labels(req.user, id_, req.labels)
            return {"recorded": True, "requestId": id_, "rows": len(req.labels)}
        if adapter == "unsup":
            if p.manifest["method"] == "pca":
                raise ProductionError("E_LABEL_SCHEMA", "PCA has no labels; its label-free reconstruction error is monitored instead.")
            if not all(isinstance(v, (str, int)) and not isinstance(v, bool) for v in req.labels):
                raise ProductionError("E_LABEL_SCHEMA", "External labels are strings or integers, one per record.")
            ps.add_labels(req.user, id_, req.labels)
            return {"recorded": True, "requestId": id_, "rows": len(req.labels)}
        if adapter == "rl":
            valid = all(type(v) is int and v in classes for v in req.labels)  # booleans are not actions
        elif classes is not None:
            valid = all(v in classes for v in req.labels)
        else:
            valid = all(isinstance(v, (int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in req.labels)
        if not valid:
            raise ProductionError("E_LABEL_SCHEMA", "Ground truth must match the pinned task's labels/value type.")
        ps.add_labels(req.user, id_, req.labels)
        return {"recorded": True, "requestId": id_, "rows": len(req.labels)}

    @app.get("/api/production/releases/{rid}/conversations")
    def conversations(rid: str, user: str = Query("local-user", pattern=r"^[A-Za-z0-9_-]{1,64}$"),
                      limit: int = Query(25, ge=1, le=100),
                      after: str | None = Query(None, pattern=r"^[A-Za-z0-9_-]{1,64}$"),
                      prefix: str = Query("", pattern=r"^[A-Za-z0-9_-]{0,64}$")):
        from production.discovery import discover
        return discover(ps, rid, user, limit=limit, after=after, prefix=prefix)

    @app.get("/api/production/releases/{rid}/conversation")
    def conversation(rid: str, user: str = Query("local-user", pattern=r"^[A-Za-z0-9_-]{1,64}$"),
                     session: str = Query(..., pattern=r"^[A-Za-z0-9_-]{1,64}$")):
        from tabular.core import dumps
        release = ps.get("release", rid)
        if release["config"]["sessionMode"] != "conversation":
            raise ProductionError("E_RELEASE_CONFIG", "This release has no native conversations.")
        p = rt.pipeline(release["versionId"])
        head = ps.conversation(dumps([rid, user, session]))
        return {"releaseId": rid, "versionId": release["versionId"], "user": user, "session": session, "head": head,
                "state": p.checkpoint_state(head["checkpointSha256"]) if head and head["checkpointSha256"] else None,
                "persistencePolicy": "Native checkpoint/history persists even when trace capture is off. User/session are caller-declared isolation keys, not authenticated identities."}

    @app.post("/api/production/releases/{rid}/conversation/reset")
    def reset_conversation(rid: str, req: ConversationReset):
        from production.conversations import action
        return action(rt, rid, "reset", req)

    @app.post("/api/production/releases/{rid}/conversation/fork", status_code=201)
    def fork_conversation(rid: str, req: ConversationFork):
        from production.conversations import action
        return action(rt, rid, "fork", req)

    @app.post("/api/production/requests/{id_}/replay")
    def investigate(id_: str, req: User):
        trace = ps.trace(req.user, id_)
        if not trace.get("records") or "versionId" not in trace:
            raise ProductionError("E_REPLAY_NOT_CAPTURED", "Inputs were not captured under this release's policy; replay unavailable.", 409)
        p = rt.pipeline(trace["versionId"])
        p.validate_records(trace["records"])
        agent = ps.get("version", trace["versionId"]).get("adapter") in ("agent", "conversation")
        if ps.get("version", trace["versionId"]).get("adapter") == "conversation":
            if "conversationParent" not in trace:
                raise ProductionError("E_REPLAY_CHECKPOINT", "This failed request has no recorded prior checkpoint; replay unavailable.", 409)
            result, timing = p.predict(trace["records"], capture=True, checkpoint=trace["conversationParent"]["checkpointSha256"])
            timing.pop("_checkpoint", None)
        else:
            result, timing = p.predict(trace["records"], capture=True) if agent else p.predict(trace["records"])
        return {"result": result, "timings": timing, "sourceTraceSha256": trace["traceSha256"], "releaseId": trace["releaseId"],
                "versionId": trace["versionId"], "mode": "isolated forward pass of the pinned version; no route, training or session state update",
                "replayNote": "New native Ollama call; output may differ. Model digest/runtime reverified; recorded prior checkpoint restored in isolation; live conversation/source turn untouched." if agent else None}

    @app.get("/api/production/releases/{rid}/monitor")
    def monitor(rid: str, since: float = Query(0, ge=0)):
        return monitoring(rt, rid, since)

    @app.post("/api/production/traffic", status_code=202)
    def start_traffic(req: TrafficSpec, request: Request):
        port = request.scope.get("server", (None,None))[1]
        if not isinstance(port, int) or not 1 <= port <= 65535:
            raise ProductionError("E_TRAFFIC_SERVER", "Traffic requires this app to run on a real loopback HTTP server.")
        return traffic.start(req, port)

    @app.get("/api/production/traffic/{id_}")
    def traffic_status(id_: str):
        return traffic.view(id_)

    @app.post("/api/production/traffic/{id_}/cancel")
    def stop_traffic(id_: str):
        traffic.cancel(id_)
        return {"cancelRequested": True, "policy": "stop new arrivals; drain already submitted bounded requests"}
