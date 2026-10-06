"""Registry, release preview/actions, real local serving and measured investigation."""
from __future__ import annotations

import contextvars
import json
import math
import queue
import threading
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import Field

from agent.models import TOKEN_SINK
from .accounts import current, ensure_owner
from production.models import RegisterVersion, ReleaseCreate, PredictRequest, Strict, TrafficSpec
from production.pipeline import ProductionError
from production.runtime import AGENT_ADAPTERS, CONVERSATION_ADAPTERS, JSON_ADAPTERS, PORTABLE_ADAPTERS, ProductionRuntime
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


class ConversationRestore(ConversationReset):
    expectedCheckpointSha256: str | None = Field(..., pattern=r"^[0-9a-f]{64}$")
    sourceRequestId: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    sourceTraceSha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sourceCheckpointSha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def register(app: FastAPI, sv):
    rt = ProductionRuntime(sv.store)
    traffic = TrafficRunner(rt, getattr(sv, "api_token", None))
    sv.production, sv.traffic = rt, traffic
    ps = rt.ps
    from .approval_api import register as register_approvals
    register_approvals(app, rt)

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
                    for backend in ("keras", "jax"):  # the same checkpoint served on another backend after measured agreement
                        candidates.append({"runId": row["id"], "node": f"{backend}:{out_node}", "pipelineSha256": None, "graphHash": row["graph_hash"],
                                           "adapter": f"model_{backend}", "family": "image_classifier"})
                for a in sv.store.artifacts(row["id"], "unsup_pipeline") + sv.store.artifacts(row["id"], "unsup_fitted_pipeline"):
                    candidates.append({"runId": row["id"], "node": a["meta"]["node"], "pipelineSha256": a["sha256"], "graphHash": row["graph_hash"], "adapter": "unsup",
                                       "family": json.loads(sv.store.read_artifact(a["sha256"]))["method"]})
                from production import rl_adapter
                q_node = rl_adapter.is_candidate(sv.store, row)
                if q_node:
                    candidates.append({"runId": row["id"], "node": q_node, "pipelineSha256": None, "graphHash": row["graph_hash"], "adapter": "rl", "family": "rl_policy"})
                from production import agent_adapter, conversation_adapter, json_agent_adapter
                conversation_node = conversation_adapter.is_candidate(sv.store, row)
                if conversation_node:
                    candidates.append({"runId": row["id"], "node": conversation_node, "pipelineSha256": None,
                                       "graphHash": row["graph_hash"], "adapter": "conversation", "family": "agent_turn"})
                agent_node = agent_adapter.is_candidate(sv.store, row)
                if agent_node:
                    candidates.append({"runId": row["id"], "node": agent_node, "pipelineSha256": None,
                                       "graphHash": row["graph_hash"], "adapter": "agent", "family": "agent_turn"})
                from production import approval_adapter
                approval_node = approval_adapter.is_candidate(sv.store, row)
                if approval_node:
                    candidates.append({"runId": row["id"], "node": approval_node, "pipelineSha256": None,
                                       "graphHash": row["graph_hash"], "adapter": "conversation_approval", "family": "agent_turn"})
                from production import tools_agent_adapter
                tools_node = tools_agent_adapter.is_candidate(sv.store, row)
                if tools_node:
                    candidates.append({"runId": row["id"], "node": tools_node, "pipelineSha256": None,
                                       "graphHash": row["graph_hash"], "adapter": "agent_tools", "family": "agent_turn"})
                from production import json_conversation_adapter, retrieval_agent_adapter
                retrieval_node = retrieval_agent_adapter.is_candidate(sv.store, row)
                if retrieval_node:
                    candidates.append({"runId": row["id"], "node": retrieval_node, "pipelineSha256": None,
                                       "graphHash": row["graph_hash"], "adapter": "agent_retrieval", "family": "agent_turn"})
                json_conversation_node = json_conversation_adapter.is_candidate(sv.store, row)
                if json_conversation_node:
                    candidates.append({"runId": row["id"], "node": json_conversation_node, "pipelineSha256": None,
                                       "graphHash": row["graph_hash"], "adapter": "conversation_json", "family": "agent_json"})
                json_node = json_agent_adapter.is_candidate(sv.store, row)
                if json_node:
                    candidates.append({"runId": row["id"], "node": json_node, "pipelineSha256": None,
                                       "graphHash": row["graph_hash"], "adapter": "agent_json", "family": "agent_json"})
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
                                 "streaming": "agent SSE: provisional token deltas followed by the authoritative trace", "agent": "isolated native LangGraph text or schema-validated JSON turns; local Ollama pinned by installed digest/runtime; bounded native persistent text conversations; bounded pure calculator turns and pinned retrieval; committed human approval checkpoints; no file tools or memory effects",
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
        if ps.get("version", vid).get("adapter") in AGENT_ADAPTERS:
            json_output = ps.get("version", vid).get("adapter") in JSON_ADAPTERS
            return {"records": p.reference_records(), "observedLabels": None, "family": p.manifest["family"], "inputContract": p.manifest["inputContract"],
                    "labelNote": "Source run input only. No ground truth is inferred. Supply a schema-valid reference object for canonical JSON agreement; not semantic accuracy." if json_output else "Source run input only. No ground truth is inferred from its response. Supply reference text for exact string agreement; this is not semantic accuracy.",
                    "provenance": {"versionId": vid, "runId": p.manifest["runId"], "referenceSha256": p.manifest["referenceSha256"],
                                   "partition": p.manifest["referencePartition"], "provider": p.manifest["provider"]}}
        if ps.get("version", vid).get("adapter") == "unsup":
            return {"records": p.reference_records(3), "observedLabels": None, "family": p.manifest["method"],
                    "inputContract": (f"records: [{{{', '.join(c['name'] + ': ' + c['dtype'] + (' or null' if c['nullable'] else '') for c in p.manifest['inputSchema'])}}}] "
                                      "(raw source columns; the run's fitted preprocessing is replayed)") if "inputSchema" in p.manifest
                                     else f"records: [{{{', '.join(p.manifest['features'])}: finite numbers}}]",
                    "labelNote": "Rows the estimator was fitted on (in-sample). Clusters have no ground truth; supply external labels only if you have them.",
                    "provenance": {"versionId": vid, "runId": p.manifest["runId"], "referenceSha256": p.manifest["referenceSha256"], "partition": "fitted rows (in-sample)"}}
        if ps.get("version", vid).get("adapter") == "rl":
            return {"records": p.reference_records(3), "observedLabels": None, "family": "rl_policy", "inputContract": p.manifest["inputContract"],
                    "labelNote": "Replay-buffer observations of the source run. Their recorded actions came from the epsilon-greedy behaviour policy, so they are not offered as ground truth.",
                    "provenance": {"versionId": vid, "runId": p.manifest["runId"], "referenceSha256": p.manifest["referenceSha256"], "partition": "replay buffer (training experience)",
                                   "source": p.manifest["source"]}}
        if ps.get("version", vid).get("adapter") in ("model", *PORTABLE_ADAPTERS):
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
        agent = ps.get("version", r["versionId"]).get("adapter") in AGENT_ADAPTERS
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
        ensure_owner(req.user)
        result = rt.predict(target, namespace, req)
        return JSONResponse(result, status_code=result["status"])

    @app.post("/api/serve/{target}/{namespace}/predict/stream")
    def predict_stream(target: str, namespace: str, req: PredictRequest):
        """Same admission, execution and recorded trace as predict, with provider text deltas relayed as server-sent events.

        Events: `token` ({"delta"}) for each provider delta in arrival order, then `result` (the exact recorded trace the
        non-streaming route returns) and `end` ({"deltas", "chars", "status"}). The recorded trace stays authoritative: a
        request whose output is later refused (deadline, cancellation, budget) still streamed the text it received. A client
        disconnect does not cancel the request; use the cancel route.
        """
        bad = first_non_finite(req.records, "records")
        if bad:
            raise ProductionError("E_REQUEST_SCHEMA", f"Non-finite number at {bad}; NaN and Infinity are not valid inputs.", 422)
        ensure_owner(req.user)
        events: queue.SimpleQueue = queue.SimpleQueue()

        def work():
            TOKEN_SINK.set(lambda delta: events.put(("token", delta)))
            try:
                events.put(("result", rt.predict(target, namespace, req)))
            except Exception as e:  # noqa: BLE001 - surfaced as a final event instead of a broken stream
                events.put(("error", {"code": getattr(e, "code", "E_INFERENCE"), "message": str(e)}))

        threading.Thread(target=contextvars.copy_context().run, args=(work,), daemon=True, name=f"stream-{req.requestId}").start()

        def frames():
            count = chars = 0
            while True:
                kind, value = events.get()
                if kind == "token":
                    count, chars = count + 1, chars + len(value)
                    yield f"event: token\ndata: {json.dumps({'delta': value})}\n\n"
                    continue
                status = value["status"] if kind == "result" else 500
                yield f"event: {kind}\ndata: {json.dumps(value)}\n\n"
                yield f"event: end\ndata: {json.dumps({'deltas': count, 'chars': chars, 'status': status})}\n\n"
                return

        return StreamingResponse(frames(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.get("/api/production/requests")
    def requests(release: str | None = None):
        acc = current()
        traces = ps.traces(release)
        return {"requests": traces if acc is None or acc.role == "admin" else [t for t in traces if t.get("user") == acc.name], "max": 1000}

    @app.get("/api/production/requests/{id_}")
    def request_trace(id_: str, user: str = "local-user"):
        ensure_owner(user)
        return ps.trace(user, id_)

    @app.post("/api/production/requests/{id_}/cancel")
    def cancel_request(id_: str, req: User):
        ensure_owner(req.user)
        return ps.cancel(req.user, id_)

    @app.post("/api/production/requests/{id_}/labels")
    def label_request(id_: str, req: Labels):
        ensure_owner(req.user)
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
        if adapter in JSON_ADAPTERS:
            from production.json_agent_adapter import output_value
            try:
                for value in req.labels:
                    output_value(value, p.cfg)
            except ProductionError as exc:
                raise ProductionError("E_LABEL_SCHEMA", "Supply one finite bounded schema-valid reference JSON object per turn.") from exc
            ps.add_labels(req.user, id_, req.labels)
            return {"recorded": True, "requestId": id_, "rows": len(req.labels)}
        if adapter in AGENT_ADAPTERS:
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
        ensure_owner(user)
        from production.discovery import discover
        return discover(ps, rid, user, limit=limit, after=after, prefix=prefix)

    @app.get("/api/production/releases/{rid}/conversation")
    def conversation(rid: str, user: str = Query("local-user", pattern=r"^[A-Za-z0-9_-]{1,64}$"),
                     session: str = Query(..., pattern=r"^[A-Za-z0-9_-]{1,64}$")):
        ensure_owner(user)
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
        ensure_owner(req.user)
        from production.conversations import action
        return action(rt, rid, "reset", req)

    @app.post("/api/production/releases/{rid}/conversation/fork", status_code=201)
    def fork_conversation(rid: str, req: ConversationFork):
        ensure_owner(req.user)
        from production.conversations import action
        return action(rt, rid, "fork", req)

    @app.get("/api/production/releases/{rid}/conversation/history/{request_id}")
    def historical_checkpoint(rid: str, request_id: str,
                              user: str = Query("local-user", pattern=r"^[A-Za-z0-9_-]{1,64}$"),
                              session: str = Query(..., pattern=r"^[A-Za-z0-9_-]{1,64}$")):
        ensure_owner(user)
        from production.history import inspect_history
        return inspect_history(rt, rid, user, session, request_id)

    @app.post("/api/production/releases/{rid}/conversation/restore")
    def restore_conversation(rid: str, req: ConversationRestore):
        ensure_owner(req.user)
        from production.conversations import action
        return action(rt, rid, "restore", req)

    @app.post("/api/production/requests/{id_}/replay")
    def investigate(id_: str, req: User):
        ensure_owner(req.user)
        trace = ps.trace(req.user, id_)
        if not trace.get("records") or "versionId" not in trace:
            raise ProductionError("E_REPLAY_NOT_CAPTURED", "Inputs were not captured under this release's policy; replay unavailable.", 409)
        p = rt.pipeline(trace["versionId"])
        p.validate_records(trace["records"])
        agent = ps.get("version", trace["versionId"]).get("adapter") in AGENT_ADAPTERS
        if ps.get("version", trace["versionId"]).get("adapter") in CONVERSATION_ADAPTERS:
            if "conversationParent" not in trace:
                raise ProductionError("E_REPLAY_CHECKPOINT", "This failed request has no recorded prior checkpoint; replay unavailable.", 409)
            kwargs = {}
            if ps.get("version", trace["versionId"]).get("adapter") == "conversation_approval":
                from production.approval_requests import Review
                kwargs["approval"] = Review.model_validate(trace["approval"]) if trace.get("approval") else None
            result, timing = p.predict(trace["records"], capture=True, checkpoint=trace["conversationParent"]["checkpointSha256"], **kwargs)
            timing.pop("_checkpoint", None)
            timing.pop("_status", None)
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
        return traffic.start(req, port, request.headers.get("authorization") if current() else None)

    @app.get("/api/production/traffic/{id_}")
    def traffic_status(id_: str):
        return traffic.view(id_)

    @app.post("/api/production/traffic/{id_}/cancel")
    def stop_traffic(id_: str):
        traffic.cancel(id_)
        return {"cancelRequested": True, "policy": "stop new arrivals; drain already submitted bounded requests"}
