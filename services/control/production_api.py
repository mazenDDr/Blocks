"""Registry, release preview/actions, real local serving and measured investigation."""
from __future__ import annotations

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


def register(app: FastAPI, sv):
    rt = ProductionRuntime(sv.store)
    traffic = TrafficRunner(rt)
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
                    candidates.append({"runId": row["id"], "node": a["meta"]["node"], "pipelineSha256": a["sha256"], "graphHash": row["graph_hash"]})
        events = ps.query("SELECT * FROM lifecycle ORDER BY seq DESC LIMIT 100")
        import json
        return {"candidates": candidates, "versions": ps.list("version"), "releases": ps.list("release"),
                "routes": ps.query("SELECT * FROM routes"), "aliases": ps.query("SELECT * FROM aliases"),
                "lifecycle": [{**r, "data": json.loads(r["data"])} for r in events], "traffic": ps.list("traffic"),
                "capabilities": {"adapter": "native scikit-learn tabular pipeline", "targets": ["local", "staging"], "replicas": 1,
                                 "mode": "real CPU serving in this local FastAPI process; staging is a separate local route",
                                 "remoteDeployment": "not implemented: no infrastructure configured", "batch": "bounded synchronous batch",
                                 "streaming": "not implemented", "session": "optional durable per-release/user/session counter; native estimator is stateless",
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
        return {"ready": True, "releaseId": r["id"], "versionId": r["versionId"], "replicas": 1, "placement": "local CPU",
                "limits": r["config"]}

    @app.post("/api/serve/{target}/{namespace}/predict")
    def predict(target: str, namespace: str, req: PredictRequest):
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
        if classes is not None:
            valid = all(v in classes for v in req.labels)
        else:
            valid = all(isinstance(v, (int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in req.labels)
        if not valid:
            raise ProductionError("E_LABEL_SCHEMA", "Ground truth must match the pinned task's labels/value type.")
        ps.add_labels(req.user, id_, req.labels)
        return {"recorded": True, "requestId": id_, "rows": len(req.labels)}

    @app.post("/api/production/requests/{id_}/replay")
    def investigate(id_: str, req: User):
        trace = ps.trace(req.user, id_)
        if not trace.get("records") or "versionId" not in trace:
            raise ProductionError("E_REPLAY_NOT_CAPTURED", "Inputs were not captured under this release's policy; replay unavailable.", 409)
        p = rt.pipeline(trace["versionId"])
        p.validate_records(trace["records"])
        result, timing = p.predict(trace["records"])
        return {"result": result, "timings": timing, "sourceTraceSha256": trace["traceSha256"], "releaseId": trace["releaseId"],
                "versionId": trace["versionId"], "mode": "isolated forward pass of the pinned version; no route, training or session state update"}

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
