"""Read-only native domain checkpoint/export/inference routes; no arbitrary model uploads."""
from __future__ import annotations
import threading
import json
from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, ConfigDict
from tabular.core import ExecutionError
from domain import checkpoints as CP
from domain.inference import predict

class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    records: list[dict] = Field(min_length=1, max_length=4)


def register(app, sv):
    slots = threading.BoundedSemaphore(2)
    def call(fn):
        try:
            return fn()
        except ExecutionError as e:
            raise HTTPException(422, {"code": e.code, "message": e.message}) from e
    @app.get("/api/domain/models")
    def models(runId: str | None = None):
        rows = [sv.store.get_run(runId)] if runId else sv.store.list_runs()
        models = []
        for r in rows:
            if r and r["status"] == "completed":
                for a in sv.store.artifacts(r["id"], "domain_model"):
                    try:
                        m = CP.read_manifest(sv.store, a["sha256"])
                        models.append({"modelId": a["sha256"], "available": True, **m})
                    except ExecutionError as e:
                        models.append({"modelId": a["sha256"], "available": False, "runId": r["id"], "error": {"code": e.code, "message": e.message}})
        return {"models": models, "limits": {"maxConcurrent": 2, "maxBatch": 4, "maxPayloadBytes": 1500000},
                "scope": "local CPU domain inference; independent of tabular production releases"}
    @app.get("/api/domain/models/{identity}")
    def manifest(identity: str):
        return {"modelId": identity, **call(lambda: CP.read_manifest(sv.store, identity))}
    @app.get("/api/domain/models/{identity}/example")
    def example(identity: str):
        m = call(lambda: CP.read_manifest(sv.store, identity))
        if not any(a["kind"] == "domain_inference_example" and a["sha256"] == m["exampleSha256"] and a["meta"].get("node") == m["node"] for a in sv.store.artifacts(m["runId"])):
            raise HTTPException(422, {"code": "E_DOMAIN_CHECKPOINT_TRUST", "message": "No recorded example for this model."})
        return {"records": [json.loads(call(lambda: CP.verified(sv.store, m["exampleSha256"])))], "synthetic": True, "partition": "validation", "modelId": identity}
    @app.get("/api/domain/models/{identity}/checkpoint")
    def export(identity: str):
        m = call(lambda: CP.read_manifest(sv.store, identity))
        raw = call(lambda: CP.verified(sv.store, m["checkpointSha256"]))
        return Response(raw, media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="{identity}.pt"', "X-Checkpoint-Sha256": m["checkpointSha256"]})
    @app.post("/api/domain/models/{identity}/predict")
    def inference(identity: str, req: Input):
        if not slots.acquire(blocking=False):
            raise HTTPException(429, {"code": "E_DOMAIN_BUSY", "message": "Two local inference requests are active."})
        try:
            return call(lambda: predict(sv.store, identity, req.records))
        finally:
            slots.release()
