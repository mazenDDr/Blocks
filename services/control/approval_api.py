"""Serving approval inspection/resume routes; no standalone execution path."""
import json
from fastapi import Query
from fastapi.responses import JSONResponse
from production.approval_requests import ResumeRequest, inspect
from production.pipeline import ProductionError
from .accounts import ensure_owner

def register(app, runtime):
    @app.get("/api/production/releases/{rid}/approval")
    def approval(rid: str, user: str = Query("local-user", pattern=r"^[A-Za-z0-9_-]{1,64}$"), session: str = Query(..., pattern=r"^[A-Za-z0-9_-]{1,64}$")):
        ensure_owner(user)
        return inspect(runtime, rid, user, session)

    @app.post("/api/serve/{target}/{namespace}/resume")
    def resume(target: str, namespace: str, req: ResumeRequest):
        try:
            json.dumps(req.model_dump(), allow_nan=False)
        except ValueError as exc:
            raise ProductionError("E_REQUEST_SCHEMA", "Non-finite numbers are not valid resume inputs.") from exc
        ensure_owner(req.user)
        result = runtime.predict(target, namespace, req)
        return JSONResponse(result, status_code=result["status"])
