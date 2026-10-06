"""Start an evaluation of a deployed release (ADR 0078); results are read through /api/agent/evals/{id}."""
import threading
import uuid

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from production.pipeline import ProductionError
from production.release_eval import ReleaseEvalRequest, run_release_eval
from .accounts import current


def register(app, sv):
    @app.post("/api/production/releases/{rid}/evaluate")
    def evaluate(rid: str, req: ReleaseEvalRequest):
        acc = current()
        if acc is not None and acc.role != "admin":
            raise HTTPException(403, {"code": "E_ROLE", "message": "Evaluating a release sends requests as evaluation users; admin only."})
        rt = sv.production
        release = rt.ps.get("release", rid)
        route = next((r for r in rt.ps.query("SELECT * FROM routes") if r["target"] == release["config"]["target"] and r["namespace"] == release["config"]["namespace"]), None)
        if not route or route["release"] != rid:
            raise ProductionError("E_RELEASE_NOT_DEPLOYED", "Deploy this release first: evaluation sends real requests through its route.", 409)
        run_id = uuid.uuid4().hex[:12]
        graph_hash = rt.ps.get("version", release["versionId"])["manifest"].get("graphHash") or "release"
        sv.store.create_run(run_id, graph_hash, {"kind": "release_eval", "releaseId": rid, "name": req.name, "cases": [c.model_dump() for c in req.cases]})
        threading.Thread(target=run_release_eval, args=(rt, release, req, run_id), daemon=True, name=f"release-eval-{run_id}").start()
        return JSONResponse({"runId": run_id, "status": "queued", "cases": len(req.cases)}, status_code=201)
