"""Explicit per-project automatic cache-retention policies and recorded receipts."""
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from maintenance.cache_retention import RetentionPolicy


class Configure(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expectedRevision: int = Field(0, ge=0)
    policy: RetentionPolicy


def register(app, sv):
    def project(project_id):
        path = sv.project_path(project_id)
        if not path.exists():
            raise HTTPException(404, {"code": "E_CACHE_POLICY_PROJECT", "message": "Save this project before configuring automatic retention."})

    @app.get("/api/cache/nodes/policies/{project_id}")
    def inspect(project_id: str):
        sv.project_path(project_id)  # existing bounded, traversal-safe project id validation
        return sv.cache_retention.inspect(project_id)

    @app.post("/api/cache/nodes/policies/{project_id}/preview")
    def preview(project_id: str, policy: RetentionPolicy):
        project(project_id)
        return sv.cache_retention.preview(project_id, policy)

    @app.put("/api/cache/nodes/policies/{project_id}")
    def configure(project_id: str, req: Configure):
        project(project_id)
        try:
            return sv.cache_retention.configure(project_id, req.policy, req.expectedRevision)
        except ValueError as exc:
            message = str(exc)
            raise HTTPException(409 if message.startswith("E_CACHE_POLICY_STALE") else 422,
                                {"code": message.split(":", 1)[0], "message": message}) from exc
