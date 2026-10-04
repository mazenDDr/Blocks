"""Node result cache retention (ADR 0015): what is stored, and explicit pruning. Pruning removes cache entries only; recorded run
artifacts are never touched, and a pruned result is recomputed (a cache miss) the next time a run needs it.

GET  /api/cache/nodes          entries, bytes and per-project counts
POST /api/cache/nodes/prune    {projectId | allProjects, keepLatestPerNode?, olderThanDays?, dryRun}"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tabular.cache import cache_summary, prune


class PruneReq(BaseModel):
    model_config = {"extra": "forbid"}
    projectId: str | None = None
    allProjects: bool = False
    keepLatestPerNode: int | None = Field(None, ge=0, le=1000)
    olderThanDays: float | None = Field(None, ge=0)
    dryRun: bool = True  # the default only reports what would be removed


def register(app: FastAPI, sv: Any) -> None:
    @app.get("/api/cache/nodes")
    def summary():
        return cache_summary(sv.store)

    @app.post("/api/cache/nodes/prune")
    def do_prune(req: PruneReq):
        if req.projectId is None and not req.allProjects:
            raise HTTPException(422, {"code": "request_invalid", "message": "Name a projectId, or set allProjects to prune every project's entries."})
        return prune(sv.store, project_id=req.projectId, all_projects=req.allProjects, keep_latest_per_node=req.keepLatestPerNode,
                     older_than_seconds=None if req.olderThanDays is None else req.olderThanDays * 86400, dry_run=req.dryRun)
