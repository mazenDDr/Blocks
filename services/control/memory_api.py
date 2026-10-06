"""Release long-term memory inspection and deletion (ADR0075). Records belong to a release and a request user."""
from fastapi import Query
from production.pipeline import ProductionError
from production.runtime import MEMORY_ADAPTERS, MEMORY_MAX_RECORDS
from .accounts import ensure_owner

USER = Query("local-user", pattern=r"^[A-Za-z0-9_-]{1,64}$")


def register(app, sv):
    def memory_release(rid):
        release = sv.production.ps.get("release", rid)
        if sv.production.ps.get("version", release["versionId"]).get("adapter") not in MEMORY_ADAPTERS:
            raise ProductionError("E_RELEASE_CONFIG", "This release does not keep long-term memory.")
        return release

    @app.get("/api/production/releases/{rid}/memory")
    def memory(rid: str, user: str = USER):
        ensure_owner(user)
        memory_release(rid)
        records = sv.production.ps.release_memory(rid, user)
        return {"releaseId": rid, "user": user, "records": records, "count": len(records), "maxRecords": MEMORY_MAX_RECORDS,
                "policy": "records of this user in this release only; written by successful turns; deletion removes the record and its text"}

    @app.delete("/api/production/releases/{rid}/memory/{mid}")
    def delete_memory(rid: str, mid: str, user: str = USER):
        ensure_owner(user)
        memory_release(rid)
        return sv.production.ps.delete_release_memory(rid, user, mid)
