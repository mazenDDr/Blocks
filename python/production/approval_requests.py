"""Strict reviewed approval requests and scope checks (ADR0063)."""
from typing import Literal
from pydantic import Field, model_validator
from .models import PredictRequest, Strict
from .pipeline import ProductionError
from tabular.core import dumps

class Review(Strict):
    expectedRevision: int = Field(ge=1)
    expectedCheckpointSha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    interruptId: str = Field(min_length=1, max_length=128)
    action: Literal["approve", "reject", "edit"]
    value: str | None = Field(None, max_length=2000)

    @model_validator(mode="after")
    def validate_edit(self):
        if (self.action == "edit") != (self.value is not None):
            raise ValueError("Only edit requires a bounded text value.")
        return self

class ResumeRequest(PredictRequest):
    session: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    expectedRelease: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval: Review

def reviewed_parent(previous, approval):
    if approval is not None and (not previous or (previous["revision"], previous["checkpointSha256"]) != (approval.expectedRevision, approval.expectedCheckpointSha256)):
        raise ProductionError("E_SESSION_CONFLICT", "Reviewed paused checkpoint changed; inspect the session before resuming.", 409)

def inspect(runtime, release_id, user, session):
    release = runtime.ps.get("release", release_id)
    if runtime.ps.get("version", release["versionId"]).get("adapter") != "conversation_approval":
        raise ProductionError("E_RELEASE_CONFIG", "This release does not support approval checkpoints.")
    pipeline = runtime.pipeline(release["versionId"])
    head = runtime.ps.conversation(dumps([release_id, user, session]))
    value = pipeline.inspect_checkpoint(head["checkpointSha256"]) if head and head["checkpointSha256"] else {"state": None, "pending": None, "budget": None}
    return {"releaseId": release_id, "versionId": release["versionId"], "user": user, "session": session, "head": head, **value,
            "records": [{key: value["state"][key] for key in pipeline.manifest["inputFields"]}] if value["state"] else None,
            "readOnly": True, "persistencePolicy": "Native paused checkpoint and review persist independently of trace capture; inspection makes no model call."}
