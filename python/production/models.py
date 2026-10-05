from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RegisterVersion(Strict):
    runId: str
    node: str
    name: str = Field(min_length=1, max_length=100)
    owner: str = Field(min_length=1, max_length=100)
    intendedUse: str = Field(min_length=1, max_length=2000)
    limitations: str = Field(min_length=1, max_length=2000)


class ServingConfig(Strict):
    target: Literal["local", "staging"] = "local"
    namespace: str = Field("lab", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    concurrency: int = Field(2, ge=1, le=16)
    queueLimit: int = Field(8, ge=0, le=64)
    timeoutSeconds: float = Field(10, ge=0.01, le=30)
    maxBatch: int = Field(32, ge=1, le=128)
    sessionMode: Literal["stateless", "counter", "conversation"] = "stateless"
    captureInputs: bool = False


class ReleaseCreate(Strict):
    versionId: str
    config: ServingConfig = Field(default_factory=ServingConfig)


class PredictRequest(Strict):
    requestId: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    records: list[dict[str, Any]] = Field(min_length=1, max_length=128)
    user: str = Field("local-user", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    session: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    expectedRelease: str | None = None


class TrafficSpec(Strict):
    releaseId: str
    payloads: list[list[dict[str, Any]]] = Field(min_length=1, max_length=16)
    pattern: Literal["steady", "ramp", "burst", "closed_loop"] = "steady"
    rate: float = Field(5, gt=0, le=100)
    durationSeconds: float = Field(2, ge=0.2, le=30)
    concurrency: int = Field(2, ge=1, le=16)
    maxRequests: int = Field(100, ge=1, le=500)
    warmupRequests: int = Field(1, ge=0, le=10)
    expectedStatuses: list[int] = Field(default_factory=lambda: [200], min_length=1, max_length=10)
