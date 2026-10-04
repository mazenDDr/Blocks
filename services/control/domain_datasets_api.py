"""Bounded local imports, protected by the service's existing optional token."""
import threading
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator
from connectors.domain_import import import_dataset, provenance
from tabular.core import ExecutionError
import json


class Import(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["coco", "wav", "conll"]
    path: str = Field(min_length=1, max_length=4096)
    root: str | None = Field(None, min_length=1, max_length=4096)
    license: str = Field(min_length=1, max_length=2000)
    synthetic: StrictBool
    flip_pairs: list[list[int]] = Field(default_factory=list, max_length=64)
    token_column: int = Field(0, ge=-128, le=127)
    label_column: int = Field(-1, ge=-128, le=127)

    @field_validator("license", "path", "root")
    @classmethod
    def nonblank(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Declare a nonempty source path or license/provenance statement.")
        return v


def register(app, sv):
    slot = threading.BoundedSemaphore(1)

    @app.post("/api/domain/datasets", status_code=201)
    def convert(req: Import):
        if req.kind in ("coco", "wav") and req.root is None:
            raise HTTPException(422, {"code": "E_DATASET_PATH", "message": "Declare the source directory for image/WAV members."})
        if not slot.acquire(blocking=False):
            raise HTTPException(429, {"code": "E_DATASET_BUSY", "message": "One local dataset import is active."})
        try:
            return import_dataset(sv.store, req.model_dump())
        except ExecutionError as e:
            raise HTTPException(422, {"code": e.code, "message": e.message}) from e
        finally:
            slot.release()

    @app.get("/api/domain/datasets")
    def datasets():
        rows = []
        folder = sv.store.root / "domain-datasets"
        for p in sorted(folder.glob("*/manifest.json")) if folder.exists() else []:
            try:
                m = json.loads(p.read_bytes())
                if m["extension"] not in (".jsonl", ".npz"):
                    raise ValueError("Invalid extension")
                payload = p.parent / ("data"+m["extension"])
                provenance(payload, None, "")
                rows.append({"id": p.parent.name, "path": str(payload), **m})
            except (ExecutionError, KeyError, ValueError, TypeError, OSError):
                rows.append({"id": p.parent.name, "available": False, "error": "E_DATASET_INTEGRITY"})
        return {"datasets": rows, "limits": {"records": [2, 256], "inputBytes": 67108864,
                "decodedBytes": 67108864, "concurrent": 1}, "scope": "server-local files; explicit user declarations; no downloads"}
