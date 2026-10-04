"""Source snapshots: the recorded identity of what a run actually read (A41, A42).

A snapshot manifest is canonical JSON stored content-addressed in the artifact store; its sha256 is the snapshot id. Extracts
(materialized copies) are stored beside it. Every run that reads a connector source gets a `source_snapshot` artifact, so the run
can be repeated from the pinned identity. A hash verifies identity; it does not preserve deleted source data by itself."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .errors import SourceError

VERSION = 1


def canon(m: dict[str, Any]) -> bytes:
    return json.dumps(m, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record(ctx, manifest: dict[str, Any], extract: bytes | None = None) -> str:
    """Store manifest (+ extract) and attach them to the run. Returns the snapshot id."""
    store = ctx.store
    m = {**manifest, "snapshotVersion": VERSION, "producedByNode": ctx.node_id}
    if extract is not None:
        sha = store.put_bytes(extract)
        m["extract"] = {"sha256": sha, "bytes": len(extract)}
        if ctx.run_id:
            store.add_artifact(ctx.run_id, "source_extract", extract, "complete", None, {"node": ctx.node_id, "contentSha256": sha})
    body = canon(m)
    sid = store.put_bytes(body)
    if ctx.run_id:
        store.add_artifact(ctx.run_id, "source_snapshot", body, "complete", None, {"node": ctx.node_id, "snapshotId": sid, "kind": m.get("kind"),
                                                                                    "reproducibility": m.get("reproducibility", {}).get("level")})
    return sid


def reference(ctx, sid: str) -> None:
    """A pinned run does not create a new snapshot; it links the existing one to the run."""
    if ctx.run_id:
        ctx.store.add_artifact(ctx.run_id, "source_snapshot", ctx.store.read_artifact(sid), "complete", None,
                               {"node": ctx.node_id, "snapshotId": sid, "pinned": True})


def load(store, sid: str, want_extract: bool = True) -> tuple[dict[str, Any], bytes | None]:
    if not (isinstance(sid, str) and len(sid) == 64 and all(c in "0123456789abcdef" for c in sid)) or not store.path_of(sid).exists():
        raise SourceError("E_SRC_SNAPSHOT_UNAVAILABLE", f"Snapshot {sid[:12]}... is not in this workbench.", resource=sid)
    if not store.verify(sid):
        raise SourceError("E_SRC_SNAPSHOT_CORRUPT", f"Snapshot manifest {sid[:12]}... does not match its hash.", resource=sid)
    m = json.loads(store.read_artifact(sid))
    if not isinstance(m, dict) or m.get("snapshotVersion") != VERSION:
        raise SourceError("E_SRC_SNAPSHOT_UNAVAILABLE", f"{sid[:12]}... is not a source snapshot.", resource=sid)
    raw = None
    if want_extract and m.get("extract"):
        ex = m["extract"]["sha256"]
        if not store.path_of(ex).exists():
            raise SourceError("E_SRC_SNAPSHOT_UNAVAILABLE", f"The extract of snapshot {sid[:12]}... is missing.", resource=sid)
        if not store.verify(ex):
            raise SourceError("E_SRC_SNAPSHOT_CORRUPT", f"The extract of snapshot {sid[:12]}... does not match its hash.", resource=sid)
        raw = store.read_artifact(ex)
    return m, raw


def public(m: dict[str, Any], sid: str) -> dict[str, Any]:
    return {"id": sid, **m}


def pinned_for(cfg_pin: str | None, ctx) -> str | None:
    return ctx.pins.get(ctx.node_id) or cfg_pin
