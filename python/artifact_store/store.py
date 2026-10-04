"""Content-addressed artifact files + SQLite metadata for runs, events, artifacts.

Layout under `root` (default `.workbench`): `meta.db`, `artifacts/<sha256>`.
Each call opens its own short-lived connection, so the store is safe to use from the
control process and from a worker process at the same time (SQLite WAL)."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

RUN_STATES = ("queued", "preparing", "running", "cancelling", "completed", "failed", "cancelled")
_TRANSITIONS = {
    "queued": {"preparing", "failed", "cancelling", "cancelled"},
    "preparing": {"running", "failed", "cancelling", "cancelled"},
    "running": {"completed", "failed", "cancelling"},
    "cancelling": {"cancelled", "failed"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, graph_hash TEXT NOT NULL, status TEXT NOT NULL,
  config TEXT NOT NULL, error TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS events (
  run_id TEXT NOT NULL, seq INTEGER NOT NULL, ts REAL NOT NULL, type TEXT NOT NULL,
  node_id TEXT, graph_hash TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (run_id, seq));
CREATE TABLE IF NOT EXISTS artifacts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, sha256 TEXT NOT NULL, run_id TEXT NOT NULL, kind TEXT NOT NULL,
  status TEXT NOT NULL, step INTEGER, size INTEGER NOT NULL, meta TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS idempotency (
  key TEXT PRIMARY KEY, request_hash TEXT NOT NULL, run_id TEXT NOT NULL, created_at REAL NOT NULL);
"""


class IllegalTransition(Exception):
    pass


class ArtifactStore:
    def __init__(self, root: str | Path = ".workbench"):
        self.root = Path(root)
        self.artifact_dir = self.root / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "meta.db"
        # Opening/closing many connections to one file from several threads of the same process (the control
        # service) deadlocked inside libsqlite3 (observed on macOS, SQLite 3.51.1), so in-process access is serialized.
        # Other processes (the worker) are coordinated by SQLite's own file locking.
        self._lock = threading.RLock()
        with self._lock, closing(self._db()) as db:
            with db:
                db.execute("PRAGMA journal_mode=WAL")
                db.executescript(_SCHEMA)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def _exec(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock, closing(self._db()) as db:
            with db:
                return db.execute(sql, args).fetchall()

    # ------------------------------------------------------------ files
    def put_bytes(self, data: bytes) -> str:
        """Store bytes under their sha256; atomic (temp file + rename). Returns the hash."""
        sha = hashlib.sha256(data).hexdigest()
        dest = self.artifact_dir / sha
        if not dest.exists():
            fd, tmp = tempfile.mkstemp(dir=self.artifact_dir, suffix=".tmp")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
        return sha

    def path_of(self, sha: str) -> Path:
        return self.artifact_dir / sha

    def verify(self, sha: str) -> bool:
        p = self.path_of(sha)
        return p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == sha

    # ------------------------------------------------------------ runs
    def create_run(self, run_id: str, graph_hash: str, config: dict[str, Any]) -> None:
        now = time.time()
        self._exec("INSERT INTO runs VALUES (?,?,?,?,?,?,?)", (run_id, graph_hash, "queued", json.dumps(config), None, now, now))

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self._exec("SELECT * FROM runs WHERE id=?", (run_id,))
        if not rows:
            return None
        d = dict(rows[0])
        d["config"] = json.loads(d["config"])
        return d

    def list_runs(self) -> list[dict[str, Any]]:
        out = []
        for r in self._exec("SELECT id, graph_hash, status, config, error, created_at, updated_at FROM runs ORDER BY created_at, rowid"):
            d = dict(r)
            d["config"] = json.loads(d["config"])
            out.append(d)
        return out

    # ------------------------------------------------------------ idempotency keys
    def get_idempotent(self, key: str) -> dict[str, Any] | None:
        rows = self._exec("SELECT key, request_hash, run_id FROM idempotency WHERE key=?", (key,))
        return dict(rows[0]) if rows else None

    def put_idempotent(self, key: str, request_hash: str, run_id: str) -> None:
        self._exec("INSERT INTO idempotency VALUES (?,?,?,?)", (key, request_hash, run_id, time.time()))

    def set_status(self, run_id: str, new: str, error: str | None = None) -> None:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        if new not in _TRANSITIONS.get(run["status"], set()):
            raise IllegalTransition(f"{run['status']} -> {new}")
        self._exec("UPDATE runs SET status=?, error=?, updated_at=? WHERE id=?", (new, error, time.time(), run_id))

    # ------------------------------------------------------------ events
    def append_event(self, run_id: str, seq: int, ts: float, type_: str, graph_hash: str, node_id: str | None, data: dict[str, Any]) -> None:
        self._exec("INSERT INTO events VALUES (?,?,?,?,?,?,?)", (run_id, seq, ts, type_, node_id, graph_hash, json.dumps(data)))

    def events(self, run_id: str, after_seq: int = -1, types: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM events WHERE run_id=? AND seq>?", [run_id, after_seq]
        if types:
            sql += " AND type IN (" + ",".join("?" * len(types)) + ")"
            args += list(types)
        rows = self._exec(sql + " ORDER BY seq", tuple(args))
        out = []
        for r in rows:
            d = dict(r)
            d["data"] = json.loads(d["data"])
            out.append(d)
        return out

    # ------------------------------------------------------------ artifact records
    def add_artifact(self, run_id: str, kind: str, data: bytes, status: str, step: int | None, meta: dict[str, Any]) -> dict[str, Any]:
        sha = self.put_bytes(data)
        self._exec("INSERT INTO artifacts (sha256, run_id, kind, status, step, size, meta, created_at) VALUES (?,?,?,?,?,?,?,?)",
                   (sha, run_id, kind, status, step, len(data), json.dumps(meta), time.time()))
        return {"sha256": sha, "kind": kind, "status": status, "step": step, "size": len(data)}

    def artifacts(self, run_id: str, kind: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM artifacts WHERE run_id=?", [run_id]
        if kind:
            sql, args = sql + " AND kind=?", args + [kind]
        out = []
        for r in self._exec(sql + " ORDER BY id", tuple(args)):
            d = dict(r)
            d["meta"] = json.loads(d["meta"])
            out.append(d)
        return out

    def set_artifact_status(self, run_id: str, sha256: str, status: str, kind: str | None = None) -> None:
        """Retention marks (e.g. a checkpoint pruned by keep_last). The bytes stay: content-addressed files may be shared by other records."""
        sql, args = "UPDATE artifacts SET status=? WHERE run_id=? AND sha256=?", [status, run_id, sha256]
        if kind:
            sql, args = sql + " AND kind=?", args + [kind]
        self._exec(sql, tuple(args))

    def last_event(self, run_id: str, type_: str) -> dict[str, Any] | None:
        rows = self._exec("SELECT * FROM events WHERE run_id=? AND type=? ORDER BY seq DESC LIMIT 1", (run_id, type_))
        if not rows:
            return None
        d = dict(rows[0])
        d["data"] = json.loads(d["data"])
        return d

    def max_seq(self, run_id: str) -> int:
        rows = self._exec("SELECT MAX(seq) AS m FROM events WHERE run_id=?", (run_id,))
        return -1 if rows[0]["m"] is None else rows[0]["m"]

    def read_artifact(self, sha: str) -> bytes:
        return self.path_of(sha).read_bytes()
