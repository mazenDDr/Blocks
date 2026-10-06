"""Content-addressed artifact files + SQLite metadata for runs, events, artifacts.

Layout under `root` (default `.workbench`): `meta.db`, `artifacts/<sha256>`.
Each call opens its own short-lived connection, so the store is safe to use from the
control process and from a worker process at the same time (SQLite WAL)."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from storage.schema import enable_wal, open_database
import tempfile
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

RUN_STATES = ("queued", "preparing", "running", "paused", "cancelling", "completed", "failed", "cancelled")
_TRANSITIONS = {
    "queued": {"preparing", "failed", "cancelling", "cancelled"},
    "preparing": {"running", "failed", "cancelling", "cancelled"},
    "running": {"completed", "failed", "cancelling", "paused"},
    "paused": {"running", "cancelling", "cancelled", "failed"},  # agent runs waiting on an interrupt; a resume is a new worker process on the same run
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
CREATE TABLE IF NOT EXISTS node_cache (
  key TEXT PRIMARY KEY, sha256 TEXT NOT NULL, size INTEGER NOT NULL, run_id TEXT NOT NULL, node_id TEXT NOT NULL, op_type TEXT NOT NULL,
  project_id TEXT, node_part TEXT NOT NULL, implementation TEXT NOT NULL, environment TEXT NOT NULL, inputs TEXT NOT NULL, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS node_cache_node ON node_cache (node_id, op_type, created_at);
CREATE TABLE IF NOT EXISTS run_leases (
  run_id TEXT PRIMARY KEY, pid INTEGER NOT NULL, nonce TEXT NOT NULL, started REAL NOT NULL, heartbeat REAL NOT NULL);
"""
ACTIVE = ("queued", "preparing", "running", "cancelling")  # "paused" agent runs wait for a person with no worker process
LOST_AFTER_SECONDS = 120.0


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
                enable_wal(db)
                # Explicit migration and downgrade checks run in the connection factory.

    def _db(self) -> sqlite3.Connection:
        db = open_database(self.db_path, "meta", _SCHEMA, timeout=30)
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

    # ------------------------------------------------------------ node result cache index (tabular.cache; written only by the local worker)
    def get_node_cache(self, key: str) -> dict[str, Any] | None:
        rows = self._exec("SELECT * FROM node_cache WHERE key=?", (key,))
        return dict(rows[0]) if rows else None

    def latest_node_cache(self, node_id: str, op_type: str, project_id: str | None) -> dict[str, Any] | None:
        rows = self._exec("SELECT * FROM node_cache WHERE node_id=? AND op_type=? AND project_id IS ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                          (node_id, op_type, project_id))
        return dict(rows[0]) if rows else None

    def put_node_cache(self, key: str, sha256: str, size: int, run_id: str, node_id: str, op_type: str, project_id: str | None,
                       node_part: str, implementation: str, environment: str, inputs: str) -> None:
        self._exec("INSERT OR IGNORE INTO node_cache VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (key, sha256, size, run_id, node_id, op_type, project_id, node_part, implementation, environment, inputs, time.time()))

    def node_cache_entries(self, project_id: str | None = None, any_project: bool = True) -> list[dict[str, Any]]:
        sql, args = "SELECT key, sha256, size, run_id, node_id, op_type, project_id, created_at FROM node_cache", ()
        if not any_project:
            sql, args = sql + " WHERE project_id IS ?", (project_id,)
        return [dict(r) for r in self._exec(sql + " ORDER BY created_at, rowid", args)]

    def delete_node_cache(self, keys: list[str]) -> int:
        """Remove index rows, then each entry file that no remaining index row or artifact record refers to. Returns bytes freed."""
        freed = 0
        with self._lock, closing(self._db()) as db:
            with db:
                shas = {r["sha256"] for k in keys for r in db.execute("SELECT sha256 FROM node_cache WHERE key=?", (k,))}
                db.executemany("DELETE FROM node_cache WHERE key=?", [(k,) for k in keys])
                for sha in shas:
                    used = db.execute("SELECT 1 FROM node_cache WHERE sha256=? UNION ALL SELECT 1 FROM artifacts WHERE sha256=? LIMIT 1", (sha, sha)).fetchone()
                    p = self.path_of(sha)
                    if used is None and p.exists():
                        freed += p.stat().st_size
                        p.unlink()
        return freed

    def set_status(self, run_id: str, new: str, error: str | None = None) -> None:
        # Check and update in ONE write transaction: the control process (cancel) and the worker (cancelling -> cancelled) race on the
        # same row, and a separate read then write let a late "cancelling" overwrite a run the worker had already marked "cancelled".
        with self._lock, closing(self._db()) as db:
            db.isolation_level = None
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()
                if row is None:
                    raise KeyError(run_id)
                if new not in _TRANSITIONS.get(row["status"], set()):
                    raise IllegalTransition(f"{row['status']} -> {new}")
                db.execute("UPDATE runs SET status=?, error=?, updated_at=? WHERE id=?", (new, error, time.time(), run_id))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

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

    # ------------------------------------------------------------ worker liveness
    def lease(self, run_id: str, pid: int, nonce: str) -> None:
        """A worker process claims its run; a later worker (e.g. an agent resume) replaces the claim."""
        now = time.time()
        self._exec("INSERT OR REPLACE INTO run_leases VALUES (?,?,?,?,?)", (run_id, pid, nonce, now, now))

    def heartbeat(self, run_id: str, nonce: str) -> None:
        self._exec("UPDATE run_leases SET heartbeat=? WHERE run_id=? AND nonce=?", (time.time(), run_id, nonce))

    def reconcile_lost_workers(self, lost_after: float = LOST_AFTER_SECONDS, now: float | None = None) -> list[dict[str, Any]]:
        """Fail active runs whose worker left no sign of life (heartbeat, event or status change) for `lost_after` seconds.

        A crashed control process or killed/OOM worker otherwise leaves a run queued/running forever, blocking backup and
        collection. Recorded events and artifacts are kept; nothing is resumed or invented."""
        now = time.time() if now is None else now
        rows = self._exec(f"""SELECT r.id, r.status, r.graph_hash, r.updated_at, l.heartbeat, l.pid,
                                     (SELECT MAX(ts) FROM events e WHERE e.run_id=r.id) AS last_event
                              FROM runs r LEFT JOIN run_leases l ON l.run_id=r.id
                              WHERE r.status IN ({','.join('?' * len(ACTIVE))})""", ACTIVE)
        lost = []
        for r in rows:
            evidence = max(x for x in (r["updated_at"], r["heartbeat"], r["last_event"]) if x is not None)
            if now - evidence < lost_after:
                continue
            message = (f"E_WORKER_LOST: no worker heartbeat, event or status change for {now - evidence:.0f}s while {r['status']} "
                       f"(control or worker process ended); recorded events/artifacts are kept and the run was not resumed")
            try:
                self.set_status(r["id"], "failed", message)
            except IllegalTransition:
                continue  # the worker finished meanwhile
            self.append_event(r["id"], self.max_seq(r["id"]) + 1, now, "run_finished", r["graph_hash"], None,
                              {"status": "failed", "error": message, "recovered": True, "previousStatus": r["status"], "workerPid": r["pid"]})
            lost.append({"runId": r["id"], "previousStatus": r["status"], "silentSeconds": round(now - evidence, 1), "workerPid": r["pid"]})
        return lost

    def max_seq(self, run_id: str) -> int:
        rows = self._exec("SELECT MAX(seq) AS m FROM events WHERE run_id=?", (run_id,))
        return -1 if rows[0]["m"] is None else rows[0]["m"]

    def read_artifact(self, sha: str) -> bytes:
        return self.path_of(sha).read_bytes()
