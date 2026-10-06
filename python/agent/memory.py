"""Explicit memory stores (VISION 12.4-12.5).

* short-term  : the ordered messages of ONE thread. They live in a state field of the LangGraph checkpoint (native thread state), so
                they are versioned, replayable and scoped by thread id; this module only reads them.
* long-term   : records with metadata in `<workbench>/agent/memory.db` (SQLite), scoped by namespace + scope string. Shared across threads.

Also here: the audit of every write (producing node, evidence, old/new values, validation, scope, run/event), the recorded results of
policy applications (every record with its decision at every stage), and the effect ledger that keeps protected effects from running
twice when a node is replayed after an interrupt or a restart."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from storage.schema import open_database
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

KINDS = ("semantic", "episodic", "procedural", "summary", "message")
SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
  id TEXT PRIMARY KEY, namespace TEXT NOT NULL, scope TEXT NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL, metadata TEXT NOT NULL,
  importance REAL NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL, expires_at REAL, generated INTEGER NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 1, source TEXT NOT NULL, deleted_at REAL);
CREATE TABLE IF NOT EXISTS memory_writes (
  id TEXT PRIMARY KEY, ts REAL NOT NULL, run_id TEXT, thread_id TEXT, node TEXT, op TEXT NOT NULL, record_id TEXT, old TEXT, new TEXT,
  validation TEXT NOT NULL, scope TEXT, evidence TEXT, generated INTEGER, event_seq INTEGER);
CREATE TABLE IF NOT EXISTS applications (
  id TEXT PRIMARY KEY, ts REAL NOT NULL, run_id TEXT, thread_id TEXT, node TEXT, policy_id TEXT, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS retrievals (
  id TEXT PRIMARY KEY, ts REAL NOT NULL, run_id TEXT, thread_id TEXT, node TEXT, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS effects (
  key TEXT PRIMARY KEY, ts REAL NOT NULL, run_id TEXT, thread_id TEXT, node TEXT, tool TEXT, args_hash TEXT, status TEXT NOT NULL, result TEXT);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class MemoryStore:
    def __init__(self, workbench: str | Path):
        self.root = Path(workbench) / "agent"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "memory.db"
        with closing(self._db()) as db:
            # Explicit migration and downgrade checks run in the connection factory.
            db.commit()

    def _db(self) -> sqlite3.Connection:
        db = open_database(self.path, "memory", SCHEMA, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def _q(self, sql: str, args: tuple = (), write: bool = False) -> list[sqlite3.Row]:
        with closing(self._db()) as db:
            rows = db.execute(sql, args).fetchall()
            if write:
                db.commit()
            return rows

    # ------------------------------------------------------------------ long-term records
    @staticmethod
    def _rec(r: sqlite3.Row) -> dict[str, Any]:
        return {"id": r["id"], "store": "long_term", "namespace": r["namespace"], "scope": r["scope"], "kind": r["kind"], "text": r["text"],
                "metadata": json.loads(r["metadata"]), "importance": r["importance"], "created_at": r["created_at"], "updated_at": r["updated_at"],
                "expires_at": r["expires_at"], "generated": bool(r["generated"]), "version": r["version"], "source": json.loads(r["source"]),
                "deleted_at": r["deleted_at"]}

    def list_records(self, namespace: str | None = None, scope: str | None = None, kind: str | None = None, include_deleted: bool = False) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM records WHERE 1=1", []
        if namespace:
            sql, args = sql + " AND namespace=?", args + [namespace]
        if scope:
            sql, args = sql + " AND scope=?", args + [scope]
        if kind:
            sql, args = sql + " AND kind=?", args + [kind]
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        return [self._rec(r) for r in self._q(sql + " ORDER BY created_at, id", tuple(args))]

    def get_record(self, rid: str) -> dict[str, Any] | None:
        rows = self._q("SELECT * FROM records WHERE id=?", (rid,))
        return self._rec(rows[0]) if rows else None

    def put_record(self, rec: dict[str, Any], *, run_id: str | None = None, thread_id: str | None = None, node: str | None = None,
                   evidence: Any = None, validation: dict | None = None, event_seq: int | None = None) -> dict[str, Any]:
        """Insert or replace a record and audit the write (old/new values)."""
        now = time.time()
        old = self.get_record(rec["id"]) if rec.get("id") else None
        rid = rec.get("id") or new_id("mem")
        new = {"id": rid, "namespace": rec.get("namespace", "default"), "scope": rec.get("scope", "global"), "kind": rec.get("kind", "semantic"),
               "text": rec["text"], "metadata": rec.get("metadata") or {}, "importance": float(rec.get("importance", 0.5)),
               "created_at": rec.get("created_at") or (old["created_at"] if old else now), "updated_at": now, "expires_at": rec.get("expires_at"),
               "generated": bool(rec.get("generated", False)), "version": (old["version"] + 1) if old else 1, "source": rec.get("source") or {}}
        self._q("INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
                (rid, new["namespace"], new["scope"], new["kind"], new["text"], json.dumps(new["metadata"]), new["importance"], new["created_at"], now,
                 new["expires_at"], int(new["generated"]), new["version"], json.dumps(new["source"])), write=True)
        self._audit("update" if old else "insert", rid, old, new, run_id, thread_id, node, evidence, validation or {"ok": True}, new["scope"], new["generated"], event_seq)
        return {**new, "store": "long_term", "deleted_at": None}

    def delete_record(self, rid: str, *, run_id=None, thread_id=None, node=None, evidence=None) -> bool:
        old = self.get_record(rid)
        if not old or old["deleted_at"]:
            return False
        self._q("UPDATE records SET deleted_at=? WHERE id=?", (time.time(), rid), write=True)
        self._audit("delete", rid, old, None, run_id, thread_id, node, evidence, {"ok": True}, old["scope"], old["generated"], None)
        return True

    def _audit(self, op, rid, old, new, run_id, thread_id, node, evidence, validation, scope, generated, event_seq) -> None:
        self._q("INSERT INTO memory_writes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (new_id("w"), time.time(), run_id, thread_id, node, op, rid, json.dumps(old) if old else None, json.dumps(new) if new else None,
                 json.dumps(validation), scope, json.dumps(evidence), int(bool(generated)), event_seq), write=True)

    def writes(self, record_id: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM memory_writes WHERE 1=1", []
        if record_id:
            sql, args = sql + " AND record_id=?", args + [record_id]
        if run_id:
            sql, args = sql + " AND run_id=?", args + [run_id]
        out = []
        for r in self._q(sql + " ORDER BY ts", tuple(args)):
            d = dict(r)
            for k in ("old", "new", "validation", "evidence"):
                d[k] = json.loads(d[k]) if d[k] else None
            out.append(d)
        return out

    # ------------------------------------------------------------------ recorded policy applications / retrievals
    def save_application(self, app: dict[str, Any]) -> None:
        self._q("INSERT OR REPLACE INTO applications VALUES (?,?,?,?,?,?,?)", (app["id"], time.time(), app.get("runId"), app.get("threadId"), app.get("node"),
                                                                                app.get("policyId"), json.dumps(app)), write=True)

    def get_application(self, aid: str) -> dict[str, Any] | None:
        rows = self._q("SELECT payload FROM applications WHERE id=?", (aid,))
        return json.loads(rows[0]["payload"]) if rows else None

    def applications(self, run_id: str | None = None, thread_id: str | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT payload FROM applications WHERE 1=1", []
        if run_id:
            sql, args = sql + " AND run_id=?", args + [run_id]
        if thread_id:
            sql, args = sql + " AND thread_id=?", args + [thread_id]
        return [json.loads(r["payload"]) for r in self._q(sql + " ORDER BY ts", tuple(args))]

    def save_retrieval(self, rid: str, payload: dict[str, Any]) -> None:
        self._q("INSERT OR REPLACE INTO retrievals VALUES (?,?,?,?,?,?)", (rid, time.time(), payload.get("runId"), payload.get("threadId"), payload.get("node"), json.dumps(payload)), write=True)

    def get_retrieval(self, rid: str) -> dict[str, Any] | None:
        rows = self._q("SELECT payload FROM retrievals WHERE id=?", (rid,))
        return json.loads(rows[0]["payload"]) if rows else None

    # ------------------------------------------------------------------ effect ledger (protected effects run once)
    def claim_effect(self, key: str, *, run_id: str, thread_id: str, node: str, tool: str, args_hash: str) -> tuple[str, Any]:
        """Atomically claim an effect. Returns ("claimed", None) when the caller must perform it, ("done", result) when it already ran
        (replay after interrupt/restart: do NOT run it again), or ("uncertain", None) when a previous attempt started but never recorded
        completion (crash mid-effect: not repeated automatically)."""
        with closing(self._db()) as db:
            try:
                db.execute("INSERT INTO effects VALUES (?,?,?,?,?,?,?,?,NULL)", (key, time.time(), run_id, thread_id, node, tool, args_hash, "started"))
                db.commit()
                return "claimed", None
            except sqlite3.IntegrityError:
                row = db.execute("SELECT status, result FROM effects WHERE key=?", (key,)).fetchone()
                if row["status"] == "done":
                    return "done", json.loads(row["result"]) if row["result"] else None
                return "uncertain", None

    def complete_effect(self, key: str, result: Any) -> None:
        self._q("UPDATE effects SET status='done', result=? WHERE key=?", (json.dumps(result), key), write=True)

    def effects(self, thread_id: str | None = None) -> list[dict[str, Any]]:
        rows = self._q("SELECT * FROM effects" + (" WHERE thread_id=?" if thread_id else "") + " ORDER BY ts", (thread_id,) if thread_id else ())
        return [dict(r) for r in rows]


def args_hash(args: Any) -> str:
    return hashlib.sha256(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()[:16]
