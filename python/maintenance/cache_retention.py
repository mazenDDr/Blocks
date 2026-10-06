"""Opt-in per-project periodic cache retention, reusing native run-artifact protection.

The owning control process manages scheduling and shutdown.
Policies and receipts are local metadata, never a native model/cache identity input.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import hashlib
from pathlib import Path
import sqlite3
import threading
import time

from pydantic import BaseModel, ConfigDict, Field
from tabular.cache import prune


class RetentionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    enabled: bool = False
    keepLatestPerNode: int = Field(1, ge=1, le=1000)
    olderThanHours: float = Field(24, ge=0, le=87600)
    intervalSeconds: int = Field(3600, ge=5, le=604800)


class CacheRetention:
    def __init__(self, store, idle_lock=None):
        self.store = store
        self.path = Path(store.root) / "cache-retention.sqlite"
        self._stop = threading.Event()
        self._thread = None
        self._last_error = None
        self._lifecycle = threading.Lock()
        self._idle_lock = idle_lock or threading.Lock()

    @contextmanager
    def database(self):
        # Created only when explicitly configuring a policy (or reading existing metadata).
        with sqlite3.connect(self.path, timeout=30) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA busy_timeout=30000")
            db.execute("CREATE TABLE IF NOT EXISTS policies (project TEXT PRIMARY KEY, config TEXT NOT NULL, revision INTEGER NOT NULL, next_due REAL NOT NULL, updated REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS receipts (seq INTEGER PRIMARY KEY AUTOINCREMENT, project TEXT NOT NULL, revision INTEGER NOT NULL, started REAL NOT NULL, finished REAL NOT NULL, result TEXT, error TEXT)")
            yield db

    def inspect(self, project):
        if not self.path.exists():
            return {"projectId": project, "policy": None, "receipts": [], "running": bool(self._thread and self._thread.is_alive()), "lastError": self._last_error}
        with self.database() as db:
            row = db.execute("SELECT * FROM policies WHERE project=?", (project,)).fetchone()
            receipts = [dict(r) for r in db.execute("SELECT * FROM receipts WHERE project=? ORDER BY seq DESC LIMIT 100", (project,))]
        for receipt in receipts:
            receipt["result"] = json.loads(receipt["result"]) if receipt["result"] else None
        policy = None if row is None else {"config": json.loads(row["config"]), "revision": row["revision"], "nextDue": row["next_due"], "updatedAt": row["updated"]}
        return {"projectId": project, "policy": policy, "receipts": receipts, "running": bool(self._thread and self._thread.is_alive()), "lastError": self._last_error}

    def configure(self, project, policy, expected_revision):
        if not isinstance(project, str) or not 1 <= len(project) <= 128:
            raise ValueError("E_CACHE_POLICY: Declare a bounded project id.")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("E_CACHE_POLICY: Declare a nonnegative integer revision.")
        policy = RetentionPolicy.model_validate(policy)
        with self.database() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT revision FROM policies WHERE project=?", (project,)).fetchone()
            revision = row[0] if row else 0
            if expected_revision != revision:
                raise ValueError("E_CACHE_POLICY_STALE: Policy revision changed; reload before configuring.")
            if row is None and db.execute("SELECT COUNT(*) FROM policies").fetchone()[0] >= 100:
                raise ValueError("E_CACHE_POLICY_BOUNDS: At most100 project policies.")
            now = time.time()
            db.execute("INSERT OR REPLACE INTO policies VALUES (?,?,?,?,?)", (project, policy.model_dump_json(), revision + 1, now + policy.intervalSeconds, now))
        return self.inspect(project)

    def preview(self, project, policy):
        policy = RetentionPolicy.model_validate(policy)
        return prune(self.store, project_id=project, keep_latest_per_node=policy.keepLatestPerNode,
                     older_than_seconds=policy.olderThanHours * 3600, dry_run=True)

    def tick(self):
        if not self.path.exists():
            return []
        receipts = []
        # Serialize scheduling/configuration across owning-service threads. Metadata transaction
        # cannot make CAS deletion atomic with receipt publication; a crash can lose that receipt.
        with self._idle_lock, self.database() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in db.execute("SELECT * FROM policies WHERE next_due<=? ORDER BY project", (time.time(),)).fetchall():
                policy = RetentionPolicy.model_validate_json(row["config"])
                if not policy.enabled:
                    continue
                started = time.time()
                result, error = None, None
                try:
                    busy = [r["id"] for r in self.store.list_runs() if r["status"] in ("queued", "preparing", "running", "cancelling", "paused")]
                    if busy:
                        result = {"deferred": True, "reason": "Native research runs are active or paused; no prune attempted.", "runIds": busy[:100], "runsTruncated": len(busy) > 100}
                    else:
                        result = self._prune(row["project"], policy)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                finished = time.time()
                db.execute("INSERT INTO receipts(project,revision,started,finished,result,error) VALUES (?,?,?,?,?,?)",
                           (row["project"], row["revision"], started, finished, json.dumps(result, allow_nan=False) if result is not None else None, error))
                db.execute("UPDATE policies SET next_due=? WHERE project=?", (finished + policy.intervalSeconds, row["project"]))
                db.execute("DELETE FROM receipts WHERE project=? AND seq NOT IN (SELECT seq FROM receipts WHERE project=? ORDER BY seq DESC LIMIT 100)", (row["project"], row["project"]))
                receipts.append({"projectId": row["project"], "revision": row["revision"], "result": result, "error": error})
        return receipts

    def _prune(self, project, policy):
        native = prune(self.store, project_id=project, keep_latest_per_node=policy.keepLatestPerNode,
                       older_than_seconds=policy.olderThanHours * 3600)
        entries = native["entries"]
        return {**native, "entries": entries[:1000], "entriesTruncated": len(entries) > 1000,
                "entriesSha256": hashlib.sha256(json.dumps(entries, sort_keys=True, allow_nan=False).encode()).hexdigest()}

    def start(self):
        with self._lifecycle:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            def loop():
                while not self._stop.wait(1):
                    try:
                        self.tick()
                        self._last_error = None
                    except Exception as exc:
                        # Database/config failures must not terminate model serving. No successful
                        # receipt is invented; a subsequent tick retries unchanged due metadata.
                        self._last_error = f"{type(exc).__name__}: {exc}"
            self._thread = threading.Thread(target=loop, name="void-cache-retention", daemon=True)
            self._thread.start()

    def stop(self):
        with self._lifecycle:
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=30)
            if self._thread and self._thread.is_alive():
                raise RuntimeError("Cache-retention shutdown did not complete; owning service must remain stopped before offline backup.")
