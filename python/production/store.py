"""Durable production records, atomic routing/state and append-only lifecycle history.

One local control process owns admission/session locks. This adapter does not claim
distributed replicas. SQLite transactions protect persisted route/request updates.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import closing

from tabular.core import dumps
from .pipeline import ProductionError, read_verified

SCHEMA = """
CREATE TABLE IF NOT EXISTS records(kind TEXT, id TEXT, sha TEXT, created REAL, PRIMARY KEY(kind,id));
CREATE TABLE IF NOT EXISTS routes(target TEXT, namespace TEXT, release TEXT, PRIMARY KEY(target,namespace));
CREATE TABLE IF NOT EXISTS aliases(name TEXT PRIMARY KEY, version TEXT);
CREATE TABLE IF NOT EXISTS lifecycle(seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, type TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS requests(user TEXT, id TEXT, fingerprint TEXT, release TEXT, state TEXT, cancelled INTEGER, trace TEXT, created REAL, PRIMARY KEY(user,id));
CREATE TABLE IF NOT EXISTS sessions(scope TEXT PRIMARY KEY, count INTEGER, last TEXT);
CREATE TABLE IF NOT EXISTS labels(user TEXT, request TEXT, labels TEXT, ts REAL, PRIMARY KEY(user,request));
"""


class ProductionStore:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.path = artifacts.root / "production.sqlite"
        self.lock = threading.RLock()
        with self.lock, closing(self.db()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    def db(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def query(self, sql, args=()):
        with self.lock, closing(self.db()) as db, db:
            return [dict(r) for r in db.execute(sql, args).fetchall()]

    def event(self, db, typ, data):
        db.execute("INSERT INTO lifecycle(ts,type,data) VALUES(?,?,?)", (time.time(), typ, dumps(data)))

    def save(self, kind, data):
        sha = self.artifacts.put_bytes(dumps(data).encode())
        with self.lock, closing(self.db()) as db, db:
            cur = db.execute("INSERT OR IGNORE INTO records VALUES(?,?,?,?)", (kind, sha, sha, time.time()))
            if cur.rowcount:
                self.event(db, f"{kind}_created", {"id": sha})
        return {"id": sha, **data}

    def get(self, kind, id_):
        rows = self.query("SELECT * FROM records WHERE kind=? AND id=?", (kind, id_))
        if not rows:
            raise ProductionError("E_PRODUCTION_NOT_FOUND", f"Unknown {kind} {id_}.", 404)
        return {"id": id_, **json.loads(read_verified(self.artifacts, rows[0]["sha"]))}

    def list(self, kind):
        return [self.get(kind, r["id"]) for r in self.query("SELECT id FROM records WHERE kind=? ORDER BY created DESC LIMIT 100", (kind,))]

    def alias(self, name, version):
        self.get("version", version)
        with self.lock, closing(self.db()) as db, db:
            db.execute("INSERT INTO aliases VALUES(?,?) ON CONFLICT(name) DO UPDATE SET version=excluded.version", (name, version))
            self.event(db, "alias_updated", {"name": name, "versionId": version})

    def resolve_version(self, value):
        aliases = self.query("SELECT version FROM aliases WHERE name=?", (value,))
        return self.get("version", aliases[0]["version"] if aliases else value)

    def route(self, target, namespace):
        rows = self.query("SELECT release FROM routes WHERE target=? AND namespace=?", (target, namespace))
        if not rows:
            raise ProductionError("E_NO_RELEASE", "No release has been deployed to this local route.", 404)
        return self.get("release", rows[0]["release"])

    def activate(self, id_, expected, rollback=False):
        release = self.get("release", id_)
        config = release["config"]
        target, ns = config["target"], config["namespace"]
        with self.lock, closing(self.db()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT release FROM routes WHERE target=? AND namespace=?", (target, ns)).fetchone()
            current = row[0] if row else None
            if current != expected:
                raise ProductionError("E_ROUTE_CONFLICT", "Route changed since preview; refresh before deploying.", 409)
            if rollback:
                history = db.execute("SELECT data FROM lifecycle WHERE type IN ('deployed','rolled_back')").fetchall()
                if not any(json.loads(r[0]).get("releaseId") == id_ for r in history):
                    raise ProductionError("E_ROLLBACK_UNKNOWN", "Rollback requires a previously deployed release.", 409)
            db.execute("INSERT INTO routes VALUES(?,?,?) ON CONFLICT(target,namespace) DO UPDATE SET release=excluded.release", (target, ns, id_))
            self.event(db, "rolled_back" if rollback else "deployed", {"releaseId": id_, "previousRelease": current, "target": target, "namespace": ns,
                                                                    "versionId": release["versionId"], "mode": "live local CPU endpoint; no remote infrastructure"})
        return {"releaseId": id_, "previousRelease": current, "target": target, "namespace": ns}

    def begin_request(self, user, id_, fingerprint, release):
        with self.lock, closing(self.db()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM requests WHERE user=? AND id=?", (user, id_)).fetchone()
            if row:
                if row["fingerprint"] != fingerprint:
                    raise ProductionError("E_REQUEST_CONFLICT", "Request id already belongs to different input or release.", 409)
                if row["trace"]:
                    return json.loads(read_verified(self.artifacts, row["trace"]))
                raise ProductionError("E_REQUEST_IN_PROGRESS", "Request is already queued/running; no duplicate state update.", 409)
            db.execute("INSERT INTO requests VALUES(?,?,?,?,?,?,?,?)", (user, id_, fingerprint, release, "queued", 0, None, time.time()))
        return None

    def cancel(self, user, id_):
        with self.lock, closing(self.db()) as db, db:
            row = db.execute("SELECT state FROM requests WHERE user=? AND id=?", (user, id_)).fetchone()
            if not row:
                raise ProductionError("E_REQUEST_NOT_FOUND", "Request does not exist for this user.", 404)
            if row[0] in ("completed", "failed", "cancelled"):
                raise ProductionError("E_REQUEST_TERMINAL", "Request already finished; committed state cannot be cancelled.", 409)
            db.execute("UPDATE requests SET cancelled=1 WHERE user=? AND id=?", (user, id_))
            self.event(db, "request_cancel_requested", {"user": user, "requestId": id_})
        return {"cancelRequested": True, "policy": "no state commit after cancellation; an active native forward pass may finish"}

    def cancelled(self, user, id_):
        rows = self.query("SELECT cancelled FROM requests WHERE user=? AND id=?", (user, id_))
        return bool(rows and rows[0]["cancelled"])

    def finish_request(self, user, id_, trace, scope=None):
        with self.lock, closing(self.db()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT cancelled FROM requests WHERE user=? AND id=?", (user, id_)).fetchone()
            if row[0] and trace["status"] == 200:
                trace.update(status=409, error={"code": "E_REQUEST_CANCELLED", "message": "Cancelled before state commit."}, result=None)
            if scope and trace["status"] == 200:
                db.execute("INSERT INTO sessions VALUES(?,1,?) ON CONFLICT(scope) DO UPDATE SET count=count+1,last=excluded.last", (scope, id_))
                state = db.execute("SELECT count,last FROM sessions WHERE scope=?", (scope,)).fetchone()
                trace["sessionState"] = dict(state)
            else:
                trace["sessionState"] = None
            sha = self.artifacts.put_bytes(dumps(trace).encode())
            state = "completed" if trace["status"] == 200 else "cancelled" if trace.get("error", {}).get("code") == "E_REQUEST_CANCELLED" else "failed"
            db.execute("UPDATE requests SET state=?,trace=? WHERE user=? AND id=?", (state, sha, user, id_))
        return {**trace, "traceSha256": sha}

    def traces(self, release=None):
        rows = self.query("SELECT trace FROM requests WHERE trace IS NOT NULL" + (" AND release=?" if release else "") + " ORDER BY created DESC LIMIT 1000",
                          (release,) if release else ())
        return [{**json.loads(read_verified(self.artifacts, r["trace"])), "traceSha256": r["trace"]} for r in rows]

    def trace(self, user, id_):
        rows = self.query("SELECT trace,state FROM requests WHERE user=? AND id=?", (user, id_))
        if not rows:
            raise ProductionError("E_REQUEST_NOT_FOUND", "Unknown request for this user.", 404)
        if not rows[0]["trace"]:
            return {"requestId": id_, "state": rows[0]["state"], "available": False}
        return {**json.loads(read_verified(self.artifacts, rows[0]["trace"])), "traceSha256": rows[0]["trace"]}

    def add_labels(self, user, id_, labels):
        t = self.trace(user, id_)
        if t.get("status") != 200 or len(labels) != len(t["result"]["predictions"]):
            raise ProductionError("E_LABEL_ALIGNMENT", "Labels must align with a completed prediction batch.")
        data = dumps(labels)
        with self.lock, closing(self.db()) as db, db:
            old = db.execute("SELECT labels FROM labels WHERE user=? AND request=?", (user, id_)).fetchone()
            if old and old[0] != data:
                raise ProductionError("E_LABEL_CONFLICT", "Recorded ground truth is immutable; conflicting overwrite refused.", 409)
            db.execute("INSERT OR IGNORE INTO labels VALUES(?,?,?,?)", (user, id_, data, time.time()))

    def recover_incomplete(self):
        """Single-owner restart: inference is not resumed and session updates are not replayed."""
        with self.lock, closing(self.db()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT * FROM requests WHERE trace IS NULL").fetchall()
            for row in rows:
                trace = {"requestId": row["id"], "user": row["user"], "releaseId": row["release"], "receivedAt": row["created"],
                         "status": 503, "error": {"code": "E_SERVING_RESTART", "message": "Single-owner server restarted before request commit; inference was not resumed."},
                         "result": None, "sessionState": None, "records": None, "batchSize": None, "totalMs": None,
                         "capturePolicy": "incomplete request; inputs not retained"}
                sha = self.artifacts.put_bytes(dumps(trace).encode())
                db.execute("UPDATE requests SET state='failed',trace=? WHERE user=? AND id=?", (sha,row["user"],row["id"]))
                self.event(db,"request_recovered_as_failed",{"user":row["user"],"requestId":row["id"],"releaseId":row["release"],"traceSha256":sha})
