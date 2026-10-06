"""Durable production records, atomic routing/state and append-only lifecycle history.

One local control process owns admission/session locks. This adapter does not claim
distributed replicas. SQLite transactions protect persisted route/request updates.
"""
from __future__ import annotations

import json
import sqlite3
from storage.schema import enable_wal, open_database
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
CREATE TABLE IF NOT EXISTS agent_sessions(scope TEXT PRIMARY KEY, revision INTEGER, checkpoint TEXT, last TEXT);
CREATE TABLE IF NOT EXISTS conversation_actions(user TEXT, id TEXT, fingerprint TEXT, result TEXT, PRIMARY KEY(user,id));
CREATE TABLE IF NOT EXISTS labels(user TEXT, request TEXT, labels TEXT, ts REAL, PRIMARY KEY(user,request));
"""
# Version 2 (ADR0075): long-term memory owned by a release and a request user, committed only with a successful request trace.
RELEASE_MEMORY = """
CREATE TABLE release_memory(release TEXT NOT NULL, user TEXT NOT NULL, id TEXT NOT NULL, namespace TEXT NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL,
  metadata TEXT NOT NULL, importance REAL NOT NULL, generated INTEGER NOT NULL, evidence TEXT, request TEXT NOT NULL, created REAL NOT NULL,
  PRIMARY KEY(release,user,id));
"""
STEPS = (SCHEMA, RELEASE_MEMORY)


class ProductionStore:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.path = artifacts.root / "production.sqlite"
        self.lock = threading.RLock()
        with self.lock, closing(self.db()) as db, db:
            enable_wal(db)
            # Explicit migration and downgrade checks run in the connection factory.

    def db(self):
        db = open_database(self.path, "production", STEPS, timeout=30)
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

    def conversation(self, scope):
        rows = self.query("SELECT revision,checkpoint,last FROM agent_sessions WHERE scope=?", (scope,))
        if not rows:
            return None
        if rows[0]["checkpoint"] is not None:
            read_verified(self.artifacts, rows[0]["checkpoint"])
        return {"revision": rows[0]["revision"], "checkpointSha256": rows[0]["checkpoint"], "lastRequestId": rows[0]["last"]}

    def conversation_action(self, user, id_, fingerprint):
        rows = self.query("SELECT fingerprint,result FROM conversation_actions WHERE user=? AND id=?", (user, id_))
        if not rows:
            return None
        if rows[0]["fingerprint"] != fingerprint:
            raise ProductionError("E_SESSION_ACTION_CONFLICT", "Action id already belongs to another operation/input/release.", 409)
        return {**json.loads(read_verified(self.artifacts, rows[0]["result"])), "actionSha256": rows[0]["result"], "idempotentReplay": True}

    def commit_conversation_action(self, data, fingerprint, source, target, expected, checkpoint, thread, deadline):
        with self.lock, closing(self.db()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            duplicate = db.execute("SELECT fingerprint,result FROM conversation_actions WHERE user=? AND id=?", (data["user"], data["actionId"])).fetchone()
            if duplicate:
                if duplicate["fingerprint"] != fingerprint:
                    raise ProductionError("E_SESSION_ACTION_CONFLICT", "Action id conflicts with a recorded operation.", 409)
                return {**json.loads(read_verified(self.artifacts, duplicate["result"])), "actionSha256": duplicate["result"], "idempotentReplay": True}
            current = db.execute("SELECT revision,checkpoint FROM agent_sessions WHERE scope=?", (source,)).fetchone()
            if not current or (current["revision"], current["checkpoint"]) != (expected["revision"], expected["checkpointSha256"]):
                raise ProductionError("E_SESSION_CONFLICT", "Reviewed checkpoint changed before action commit.", 409)
            if data["operation"] == "fork" and db.execute("SELECT 1 FROM agent_sessions WHERE scope=?", (target,)).fetchone():
                raise ProductionError("E_SESSION_DESTINATION", "Destination already exists; no overwrite.", 409)
            if data["operation"] == "restore":
                historical = db.execute("SELECT trace,state,release FROM requests WHERE user=? AND id=?", (data["user"], data["sourceRequestId"])).fetchone()
                if not historical or tuple(historical) != (data["sourceTraceSha256"], "completed", data["releaseId"]):
                    raise ProductionError("E_SESSION_HISTORY_CONFLICT", "Historical request changed before restore commit.", 409)
                read_verified(self.artifacts, data["sourceTraceSha256"])
                read_verified(self.artifacts, data["sourceCheckpointSha256"])
            revision = 1 if data["operation"] == "fork" else current["revision"]+1
            head = {"revision": revision, "checkpointSha256": checkpoint, "lastRequestId": None}
            result = {**data, "versionId": self.get("release", data["releaseId"])["versionId"],
                      "sourceHead": expected, "head": head, "threadId": thread, "recordedAt": time.time(),
                      "policy": ("restore clones the reviewed historical request checkpoint into a fresh native thread in this session; monotonic revision; all earlier evidence retained; caller-declared user is not authenticated"
                                 if data["operation"] == "restore" else "reset changes only the live head; fork clones the reviewed native END checkpoint into a new thread; old traces/snapshots retained; caller-declared user is not authenticated")}
            sha = self.artifacts.put_bytes(dumps(result).encode())
            if time.perf_counter() >= deadline:
                raise ProductionError("E_REQUEST_TIMEOUT", "Action deadline passed before commit; no head changed.", 504)
            if data["operation"] == "fork":
                db.execute("INSERT INTO agent_sessions VALUES(?,?,?,NULL)", (target, revision, checkpoint))
            elif data["operation"] == "restore":
                db.execute("UPDATE agent_sessions SET revision=?,checkpoint=?,last=NULL WHERE scope=?", (revision, checkpoint, source))
            else:
                db.execute("UPDATE agent_sessions SET revision=?,checkpoint=NULL,last=NULL WHERE scope=?", (revision, source))
            db.execute("INSERT INTO conversation_actions VALUES(?,?,?,?)", (data["user"], data["actionId"], fingerprint, sha))
            event_data = {"actionId": data["actionId"], "actionSha256": sha, "releaseId": data["releaseId"],
                                                            "user": data["user"], "session": data["session"], "destinationSession": data.get("destinationSession"),
                                                            "sourceHead": expected, "head": head, "reason": data["reason"]}
            if data["operation"] == "restore":
                event_data.update({key: data[key] for key in ("sourceRequestId", "sourceTraceSha256", "sourceCheckpointSha256")})
            self.event(db, "conversation_"+data["operation"], event_data)
        return {**result, "actionSha256": sha, "idempotentReplay": False}

    def release_memory(self, release, user):
        rows = self.query("SELECT * FROM release_memory WHERE release=? AND user=? ORDER BY created, id", (release, user))
        return [{**r, "metadata": json.loads(r["metadata"]), "generated": bool(r["generated"]), "evidence": json.loads(r["evidence"]) if r["evidence"] else None} for r in rows]

    def delete_release_memory(self, release, user, id_):
        """Physical delete of one record (its text is gone); the lifecycle keeps only ids."""
        with self.lock, closing(self.db()) as db, db:
            cur = db.execute("DELETE FROM release_memory WHERE release=? AND user=? AND id=?", (release, user, id_))
            if not cur.rowcount:
                raise ProductionError("E_MEMORY_NOT_FOUND", "No such memory record for this release and user.", 404)
            self.event(db, "release_memory_deleted", {"releaseId": release, "user": user, "recordId": id_})
        return {"deleted": id_}

    def finish_request(self, user, id_, trace, scope=None, *, conversation=None, deadline=None, memory=None):
        # CAS writes may leave orphan bytes after cancellation/crash; heads never reference them.
        candidate_sha = None
        if conversation and conversation[2] is not None and trace["status"] == 200:
            candidate_sha = self.artifacts.put_bytes(dumps(conversation[2]).encode())
        with self.lock, closing(self.db()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT cancelled FROM requests WHERE user=? AND id=?", (user, id_)).fetchone()
            if row[0] and trace["status"] == 200:
                trace.update(status=409, error={"code": "E_REQUEST_CANCELLED", "message": "Cancelled before state commit."}, result=None)
            trace["conversationState"], trace["sessionState"] = None, None
            update = None
            if conversation and trace["status"] == 200:
                key, previous, candidate = conversation
                head = db.execute("SELECT revision,checkpoint FROM agent_sessions WHERE scope=?", (key,)).fetchone()
                actual = (head["revision"], head["checkpoint"]) if head else (0, None)
                expected = (previous["revision"], previous["checkpointSha256"]) if previous else (0, None)
                if actual != expected or candidate_sha is None:
                    trace.update(status=409, error={"code": "E_SESSION_CONFLICT", "message": "Conversation checkpoint changed before commit."}, result=None)
                else:
                    update = (key, actual[0]+1, candidate_sha, id_)
                    trace["conversationState"] = {"revision": actual[0]+1, "checkpointSha256": candidate_sha, "parentCheckpointSha256": actual[1],
                                                  "threadId": candidate["threadId"], "lastRequestId": id_}
            if scope and trace["status"] == 200:
                head = db.execute("SELECT count FROM sessions WHERE scope=?", (scope,)).fetchone()
                trace["sessionState"] = {"count": (head[0] if head else 0)+1, "last": id_}
            memory_rows = []
            if memory and memory[2] and trace["status"] == 200:
                release, owner, writes, cap = memory
                live = db.execute("SELECT COUNT(*) FROM release_memory WHERE release=? AND user=?", (release, owner)).fetchone()[0]
                if live + len(writes) > cap:
                    trace.update(status=409, error={"code": "E_AGENT_MEMORY_FULL", "message": f"This user's memory in this release holds {live} of {cap} records; delete records before new ones are stored."}, result=None)
                else:
                    now = time.time()
                    memory_rows = [(release, owner, w["id"], w["namespace"], w["kind"], w["text"], dumps(w["metadata"]), w["importance"], int(w["generated"]),
                                    dumps(w["evidence"]) if w["evidence"] is not None else None, id_, now) for w in writes]
                    trace["memoryWrites"] = [w["id"] for w in writes]
            sha = self.artifacts.put_bytes(dumps(trace).encode())
            # Check after serialization/CAS too, before any checkpoint head mutation.
            if deadline is not None and time.perf_counter() >= deadline and trace["status"] == 200:
                trace.update(status=504, error={"code": "E_REQUEST_TIMEOUT", "message": "Deadline passed before state commit."},
                             result=None, conversationState=None, sessionState=None)
                trace.pop("memoryWrites", None)
                sha = self.artifacts.put_bytes(dumps(trace).encode())
            if trace["status"] == 200:
                if update:
                    db.execute("INSERT INTO agent_sessions VALUES(?,?,?,?) ON CONFLICT(scope) DO UPDATE SET revision=excluded.revision,checkpoint=excluded.checkpoint,last=excluded.last", update)
                if scope:
                    db.execute("INSERT INTO sessions VALUES(?,1,?) ON CONFLICT(scope) DO UPDATE SET count=count+1,last=excluded.last", (scope, id_))
                db.executemany("INSERT INTO release_memory VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", memory_rows)
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
