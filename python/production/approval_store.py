"""Atomic paused/END head and trace commits; original END-only store stays pinned."""
from contextlib import closing
import time
from tabular.core import dumps

def finish_request(self, user, id_, trace, scope=None, *, conversation=None, deadline=None):
    # CAS writes may leave orphan bytes after cancellation/crash; heads never reference them.
    candidate_sha = None
    if conversation and conversation[2] is not None and trace["status"] in (200, 202):
        candidate_sha = self.artifacts.put_bytes(dumps(conversation[2]).encode())
    with self.lock, closing(self.db()) as db, db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT cancelled FROM requests WHERE user=? AND id=?", (user, id_)).fetchone()
        if row[0] and trace["status"] in (200, 202):
            trace.update(status=409, error={"code": "E_REQUEST_CANCELLED", "message": "Cancelled before state commit."}, result=None)
        trace["conversationState"], trace["sessionState"] = None, None
        update = None
        if conversation and trace["status"] in (200, 202):
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
        if scope and trace["status"] in (200, 202):
            head = db.execute("SELECT count FROM sessions WHERE scope=?", (scope,)).fetchone()
            trace["sessionState"] = {"count": (head[0] if head else 0)+1, "last": id_}
        sha = self.artifacts.put_bytes(dumps(trace).encode())
        # Check after serialization/CAS too, before any checkpoint head mutation.
        if deadline is not None and time.perf_counter() >= deadline and trace["status"] in (200, 202):
            trace.update(status=504, error={"code": "E_REQUEST_TIMEOUT", "message": "Deadline passed before state commit."},
                         result=None, conversationState=None, sessionState=None)
            sha = self.artifacts.put_bytes(dumps(trace).encode())
        if trace["status"] in (200, 202):
            if update:
                db.execute("INSERT INTO agent_sessions VALUES(?,?,?,?) ON CONFLICT(scope) DO UPDATE SET revision=excluded.revision,checkpoint=excluded.checkpoint,last=excluded.last", update)
            if scope:
                db.execute("INSERT INTO sessions VALUES(?,1,?) ON CONFLICT(scope) DO UPDATE SET count=count+1,last=excluded.last", (scope, id_))
        state = "completed" if trace["status"] in (200, 202) else "cancelled" if (trace.get("error") or {}).get("code") == "E_REQUEST_CANCELLED" else "failed"
        db.execute("UPDATE requests SET state=?,trace=? WHERE user=? AND id=?", (state, sha, user, id_))
    return {**trace, "traceSha256": sha}

