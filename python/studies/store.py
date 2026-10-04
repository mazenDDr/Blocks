"""Study bookkeeping in SQLite (`studies.db`): studies, trials, attempts. Resumable: a restart finds every planned trial and every attempt."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (id TEXT PRIMARY KEY, created_at REAL NOT NULL, updated_at REAL NOT NULL, spec TEXT NOT NULL,
  state TEXT NOT NULL, baseline_run_id TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS trials (study_id TEXT NOT NULL, id TEXT NOT NULL, idx INTEGER NOT NULL, group_idx INTEGER NOT NULL, group_key TEXT NOT NULL,
  is_baseline INTEGER NOT NULL, assignments TEXT NOT NULL, seed INTEGER, fold INTEGER, status TEXT NOT NULL, diagnostics TEXT NOT NULL,
  PRIMARY KEY (study_id, id));
CREATE TABLE IF NOT EXISTS attempts (study_id TEXT NOT NULL, trial_id TEXT NOT NULL, attempt INTEGER NOT NULL, run_id TEXT, status TEXT NOT NULL,
  error TEXT, kind TEXT NOT NULL, started_at REAL NOT NULL, finished_at REAL, PRIMARY KEY (study_id, trial_id, attempt));
"""


class StudyStore:
    def __init__(self, workbench: str | Path):
        self.path = Path(workbench) / "studies.db"
        Path(workbench).mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._lock, closing(self._db()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)

    def _db(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def _x(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock, closing(self._db()) as db, db:
            return db.execute(sql, args).fetchall()

    # ------------------------------------------------------------ studies
    def create(self, sid: str, spec: dict[str, Any], trials: list[dict[str, Any]]) -> None:
        now = time.time()
        with self._lock, closing(self._db()) as db, db:
            db.execute("INSERT INTO studies (id, created_at, updated_at, spec, state) VALUES (?,?,?,?,?)", (sid, now, now, json.dumps(spec), "planned"))
            for t in trials:
                db.execute("INSERT INTO trials VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                           (sid, t["id"], t["idx"], t["groupIdx"], t["groupKey"], int(t["isBaseline"]), json.dumps(t["assignments"]), t["seed"], t["fold"],
                            t["status"], json.dumps(t.get("diagnostics", []))))

    def get(self, sid: str) -> dict[str, Any] | None:
        r = self._x("SELECT * FROM studies WHERE id=?", (sid,))
        if not r:
            return None
        d = dict(r[0])
        d["spec"] = json.loads(d["spec"])
        return d

    def list(self) -> list[dict[str, Any]]:
        out = []
        for r in self._x("SELECT * FROM studies ORDER BY created_at, rowid"):
            d = dict(r)
            d["spec"] = json.loads(d["spec"])
            out.append(d)
        return out

    def set_state(self, sid: str, state: str) -> None:
        self._x("UPDATE studies SET state=?, updated_at=? WHERE id=?", (state, time.time(), sid))

    def set_cancel(self, sid: str, v: bool) -> None:
        self._x("UPDATE studies SET cancel_requested=? WHERE id=?", (int(v), sid))

    def set_baseline(self, sid: str, run_id: str | None) -> None:
        self._x("UPDATE studies SET baseline_run_id=?, updated_at=? WHERE id=?", (run_id, time.time(), sid))

    def update_spec(self, sid: str, spec: dict[str, Any]) -> None:
        self._x("UPDATE studies SET spec=?, updated_at=? WHERE id=?", (json.dumps(spec), time.time(), sid))

    # ------------------------------------------------------------ trials / attempts
    def trials(self, sid: str) -> list[dict[str, Any]]:
        out = []
        for r in self._x("SELECT * FROM trials WHERE study_id=? ORDER BY idx", (sid,)):
            d = dict(r)
            d["assignments"], d["diagnostics"], d["is_baseline"] = json.loads(d["assignments"]), json.loads(d["diagnostics"]), bool(d["is_baseline"])
            out.append(d)
        return out

    def trial(self, sid: str, tid: str) -> dict[str, Any] | None:
        return next((t for t in self.trials(sid) if t["id"] == tid), None)

    def set_trial_status(self, sid: str, tid: str, status: str) -> None:
        self._x("UPDATE trials SET status=? WHERE study_id=? AND id=?", (status, sid, tid))

    def attempts(self, sid: str, tid: str | None = None) -> list[dict[str, Any]]:
        if tid:
            rows = self._x("SELECT * FROM attempts WHERE study_id=? AND trial_id=? ORDER BY attempt", (sid, tid))
        else:
            rows = self._x("SELECT * FROM attempts WHERE study_id=? ORDER BY trial_id, attempt", (sid,))
        return [dict(r) for r in rows]

    def add_attempt(self, sid: str, tid: str, kind: str) -> int:
        with self._lock:
            n = len(self.attempts(sid, tid)) + 1
            self._x("INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?)", (sid, tid, n, None, "running", None, kind, time.time(), None))
            return n

    def finish_attempt(self, sid: str, tid: str, attempt: int, run_id: str | None, status: str, error: str | None) -> None:
        self._x("UPDATE attempts SET run_id=?, status=?, error=?, finished_at=? WHERE study_id=? AND trial_id=? AND attempt=?",
                (run_id, status, error, time.time(), sid, tid, attempt))

    def set_attempt_run(self, sid: str, tid: str, attempt: int, run_id: str) -> None:
        self._x("UPDATE attempts SET run_id=? WHERE study_id=? AND trial_id=? AND attempt=?", (run_id, sid, tid, attempt))
