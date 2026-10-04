"""Sequential trial scheduler (max concurrency 1) with resumable bookkeeping.

Each trial attempt is an ordinary run (own OS process). A failed attempt marks the trial `failed` and keeps the attempt; retrying adds a
new attempt (same trial identity). Planned trials left over from an earlier session are picked up by `start`."""
from __future__ import annotations

import re
import threading
import time
from typing import Any, Callable

from graph_core.schema import Graph

from . import planner
from .store import StudyStore

TERMINAL = ("completed", "failed", "cancelled")
CODE_RE = re.compile(r"^([A-Z][A-Z0-9_]+) at ")


class StudyRunner:
    def __init__(self, ss: StudyStore, run_store, launch: Callable[[Graph, dict[str, Any]], str], cancel_run: Callable[[str], None]):
        self.ss, self.rs, self.launch, self.cancel_run = ss, run_store, launch, cancel_run
        self.threads: dict[str, threading.Thread] = {}
        self.lock = threading.Lock()

    def is_running(self, sid: str) -> bool:
        t = self.threads.get(sid)
        return bool(t and t.is_alive())

    def start(self, sid: str) -> bool:
        with self.lock:
            if self.is_running(sid):
                return False
            self.ss.set_cancel(sid, False)
            self.ss.set_state(sid, "running")
            t = threading.Thread(target=self._loop, args=(sid,), name=f"study-{sid}", daemon=True)
            self.threads[sid] = t
            t.start()
            return True

    def request_cancel(self, sid: str) -> None:
        self.ss.set_cancel(sid, True)

    def join(self, sid: str, timeout: float | None = None) -> None:
        t = self.threads.get(sid)
        if t:
            t.join(timeout)

    # -------------------------------------------------------------------------------------
    def _cancelled(self, sid: str) -> bool:
        return bool(self.ss.get(sid)["cancel_requested"])

    def _loop(self, sid: str) -> None:
        try:
            while True:
                if self._cancelled(sid):
                    for t in self.ss.trials(sid):
                        if t["status"] == "planned":
                            self.ss.set_trial_status(sid, t["id"], "cancelled")
                    self.ss.set_state(sid, "cancelled")
                    return
                nxt = next((t for t in self.ss.trials(sid) if t["status"] == "planned"), None)
                if nxt is None:
                    break
                self._run_trial(sid, nxt)
            ts = self.ss.trials(sid)
            self.ss.set_state(sid, "completed_with_failures" if any(t["status"] in ("failed", "cancelled") for t in ts) else "completed")
        except Exception as e:  # noqa: BLE001  (a scheduler bug must be visible, not silent)
            self.ss.set_state(sid, "scheduler_error")
            spec = self.ss.get(sid)["spec"]
            spec["schedulerError"] = f"{type(e).__name__}: {e}"
            self.ss.update_spec(sid, spec)

    def _run_trial(self, sid: str, trial: dict[str, Any]) -> None:
        st = self.ss.get(sid)
        spec = st["spec"]
        graph = Graph.model_validate(spec["graph"])
        prior = self.ss.attempts(sid, trial["id"])
        attempt = self.ss.add_attempt(sid, trial["id"], "retry" if prior else "initial")
        self.ss.set_trial_status(sid, trial["id"], "running")
        run_id, status, error = None, "failed", None
        try:
            full = list(trial["assignments"])
            rp = spec["repeats"]
            if trial["seed"] is not None:
                full.append({"target": (rp.get("seed_target") or {"scope": "run", "field": "seed"}), "value": trial["seed"]})
            if trial["fold"] is not None:
                full += [{"target": {"scope": "node", "node": rp["fold_node"], "field": "n_folds"}, "value": rp["folds"]},
                         {"target": {"scope": "node", "node": rp["fold_node"], "field": "fold"}, "value": trial["fold"]}]
            g, cfg = planner.apply(graph, spec["run_config"], full)
            cfg["trial"] = {"studyId": sid, "trialId": trial["id"], "attempt": attempt, "groupKey": trial["group_key"], "seed": trial["seed"], "fold": trial["fold"],
                            "isBaseline": trial["is_baseline"]}
            run_id = self.launch(g, cfg)
            self.ss.set_attempt_run(sid, trial["id"], attempt, run_id)
            while True:
                row = self.rs.get_run(run_id)
                if row["status"] in TERMINAL:
                    break
                if self._cancelled(sid):
                    self.cancel_run(run_id)
                time.sleep(0.15)
            status, error = row["status"], row["error"]
        except Exception as e:  # noqa: BLE001
            status = "failed"
            code = getattr(e, "code", None) or getattr(getattr(e, "detail", None), "get", lambda *_: None)("code")
            error = f"{code + ': ' if code else ''}{getattr(e, 'message', None) or getattr(e, 'detail', None) or e}"
        self.ss.finish_attempt(sid, trial["id"], attempt, run_id, status, error)
        self.ss.set_trial_status(sid, trial["id"], status)
