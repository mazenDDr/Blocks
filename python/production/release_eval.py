"""Evaluate a deployed release on labelled cases (ADR 0078).

Each case is one real serving request through the release's route (recorded with its trace like any request), as a user and session
unique to this evaluation and case, so conversation and long-term memory state never leaks between cases or into real users.
Checks read `output` (the served prediction) and `output.<key>` (JSON outputs), with the same literal checks as research evaluations.
The evaluation itself is recorded as a run of kind `release_eval` with `eval_case` events and an `agent_eval_report` artifact.
"""
from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from worker.agent_eval import EvalCase, evaluate_check, wilson
from worker.events import Emitter

from .models import PredictRequest


class ReleaseEvalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field("release evaluation", max_length=80)
    cases: list[EvalCase] = Field(min_length=1, max_length=100)

    def model_post_init(self, _):
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("case ids must be unique")


def run_release_eval(runtime, release: dict[str, Any], req: ReleaseEvalRequest, run_id: str) -> str:
    store = runtime.store
    graph_hash = runtime.pipeline(release["versionId"]).manifest.get("graphHash") or "release"
    em = Emitter(store, run_id, graph_hash)
    cfg = release["config"]

    def finish(status, error=None, **data):
        store.set_status(run_id, status, error)
        em.emit("run_finished", status=status, error=error, **data)
        return status

    try:
        store.set_status(run_id, "preparing")
        store.set_status(run_id, "running")
        em.emit("run_started", name=req.name, cases=len(req.cases), releaseId=release["id"], versionId=release["versionId"],
                grading="declared checks on the served output; no model grades answers")
        results, t_all = [], time.perf_counter()
        for i, case in enumerate(req.cases):
            ident = f"ev{run_id[:10]}-{i:03d}"
            pr = PredictRequest(requestId=ident, records=[case.input], user=ident, expectedRelease=release["id"],
                                session=ident if cfg["sessionMode"] in ("counter", "conversation") else None)
            t = time.perf_counter()
            trace = runtime.predict(cfg["target"], cfg["namespace"], pr)
            latency = (time.perf_counter() - t) * 1000
            status = trace.get("status")
            out = (trace.get("result") or {}).get("predictions", [None])[0] if status == 200 else None
            state = {"output": out} if out is not None else {}
            checks = [evaluate_check(c, state) for c in case.checks]
            row = {"case": case.id, "seed": None, "childRunId": None, "requestId": ident, "user": ident, "traceSha256": trace.get("traceSha256"),
                   "status": "completed" if status == 200 else f"http {status}", "error": (trace.get("error") or {}).get("message"),
                   "passed": status == 200 and all(c["passed"] for c in checks), "checks": checks, "latencyMs": round(latency, 1),
                   "modelCalls": ((trace.get("result") or {}).get("agent") or {}).get("modelCalls"), "outputTokens": None, "stoppedBy": None}
            results.append(row)
            em.emit("eval_case", **row)
        k, n = sum(r["passed"] for r in results), len(results)
        report = {"name": req.name, "graphHash": graph_hash, "releaseId": release["id"], "versionId": release["versionId"], "cases": n, "passed": k,
                  "passRate": round(k / n, 4), "wilson95": wilson(k, n), "caseCount": n, "seeds": [], "perSeed": None, "stability": None,
                  "byStatus": {s: sum(r["status"] == s for r in results) for s in sorted({r["status"] for r in results})},
                  "failedChecks": [{"case": r["case"], **c} for r in results for c in r["checks"] if not c["passed"]],
                  "totalSeconds": round(time.perf_counter() - t_all, 2), "results": results,
                  "interpretation": "Pass rate of this deployed release over these cases (each a recorded serving request), with a Wilson 95% interval over cases. Checks are literal."}
        store.add_artifact(run_id, "agent_eval_report", json.dumps(report).encode(), "complete", None, {"graph_hash": graph_hash})
        em.emit("eval_summary", **{k_: v for k_, v in report.items() if k_ not in ("results", "failedChecks")})
        return finish("completed", passed=k, cases=n)
    except Exception as e:  # noqa: BLE001
        return finish("failed", f"{type(e).__name__}: {e}")
