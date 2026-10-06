"""Agent evaluation runs (ADR 0077): one agent graph over a labelled case set, each case a real child agent run.

Each case's input starts a fresh thread; the child run is an ordinary recorded agent run (inspectable like any other). Declared
checks compare fields of the child's final state with expectations. The evaluation run records one `eval_case` event per case and
an `agent_eval_report` artifact: pass count, pass rate with a Wilson 95% interval over cases, and every failed check with the observed
value. A case passes only if its run completed and every check passed. Nothing is graded by a model.
"""
from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from artifact_store import ArtifactStore
from graph_core.hashing import semantic_hash
from graph_core.schema import Graph
from worker.events import Emitter

from .agent_run import AgentRunConfig, run_agent

OBSERVED_CHARS = 500


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)*$")  # state field; dotted path into objects
    kind: Literal["equals", "contains", "not_contains", "regex", "one_of", "number_close"]
    value: Any = None
    case_sensitive: bool = False
    tolerance: float = Field(0.0, ge=0)  # number_close: |observed - value| <= tolerance


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")
    input: dict[str, Any]
    checks: list[Check] = Field(min_length=1, max_length=8)


class AgentEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["agent_eval"] = "agent_eval"
    project_id: str | None = None
    name: str = Field("evaluation", max_length=80)
    cases: list[EvalCase] = Field(min_length=1, max_length=200)
    # Repeats (ADR 0078): each seed is written into every model node's `seed`; one child run per case and seed. Empty = the graph as is.
    seeds: list[int] = Field(default_factory=list, max_length=5)

    def model_post_init(self, _):
        ids = [c.id for c in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("case ids must be unique")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if len(self.cases) * max(1, len(self.seeds)) > 400:
            raise ValueError("cases × seeds must be at most 400 runs")


def seeded(graph: Graph, seed: int) -> Graph:
    """The same graph with `seed` on every model invocation (chat and structured output)."""
    doc = graph.to_json()
    for n in doc["nodes"]:
        if isinstance(n.get("config", {}).get("model"), dict):
            n["config"]["model"] = {**n["config"]["model"], "seed": seed}
    return Graph.model_validate(doc)


def _path(state: dict[str, Any], path: str) -> Any:
    cur: Any = state
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _text(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True)


def evaluate_check(check: Check, state: dict[str, Any]) -> dict[str, Any]:
    observed = _path(state, check.field)
    fold = (lambda s: s) if check.case_sensitive else (lambda s: s.casefold())
    if observed is None:
        ok, note = False, "field missing in the final state"
    elif check.kind == "equals":
        ok, note = (fold(observed.strip()) == fold(str(check.value).strip())) if isinstance(observed, str) else observed == check.value, None
    elif check.kind == "contains":
        ok, note = fold(str(check.value)) in fold(_text(observed)), None
    elif check.kind == "not_contains":
        ok, note = fold(str(check.value)) not in fold(_text(observed)), None
    elif check.kind == "regex":
        ok, note = re.search(str(check.value), _text(observed), 0 if check.case_sensitive else re.IGNORECASE) is not None, None
    elif check.kind == "one_of":
        options = check.value if isinstance(check.value, list) else [check.value]
        ok, note = any((fold(observed.strip()) == fold(str(o).strip())) if isinstance(observed, str) else observed == o for o in options), None
    else:  # number_close: the first number in a text, or the number itself
        num = observed if isinstance(observed, (int, float)) and not isinstance(observed, bool) else None
        if num is None:
            m = re.search(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", _text(observed))
            num = float(m.group()) if m else None
        ok = num is not None and isinstance(check.value, (int, float)) and abs(num - check.value) <= check.tolerance
        note = None if num is not None else "no number in the observed value"
    return {"field": check.field, "kind": check.kind, "expected": check.value, "observed": _text(observed)[:OBSERVED_CHARS] if observed is not None else None,
            "passed": bool(ok), "note": note}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)


def run_agent_eval(graph: Graph, cfg: AgentEvalConfig, store: ArtifactStore, run_id: str, should_cancel: Callable[[], bool] = lambda: False) -> str:
    graph_hash = semantic_hash(graph)
    em = Emitter(store, run_id, graph_hash)

    def finish(status: str, error: str | None = None, **data) -> str:
        store.set_status(run_id, status, error)
        em.emit("run_finished", status=status, error=error, **data)
        return status

    try:
        store.set_status(run_id, "preparing")
        store.set_status(run_id, "running")
        seeds = cfg.seeds or [None]
        em.emit("run_started", name=cfg.name, cases=len(cfg.cases), seeds=cfg.seeds, runs=len(cfg.cases) * len(seeds),
                grading="declared checks on the final state; no model grades answers")
        results = []
        t_all = time.perf_counter()
        for si, seed in enumerate(seeds):
            g = graph if seed is None else seeded(graph, seed)
            for i, case in enumerate(cfg.cases):
                if should_cancel():
                    return finish("cancelled", "cancelled between cases", completedCases=len(results))
                suffix = f"c{i:03d}" if seed is None else f"s{si}c{i:03d}"
                child = f"{run_id}-{suffix}"
                child_cfg = AgentRunConfig(project_id=cfg.project_id, thread_id=f"eval-{run_id}-{suffix}", input=case.input)
                store.create_run(child, semantic_hash(g), {**child_cfg.model_dump(), "evaluationOf": run_id, "caseId": case.id, "seed": seed})
                t = time.perf_counter()
                status = run_agent(g, child_cfg, store, child, should_cancel)
                latency = (time.perf_counter() - t) * 1000
                finals = store.artifacts(child, "final_state")
                state = json.loads(store.read_artifact(finals[-1]["sha256"])) if finals else {}
                checks = [evaluate_check(c, state) for c in case.checks]
                fin = store.last_event(child, "run_finished")
                calls = store.events(child, -1, ("model_call",))
                passed = status == "completed" and all(c["passed"] for c in checks)
                row = {"case": case.id, "seed": seed, "childRunId": child, "status": status, "error": store.get_run(child)["error"], "passed": passed, "checks": checks,
                       "latencyMs": round(latency, 1), "modelCalls": len(calls),
                       "outputTokens": sum((c["data"].get("usage") or {}).get("outputTokens") or 0 for c in calls),
                       "stoppedBy": (fin["data"].get("stoppedBy") if fin else None)}
                results.append(row)
                em.emit("eval_case", **row)
        k, n = sum(r["passed"] for r in results), len(results)
        per_seed, stability = None, None
        if cfg.seeds:
            per_seed = [{"seed": s, "passed": sum(r["passed"] for r in results if r["seed"] == s), "cases": len(cfg.cases),
                         "passRate": round(sum(r["passed"] for r in results if r["seed"] == s) / len(cfg.cases), 4),
                         "wilson95": wilson(sum(r["passed"] for r in results if r["seed"] == s), len(cfg.cases))} for s in cfg.seeds]
            outcome = {c.id: [r["passed"] for r in results if r["case"] == c.id] for c in cfg.cases}
            stability = {"alwaysPass": sorted(c for c, v in outcome.items() if all(v)), "neverPass": sorted(c for c, v in outcome.items() if not any(v)),
                         "varies": sorted(c for c, v in outcome.items() if any(v) and not all(v))}
        report = {"name": cfg.name, "graphHash": graph_hash, "cases": n, "passed": k, "passRate": round(k / n, 4), "wilson95": wilson(k, n),
                  "caseCount": len(cfg.cases), "seeds": cfg.seeds, "perSeed": per_seed, "stability": stability,
                  "byStatus": {s: sum(r["status"] == s for r in results) for s in sorted({r["status"] for r in results})},
                  "failedChecks": [{"case": r["case"], **c} for r in results for c in r["checks"] if not c["passed"]],
                  "totalSeconds": round(time.perf_counter() - t_all, 2), "results": results,
                  "interpretation": ("Pass rate over these cases only, with a Wilson 95% interval over cases; a different case set or model can differ. Checks are literal; they do not judge meaning."
                                     if not cfg.seeds else "With seeds, `cases` counts case×seed runs; runs of one case are not independent, so read the per-seed rates (each with its own interval over cases) and which cases vary with the seed.")}
        store.add_artifact(run_id, "agent_eval_report", json.dumps(report).encode(), "complete", None, {"graph_hash": graph_hash})
        em.emit("eval_summary", **{k_: v for k_, v in report.items() if k_ not in ("results", "failedChecks")})
        return finish("completed", passed=k, cases=n)
    except Exception as e:  # noqa: BLE001  (record any failure on the run)
        return finish("failed", f"{type(e).__name__}: {e}")
