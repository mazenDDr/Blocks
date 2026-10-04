"""Assemble the study results table: trials with identities and attempts, objective values with semantics, groups, baseline comparison."""
from __future__ import annotations

import math
from typing import Any

from artifact_store import ArtifactStore

from . import metrics
from .models import MetricSpec
from .store import StudyStore


def _stats(vals: list[float]) -> dict[str, Any]:
    n = len(vals)
    if not n:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": m, "std": sd, "min": min(vals), "max": max(vals)}


def _better(direction: str, a: float, b: float) -> bool:
    return a < b if direction == "minimize" else a > b


def study_view(ss: StudyStore, rs: ArtifactStore, sid: str, running: bool) -> dict[str, Any]:
    st = ss.get(sid)
    spec = st["spec"]
    kind = spec["graphKind"]
    mspec = MetricSpec.model_validate(spec["objective"]["metric"])
    direction = spec["objective"]["direction"]
    attempts = ss.attempts(sid)
    by_trial: dict[str, list[dict[str, Any]]] = {}
    for a in attempts:
        by_trial.setdefault(a["trial_id"], []).append(a)
    rows = []
    for t in ss.trials(sid):
        at = by_trial.get(t["id"], [])
        done = [a for a in at if a["status"] == "completed"]
        best_attempt = done[-1] if done else None
        metric = metrics.extract(rs, best_attempt["run_id"], kind, mspec) if best_attempt and best_attempt["run_id"] else None
        rows.append({
            "trialId": t["id"], "index": t["idx"], "group": t["group_idx"], "groupKey": t["group_key"], "isBaseline": t["is_baseline"],
            "label": next((x for x in [spec["labels"].get(t["group_key"])] if x), "baseline" if t["is_baseline"] else t["group_key"]),
            "assignments": t["assignments"], "seed": t["seed"], "fold": t["fold"], "status": t["status"], "diagnostics": t["diagnostics"],
            "attemptCount": len(at), "retried": len(at) > 1,
            "attempts": [{"attempt": a["attempt"], "kind": a["kind"], "runId": a["run_id"], "status": a["status"], "error": a["error"],
                          "startedAt": a["started_at"], "finishedAt": a["finished_at"]} for a in at],
            "runId": best_attempt["run_id"] if best_attempt else (at[-1]["run_id"] if at else None),
            "metric": metric, "error": (at[-1]["error"] if at and t["status"] in ("failed", "cancelled") else None),
        })
    # baseline
    base: dict[str, Any] = {"mode": None, "value": None, "runId": None, "n": 0}
    if st["baseline_run_id"]:
        m = metrics.extract(rs, st["baseline_run_id"], kind, mspec)
        base = {"mode": "pinned_run", "runId": st["baseline_run_id"], "value": m.get("value") if m.get("available") else None, "n": 1, "metric": m,
                "note": "Every trial is compared with this single pinned run."}
    else:
        bvals = [r["metric"]["value"] for r in rows if r["isBaseline"] and r["metric"] and r["metric"].get("available")]
        if bvals:
            s = _stats(bvals)
            base = {"mode": "baseline_group_mean", "runId": None, "value": s["mean"], "n": s["n"], "std": s["std"],
                    "note": "Mean of the baseline configuration's completed repeats; each trial is compared with it."}
    for r in rows:
        v = r["metric"]["value"] if r["metric"] and r["metric"].get("available") else None
        r["value"] = v
        r["delta"] = (v - base["value"]) if (v is not None and base["value"] is not None) else None
        r["better"] = _better(direction, v, base["value"]) if r["delta"] not in (None, 0.0) else None
        r["changedFields"] = [a["target"]["field"] if a["target"]["scope"] == "run" else f"{a['target']['node']}.{a['target']['field']}" for a in r["assignments"]]
        r["attributionWarning"] = ("several fields differ from the baseline in this configuration; an outcome change is not attributable to any one of them"
                                   if len(r["changedFields"]) > 1 else None)
    # groups (configurations) with repeat aggregation
    groups = []
    for gi in sorted({r["group"] for r in rows}):
        rs_ = [r for r in rows if r["group"] == gi]
        vals = [r["value"] for r in rs_ if r["value"] is not None]
        s = _stats(vals)
        g = {"group": gi, "groupKey": rs_[0]["groupKey"], "label": rs_[0]["label"], "isBaseline": rs_[0]["isBaseline"], "assignments": rs_[0]["assignments"],
             "trials": len(rs_), "completed": sum(1 for r in rs_ if r["status"] == "completed"), "failed": sum(1 for r in rs_ if r["status"] == "failed"),
             "cancelled": sum(1 for r in rs_ if r["status"] == "cancelled"), "invalid": sum(1 for r in rs_ if r["status"] == "invalid"),
             "retried": sum(1 for r in rs_ if r["retried"]), **s,
             "repeats": [{"trialId": r["trialId"], "seed": r["seed"], "fold": r["fold"], "value": r["value"], "status": r["status"]} for r in rs_]}
        g["delta"] = (s["mean"] - base["value"]) if (s["mean"] is not None and base["value"] is not None) else None
        g["better"] = _better(direction, s["mean"], base["value"]) if g["delta"] not in (None, 0.0) else None
        groups.append(g)
    ranked = sorted([g for g in groups if g["mean"] is not None], key=lambda g: g["mean"], reverse=(direction == "maximize"))
    for i, g in enumerate(ranked):
        g["rank"] = i + 1
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"id": sid, "name": spec["name"], "hypothesis": spec["hypothesis"], "notes": spec["notes"], "state": "running" if running else st["state"],
            "createdAt": st["created_at"], "updatedAt": st["updated_at"], "graphKind": kind, "graphHash": spec["graphHash"], "projectId": spec.get("projectId"),
            "objective": {**spec["objective"], "semantics": ("single evaluation after fitting; no steps" if kind == "tabular" else
                                                             "final greedy evaluation on the declared evaluation seeds (separate environment instances), after training" if kind == "rl" else
                                                             f"{mspec.select} over epoch-end evaluations (value, step and epoch are recorded per trial)")},
            "search": spec["search"], "repeats": spec["repeats"], "limits": spec["limits"], "runConfig": spec["run_config"],
            "plan": spec["planSummary"], "baseline": base, "counts": counts, "trials": rows, "groups": groups,
            "best": ({"group": ranked[0]["group"], "label": ranked[0]["label"], "mean": ranked[0]["mean"], "n": ranked[0]["n"]} if ranked else None),
            "schedulerError": spec.get("schedulerError"),
            "notice": ("Repeats of one configuration (different seeds or folds) are aggregated as mean and sample standard deviation across the completed repeats; failed, "
                       "cancelled and invalid trials stay in the table and are not hidden. Parameter importance and ranking describe this sampled search only.")}
