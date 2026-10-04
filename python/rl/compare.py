"""Reward / policy variant comparison across independently repeated runs (VISION 9.9 'Evaluation and scientific control', 19.3, A54).

A variant is a group of runs that differ only in their repeat identity (seed). Per run we read what the worker recorded; per variant we aggregate across
runs: mean and a Student t interval of the mean of the run-level evaluation return (the unit of replication is the RUN, not the episode), per-component
returns, episode length, termination vs truncation counts, success rate and interactions. Returns under different reward definitions are not comparable,
so the primary metric is the TASK return (the environment's default reward); the return under each variant's own training reward is reported next to it."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from artifact_store import ArtifactStore

from .evaluate import mean_ci


def _art(store: ArtifactStore, rid: str, kind: str) -> dict[str, Any] | None:
    a = store.artifacts(rid, kind)
    return json.loads(store.read_artifact(a[-1]["sha256"])) if a else None


def run_record(store: ArtifactStore, rid: str) -> dict[str, Any]:
    row = store.get_run(rid)
    rep = _art(store, rid, "rl_eval_report")
    summ = _art(store, rid, "rl_summary")
    setup = store.last_event(rid, "rl_setup")
    started = store.last_event(rid, "run_started")
    base = {"runId": rid, "status": row["status"], "seed": row["config"].get("seed"), "available": rep is not None}
    if rep is None:
        return {**base, "reason": f"run is {row['status']} and recorded no evaluation report"}
    f = rep["final"]
    first = rep["history"][0] if rep["history"] and not rep["history"][0]["final"] else None
    return {**base, "taskReturn": f["taskReturn"]["mean"], "trainReturn": f["return"]["mean"], "length": f["length"]["mean"], "terminated": f["terminatedCount"], "truncated": f["truncatedCount"],
            "episodes": len(f["episodes"]), "successRate": f["successRate"],
            "components": {k: v["mean"] for k, v in f["componentReturns"].items()}, "rawComponents": {k: v["mean"] for k, v in f["rawComponentReturns"].items()},
            "initialTaskReturn": first["taskReturn"]["mean"] if first else None, "envSteps": summ["envSteps"] if summ else None, "updates": summ["gradientUpdates"] if summ else None,
            "weights": setup["data"]["effectiveWeights"] if setup else None, "evaluationSeeds": rep["evaluationSeeds"], "wrapperChain": [w["name"] for w in setup["data"]["wrapperChain"]] if setup else None,
            "graphHash": row["graph_hash"], "algorithm": started["data"]["algorithm"] if started else None}


def _agg(vals: list[float]) -> dict[str, Any]:
    r = mean_ci([v for v in vals if v is not None])
    if r["n"] and r["n"] < 3:
        r["warning"] = f"only {r['n']} independent run(s): the interval is not reliable; add seeds"
    return r


def compare_variants(store: ArtifactStore, variants: list[dict[str, Any]]) -> dict[str, Any]:
    """variants: [{label, runIds, assignments?}]. Everything is read from recorded runs."""
    out = []
    eval_seeds = set()
    for v in variants:
        recs = [run_record(store, r) for r in v["runIds"]]
        ok = [r for r in recs if r["available"]]
        comps = sorted({k for r in ok for k in r["components"]})
        for r in ok:
            eval_seeds.add(tuple(r["evaluationSeeds"]))
        out.append({
            "label": v["label"], "assignments": v.get("assignments", []), "runs": recs, "nRuns": len(recs), "nCompleted": len(ok),
            "taskReturn": _agg([r["taskReturn"] for r in ok]), "trainReturn": _agg([r["trainReturn"] for r in ok]), "length": _agg([r["length"] for r in ok]),
            "successRate": _agg([r["successRate"] for r in ok if r["successRate"] is not None]),
            "terminated": sum(r["terminated"] for r in ok), "truncated": sum(r["truncated"] for r in ok), "episodes": sum(r["episodes"] for r in ok),
            "components": {k: _agg([r["components"].get(k) for r in ok]) for k in comps}, "rawComponents": {k: _agg([r["rawComponents"].get(k) for r in ok]) for k in comps},
            "initialTaskReturn": _agg([r["initialTaskReturn"] for r in ok if r["initialTaskReturn"] is not None]),
            "envSteps": float(np.mean([r["envSteps"] for r in ok])) if ok else None, "updates": float(np.mean([r["updates"] for r in ok])) if ok else None,
            "weights": ok[0]["weights"] if ok else None,
        })
    notes = ["The unit of replication is the run (seed). Intervals are Student t intervals of the mean across runs; they describe this sample, not the algorithm in general.",
             "taskReturn is the return under the environment's DEFAULT reward and is comparable across reward variants; trainReturn is under each variant's own reward and is not."]
    if len(eval_seeds) > 1:
        notes.append("WARNING: the variants were evaluated on different evaluation seeds, which weakens the comparison.")
    return {"variants": out, "evaluationSeedSets": [list(s) for s in sorted(eval_seeds)], "notes": notes}
