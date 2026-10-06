"""SYNTHETIC agent evaluation of installed local models (ADR 0077).

Runs the example's 20 cases through the real evaluation runner (each case a recorded child agent run) for each model.
    .venv/bin/python benchmarks/agent_eval_quality.py --models qwen3.5:0.8b qwen3.5:2b --out benchmarks/results/agent_eval_local_models.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "services")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["qwen3.5:0.8b", "qwen3.5:2b"])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    from artifact_store import ArtifactStore
    from agent.models import ollama_status
    from worker.agent_eval import AgentEvalConfig, run_agent_eval

    spec = importlib.util.spec_from_file_location("fx", ROOT / "examples" / "make_agent_eval_fixture.py")
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    status = ollama_status()
    results = []
    for model in a.models:
        with tempfile.TemporaryDirectory(prefix="void-agent-eval-") as d:
            store = ArtifactStore(Path(d))
            cfg = AgentEvalConfig(name=f"SYNTHETIC computable answers · {model}", cases=fx.cases())
            store.create_run("eval", "g", cfg.model_dump())
            run_status = run_agent_eval(fx.graph(model), cfg, store, "eval")
            rep = json.loads(store.read_artifact(store.artifacts("eval", "agent_eval_report")[-1]["sha256"]))
        by_family = {}
        for r in rep["results"]:
            fam = r["case"].split("-")[0]
            by_family.setdefault(fam, [0, 0])
            by_family[fam][0] += r["passed"]
            by_family[fam][1] += 1
        row = {"model": model, "status": run_status, "cases": rep["cases"], "passed": rep["passed"], "passRate": rep["passRate"], "wilson95": rep["wilson95"],
               "byFamily": {k: {"passed": v[0], "cases": v[1]} for k, v in by_family.items()}, "totalSeconds": rep["totalSeconds"],
               "failures": [{"case": f["case"], "expected": f["expected"], "observed": f["observed"]} for f in rep["failedChecks"]]}
        results.append(row)
        print(json.dumps({k: row[k] for k in ("model", "passed", "cases", "passRate", "wilson95", "byFamily", "totalSeconds")}), flush=True)
    out = {"benchmark": "agent evaluation on 20 SYNTHETIC cases with computable answers", "host": platform.node(), "platform": f"{platform.system()} {platform.machine()}",
           "ollama": {"runtimeVersion": status.get("version"), "reachable": status.get("reachable")}, "settings": "temperature 0, seed 7, think off, max_tokens 32",
           "results": results, "notes": "One run per model. Literal checks (number/exact string); a pass rate over these cases only, not a general quality claim."}
    if a.out:
        a.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
