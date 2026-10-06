"""Tabular data-scale benchmark: the production_sensors pipeline on SYNTHETIC CSVs of growing size.

Each size runs in a fresh process (so peak RSS belongs to that size): generate the CSV with the same columns and label rule
as the example fixture, validate the graph natively (static source read), then execute it with the real worker
(`run_tabular`: split, fit standardize, logistic regression, metrics). Records wall times, peak RSS, file size and the
validation accuracy as a sanity check (the label is a noisy linear rule, so accuracy should stay near its ceiling).

    .venv/bin/python benchmarks/tabular_scale.py --rows 10000 100000 1000000 --out benchmarks/results/tabular_scale_mac.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def one(rows: int, workdir: Path) -> dict:
    sys.path[:0] = [str(ROOT / "python"), str(ROOT / "services")]
    import numpy as np
    import pandas as pd
    from artifact_store import ArtifactStore
    from graph_core.schema import Graph
    from graph_core.validate import validate as validate_graph
    from worker.tabular_run import TabularRunConfig, run_tabular

    rng = np.random.default_rng(rows)
    signal, background = rng.normal(0, 1, rows), rng.normal(0, 1, rows)
    label = (signal + 0.25 * rng.normal(0, 1, rows) > 0).astype(int)  # SYNTHETIC: noisy threshold on `signal`
    csv = workdir / f"synthetic_sensors_{rows}.csv"
    t = time.perf_counter()
    pd.DataFrame({"signal": signal.round(6), "background": background.round(6), "label": label}).to_csv(csv, index=False)
    generate_s = time.perf_counter() - t
    doc = json.loads((ROOT / "examples/production_sensors.project.json").read_text())
    next(n for n in doc["nodes"] if n["id"] == "sensors")["config"]["path"] = str(csv)
    graph = Graph.model_validate(doc)
    t = time.perf_counter()
    report = validate_graph(graph)
    validate_s = time.perf_counter() - t
    errors = [d.message for d in report.diagnostics if d.severity == "error"]
    store = ArtifactStore(workdir / "wb")
    store.create_run("r1", "g", {"kind": "tabular"})
    t = time.perf_counter()
    status = run_tabular(graph, TabularRunConfig(), store, "r1")
    run_s = time.perf_counter() - t
    finished = next(e for e in store.events("r1") if e["type"] == "node_finished" and e["node_id"] == "metrics")
    values = json.loads(store.read_artifact(finished["data"]["summarySha256"]))["values"]
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak_kb / (1024 * 1024) if platform.system() == "Darwin" else peak_kb / 1024  # bytes on macOS, KiB on Linux
    return {"rows": rows, "csvBytes": csv.stat().st_size, "generateSeconds": round(generate_s, 3), "validateSeconds": round(validate_s, 3),
            "validationErrors": errors, "staticSourceExact": csv.stat().st_size <= 50_000_000, "runStatus": status,
            "runError": store.get_run("r1")["error"], "runSeconds": round(run_s, 3), "peakRssMiB": round(peak_mb, 1),
            "artifactsBytes": sum(a["size"] for a in store.artifacts("r1")), "validationAccuracy": round(values["accuracy"], 4), "rocAuc": round(values["roc_auc"], 4),
            "accuracyCeiling": round(1 - math.atan(0.25) / math.pi, 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--one", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--workdir", type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        print(json.dumps(one(a.one, a.workdir)))
        return
    results = []
    for rows in a.rows:
        with tempfile.TemporaryDirectory(prefix="void-tabular-scale-") as d:
            p = subprocess.run([sys.executable, __file__, "--one", str(rows), "--workdir", d], capture_output=True, text=True, timeout=3600)
            row = json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else {"rows": rows, "error": p.stderr[-2000:]}
        results.append(row)
        print(json.dumps(row), flush=True)
    import os
    out = {"benchmark": "tabular pipeline scale (production_sensors graph on SYNTHETIC CSVs)", "host": platform.node(), "platform": f"{platform.system()} {platform.machine()}",
           "python": platform.python_version(), "cpus": os.cpu_count(), "results": results,
           "notes": "One fresh process per size; peak RSS covers generation, validation and execution in that process. Single run per size, no repeats: timings indicate scale, not a precise rate."}
    if a.out:
        a.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
