"""Summarize a raw benchmark file (also usable alone: python benchmarks/summarize.py benchmarks/results/m6a_raw.json). Only measured numbers are printed."""
from __future__ import annotations

import json
import statistics
import sys


def _med(xs):
    return statistics.median(xs)


def _ms(x):
    return f"{x * 1000:.3f}"


def summarize(raw: dict) -> str:
    env, runs = raw["environment"], [r for r in raw["runs"] if "error" not in r]
    failed = [r for r in raw["runs"] if "error" in r]
    s = env["settings"]
    o = ["# Benchmark summary (generated from the raw file; do not edit)", ""]
    o.append(f"Raw data: every timed call of every process is in the raw JSON. {s['reps']} fresh-process repetitions x {s['iters']} timed calls "
             f"(after 5 warmup calls) per (backend, implementation, workload); batch {s['batch']}; plain SGD lr {s['lr']}.")
    o.append("")
    o.append(f"Environment: {env['cpu']} ({env['logical_cpus']} logical CPUs, {env['performance_cores']} performance), {env['platform']}, Python {env['python'].split()[0]}, "
             f"{', '.join(f'{k} {v}' for k, v in env['versions'].items())}; commit {env['git_commit'][:10]}{' (working tree dirty at run time)' if env['git_dirty'] else ''}; {env['date_utc']}. "
             f"Threads requested: {s['threads']} (torch, TensorFlow); the JAX flag was requested, its effect unverified. {env['note']}.")
    o.append("")
    if failed:
        o.append(f"**{len(failed)} worker process(es) failed**; see the raw file: " + ", ".join(f"{r['backend']}/{r['impl']} rep {r['rep']}" for r in failed))
        o.append("")
    by = {}
    for r in runs:
        by.setdefault((r["backend"], r["impl"]), []).append(r)
    for wl, title in (("cnn", "Reference CNN (VISION 8.1), batch 16, 3x64x64"), ("mse", "MSE fixture (3 elements; absolute overhead of a 4-node graph)")):
        for mode, key in (("forward (loss only)", "warm_forward_s"), ("train step (forward + backward + SGD)" if wl == "cnn" else "loss + input gradient", "warm_step_s")):
            o.append(f"## {title}: warm {mode}, CPU")
            o.append("")
            o.append("Per-process median of the timed calls, then the median (min-max) of those medians across processes, in ms. Ratio = lowered / native using the medians; "
                     "difference = absolute overhead in ms. Spread = (max - min) / median across processes, per implementation.")
            o.append("")
            o.append("| Backend | native median (min-max) | lowered median (min-max) | ratio | difference (ms) | native spread | lowered spread |")
            o.append("|---|---|---|---|---|---|---|")
            for b in ("pytorch", "keras", "jax"):
                cells = {}
                for impl in ("native", "lowered"):
                    meds = [_med(r["workloads"][wl][key]) for r in by.get((b, impl), [])]
                    if meds:
                        cells[impl] = (_med(meds), min(meds), max(meds), (max(meds) - min(meds)) / _med(meds), len(meds))
                if len(cells) < 2:
                    o.append(f"| {b} | missing data | | | | | |")
                    continue
                n, l = cells["native"], cells["lowered"]
                o.append(f"| {b} | {_ms(n[0])} ({_ms(n[1])}-{_ms(n[2])}) | {_ms(l[0])} ({_ms(l[1])}-{_ms(l[2])}) | {l[0] / n[0]:.3f} | {(l[0] - n[0]) * 1000:+.3f} | {n[3]:.0%} | {l[3]:.0%} |")
            o.append("")
    o.append("## Cold start (median across processes, seconds)")
    o.append("")
    o.append("Import = process start to libraries imported; build = construct the model (lowered: validate + lower + copy weights); first forward / first train step include tracing and compilation.")
    o.append("")
    o.append("| Workload | Backend | Impl | import | build | first forward | first step |")
    o.append("|---|---|---|---|---|---|---|")
    for wl in ("cnn", "mse"):
        for b in ("pytorch", "keras", "jax"):
            for impl in ("native", "lowered"):
                rs = by.get((b, impl), [])
                if rs:
                    o.append(f"| {wl} | {b} | {impl} | {_med([r['import_s'] for r in rs]):.3f} | {_med([r['workloads'][wl]['build_s'] for r in rs]):.3f} | "
                             f"{_med([r['workloads'][wl]['cold_forward_s'] for r in rs]):.3f} | {_med([r['workloads'][wl]['cold_step_s'] for r in rs]):.3f} |")
    o.append("")
    o.append("## Process peak RSS (MB, median; whole process incl. libraries and both workloads; not a model-memory measurement)")
    o.append("")
    o.append("| Backend | native | lowered |")
    o.append("|---|---|---|")
    for b in ("pytorch", "keras", "jax"):
        o.append(f"| {b} | " + " | ".join(f"{_med([r['max_rss_mb_process'] for r in by[(b, i)]]):.0f}" if by.get((b, i)) else "-" for i in ("native", "lowered")) + " |")
    o.append("")
    o.append("## Reading these numbers")
    o.append("")
    o.append("VISION 18.3 proposes a warm training overhead of at most 1.05x native as a TARGET; the train-step ratios above are the measurement against it, with the spreads shown. "
             "These are single-machine laptop runs without isolation: differences smaller than the spread are not distinguishable. "
             "Accelerator memory, GPU, large models, the tabular pipeline and the language-model workflow are NOT measured here.")
    o.append("")
    return "\n".join(o)


if __name__ == "__main__":
    print(summarize(json.load(open(sys.argv[1]))))
