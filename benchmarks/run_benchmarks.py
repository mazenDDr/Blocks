"""A19 / VISION 18.4: graph-lowered execution vs handwritten native models, per backend, CPU.

    python benchmarks/run_benchmarks.py                 # full run: writes benchmarks/results/m6a_raw.json and m6a_summary.md
    python benchmarks/run_benchmarks.py --reps 1 --iters 5 --out /tmp/quick    # a quick smoke run

Method (all of it is in the raw file):
  * every (backend, implementation) pair runs in a FRESH Python process, so imports, tracing and compilation are cold; process repetitions are
    interleaved (rep 0: all pairs, rep 1: all pairs, ...) and the order of the two implementations alternates, so slow drift of the machine hits both alike
  * cold = import time, build time (lowered: validate + lower + weights), first forward call, first train step (these include tracing/compilation)
  * warm = 5 untimed warmup calls, then --iters individually timed calls, results materialized/synchronized inside the timed region
    (torch CPU is synchronous; tf.function results are read with .numpy(); JAX uses block_until_ready)
  * both implementations get identical weights (graph layout, converted by hand for the native model), identical data, batch, learning rate (plain SGD)
  * workloads: `cnn` = the reference CNN (batch 16, 3x64x64, 10 classes, cross-entropy; forward = loss, step = forward+backward+SGD update);
    `mse` = the 3-element MSE fixture (forward = loss, step = loss + gradient w.r.t. the input; there are no parameters to update)
  * threads: torch and TensorFlow are asked for --threads intra-op threads; for JAX the XLA flag is REQUESTED but its effect was not verified
  * 'lowered' = the Executable's compute path on inputs converted once (NumPy -> tensor conversion at the API boundary is not timed here);
    the native Keras model also receives NHWC data directly, whereas the lowered one transposes NCHW -> NHWC inside its timed region (a real cost of the layout convention)"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "services"), str(HERE)]

BACKENDS = ("pytorch", "keras", "jax")
IMPLS = ("lowered", "native")
LR = 0.01
BATCH = 16


def _data():
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.standard_normal((BATCH, 3, 64, 64)).astype(np.float32)
    y = rng.integers(0, 10, (BATCH,)).astype(np.int64)
    pred, target = np.array([1.0, 2.5, 2.0], np.float32), np.array([1.0, 2.0, 3.0], np.float32)
    return x, y, pred, target


def _weights():
    """Deterministic weights in graph layout, uniform(+-1/sqrt(fan_in)); identical for every implementation."""
    import numpy as np

    rng = np.random.default_rng(1)

    def u(shape, fan):
        b = 1 / fan ** 0.5
        return rng.uniform(-b, b, shape).astype(np.float32)

    return {"conv_1": {"weight": u((32, 3, 3, 3), 27), "bias": u((32,), 27)}, "conv_2": {"weight": u((64, 32, 3, 3), 288), "bias": u((64,), 288)},
            "fc": {"weight": u((10, 64), 64), "bias": u((10,), 64)}}


def _timed(fn):
    t = time.perf_counter()
    out = fn()
    return time.perf_counter() - t, out


def worker(backend: str, impl: str, iters: int, threads: int) -> dict:
    t_proc = time.perf_counter()
    res: dict = {"backend": backend, "impl": impl, "workloads": {}}
    if backend == "pytorch":
        import torch
        torch.set_num_threads(threads)
        res["threads"] = {"torch": torch.get_num_threads()}
    elif backend == "keras":
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        import keras  # noqa: F401
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        res["threads"] = {"tf_intra": tf.config.threading.get_intra_op_parallelism_threads(), "tf_inter": tf.config.threading.get_inter_op_parallelism_threads()}
    else:
        import jax  # noqa: F401
        res["threads"] = {"jax": "XLA default; the requested flag's effect is not verified", "XLA_FLAGS": os.environ.get("XLA_FLAGS")}
    if impl == "lowered":
        import backends  # noqa: F401
    res["import_s"] = time.perf_counter() - t_proc
    x, y, pred, target = _data()
    params = _weights()

    for wl in ("cnn", "mse"):
        rec: dict = {}
        t0 = time.perf_counter()
        if impl == "native":
            from native_models import NATIVE
            fwd, step = NATIVE[(backend, wl)](params, x, y, LR) if wl == "cnn" else NATIVE[(backend, wl)](pred, target)
        else:
            import backends
            from backends.workloads import cnn_with_loss, mse_graph
            ex = backends.compile_graph(cnn_with_loss() if wl == "cnn" else mse_graph(), backend, compiled=True)
            if wl == "cnn":
                ex.set_params(params)
                fwd, step = ex.bench_closures({"images": x, "labels": y}, LR, "ce")
            else:
                fwd, step = ex.bench_closures({"pred": pred, "target": target}, LR, "loss", grad_inputs=("pred",))
        rec["build_s"] = time.perf_counter() - t0
        rec["cold_forward_s"], _ = _timed(fwd)
        rec["cold_step_s"], _ = _timed(step)
        for _ in range(5):
            fwd(), step()
        rec["warm_forward_s"] = [_timed(fwd)[0] for _ in range(iters)]
        rec["warm_step_s"] = [_timed(step)[0] for _ in range(iters)]
        res["workloads"][wl] = rec
    res["max_rss_mb_process"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # bytes on macOS
    return res


def environment(args) -> dict:
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    import importlib.metadata as md
    vers = {p: md.version(p) for p in ("torch", "tensorflow", "keras", "jax", "jaxlib", "numpy")}
    return {"date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "cpu": sh("sysctl", "-n", "machdep.cpu.brand_string"), "logical_cpus": os.cpu_count(), "performance_cores": sh("sysctl", "-n", "hw.perflevel0.logicalcpu"),
            "memory_bytes": sh("sysctl", "-n", "hw.memsize"), "versions": vers, "git_commit": sh("git", "-C", str(ROOT), "rev-parse", "HEAD"),
            "git_dirty": bool(sh("git", "-C", str(ROOT), "status", "--porcelain")),
            "env": {k: os.environ[k] for k in ("KERAS_BACKEND", "XLA_FLAGS", "OMP_NUM_THREADS", "TF_CPP_MIN_LOG_LEVEL") if k in os.environ},
            "settings": {"reps": args.reps, "iters": args.iters, "threads": args.threads, "batch": BATCH, "lr": LR},
            "note": "run on a developer laptop with other processes alive (including a resident uvicorn); no isolation, no frequency pinning"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2, metavar=("BACKEND", "IMPL"))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "results" / "m6a"))
    args = ap.parse_args()
    if args.worker:
        print("RESULT " + json.dumps(worker(args.worker[0], args.worker[1], args.iters, args.threads)))
        return 0
    from summarize import summarize
    runs, env = [], environment(args)
    xla = f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={args.threads}"
    for rep in range(args.reps):
        for b in BACKENDS:
            for impl in (IMPLS if rep % 2 == 0 else IMPLS[::-1]):
                e = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2", XLA_FLAGS=xla)
                t = time.perf_counter()
                p = subprocess.run([sys.executable, __file__, "--worker", b, impl, "--iters", str(args.iters), "--threads", str(args.threads)],
                                   capture_output=True, text=True, env=e)
                line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
                if p.returncode != 0 or not line:
                    print(f"FAILED rep {rep} {b} {impl}:\n{p.stderr[-2000:]}", file=sys.stderr)
                    runs.append({"backend": b, "impl": impl, "rep": rep, "error": p.stderr[-2000:]})
                    continue
                r = json.loads(line[0][7:])
                r["rep"], r["process_wall_s"] = rep, time.perf_counter() - t
                runs.append(r)
                print(f"rep {rep} {b:8s} {impl:8s} ok ({r['process_wall_s']:.1f}s)", flush=True)
    env["XLA_FLAGS_requested"] = xla
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = {"environment": env, "runs": runs}
    Path(f"{out}_raw.json").write_text(json.dumps(raw, indent=1))
    Path(f"{out}_summary.md").write_text(summarize(raw))
    print(f"wrote {out}_raw.json and {out}_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
