"""Measure real editor/native HTTP latency on declared SYNTHETIC metadata graphs.

No learned values, existing services or user workbenches are used. Timing is
descriptive evidence rather than a pass/fail threshold or performance certification.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import platform
import subprocess
import sys
import editor_smoke as smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="new evidence directory outside the repository")
    parser.add_argument("--chrome")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600:
        parser.error("timeout must be 1–600 seconds")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=smoke.ROOT, text=True).strip()
    environment = {"python": platform.python_version(),
                   "packages": {name: version(name) for name in ("fastapi", "pydantic", "torch", "numpy")},
                   "editorSourceSha256": {str(p.relative_to(smoke.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in (smoke.EDITOR / "src/App.tsx", smoke.EDITOR / "src/graphOutline.ts",
                                                    smoke.EDITOR / "src/components/OpNode.tsx",
                                                    smoke.EDITOR / "src/components/GraphOutline.tsx",
                                                    *(smoke.EDITOR / "src/components" / name for name in (
                                                        "NodeChecklist.tsx", "KeyboardGraphTools.tsx", "GraphInsertionTools.tsx",
                                                        "GraphArrangementTools.tsx", "GraphMovementTools.tsx", "GraphClipboardTools.tsx",
                                                        "Inspector.tsx", "Library.tsx")))},
                   "benchmarkSourceSha256": {str(p.relative_to(smoke.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in (smoke.ROOT / "tools/editor_performance_benchmark.py",
                                                       smoke.EDITOR / "smoke/performance.mjs")}}
    return smoke.run(args, journey=smoke.EDITOR / "smoke/performance.mjs",
                     extra_env={"VOID_BENCHMARK_REVISION": revision, "VOID_BENCHMARK_NATIVE_ENV": json.dumps(environment)},
                     fixture="SYNTHETIC 100/500/1000-node metadata chains; actual browser/native HTTP timings")


if __name__ == "__main__":
    sys.exit(main())
