"""python -m worker.cli <project.json> --data <dir> [--epochs N ...]: train and stream JSON events."""
from __future__ import annotations

import argparse
import json
import sys
import time

from graph_core.project_io import load_project
from graph_core.validate import ExecutionBlocked

from .process import submit_run
from .train import RunConfig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="worker.cli")
    ap.add_argument("project")
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workbench", default=".workbench")
    a = ap.parse_args(argv)

    graph = load_project(a.project).graph
    cfg = RunConfig(data=a.data, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, optimizer=a.optimizer,
                    momentum=a.momentum, seed=a.seed)
    try:
        handle = submit_run(graph, cfg, a.workbench)
    except ExecutionBlocked as e:
        print(json.dumps({"blocked": [d.to_json() for d in e.diagnostics]}, indent=2), file=sys.stderr)
        return 2

    last = -1
    try:
        while True:
            alive = handle.is_alive()  # read before draining so the final events are not missed
            for ev in handle.store.events(handle.run_id, last):
                print(json.dumps(ev), flush=True)
                last = ev["seq"]
            if not alive:
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        handle.cancel()
        handle.wait()
    status = handle.wait()
    print(json.dumps({"run_id": handle.run_id, "status": status, "workbench": a.workbench}))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
