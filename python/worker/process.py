"""Run each training in its own OS process (multiprocessing spawn).

The worker writes events and artifacts straight to the SQLite/CAS store, so the run keeps
going and stays inspectable if the submitting process goes away."""
from __future__ import annotations

import json
import multiprocessing as mp
import uuid
from pathlib import Path

from artifact_store import ArtifactStore, IllegalTransition
from graph_core.hashing import semantic_hash
from graph_core.schema import Graph
from graph_core.validate import require_executable

from .procedure_run import ProcedureRunConfig, run_procedure
from .tabular_run import TabularRunConfig, run_tabular
from .train import RunConfig, run_training


def _child(graph_json: dict, cfg_json: dict, root: str, run_id: str, cancel_event) -> None:
    store = ArtifactStore(root)

    def should_cancel() -> bool:  # in-process event, or a cancel recorded in the DB by any process (e.g. after a control restart)
        return cancel_event.is_set() or store.get_run(run_id)["status"] == "cancelling"

    graph = Graph.model_validate(graph_json)
    if graph.graphKind == "tabular":
        run_tabular(graph, TabularRunConfig.model_validate(cfg_json), store, run_id, should_cancel)
    elif cfg_json.get("kind") == "procedure":
        run_procedure(graph, ProcedureRunConfig.model_validate(cfg_json), store, run_id, should_cancel)
    else:
        run_training(graph, RunConfig.model_validate(cfg_json), store, run_id, should_cancel)


class RunHandle:
    def __init__(self, run_id: str, store: ArtifactStore, process, cancel_event):
        self.run_id, self.store, self.process, self._cancel = run_id, store, process, cancel_event

    def cancel(self) -> None:
        """Ask the worker to stop at the next batch boundary."""
        self._cancel.set()
        try:
            self.store.set_status(self.run_id, "cancelling")
        except IllegalTransition:
            pass  # already finished, or already cancelling

    def wait(self, timeout: float | None = None) -> str:
        self.process.join(timeout)
        return self.store.get_run(self.run_id)["status"]

    def is_alive(self) -> bool:
        return self.process.is_alive()


def submit_run(graph: Graph, cfg: RunConfig | TabularRunConfig | ProcedureRunConfig, workbench: str | Path = ".workbench", run_id: str | None = None) -> RunHandle:
    """Validate (raises ExecutionBlocked), record the run and its exact graph, and start the worker process."""
    require_executable(graph)
    store = ArtifactStore(workbench)
    run_id = run_id or uuid.uuid4().hex[:12]
    store.create_run(run_id, semantic_hash(graph), cfg.model_dump())
    store.add_artifact(run_id, "graph", json.dumps(graph.to_json(), sort_keys=True).encode(), "complete", None, {"graph_hash": semantic_hash(graph)})
    ctx = mp.get_context("spawn")
    cancel = ctx.Event()
    proc = ctx.Process(target=_child, args=(graph.to_json(), cfg.model_dump(), str(workbench), run_id, cancel))
    proc.start()
    return RunHandle(run_id, store, proc, cancel)
