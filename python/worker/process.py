"""Run each training in its own OS process (multiprocessing spawn).

The worker writes events and artifacts straight to the SQLite/CAS store, so the run keeps
going and stays inspectable if the submitting process goes away."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
import time
import uuid
from pathlib import Path

from artifact_store import ArtifactStore, IllegalTransition
from graph_core.hashing import semantic_hash
from graph_core.schema import Graph
from graph_core.validate import require_executable

from .agent_run import AgentRunConfig, run_agent
from .rl_run import RLRunConfig, run_rl
from .procedure_run import ProcedureRunConfig, run_procedure
from .tabular_run import TabularRunConfig, run_tabular
from .train import RunConfig, run_training


HEARTBEAT_SECONDS = 5.0


def _keep_alive(store: ArtifactStore, run_id: str) -> None:
    """Claim the run and refresh a heartbeat until this process ends, so a restarted control service can tell a live worker
    from a lost one (ArtifactStore.reconcile_lost_workers)."""
    nonce = uuid.uuid4().hex
    store.lease(run_id, os.getpid(), nonce)

    def beat():
        while True:
            time.sleep(HEARTBEAT_SECONDS)
            try:
                store.heartbeat(run_id, nonce)
            except Exception:  # noqa: BLE001 - a missed beat only makes the run look silent sooner
                pass

    threading.Thread(target=beat, daemon=True, name=f"heartbeat-{run_id}").start()


def _child(graph_json: dict, cfg_json: dict, root: str, run_id: str, cancel_event, resume: dict | None = None) -> None:
    store = ArtifactStore(root)
    _keep_alive(store, run_id)

    def should_cancel() -> bool:  # in-process event, or a cancel recorded in the DB by any process (e.g. after a control restart)
        return cancel_event.is_set() or store.get_run(run_id)["status"] == "cancelling"

    graph = Graph.model_validate(graph_json)
    if graph.graphKind == "agent":
        run_agent(graph, AgentRunConfig.model_validate(cfg_json), store, run_id, should_cancel, resume)
    elif graph.graphKind == "rl":
        run_rl(graph, RLRunConfig.model_validate(cfg_json), store, run_id, should_cancel)
    elif graph.graphKind in ("tabular", "domain"):
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


def submit_run(graph: Graph, cfg: RunConfig | TabularRunConfig | ProcedureRunConfig | AgentRunConfig | RLRunConfig, workbench: str | Path = ".workbench", run_id: str | None = None) -> RunHandle:
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


def submit_resume(run_id: str, resume: dict, workbench: str | Path = ".workbench") -> RunHandle:
    """Resume a paused agent run in a NEW worker process: the thread's state comes from the SQLite checkpoint, not from memory, so this
    also works after the control service (or machine) restarted. The paused -> running transition is atomic, so a double resume is rejected."""
    store = ArtifactStore(workbench)
    row = store.get_run(run_id)
    if row is None:
        raise KeyError(run_id)
    store.set_status(run_id, "running")  # raises IllegalTransition unless the run is paused
    graph_json = json.loads(store.read_artifact(store.artifacts(run_id, "graph")[0]["sha256"]))
    ctx = mp.get_context("spawn")
    cancel = ctx.Event()
    proc = ctx.Process(target=_child, args=(graph_json, row["config"], str(workbench), run_id, cancel, resume))
    proc.start()
    return RunHandle(run_id, store, proc, cancel)
