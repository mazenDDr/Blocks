"""Run an `rl` graph in a worker process: validate, build the spec, train DQN, record events and artifacts.

Events: run_queued, run_preparing, validation_error, run_started (library versions, spec, seed), rl_setup (wrapper chain, autoreset mode, effective reward weights,
update equation), episode_end (unsmoothed, one per training episode), train_update (every `log_interval` gradient steps), target_sync, episode_captured, eval,
cancel_acknowledged, error, run_finished.
Artifacts: rl_episode / rl_frame (bounded captured episodes and real rendered frames), rl_trace (bounded transition-to-update trace), rl_buffer (final replay buffer,
npz), rl_checkpoint (network state dicts at evaluations and the end), rl_eval_report (final evaluation), rl_summary, graph (stored by the submitter)."""
from __future__ import annotations

import json
import traceback
from typing import Any, Callable

from pydantic import BaseModel

from artifact_store import ArtifactStore, IllegalTransition
from graph_core.hashing import semantic_hash
from graph_core.schema import Graph
from graph_core.validate import ExecutionBlocked, require_executable

from .events import Emitter


class RLRunConfig(BaseModel):
    model_config = {"extra": "forbid"}
    kind: str = "rl"
    project_id: str | None = None
    # the run seed: training environment seeds (seed + env index), network initialisation, exploration, buffer sampling. Evaluation seeds are separate (the evaluation node).
    seed: int = 0
    trial: dict | None = None


def _advance(store: ArtifactStore, run_id: str, new: str) -> None:
    try:
        store.set_status(run_id, new)
    except IllegalTransition:
        if store.get_run(run_id)["status"] != "cancelling":
            raise


class StoreSink:
    def __init__(self, store: ArtifactStore, run_id: str, em: Emitter):
        self.store, self.run_id, self.em = store, run_id, em
        self.last: dict[str, Any] = {}

    def emit(self, type_: str, **data: Any) -> None:
        self.em.emit(type_, None, **data)

    def put_artifact(self, kind: str, data: bytes, meta: dict[str, Any]) -> str:
        a = self.store.add_artifact(self.run_id, kind, data, "complete", None, meta)
        return a["sha256"]


def run_rl(graph: Graph, cfg: RLRunConfig, store: ArtifactStore, run_id: str, should_cancel: Callable[[], bool] = lambda: False) -> str:
    graph_hash = semantic_hash(graph)
    em = Emitter(store, run_id, graph_hash)
    em.emit("run_queued", config=cfg.model_dump())

    def finish(status: str, error: str | None = None, **data) -> str:
        store.set_status(run_id, status, error)
        em.emit("run_finished", status=status, error=error, **data)
        return status

    try:
        _advance(store, run_id, "preparing")
        em.emit("run_preparing")
        try:
            report = require_executable(graph)
        except ExecutionBlocked as e:
            for d in e.diagnostics:
                em.emit("validation_error", d.nodeId, **d.to_json())
            return finish("failed", str(e))
        import gymnasium, numpy, torch

        from rl.train import Cancelled, train
        from rl.validate import spec_from_graph

        spec = spec_from_graph(graph)
        _advance(store, run_id, "running")
        em.emit("run_started", kind="rl", order=report.order, libraries={"gymnasium": gymnasium.__version__, "torch": torch.__version__, "numpy": numpy.__version__},
                seed=cfg.seed, trial=cfg.trial, spec=spec.model_dump(mode="json", exclude={"network"}), algorithm="DQN", totalSteps=spec.dqn.total_steps)
        sink = StoreSink(store, run_id, em)
        try:
            res = train(spec, cfg.seed, sink, should_cancel)
        except Cancelled:
            if store.get_run(run_id)["status"] != "cancelling":
                store.set_status(run_id, "cancelling")
            em.emit("cancel_acknowledged")
            store.set_status(run_id, "cancelled")
            em.emit("run_finished", status="cancelled", error=None)
            return "cancelled"
        final = res.final_eval
        store.add_artifact(run_id, "rl_eval_report", json.dumps({"final": final, "history": [{k: v for k, v in e.items() if k not in ("episodes", "capturedEpisodes")} for e in res.evals],
                                                                 "seed": cfg.seed, "evaluationSeeds": spec.eval.seeds}).encode(), "complete", None, {"graph_hash": graph_hash})
        store.add_artifact(run_id, "rl_summary", json.dumps({**res.summary, "components": list(final["componentReturns"]), "graphHash": graph_hash}).encode(), "complete", None, {})
        return finish("completed", summary=res.summary)
    except Exception as e:  # noqa: BLE001
        em.emit("error", message=str(e), traceback=traceback.format_exc())
        return finish("failed", f"{type(e).__name__}: {e}")
