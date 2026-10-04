"""Run a model graph with an explicit training procedure (Milestone 3) in this worker process.

Events (all carry run id, graph hash, seq, timestamp): run_queued, run_preparing, run_started (procedure, parameters, determinism),
train_step (step = optimizer step, loss, lr, grad_norm, clip decision), validation, epoch_end, scheduler_step, optimizer_trace, checkpoint,
checkpoint_pruned, capture_recorded, probe, early_stopping_check, breakpoint_hit, assertion_failed, resumed, error, run_finished.
Artifacts: graph, procedure, checkpoint (state at a safe boundary), capture (recorded activations/gradients of a step)."""
from __future__ import annotations

import json
import traceback
from typing import Any, Callable

from pydantic import BaseModel, Field

from artifact_store import ArtifactStore, IllegalTransition
from graph_core.hashing import semantic_hash
from graph_core.schema import Graph
from graph_core.validate import ExecutionBlocked
from training.spec import ProcedureSpec
from training.trainer import Trainer

from .events import Emitter


class ProcedureRunConfig(BaseModel):
    model_config = {"extra": "forbid"}
    kind: str = "procedure"
    procedure: dict[str, Any] = Field(default_factory=dict)  # a ProcedureSpec
    project_id: str | None = None
    # continue from a checkpoint of another run (same graph and procedure identity required) instead of starting from the seeded initialisation
    resume_from: dict[str, Any] | None = None  # {"run_id": ..., "step": ...}
    until_step: int | None = None  # stop at this optimizer step (a safe boundary)
    rerun_of: str | None = None  # this run re-executes another run to capture values that were not recorded
    label: str = ""


def _advance(store: ArtifactStore, run_id: str, new: str) -> None:
    try:
        store.set_status(run_id, new)
    except IllegalTransition:
        if store.get_run(run_id)["status"] != "cancelling":
            raise


def find_checkpoint(store: ArtifactStore, run_id: str, max_step: int | None = None, step: int | None = None) -> dict[str, Any] | None:
    cks = [c for c in store.artifacts(run_id, "checkpoint") if c["status"] != "pruned"]
    if step is not None:
        cks = [c for c in cks if c["step"] == step]
    elif max_step is not None:
        cks = [c for c in cks if c["step"] is not None and c["step"] <= max_step]
    return max(cks, key=lambda c: (c["step"], c["id"])) if cks else None


def run_procedure(graph: Graph, cfg: ProcedureRunConfig, store: ArtifactStore, run_id: str, should_cancel: Callable[[], bool] = lambda: False) -> str:
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
        spec = ProcedureSpec.model_validate(cfg.procedure)
        store.add_artifact(run_id, "procedure", json.dumps(cfg.procedure, sort_keys=True).encode(), "complete", None, {})

        def on_event(type_: str, node: str | None = None, **data: Any) -> None:
            if type_ == "checkpoint_pruned":
                store.set_artifact_status(run_id, data["sha256"], "pruned", "checkpoint")
            em.emit(type_, node, **data)

        def on_checkpoint(step: int, data: bytes, meta: dict[str, Any]) -> None:
            art = store.add_artifact(run_id, "checkpoint", data, "complete", step, {k: v for k, v in meta.items() if k != "sha256"})
            meta["sha256"] = art["sha256"]

        def on_capture(step: int, data: bytes, meta: dict[str, Any]) -> None:
            store.add_artifact(run_id, "capture", data, "complete", step, meta)

        try:
            trainer = Trainer(graph, spec, emit=on_event, on_checkpoint=on_checkpoint, on_capture=on_capture, cancel=should_cancel)
        except ExecutionBlocked as e:
            for d in e.diagnostics:
                em.emit("validation_error", d.nodeId, **d.to_json())
            return finish("failed", str(e))
        resumed_from = None
        if cfg.resume_from:
            src = find_checkpoint(store, cfg.resume_from["run_id"], max_step=cfg.resume_from.get("max_step"), step=cfg.resume_from.get("step"))
            if src is None:
                raise ValueError(f"run {cfg.resume_from['run_id']} has no usable checkpoint for {cfg.resume_from}")
            trainer.load_state(Trainer.read_state(store.read_artifact(src["sha256"])), strict=True)
            resumed_from = {"run_id": cfg.resume_from["run_id"], "step": src["step"], "sha256": src["sha256"]}
        _advance(store, run_id, "running")
        em.emit("run_started", total_params=trainer.report.total_params, procedure=cfg.procedure, deterministic=spec.deterministic,
                spec_hash=trainer.spec_hash, resumed_from=resumed_from, warnings=[w.to_json() for w in trainer.warnings], rerun_of=cfg.rerun_of,
                data=trainer.train_ds.description, synthetic=trainer.train_ds.synthetic, n_train=len(trainer.train_ds), n_val=len(trainer.val_ds),
                torch=__import__("torch").__version__, steps_per_epoch=trainer._n_micro() // spec.accumulation.steps)
        result = trainer.run(until_opt_step=cfg.until_step)
        if result["stopped_by"] == "cancelled":
            if store.get_run(run_id)["status"] != "cancelling":
                store.set_status(run_id, "cancelling")
            trainer.save_checkpoint("cancelled")
            em.emit("cancel_acknowledged", step=trainer.opt_step)
            store.set_status(run_id, "cancelled")
            em.emit("run_finished", status="cancelled", error=None, **result)
            return "cancelled"
        if cfg.rerun_of:
            _verify_rerun(store, em, cfg, trainer)
        return finish("completed", **result)
    except Exception as e:  # noqa: BLE001
        em.emit("error", message=str(e), traceback=traceback.format_exc())
        return finish("failed", f"{type(e).__name__}: {e}")


def _verify_rerun(store: ArtifactStore, em: Emitter, cfg: ProcedureRunConfig, trainer: Trainer) -> None:
    """A rerun is only evidence about the original run if it reproduced it: compare the recomputed losses with the recorded ones."""
    orig = {e["data"]["step"]: e["data"]["loss"] for e in store.events(cfg.rerun_of, -1, ("train_step",))}
    mine = {h["step"]: h["loss"] for h in trainer.history}
    common = sorted(set(orig) & set(mine))
    same = [s for s in common if orig[s] == mine[s]]
    em.emit("rerun_verified", rerun_of=cfg.rerun_of, compared_steps=common, identical_steps=same, identical=len(common) > 0 and len(same) == len(common),
            note="recomputed per-step losses compared bit for bit with the original run's recorded losses")
