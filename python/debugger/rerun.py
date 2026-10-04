"""Capture-and-rerun: obtain values that a run did not record, by re-executing it deterministically with a capture at the wanted step."""
from __future__ import annotations

from typing import Any

from artifact_store import ArtifactStore
from worker.procedure_run import ProcedureRunConfig


def plan_rerun(store: ArtifactStore, run_id: str, step: int, nodes: list[str] | None = None, max_elements: int = 400_000) -> ProcedureRunConfig:
    row = store.get_run(run_id)
    if row is None or row["config"].get("kind") != "procedure":
        raise ValueError(f"run '{run_id}' is not a training-procedure run")
    cfg = ProcedureRunConfig.model_validate(row["config"])
    proc = dict(cfg.procedure)
    proc["capture"] = {"steps": [step], "nodes": nodes, "max_elements": max_elements}
    proc["checkpoint"] = {**(proc.get("checkpoint") or {}), "every": None}
    cks = [c for c in store.artifacts(run_id, "checkpoint") if c["status"] != "pruned" and c["step"] is not None and c["step"] <= step - 1]
    resume = {"run_id": run_id, "max_step": step - 1} if cks else None
    return ProcedureRunConfig(procedure=proc, project_id=cfg.project_id, resume_from=resume, until_step=step, rerun_of=run_id, label=f"capture of step {step} (rerun of {run_id})")
