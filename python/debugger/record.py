"""Reading what a procedure run actually recorded: the scrubber (every optimizer step, which of them have captured intermediate values), wire values,
gradients of captured steps. Historical values that were not captured are reported as "not recorded" together with the way to obtain them
(a capture-and-rerun, which is deterministic and verified against the original); nothing is reconstructed from final outputs."""
from __future__ import annotations

import json
from typing import Any

from artifact_store import ArtifactStore

from .capture import load_payload, tensor_summary

MAX_VALUES = 4096


def _graph_of(store: ArtifactStore, run_id: str):
    from graph_core.schema import Graph

    gs = store.artifacts(run_id, "graph")
    return Graph.model_validate(json.loads(store.read_artifact(gs[0]["sha256"]))) if gs else None


def scrubber(store: ArtifactStore, run_id: str) -> dict[str, Any]:
    row = store.get_run(run_id)
    steps = {}
    for e in store.events(run_id, -1, ("train_step",)):
        d = e["data"]
        steps[d["step"]] = {"step": d["step"], "epoch": d["epoch"], "loss": d["loss"], "lr": d.get("lr"), "gradNorm": d.get("grad_norm"), "captured": False, "checkpoint": False}
    for a in store.artifacts(run_id, "capture"):
        if a["step"] in steps:
            steps[a["step"]]["captured"] = True
            steps[a["step"]]["captureNodes"] = len(a["meta"].get("nodes", []))
    for a in store.artifacts(run_id, "checkpoint"):
        if a["step"] in steps and a["status"] != "pruned":
            steps[a["step"]]["checkpoint"] = True
    breakpoints = [{"step": e["data"].get("step"), **e["data"]} for e in store.events(run_id, -1, ("breakpoint_hit", "assertion_failed"))]
    return {"runId": run_id, "status": row["status"], "graphHash": row["graph_hash"], "steps": [steps[k] for k in sorted(steps)], "breakpointsHit": breakpoints,
            "captureSteps": sorted(a["step"] for a in store.artifacts(run_id, "capture")), "provenance": {"runId": run_id, "source": "events and capture artifacts recorded by the worker"}}


def _capture(store: ArtifactStore, run_id: str, step: int):
    arts = [a for a in store.artifacts(run_id, "capture") if a["step"] == step]
    return (arts[-1], load_payload(store.read_artifact(arts[-1]["sha256"]))) if arts else (None, None)


def not_recorded(run_id: str, step: int, what: str, nodes: list[str] | None = None) -> dict[str, Any]:
    return {"available": False, "reason": "not_recorded", "runId": run_id, "step": step,
            "message": f"{what} was not recorded at optimizer step {step} of run {run_id}. Nothing is shown rather than a reconstructed value.",
            "action": {"kind": "capture_and_rerun", "label": "Capture and rerun",
                       "description": "Re-execute the run deterministically (from the nearest checkpoint) with a capture at this step; the rerun is verified against the original losses.",
                       "endpoint": f"/api/runs/{run_id}/debug/capture", "body": {"step": step, "nodes": nodes}}}


def wire_value(store: ArtifactStore, run_id: str, step: int, node: str, port: str | None = None, limit: int = 64) -> dict[str, Any]:
    art, payload = _capture(store, run_id, step)
    if payload is None:
        return not_recorded(run_id, step, f"the value on {node}", [node])
    t = payload["activations"].get(node)
    if t is None:
        return not_recorded(run_id, step, f"the value of node '{node}' (the capture at this step covered {len(payload['activations'])} other nodes)", [node])
    if node in payload["truncated"]:
        return {"available": False, "reason": "over_budget", "runId": run_id, "step": step, "message": f"'{node}' ({payload['truncated'][node]} elements) exceeded the capture budget and was dropped."}
    out = {"available": True, "runId": run_id, "step": step, "node": node, "port": port, "summary": tensor_summary(t, min(limit, MAX_VALUES)),
           "provenance": {"runId": run_id, "step": step, "graphHash": store.get_run(run_id)["graph_hash"], "captured": True, "micro_batch": art["meta"].get("micro_batch", 0),
                          "note": payload["note"], "sha256": art["sha256"]}}
    if t.dim() >= 1:
        out["rows"] = t[:min(8, t.shape[0])].double().flatten(1).tolist() if t.dim() > 1 else None
    return out


def gradient_view(store: ArtifactStore, run_id: str, step: int, node: str | None = None, param: str | None = None, limit: int = 64) -> dict[str, Any]:
    art, payload = _capture(store, run_id, step)
    if payload is None:
        return not_recorded(run_id, step, "gradients", [node] if node else None)
    prov = {"runId": run_id, "step": step, "graphHash": store.get_run(run_id)["graph_hash"], "captured": True, "micro_batch": art["meta"].get("micro_batch", 0), "note": payload["note"]}
    if param:
        g = payload["param_grads"].get(param)
        if g is None:
            return {"available": False, "reason": "no_such_gradient", "message": f"no gradient for parameter '{param}' in the capture", "provenance": prov}
        return {"available": True, "kind": "parameter", "name": param, "summary": tensor_summary(g, limit), "provenance": prov}
    if node is not None:
        g = payload["gradients"].get(node)
        pg = {n: tensor_summary(t, 8) for n, t in payload["param_grads"].items() if n == node or n.startswith(node + ".")}
        if g is None and not pg:
            return {"available": False, "reason": "no_gradient", "message": f"no gradient was captured for '{node}': it is not on a path to the loss, is non-differentiable, or was outside the capture scope", "provenance": prov}
        return {"available": True, "kind": "node", "node": node, "output": tensor_summary(g, limit) if g is not None else None, "parameters": pg, "provenance": prov}
    return {"available": True, "kind": "overview", "nodes": {n: tensor_summary(t, 0) for n, t in payload["gradients"].items()},
            "parameters": {n: tensor_summary(t, 0) for n, t in payload["param_grads"].items()}, "provenance": prov}
