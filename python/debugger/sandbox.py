"""Sandbox interventions (VISION 13.3, A36): branch from a recorded step of a procedure run, change a value or a hyper-parameter, and rerun in isolation.

How the original stays immutable: the branch is built from a COPY of the state at the step (nearest checkpoint, replayed deterministically in memory to the step
and verified against the recorded losses). Nothing is written to the original run: the sandbox result is stored as a new run of kind 'sandbox'. The original
run's row, events and artifacts are fingerprinted before and after and the two fingerprints are reported (and tested) to be equal.

Supported interventions (anything else is rejected, not ignored):
  {"kind": "hyper", "path": "optimizer.lr" | "optimizer.weight_decay" | "optimizer.momentum" | "clip.max_norm" | "clip.value", "value": number}
  {"kind": "activation", "node": id, "op": "set" | "scale" | "add" | "zero", "index": [..] | null, "value": number}   (first micro-batch of the branched step only)
  {"kind": "parameter", "name": "fc.weight", "op": "set" | "scale" | "add" | "zero", "index": [..] | null, "value": number}   (before the branched step)
An activation or parameter intervention changes execution inside the sandbox on purpose; that is what is being tested."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any

import torch

from artifact_store import ArtifactStore
from graph_core.schema import Graph
from training.spec import CaptureSpec, ProcedureSpec
from training.trainer import Trainer

from .capture import load_payload

MAX_REPLAY = 3000
MAX_FORWARD = 50
HYPERS = {"optimizer.lr", "optimizer.weight_decay", "optimizer.momentum", "clip.max_norm", "clip.value"}
OPS = {"set", "scale", "add", "zero"}


class SandboxError(Exception):
    pass


def fingerprint(store: ArtifactStore, run_id: str) -> dict[str, Any]:
    """Hash of everything recorded about a run: row, events, artifact records, and the bytes of every artifact file."""
    row = store.get_run(run_id)
    h = hashlib.sha256()
    h.update(json.dumps({k: row[k] for k in ("status", "error", "graph_hash", "config", "updated_at", "created_at")}, sort_keys=True).encode())
    n_events = 0
    for e in store.events(run_id, -1):
        h.update(json.dumps([e["seq"], e["type"], e["node_id"], e["data"]], sort_keys=True).encode())
        n_events += 1
    arts = store.artifacts(run_id)
    for a in arts:
        h.update(json.dumps([a["sha256"], a["kind"], a["status"], a["step"], a["meta"]], sort_keys=True).encode())
        h.update(store.read_artifact(a["sha256"]))
    return {"sha256": h.hexdigest(), "events": n_events, "artifacts": len(arts), "status": row["status"]}


def _load(store: ArtifactStore, run_id: str) -> tuple[Graph, ProcedureSpec, dict[str, Any]]:
    row = store.get_run(run_id)
    if row is None or row["config"].get("kind") != "procedure":
        raise SandboxError(f"run '{run_id}' is not a training-procedure run")
    gs = store.artifacts(run_id, "graph")
    graph = Graph.model_validate(json.loads(store.read_artifact(gs[0]["sha256"])))
    return graph, ProcedureSpec.model_validate(row["config"]["procedure"]), row


def _plain(spec: ProcedureSpec) -> ProcedureSpec:
    """The same training trajectory without anything that only observes or stops it (breakpoints, probes, captures, checkpoint writing)."""
    s = spec.model_copy(deep=True)
    s.breakpoints, s.probes, s.capture, s.max_optimizer_steps, s.watch = [], [], None, None, None
    s.checkpoint.every = None
    s.checkpoint.best = None
    return s


def reconstruct(store: ArtifactStore, run_id: str, target_step: int, graph: Graph, spec: ProcedureSpec) -> tuple[bytes, dict[str, Any]]:
    """State after `target_step` optimizer steps of the original run: nearest checkpoint at or before it, replayed deterministically. The replay is checked against the recorded losses."""
    cks = [c for c in store.artifacts(run_id, "checkpoint") if c["status"] != "pruned" and c["step"] is not None and c["step"] <= target_step]
    ck = max(cks, key=lambda c: (c["step"], c["id"])) if cks else None
    spec = _plain(spec)
    t = Trainer(graph, spec)
    start = 0
    if ck is not None:
        t.load_state(Trainer.read_state(store.read_artifact(ck["sha256"])), strict=True)
        start = ck["step"]
    if target_step - start > MAX_REPLAY:
        raise SandboxError(f"replaying {target_step - start} steps from the nearest checkpoint exceeds the limit of {MAX_REPLAY}; checkpoint closer to the step")
    if target_step > start:
        t.run(until_opt_step=target_step)
    if t.opt_step != target_step:
        raise SandboxError(f"could not reach optimizer step {target_step} (the run stops at {t.opt_step})")
    recorded = {e["data"]["step"]: e["data"]["loss"] for e in store.events(run_id, -1, ("train_step",))}
    replayed = {h["step"]: h["loss"] for h in t.history}
    same = sum(1 for s, v in replayed.items() if recorded.get(s) == v)
    info = {"checkpoint": ({"step": ck["step"], "sha256": ck["sha256"]} if ck else None), "replayedSteps": len(replayed), "replayIdenticalToRecorded": same == len(replayed),
            "replayCompared": len(replayed), "startedFrom": "checkpoint" if ck else "seeded initialisation"}
    return t.checkpoint_bytes(), info


def _edit(t: torch.Tensor, op: str, index, value) -> tuple[torch.Tensor, float | None, float | None]:
    out = t.clone()
    sel = tuple(index) if index is not None else Ellipsis
    before = float(out[sel].detach().flatten()[0]) if out[sel].numel() else None
    cur = out[sel]
    new = {"set": torch.full_like(cur, value) if value is not None else cur, "scale": cur * value if value is not None else cur, "add": cur + (value or 0.0), "zero": torch.zeros_like(cur)}[op]
    out[sel] = new
    after = float(out[sel].detach().flatten()[0]) if out[sel].numel() else None
    return out, before, after


def _validate(interventions: list[dict[str, Any]]) -> None:
    if not interventions:
        raise SandboxError("give at least one intervention")
    for iv in interventions:
        k = iv.get("kind")
        if k == "hyper":
            if iv.get("path") not in HYPERS or not isinstance(iv.get("value"), (int, float)):
                raise SandboxError(f"hyper intervention needs path in {sorted(HYPERS)} and a numeric value")
        elif k in ("activation", "parameter"):
            if iv.get("op") not in OPS or ("node" if k == "activation" else "name") not in iv:
                raise SandboxError(f"{k} intervention needs op in {sorted(OPS)} and a {'node' if k == 'activation' else 'name'}")
            if iv["op"] != "zero" and not isinstance(iv.get("value"), (int, float)):
                raise SandboxError(f"{k} intervention op '{iv['op']}' needs a numeric value")
        else:
            raise SandboxError(f"unknown intervention kind '{k}' (supported: hyper, activation, parameter)")


def _run_variant(graph: Graph, spec: ProcedureSpec, state: bytes, step: int, k: int, interventions: list[dict[str, Any]], capture_nodes: list[str] | None) -> dict[str, Any]:
    cap = _plain(spec)
    cap.capture = CaptureSpec(steps=[step], nodes=capture_nodes)
    cap.checkpoint.every = None
    captured: dict[str, Any] = {}
    t = Trainer(graph, cap, on_capture=lambda s, data, meta: captured.update(load_payload(data)))
    t.load_state(Trainer.read_state(state), strict=False)
    changed: list[dict[str, Any]] = []
    start_params = {n: p.detach().clone() for n, p in t.model.named_parameters()}
    handles = []
    for iv in interventions:
        if iv["kind"] == "hyper":
            sec, key = iv["path"].split(".")
            if sec == "optimizer":
                changed.append({"kind": "hyper", "target": iv["path"], "before": t.opt.param_groups[0].get(key), "after": iv["value"]})
                for g in t.opt.param_groups:
                    g[key] = iv["value"]
                if key == "lr" and t.sched is not None:
                    t.sched.base_lrs = [iv["value"]] * len(t.sched.base_lrs)
                    for g in t.opt.param_groups:
                        g["initial_lr"] = iv["value"]
                setattr(t.spec.optimizer, key, iv["value"])
            else:
                changed.append({"kind": "hyper", "target": iv["path"], "before": getattr(t.spec.clip, key), "after": iv["value"]})
                setattr(t.spec.clip, key, iv["value"])
        elif iv["kind"] == "parameter":
            p = dict(t.model.named_parameters()).get(iv["name"])
            if p is None:
                raise SandboxError(f"unknown parameter '{iv['name']}' (parameters: {sorted(dict(t.model.named_parameters()))[:8]}...)")
            with torch.no_grad():
                new, b, a = _edit(p.data, iv["op"], iv.get("index"), iv.get("value"))
                p.data.copy_(new)
            changed.append({"kind": "parameter", "target": iv["name"], "index": iv.get("index"), "op": iv["op"], "before": b, "after": a})
        else:
            mod = t.model._modules.get(iv["node"])
            if mod is None:
                raise SandboxError(f"unknown node '{iv['node']}'")
            rec: dict[str, Any] = {"kind": "activation", "target": iv["node"], "index": iv.get("index"), "op": iv["op"]}
            fired = {"done": False}

            def hook(m, a, out, iv=iv, rec=rec, fired=fired):
                if fired["done"] or not isinstance(out, torch.Tensor):
                    return None
                fired["done"] = True
                new, b, af = _edit(out, iv["op"], iv.get("index"), iv.get("value"))
                rec["before"], rec["after"] = b, af
                return new

            handles.append(mod.register_forward_hook(hook))
            changed.append(rec)
    try:
        t.run(until_opt_step=step - 1 + k)
    finally:
        for h in handles:
            h.remove()
    end_params = {n: p.detach() for n, p in t.model.named_parameters()}
    dist = float(torch.sqrt(sum(((end_params[n] - start_params[n]) ** 2).sum() for n in end_params)))
    return {"losses": [h["loss"] for h in t.history], "gradNorms": [h["grad_norm"] for h in t.history], "lrs": [h["lr"] for h in t.history], "paramDistanceMoved": dist,
            "changed": changed, "capture": captured, "finalParams": {n: v.clone() for n, v in end_params.items()}, "stoppedBy": t.stopped_by}


def branch(store: ArtifactStore, run_id: str, step: int, interventions: list[dict[str, Any]], steps_forward: int = 1, capture_nodes: list[str] | None = None, label: str = "") -> dict[str, Any]:
    if step < 1:
        raise SandboxError("step must be >= 1 (the branch changes what happens AT that optimizer step)")
    if not 1 <= steps_forward <= MAX_FORWARD:
        raise SandboxError(f"steps_forward must be 1..{MAX_FORWARD}")
    _validate(interventions)
    graph, spec, row = _load(store, run_id)
    before = fingerprint(store, run_id)
    state, replay = reconstruct(store, run_id, step - 1, graph, spec)
    base = _run_variant(graph, spec, state, step, steps_forward, [], capture_nodes)
    var = _run_variant(graph, spec, state, step, steps_forward, interventions, capture_nodes)
    after = fingerprint(store, run_id)
    ba, va = base["capture"].get("activations", {}), var["capture"].get("activations", {})
    act_diff = {n: float((va[n].double() - ba[n].double()).abs().max()) for n in ba if n in va and ba[n].shape == va[n].shape and ba[n].numel()}
    bg, vg = base["capture"].get("param_grads", {}), var["capture"].get("param_grads", {})
    grad_diff = {n: float((vg[n].double() - bg[n].double()).norm()) for n in bg if n in vg}
    d_params = float(torch.sqrt(sum(((var["finalParams"][n] - base["finalParams"][n]) ** 2).sum() for n in var["finalParams"])))
    l0, l1 = base["losses"][0], var["losses"][0]
    sid = "sbx-" + uuid.uuid4().hex[:8]
    result = {
        "sandboxRunId": sid, "parent": {"runId": run_id, "step": step, "graphHash": row["graph_hash"], "reconstruction": replay}, "label": label,
        "interventions": interventions, "changed": var["changed"],
        "heldFixed": ["weights, buffers and optimizer state at the end of optimizer step %d (from %s)" % (step - 1, replay["startedFrom"]), "data order and the contents of every batch",
                      "the random state", "the graph and every setting not listed under 'changed'"],
        "stepsForward": steps_forward,
        "baseline": {k: base[k] for k in ("losses", "gradNorms", "lrs", "paramDistanceMoved")}, "branch": {k: var[k] for k in ("losses", "gradNorms", "lrs", "paramDistanceMoved")},
        "diff": {"lossAtStep": {"baseline": l0, "branch": l1, "delta": l1 - l0}, "lossPerStep": [{"step": step + i, "baseline": a, "branch": b, "delta": b - a} for i, (a, b) in enumerate(zip(base["losses"], var["losses"]))],
                 "paramsL2DistanceBranchVsBaseline": d_params, "activationMaxAbsDiff": act_diff, "paramGradL2Diff": grad_diff,
                 "gradNormAtStep": {"baseline": base["gradNorms"][0], "branch": var["gradNorms"][0]}},
        "outcome": f"loss at optimizer step {step}: {l0:.6g} (original) -> {l1:.6g} (branch), change {l1 - l0:+.3g}; after {steps_forward} step(s) the parameters differ by L2 {d_params:.4g}",
        "immutability": {"originalRunUnchanged": before == after, "fingerprintBefore": before, "fingerprintAfter": after,
                         "note": "the original run's row, events and artifact bytes were hashed before and after the sandbox ran"},
        "createdAt": time.time(),
    }
    # persist as a NEW run of kind 'sandbox'; the original is only read
    store.create_run(sid, row["graph_hash"], {"kind": "sandbox", "parent": run_id, "step": step, "label": label, "interventions": interventions})
    for st in ("preparing", "running", "completed"):
        store.set_status(sid, st)
    store.append_event(sid, 0, time.time(), "sandbox_result", row["graph_hash"], None, result)
    store.add_artifact(sid, "sandbox_result", json.dumps(result).encode(), "complete", step, {"parent": run_id})
    return result
