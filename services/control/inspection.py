"""Read-only inspection of recorded runs: checkpoint weights, activations of a checkpoint on a val sample,
per-sample loss and confusion matrix recorded during training, inference from a checkpoint.

Nothing here writes to the store or starts training. Weights, per-sample loss and the confusion matrix are values
that were recorded; activations and inference run a forward pass of the stored graph with the checkpoint's weights,
and say so in their provenance. Every response is bounded (MAX_ELEMS values) and carries provenance."""
from __future__ import annotations

import base64
import binascii
import io
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from artifact_store import ArtifactStore
from graph_core.lower import GraphModule, capture_activations, lower_graph
from graph_core.schema import Graph
from graph_core.validate import validate
from worker.dataset import preprocess_image
from worker.train import load_checkpoint

MAX_ELEMS = 40_000
MAX_WORST = 200


class InspectError(Exception):
    """A request that is wrong (unknown node, sample out of range), as opposed to data that is absent."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.message, self.status = message, status


def _stats(t: torch.Tensor) -> dict[str, Any]:
    f = t.detach().float()
    return {"min": float(f.min()), "max": float(f.max()), "mean": float(f.mean()), "std": float(f.std()) if f.numel() > 1 else 0.0,
            "count": int(f.numel())}


def _lst(t: torch.Tensor) -> list:
    return np.round(t.detach().float().numpy(), 6).tolist()


def _slice(t: torch.Tensor, offset: int, limit: int | None) -> tuple[torch.Tensor, dict[str, int | bool]]:
    """Bounded slice along dim 0 (rank >= 1). limit is clamped so the slice has at most MAX_ELEMS values."""
    n = t.shape[0]
    rest = max(1, t[0].numel()) if t.dim() > 1 else 1
    cap = max(1, MAX_ELEMS // rest)
    limit = cap if limit is None else max(1, min(limit, cap))
    offset = max(0, min(offset, max(0, n - 1)))
    end = min(n, offset + limit)
    return t[offset:end], {"offset": offset, "limit": end - offset, "of": n, "truncated": end - offset < n}


class RunData:
    """Everything stored about one run that inspection needs."""

    def __init__(self, store: ArtifactStore, run_id: str):
        row = store.get_run(run_id)
        if row is None:
            raise InspectError(f"unknown run '{run_id}'", 404)
        self.store, self.run_id, self.row = store, run_id, row
        gs = store.artifacts(run_id, "graph")
        self.graph = Graph.model_validate(json.loads(store.read_artifact(gs[0]["sha256"]))) if gs else None
        self.checkpoints = [c for c in store.artifacts(run_id, "checkpoint")]
        sp = store.artifacts(run_id, "split")
        self.split = json.loads(store.read_artifact(sp[0]["sha256"])) if sp else None

    @property
    def graph_hash(self) -> str:
        return self.row["graph_hash"]

    def pick_checkpoint(self, step: int | None) -> dict | None:
        if not self.checkpoints:
            return None
        if step is None:
            return max(self.checkpoints, key=lambda c: (c["step"] or 0, c["id"]))
        m = [c for c in self.checkpoints if c["step"] == step]
        if not m:
            raise InspectError(f"run {self.run_id} has no checkpoint at step {step} (have {sorted({c['step'] for c in self.checkpoints})})")
        return m[-1]  # a partial checkpoint at the same step as a complete one is the later record

    def prov(self, ck: dict | None = None, **extra) -> dict[str, Any]:
        p = {"runId": self.run_id, "graphHash": self.graph_hash, "checkpointStep": ck["step"] if ck else None,
             "checkpointStatus": ck["status"] if ck else None, "checkpointSha256": ck["sha256"] if ck else None}
        p.update(extra)
        return p


def checkpoint_list(rd: RunData) -> list[dict[str, Any]]:
    return [{"step": c["step"], "epoch": c["meta"].get("epoch"), "status": c["status"], "sha256": c["sha256"], "size": c["size"],
             "graphHash": c["meta"].get("graph_hash"), "createdAt": c["created_at"]} for c in rd.checkpoints]


def _unavailable(rd: RunData | None, kind: str, reason: str, message: str, **prov) -> dict[str, Any]:
    # runStatus lets a client ask again while an active run may still record the value (e.g. its first checkpoint)
    return {"available": False, "kind": kind, "reason": reason, "message": message, "runStatus": rd.row["status"] if rd else None,
            "provenance": rd.prov(None, **prov) if rd else {"runId": None, **prov}}


_MODELS: "OrderedDict[tuple[str, str], GraphModule]" = OrderedDict()


def _model_for(rd: RunData, ck: dict) -> GraphModule:
    key = (rd.run_id, ck["sha256"])
    if key in _MODELS:
        _MODELS.move_to_end(key)
        return _MODELS[key]
    model = lower_graph(rd.graph)
    model.load_state_dict(load_checkpoint(rd.store.path_of(ck["sha256"]))["model"])
    model.eval()
    _MODELS[key] = model
    while len(_MODELS) > 4:
        _MODELS.popitem(last=False)
    return model


def _input_hw(rd: RunData) -> tuple[int, int]:
    report = validate(rd.graph)
    model_ids = sorted(n.id for n in rd.graph.nodes if n.type == "core.tensor_input")
    it = report.output_types[model_ids[0]]["value"]
    return it.shape[2], it.shape[3]


def val_sample_image_path(rd: RunData, index: int) -> Path:
    if rd.split is None:
        raise InspectError("this run has no recorded split yet (the worker has not loaded the dataset)", 409)
    if not 0 <= index < len(rd.split["val"]):
        raise InspectError(f"validation sample index {index} out of range 0..{len(rd.split['val']) - 1}")
    root = Path(rd.split["root"]).resolve()
    p = (root / rd.split["val"][index]).resolve()
    if root not in p.parents:
        raise InspectError("recorded sample path escapes the dataset root", 400)
    return p


def sample_list(rd: RunData) -> list[dict[str, Any]]:
    if rd.split is None:
        return []
    cls = rd.split["classes"]
    return [{"index": i, "id": f, "label": lab, "labelName": cls[lab]} for i, (f, lab) in enumerate(zip(rd.split["val"], rd.split["val_labels"]))]


def _sample_tensor(rd: RunData, index: int) -> torch.Tensor:
    h, w = _input_hw(rd)
    with Image.open(val_sample_image_path(rd, index)) as im:
        return preprocess_image(im, h, w).unsqueeze(0)


# ------------------------------------------------------------------------------------------ weights
def inspect_weights(rd: RunData, node: str, step: int | None, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
    ck = rd.pick_checkpoint(step)
    if rd.graph is None or all(n.id != node for n in rd.graph.nodes):
        raise InspectError(f"node '{node}' is not in the graph of run {rd.run_id}")
    if ck is None:
        return _unavailable(rd, "weights", "not_recorded", f"Run {rd.run_id} ({rd.row['status']}) has no checkpoint yet; no weights are recorded.", nodeId=node)
    sd = load_checkpoint(rd.store.path_of(ck["sha256"]))["model"]
    params = {k[len(node) + 1:]: v for k, v in sd.items() if k.startswith(node + ".")}
    if not params:
        return {"available": False, "kind": "weights", "reason": "no_parameters", "message": f"Node '{node}' has no learnable parameters.",
                "provenance": rd.prov(ck, nodeId=node)}
    out = {}
    for name, t in params.items():
        sl, meta = _slice(t, offset, limit) if t.dim() >= 1 else (t.reshape(1), {"offset": 0, "limit": 1, "of": 1, "truncated": False})
        out[name] = {"shape": list(t.shape), "stats": _stats(t), "slice": meta, "values": _lst(sl)}
    return {"available": True, "kind": "weights", "nodeId": node, "params": out,
            "provenance": rd.prov(ck, nodeId=node, source="values stored in the checkpoint", normalization="none (raw values)")}


# ------------------------------------------------------------------------------------------ activations
def inspect_activations(rd: RunData, node: str, step: int | None, sample: int | None, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
    if rd.graph is None or all(n.id != node for n in rd.graph.nodes):
        raise InspectError(f"node '{node}' is not in the graph of run {rd.run_id}")
    if sample is None:
        raise InspectError("a validation sample index is required")
    ck = rd.pick_checkpoint(step)
    if ck is None:
        return _unavailable(rd, "activations", "not_recorded", f"Run {rd.run_id} ({rd.row['status']}) has no checkpoint yet; nothing to compute activations from.", nodeId=node, sampleIndex=sample)
    x = _sample_tensor(rd, sample)  # validates the index
    model = _model_for(rd, ck)
    with torch.no_grad(), capture_activations(model, [node]) as acts:
        model(x)
    t = acts[node]
    sample_id = rd.split["val"][sample]
    if t.dim() == 4:
        sl, meta = _slice(t[0], offset, limit)
        layout, chan_stats = "feature_maps", [_stats(c) for c in sl]
    else:
        flat = t.reshape(t.shape[0], -1)[0]
        sl, meta = _slice(flat, offset, limit)
        layout, chan_stats = "vector", []
    return {"available": True, "kind": "activations", "nodeId": node, "layout": layout, "shape": list(t.shape), "stats": _stats(t),
            "slice": meta, "channelStats": chan_stats, "values": _lst(sl),
            "provenance": rd.prov(ck, nodeId=node, sampleIndex=sample, sampleId=sample_id,
                                  source="forward pass of the run's graph with the checkpoint's weights on this validation sample",
                                  normalization="none (raw values)")}


# ------------------------------------------------------------------------------------------ recorded validation results
def _val_detail(rd: RunData, epoch: int | None):
    ev = rd.store.events(rd.run_id, -1, ("val_detail",))
    if not ev:
        return None
    if epoch is None:
        return ev[-1]
    m = [e for e in ev if e["data"]["epoch"] == epoch]
    if not m:
        raise InspectError(f"run {rd.run_id} has no validation record for epoch {epoch} (have {[e['data']['epoch'] for e in ev]})")
    return m[0]


def inspect_sample_loss(rd: RunData, epoch: int | None, limit: int = 12) -> dict[str, Any]:
    ev = _val_detail(rd, epoch)
    if ev is None or rd.split is None:
        return _unavailable(rd, "sample_loss", "not_recorded", f"Run {rd.run_id} ({rd.row['status']}) has not finished an epoch; per-sample validation loss is not recorded.")
    d, cls = ev["data"], rd.split["classes"]
    limit = max(1, min(limit, MAX_WORST))
    order = sorted(range(len(d["sample_loss"])), key=lambda i: -d["sample_loss"][i])[:limit]
    rows = [{"index": i, "id": rd.split["val"][i], "label": d["label"][i], "labelName": cls[d["label"][i]], "pred": d["pred"][i],
             "predName": cls[d["pred"][i]], "loss": d["sample_loss"][i]} for i in order]
    return {"available": True, "kind": "sample_loss", "worst": rows, "nVal": len(d["sample_loss"]),
            "provenance": rd.prov(None, epoch=d["epoch"], checkpointStep=d["step"], eventSeq=ev["seq"],
                                  source="per-sample validation loss recorded by the worker at the end of this epoch")}


def inspect_confusion(rd: RunData, epoch: int | None) -> dict[str, Any]:
    ev = _val_detail(rd, epoch)
    if ev is None or rd.split is None:
        return _unavailable(rd, "confusion", "not_recorded", f"Run {rd.run_id} ({rd.row['status']}) has not finished an epoch; no confusion matrix is recorded.")
    d = ev["data"]
    return {"available": True, "kind": "confusion", "classes": rd.split["classes"], "matrix": d["confusion"], "axes": "rows = true class, columns = predicted class",
            "provenance": rd.prov(None, epoch=d["epoch"], checkpointStep=d["step"], eventSeq=ev["seq"],
                                  source="validation set, recorded by the worker at the end of this epoch")}


# ------------------------------------------------------------------------------------------ inference
def infer(rd: RunData, step: int | None, sample: int | None, image_b64: str | None) -> dict[str, Any]:
    if (sample is None) == (image_b64 is None):
        raise InspectError("give exactly one of 'sample' (validation index) or 'imageBase64'")
    ck = rd.pick_checkpoint(step)
    if ck is None:
        return _unavailable(rd, "infer", "not_recorded", f"Run {rd.run_id} ({rd.row['status']}) has no checkpoint yet; cannot run inference.")
    h, w = _input_hw(rd)
    if sample is not None:
        x = _sample_tensor(rd, sample)
        src = {"sampleIndex": sample, "sampleId": rd.split["val"][sample], "trueLabel": rd.split["classes"][rd.split["val_labels"][sample]]}
    else:
        try:
            raw = base64.b64decode(image_b64.split(",", 1)[-1], validate=True)
            with Image.open(io.BytesIO(raw)) as im:
                x = preprocess_image(im, h, w).unsqueeze(0)
        except (binascii.Error, OSError, ValueError) as e:
            raise InspectError(f"could not decode the uploaded image: {e}")
        src = {"sampleIndex": None, "sampleId": "uploaded image", "trueLabel": None}
    model = _model_for(rd, ck)
    with torch.no_grad():
        logits = model(x)[0]
    probs = torch.softmax(logits, 0)
    classes = rd.split["classes"] if rd.split else [str(i) for i in range(len(logits))]
    top = int(probs.argmax())
    return {"available": True, "kind": "infer", "classes": classes, "logits": _lst(logits), "probabilities": _lst(probs),
            "predicted": top, "predictedName": classes[top],
            "provenance": rd.prov(ck, source="forward pass with the checkpoint's weights; probabilities = softmax(logits)",
                                  preprocessing=f"RGB, bilinear resize to {h}x{w}, scaled to [-1, 1]", **src)}
