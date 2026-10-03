"""The fixed training procedure: seeded CrossEntropy + SGD/Adam over an image-folder dataset.

Per step: forward -> cross-entropy -> zero_grad -> backward -> optimizer.step.
Cancellation is cooperative and checked before every batch."""
from __future__ import annotations

import io
import json
import traceback
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from artifact_store import ArtifactStore, IllegalTransition
from graph_core.hashing import semantic_hash
from graph_core.lower import GraphModule, lower_graph
from graph_core.schema import Graph
from graph_core.validate import ExecutionBlocked, require_executable

from .dataset import load_image_folder
from .events import Emitter


class RunConfig(BaseModel):
    data: str
    epochs: int = Field(1, gt=0)
    batch_size: int = Field(16, gt=0)
    lr: float = Field(0.05, gt=0)
    optimizer: Literal["sgd", "adam"] = "sgd"
    momentum: float = Field(0.0, ge=0)
    seed: int = 0
    val_fraction: float = Field(0.2, gt=0, lt=1)
    split_seed: int | None = None  # seed of the train/val split; None means "same as seed"
    project_id: str | None = None  # which saved project this run came from (informational)

    @property
    def effective_split_seed(self) -> int:
        return self.seed if self.split_seed is None else self.split_seed


class UnsupportedGraph(Exception):
    pass


def _checkpoint_bytes(model, opt, step, epoch, graph_hash, run_id, status) -> bytes:
    buf = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "rng": {"torch_cpu": torch.get_rng_state()},
                "step": step, "epoch": epoch, "graph_hash": graph_hash, "run_id": run_id, "status": status}, buf)
    return buf.getvalue()


def load_checkpoint(path) -> dict:
    return torch.load(path, weights_only=True)


def _io_contract(graph: Graph, report, model: GraphModule) -> tuple[int, int]:
    """Worker handles one image input [N,3,H,W] float32 and one logits output [N,C]."""
    if len(model.input_ids) != 1 or len(model.output_ids) != 1:
        raise UnsupportedGraph("training needs exactly one tensor_input and one terminal output (logits)")
    it = report.output_types[model.input_ids[0]]["value"]
    ot = report.output_types[model.output_ids[0]]["output"]
    if it.dtype != "float32" or len(it.shape) != 4 or it.shape[1] != 3 or not all(isinstance(d, int) for d in it.shape[2:]):
        raise UnsupportedGraph(f"input must be [N, 3, H, W] float32, got {list(it.shape)} {it.dtype}")
    if len(ot.shape) != 2 or not isinstance(ot.shape[1], int):
        raise UnsupportedGraph(f"output must be logits [N, C], got {list(ot.shape)}")
    return it.shape[2], ot.shape[1]


def _weight_stats(model: GraphModule, em: Emitter, step: int, epoch: int) -> None:
    for nid, mod in model.named_children():
        ps = list(mod.parameters())
        if ps:
            flat = torch.cat([p.detach().flatten() for p in ps])
            em.emit("weight_stats", nid, epoch=epoch, step=step, l2=float(flat.norm()), mean=float(flat.mean()),
                    std=float(flat.std()), min=float(flat.min()), max=float(flat.max()))


@torch.no_grad()
def _evaluate(model: GraphModule, x, y, batch_size: int, n_classes: int) -> dict:
    """Validation pass: mean loss, accuracy, confusion matrix [true][pred], per-sample loss and prediction."""
    model.eval()
    losses, preds = [], []
    for i in range(0, len(x), batch_size):
        logits = model(x[i:i + batch_size])
        losses.append(F.cross_entropy(logits, y[i:i + batch_size], reduction="none"))
        preds.append(logits.argmax(1))
    model.train()
    loss, pred = torch.cat(losses), torch.cat(preds)
    conf = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    for t, p in zip(y.tolist(), pred.tolist()):
        conf[t, p] += 1
    return {"val_loss": float(loss.mean()), "val_acc": float((pred == y).float().mean()), "confusion": conf.tolist(),
            "sample_loss": [float(v) for v in loss], "pred": pred.tolist(), "label": y.tolist()}


def _advance(store: ArtifactStore, run_id: str, new: str) -> None:
    """Move forward unless a cancel was recorded meanwhile; the first batch-boundary check then honours it."""
    try:
        store.set_status(run_id, new)
    except IllegalTransition:
        if store.get_run(run_id)["status"] != "cancelling":
            raise


def run_training(graph: Graph, cfg: RunConfig, store: ArtifactStore, run_id: str,
                 should_cancel: Callable[[], bool] = lambda: False) -> str:
    """Execute a run whose row already exists. Returns the final status."""
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
        torch.manual_seed(cfg.seed)
        model = lower_graph(graph, report)
        height, n_classes = _io_contract(graph, report, model)
        width = report.output_types[model.input_ids[0]]["value"].shape[3]
        data = load_image_folder(cfg.data, height, width, cfg.effective_split_seed, cfg.val_fraction)
        if len(data.classes) != n_classes:
            raise UnsupportedGraph(f"dataset has {len(data.classes)} classes but the graph outputs {n_classes} logits")
        split = {"root": str(cfg.data), "classes": data.classes, "split_seed": data.split_seed, "val_fraction": data.val_fraction,
                 "dataset_sha256": data.files_sha256, "train": data.train_files, "val": data.val_files,
                 "val_labels": data.y_val.tolist()}
        store.add_artifact(run_id, "split", json.dumps(split).encode(), "complete", None, {"n_train": len(data.train_files), "n_val": len(data.val_files)})
        opt = (torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum) if cfg.optimizer == "sgd"
               else torch.optim.Adam(model.parameters(), lr=cfg.lr))
        model.train()
        _advance(store, run_id, "running")
        em.emit("run_started", total_params=report.total_params, classes=data.classes, n_train=len(data.x_train),
                n_val=len(data.x_val), dataset_sha256=data.files_sha256, torch=torch.__version__,
                split_seed=data.split_seed, val_fraction=data.val_fraction, val_files=data.val_files)

        step, n = 0, len(data.x_train)
        for epoch in range(cfg.epochs):
            order = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed * 100003 + epoch))
            epoch_loss, seen = 0.0, 0
            for b, start in enumerate(range(0, n, cfg.batch_size)):
                if should_cancel():
                    return _cancel(store, em, run_id, model, opt, step, epoch, graph_hash)
                idx = order[start:start + cfg.batch_size]
                loss = F.cross_entropy(model(data.x_train[idx]), data.y_train[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
                epoch_loss += loss.item() * len(idx)
                seen += len(idx)
                em.emit("train_step", step=step, epoch=epoch, batch=b, loss=loss.item())
            ev = _evaluate(model, data.x_val, data.y_val, cfg.batch_size, n_classes)
            em.emit("epoch_end", epoch=epoch, step=step, train_loss=epoch_loss / seen, val_loss=ev["val_loss"], val_acc=ev["val_acc"])
            em.emit("val_detail", epoch=epoch, step=step, confusion=ev["confusion"], sample_loss=ev["sample_loss"],
                    pred=ev["pred"], label=ev["label"])
            _weight_stats(model, em, step, epoch)
            art = store.add_artifact(run_id, "checkpoint", _checkpoint_bytes(model, opt, step, epoch, graph_hash, run_id, "complete"),
                                     "complete", step, {"epoch": epoch, "graph_hash": graph_hash})
            em.emit("checkpoint", **art)
        if should_cancel():  # cancel requested after the last batch: honour it, do not report completed
            return _cancel(store, em, run_id, model, opt, step, cfg.epochs - 1, graph_hash)
        return finish("completed", steps=step)
    except Exception as e:  # noqa: BLE001  (record any failure on the run)
        tb = traceback.format_exc()
        em.emit("error", message=str(e), traceback=tb)
        return finish("failed", f"{type(e).__name__}: {e}")


def _cancel(store, em, run_id, model, opt, step, epoch, graph_hash) -> str:
    if store.get_run(run_id)["status"] != "cancelling":
        store.set_status(run_id, "cancelling")
    em.emit("cancel_acknowledged", step=step)
    art = store.add_artifact(run_id, "checkpoint", _checkpoint_bytes(model, opt, step, epoch, graph_hash, run_id, "partial"),
                             "partial", step, {"epoch": epoch, "graph_hash": graph_hash, "reason": "cancelled"})
    em.emit("checkpoint", **art)
    store.set_status(run_id, "cancelled")
    em.emit("run_finished", status="cancelled", error=None, steps=step)
    return "cancelled"
