"""Executor of the declared training procedure (VISION 10).

The stage list in ProcedureSpec is interpreted in order for every micro-batch of an accumulation window:

    - zero_grad     runs at the START of the window if listed before `forward`, at the END of the window if listed after `optimizer_step`
    - forward, loss, backward   every micro-batch (the loss of a micro-batch is scaled by 1/window_length when accumulation.normalize = "mean")
    - clip, optimizer_step, scheduler_step (timing = optimizer_step)   once per window, after its last micro-batch
    after the window: [validation by cadence] -> [early stopping] -> [checkpoint by cadence]
    at the end of an epoch: [validation by cadence] -> [scheduler with timing epoch] -> [early stopping] -> [checkpoint by cadence]

Checkpoints are written only at these safe boundaries (never inside an accumulation window) and contain everything needed to continue:
parameters and buffers, optimizer state, scheduler state, the torch RNG state, the position in the data order and the early-stopping and
best-metric state. With `deterministic` (CPU, single thread, torch deterministic algorithms) a run resumed from a checkpoint reproduces the
uninterrupted run bit for bit (tests/test_training_procedure.py)."""
from __future__ import annotations

import hashlib
import io
import json
import math
import time
import warnings
from contextlib import contextmanager
from typing import Any, Callable

import torch
import torch.nn as nn

from debugger.capture import StepCapture
from graph_core import diagnostics as D
from graph_core.hashing import semantic_hash
from graph_core.lower import GraphModule, lower_graph
from graph_core.schema import Graph
from graph_core.types import Diagnostic, TensorType
from graph_core.validate import ExecutionBlocked, require_executable

from .data import Dataset, make_datasets
from .losses import build_loss
from .optim import build_optimizer, build_scheduler, manual_update
from .spec import ProcedureSpec, check_procedure

Emit = Callable[..., None]
EXCLUDED_FROM_IDENTITY = ("max_optimizer_steps", "epochs", "capture", "probes", "breakpoints", "watch", "checkpoint")


@contextmanager
def deterministic_cpu(enabled: bool):
    if not enabled:
        yield
        return
    prev_alg, prev_threads = torch.are_deterministic_algorithms_enabled(), torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev_alg)
        torch.set_num_threads(prev_threads)


def spec_identity(spec: ProcedureSpec) -> str:
    d = spec.model_dump(mode="json")
    for k in EXCLUDED_FROM_IDENTITY:
        d.pop(k, None)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


class BreakpointHit(Exception):
    def __init__(self, bp_id: str, detail: str):
        super().__init__(f"breakpoint '{bp_id}': {detail}")
        self.bp_id, self.detail = bp_id, detail


class Trainer:
    def __init__(self, graph: Graph, spec: ProcedureSpec, *, emit: Emit | None = None, datasets: tuple[Dataset, Dataset] | None = None,
                 on_checkpoint: Callable[[int, bytes, dict[str, Any]], None] | None = None,
                 on_capture: Callable[[int, bytes, dict[str, Any]], None] | None = None,
                 cancel: Callable[[], bool] | None = None):
        errs = [d for d in check_procedure(spec) if d.severity == "error"]
        if errs:
            raise ExecutionBlocked(errs)
        self.graph, self.spec = graph, spec
        self.report = require_executable(graph)
        self.graph_hash, self.spec_hash = semantic_hash(graph.model_copy(update={"training": None})), spec_identity(spec)  # the procedure is hashed separately
        self.events: list[dict[str, Any]] = []
        self._emit_ext = emit
        self.on_checkpoint, self.on_capture, self.cancel = on_checkpoint, on_capture, cancel or (lambda: False)
        self.warnings = [d for d in check_procedure(spec) if d.severity == "warning"]

        with deterministic_cpu(spec.deterministic):
            torch.manual_seed(spec.seed)
            self.model: GraphModule = lower_graph(graph, self.report)
        if len(self.model.output_ids) != 1:
            raise ExecutionBlocked([Diagnostic("E_PROC_IO", f"training needs exactly one terminal output (the prediction), the graph has {self.model.output_ids}")])
        self.train_ds, self.val_ds = datasets or make_datasets(spec.data)
        self.input_map = self._input_map()
        out_t = next(iter(self.report.output_types[self.model.output_ids[0]].values()))
        ports = self._loss_port_types(out_t)
        self.loss_fn = build_loss(graph, spec.loss, ports)
        self.stage_index = {k: i for i, k in enumerate(spec.stages)}

        for nid, mod in self.model.named_children():
            if any(nid == f or nid.startswith(f + "/") for f in spec.frozen):
                for p in mod.parameters():
                    p.requires_grad_(False)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        if not self.params:
            raise ExecutionBlocked([Diagnostic("E_PROC_PARAMS", "no trainable parameters (everything is frozen or the graph has none)")])
        self.opt = build_optimizer(self.params, spec.optimizer)
        self.sched = build_scheduler(self.opt, spec.scheduler)
        self.param_names = {id(p): n for n, p in self.model.named_parameters()}

        # position and bookkeeping (all of it goes into a checkpoint)
        self.epoch = self.micro_in_epoch = self.opt_step = self.micro_total = 0
        self.es_best: float | None = None
        self.es_bad = 0
        self.best_value: float | None = None
        self.last_metrics: dict[str, float] = {}
        self.stopped_by: str | None = None
        self.saved: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self._epoch_losses: list[float] = []
        self._breakpoint_snapshot: bytes | None = None
        self._clip_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ setup helpers
    def _input_map(self) -> dict[str, str]:
        m = dict(self.spec.data.inputs)
        if not m:
            if len(self.model.input_ids) != 1:
                raise ExecutionBlocked([Diagnostic("E_PROC_IO", f"the model has inputs {self.model.input_ids}; map each to a dataset field in data.inputs")])
            m = {self.model.input_ids[0]: "x"}
        for nid in self.model.input_ids:
            if nid not in m or m[nid] not in self.train_ds.fields:
                raise ExecutionBlocked([Diagnostic("E_PROC_IO", f"model input '{nid}' has no dataset field (fields: {list(self.train_ds.fields)})")])
        return m

    def _loss_port_types(self, out_t: TensorType) -> dict[str, TensorType]:
        ports: dict[str, TensorType] = {}
        for name, src in self.spec.loss.ports.items():
            if src == "output":
                ports[name] = out_t
            elif src in self.train_ds.fields:
                f = self.train_ds.fields[src]
                ports[name] = TensorType(("N",) + tuple(f.shape[1:]), str(f.dtype).replace("torch.", ""))
            else:
                raise ExecutionBlocked([Diagnostic("E_PROC_LOSS", f"loss port '{name}' reads '{src}', which is neither 'output' nor a dataset field {list(self.train_ds.fields)}")])
        return ports

    def emit(self, type_: str, node: str | None = None, **data: Any) -> None:
        self.events.append({"type": type_, "node": node, **data})
        if self._emit_ext:
            self._emit_ext(type_, node, **data)

    # ------------------------------------------------------------------ data order
    def _order(self, epoch: int) -> torch.Tensor:
        n = len(self.train_ds)
        if not self.spec.data.shuffle:
            return torch.arange(n)
        return torch.randperm(n, generator=torch.Generator().manual_seed(self.spec.seed * 100003 + epoch))

    def _n_micro(self) -> int:
        n, b = len(self.train_ds), self.spec.data.batch_size
        full = n // b if self.spec.data.drop_last else math.ceil(n / b)
        k = self.spec.accumulation.steps
        return (full // k) * k if (self.spec.accumulation.partial_window == "drop" and k > 1) else full

    def _batch(self, order: torch.Tensor, pos: int) -> dict[str, torch.Tensor]:
        b = self.spec.data.batch_size
        return self.train_ds.batch(order[pos * b:(pos + 1) * b])

    # ------------------------------------------------------------------ forward + loss
    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(*[batch[self.input_map[nid]] for nid in self.model.input_ids])

    def _loss(self, out: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        t = {name: (out if src == "output" else batch[src]) for name, src in self.spec.loss.ports.items()}
        return self.loss_fn(**t)

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict[str, Any]:
        return {"param_names": [n for n, p in self.model.named_parameters() if p.requires_grad], "model": self.model.state_dict(), "optimizer": self.opt.state_dict(), "scheduler": self.sched.state_dict() if self.sched else None,
                "rng": {"torch_cpu": torch.get_rng_state()},
                "counters": {"epoch": self.epoch, "micro_in_epoch": self.micro_in_epoch, "opt_step": self.opt_step, "micro_total": self.micro_total},
                "early": {"best": self.es_best, "bad": self.es_bad}, "best_value": self.best_value, "last_metrics": dict(self.last_metrics),
                "graph_hash": self.graph_hash, "spec_hash": self.spec_hash, "torch": str(torch.__version__), "training": self.model.training}

    def checkpoint_bytes(self) -> bytes:
        buf = io.BytesIO()
        torch.save(self.state_dict(), buf)
        return buf.getvalue()

    def load_state(self, state: dict[str, Any], strict: bool = True) -> None:
        if strict and state["graph_hash"] != self.graph_hash:
            raise ValueError(f"checkpoint was written for graph {state['graph_hash'][:12]}, this graph is {self.graph_hash[:12]}")
        if strict and state["spec_hash"] != self.spec_hash:
            raise ValueError("checkpoint was written for a different training procedure (optimizer, loss, data, accumulation, ...); resuming would not continue the same run")
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["optimizer"])
        if self.sched is not None and state["scheduler"] is not None:
            self.sched.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"]["torch_cpu"])
        c = state["counters"]
        self.epoch, self.micro_in_epoch, self.opt_step, self.micro_total = c["epoch"], c["micro_in_epoch"], c["opt_step"], c["micro_total"]
        self.es_best, self.es_bad = state["early"]["best"], state["early"]["bad"]
        self.best_value, self.last_metrics = state["best_value"], dict(state["last_metrics"])
        self.model.train(state["training"])
        self.emit("resumed", step=self.opt_step, epoch=self.epoch, micro_in_epoch=self.micro_in_epoch)

    @staticmethod
    def read_state(data: bytes) -> dict[str, Any]:
        return torch.load(io.BytesIO(data), weights_only=True)

    def save_checkpoint(self, tag: str = "periodic", data: bytes | None = None) -> dict[str, Any]:
        data = data if data is not None else self.checkpoint_bytes()
        meta = {"step": self.opt_step, "epoch": self.epoch, "micro_in_epoch": self.micro_in_epoch, "tag": tag, "graph_hash": self.graph_hash,
                "spec_hash": self.spec_hash, "sha256": hashlib.sha256(data).hexdigest(), "exact_resume": self.spec.deterministic}
        self.saved.append(meta)
        if self.on_checkpoint:
            self.on_checkpoint(self.opt_step, data, meta)
        self.emit("checkpoint", step=self.opt_step, **{k: v for k, v in meta.items() if k != "step"})
        keep = self.spec.checkpoint.keep_last
        if keep is not None:
            periodic = [m for m in self.saved if m["tag"] == "periodic"]
            for old in periodic[:-keep]:
                if not old.get("pruned"):
                    old["pruned"] = True
                    self.emit("checkpoint_pruned", step=old["step"], sha256=old["sha256"], reason=f"retention: keep_last={keep}")
        return meta

    # ------------------------------------------------------------------ validation
    def validate(self) -> dict[str, float]:
        v = self.spec.validation
        was = self.model.training
        if v.eval_mode:
            self.model.eval()
        n, b = len(self.val_ds), self.spec.data.batch_size
        tot, correct, count = 0.0, 0, 0
        with torch.set_grad_enabled(not v.no_grad):
            for s in range(0, n, b):
                batch = self.val_ds.batch(torch.arange(s, min(n, s + b)))
                out = self._forward(batch)
                loss = self._loss(out, batch)
                m = len(batch[next(iter(batch))])
                tot += float(loss.detach()) * m
                count += m
                tgt = batch.get(self.spec.data.target)
                if out.dim() == 2 and tgt is not None and tgt.dtype == torch.int64 and tgt.dim() == 1:
                    correct += int((out.argmax(1) == tgt).sum())
        self.model.train(was)
        metrics = {"val_loss": tot / count}
        if self.val_ds.fields.get(self.spec.data.target) is not None and self.val_ds.fields[self.spec.data.target].dtype == torch.int64:
            metrics["val_acc"] = correct / count
        self.last_metrics.update(metrics)
        self.emit("validation", step=self.opt_step, epoch=self.epoch, eval_mode=v.eval_mode, no_grad=v.no_grad, n=count, **metrics)
        return metrics

    def _after_validation(self, metrics: dict[str, float]) -> None:
        sc = self.spec.scheduler
        if self.sched is not None and sc.kind == "reduce_on_plateau" and sc.timing == "validation":
            before = self.opt.param_groups[0]["lr"]
            self.sched.step(metrics[sc.metric] if sc.metric in metrics else self.last_metrics[sc.metric])
            self.emit("scheduler_step", step=self.opt_step, timing="validation", lr_before=before, lr_after=self.opt.param_groups[0]["lr"], metric=sc.metric)
        ck = self.spec.checkpoint
        if ck.best and ck.best.metric in metrics:
            val = metrics[ck.best.metric]
            better = self.best_value is None or (val < self.best_value if ck.best.mode == "min" else val > self.best_value)
            if better:
                self.best_value = val
                self.save_checkpoint("best")
        es = self.spec.early_stopping
        if es and es.metric in metrics:
            val = metrics[es.metric]
            improved = self.es_best is None or (val < self.es_best - es.min_delta if es.mode == "min" else val > self.es_best + es.min_delta)
            if improved:
                self.es_best, self.es_bad = val, 0
            else:
                self.es_bad += 1
            self.emit("early_stopping_check", step=self.opt_step, metric=es.metric, value=val, best=self.es_best, bad=self.es_bad, patience=es.patience)
            if self.es_bad >= es.patience:
                self.stopped_by = "early_stopping"

    def _cadence_hit(self, cad, *, unit: str, epoch_done: int | None = None) -> bool:
        if cad is None or cad.unit != unit:
            return False
        return (self.opt_step if unit == "optimizer_step" else (epoch_done or 0)) % cad.n == 0

    # ------------------------------------------------------------------ one accumulation window
    def _window(self, order: torch.Tensor, w_start: int, L: int) -> None:
        spec, st = self.spec, self.spec.stages
        fi = self.stage_index["forward"]
        zero_before = "zero_grad" in self.stage_index and self.stage_index["zero_grad"] < fi
        scale = (1.0 / L) if spec.accumulation.normalize == "mean" else 1.0
        micro_losses: list[float] = []
        cap = spec.capture
        capturing = bool(cap and (self.opt_step + 1) in cap.steps and self.on_capture)
        sc: StepCapture | None = None
        info: dict[str, Any] = {}
        if spec.breakpoints:
            self._breakpoint_snapshot = self.checkpoint_bytes()
        probe_hooks = self._probe_hooks() + self._breakpoint_hooks()
        for m in range(L):
            batch = self._batch(order, w_start + m)
            last = m == L - 1
            out = loss = None
            if capturing and m == 0:
                sc = StepCapture(self.model, cap.nodes, cap.max_elements).attach()
            try:
                for stage in st:
                    if stage == "zero_grad":
                        if (zero_before and m == 0) or (not zero_before and last):
                            self.opt.zero_grad(set_to_none=True)
                    elif stage == "forward":
                        out = self._forward(batch)
                    elif stage == "loss":
                        loss = self._loss(out, batch)
                        micro_losses.append(float(loss.detach()))
                        self._check_loss_breakpoints(loss)
                    elif stage == "backward":
                        (loss * scale).backward()
                        if capturing and m == 0 and sc is not None:
                            info = {"micro_batch": 0, "loss": micro_losses[0], "loss_scale": scale, "accumulated_micro_batches": 1}
                            sc.detach()
                            self._publish_capture(sc, scale, info)
                            sc = None
                    elif last and stage == "clip":
                        self._clip()
                    elif last and stage == "optimizer_step":
                        self._optimizer_step(micro_losses, L)
                    elif last and stage == "scheduler_step":
                        self._scheduler_after_step()
            finally:
                if sc is not None:
                    sc.detach()
                    sc = None
            self.micro_total += 1
        for h in probe_hooks:
            h.remove()
        self.micro_in_epoch = w_start + L
        self._epoch_losses.append(sum(micro_losses) / len(micro_losses))

    def _probe_hooks(self):
        hs = []
        for pr in self.spec.probes:
            mod = self.model._modules.get(pr.node)
            if mod is not None:
                hs.append(mod.register_forward_hook(self._probe_hook(pr)))
        return hs

    def _breakpoint_hooks(self):
        hs = []
        for bp in self.spec.breakpoints:
            if bp.kind == "nonfinite_node" and bp.node in self.model._modules:
                def hook(m, a, out, bp=bp):
                    ts = [out] if isinstance(out, torch.Tensor) else [o for o in out if isinstance(o, torch.Tensor)]
                    for t in ts:
                        if t.dtype.is_floating_point and not bool(torch.isfinite(t.detach()).all()):
                            raise BreakpointHit(bp.id, f"node '{bp.node}' produced non-finite values at optimizer step {self.opt_step + 1}")
                    return None
                hs.append(self.model._modules[bp.node].register_forward_hook(hook))
        return hs

    def _probe_hook(self, pr):
        def hook(m, a, out):
            if isinstance(out, torch.Tensor):
                with torch.no_grad():
                    o = out.detach().double()
                    fin = torch.isfinite(o)
                    rec = {"step": self.opt_step + 1, "shape": list(out.shape), "nonFinite": int((~fin).sum()),
                           "min": float(o[fin].min()) if fin.any() else None, "max": float(o[fin].max()) if fin.any() else None,
                           "mean": float(o[fin].mean()) if fin.any() else None}
                    if pr.values:
                        rec["values"] = o.flatten()[:pr.values].tolist()
                    self.emit("probe", pr.node, **rec)
            return None

        return hook

    def _publish_capture(self, sc: StepCapture, scale: float, info: dict[str, Any]) -> None:
        note = (f"captured at optimizer step {self.opt_step + 1}, first micro-batch of the window; output gradients are of loss*{scale:g} "
                "(accumulation scaling included); parameter gradients are those accumulated so far")
        data = sc.payload(self.opt_step + 1, note)
        meta = {"step": self.opt_step + 1, "nodes": sorted(sc.activations), "bytes": len(data), "truncated": sc.truncated, **info}
        self.on_capture(self.opt_step + 1, data, meta)  # type: ignore[misc]
        self.emit("capture_recorded", step=self.opt_step + 1, nodes=len(sc.activations), grads=len(sc.gradients), truncated=sc.truncated)

    def _check_loss_breakpoints(self, loss: torch.Tensor) -> None:
        for bp in self.spec.breakpoints:
            if bp.kind == "nonfinite_loss" and not bool(torch.isfinite(loss.detach())):
                raise BreakpointHit(bp.id, f"the loss is {float(loss.detach())} at optimizer step {self.opt_step + 1}")
            if bp.kind == "loss_above" and bp.threshold is not None and float(loss.detach()) > bp.threshold:
                raise BreakpointHit(bp.id, f"loss {float(loss.detach()):.6g} exceeds {bp.threshold}")

    def _clip(self) -> dict[str, Any]:
        c = self.spec.clip
        grads = [p.grad for p in self.params if p.grad is not None]
        if c.kind == "none" or not grads:
            return {}
        if c.kind == "norm":
            total = float(torch.nn.utils.clip_grad_norm_(self.params, c.max_norm, c.norm_type))
            self._clip_info = {"kind": "norm", "norm_before": total, "max_norm": c.max_norm, "clipped": total > c.max_norm,
                               "scale": min(1.0, c.max_norm / (total + 1e-6))}
        else:
            mx = max(float(g.abs().max()) for g in grads)
            torch.nn.utils.clip_grad_value_(self.params, c.value)
            self._clip_info = {"kind": "value", "max_abs_before": mx, "clip_value": c.value, "clipped": mx > c.value}
        return self._clip_info

    def _grad_norm(self) -> float | None:
        gs = [p.grad.detach().double().flatten() for p in self.params if p.grad is not None]
        return float(torch.cat(gs).norm()) if gs else None

    def _optimizer_step(self, micro_losses: list[float], L: int) -> None:
        for bp in self.spec.breakpoints:
            gn = self._grad_norm()
            if bp.kind == "grad_norm_above" and bp.threshold is not None and gn is not None and (not math.isfinite(gn) or gn > bp.threshold):
                raise BreakpointHit(bp.id, f"gradient norm {gn:.6g} exceeds {bp.threshold} at optimizer step {self.opt_step + 1}")
        gnorm = self._grad_norm()
        trace = self._watch_before()
        lr = self.opt.param_groups[0]["lr"]
        self.opt.step()
        self.opt_step += 1
        if trace is not None:
            self._watch_after(trace, lr)
        info = self._clip_info
        self._clip_info = {}
        rec = {"step": self.opt_step, "epoch": self.epoch, "batch": self.micro_in_epoch // max(1, L), "loss": sum(micro_losses) / len(micro_losses),
               "micro_losses": micro_losses, "window": L, "lr": lr, "grad_norm": gnorm, "accumulation": self.spec.accumulation.steps}
        if info:
            rec["clip"] = info
        self.history.append(rec)
        self.emit("train_step", **rec)

    def _scheduler_after_step(self) -> None:
        sc = self.spec.scheduler
        if self.sched is not None and sc.timing == "optimizer_step" and sc.kind != "reduce_on_plateau":
            before = self.opt.param_groups[0]["lr"]
            self.sched.step()
            self.emit("scheduler_step", step=self.opt_step, timing="optimizer_step", lr_before=before, lr_after=self.opt.param_groups[0]["lr"])

    # ------------------------------------------------------------------ optimizer element trace (VISION 10.2)
    def _watch_before(self):
        w = self.spec.watch
        if w is None:
            return None
        params = dict(self.model.named_parameters())
        p = params.get(w.param)
        if p is None or p.grad is None:
            return None
        idx = tuple(w.index)
        st = self.opt.state.get(p, {})
        return {"p": p, "idx": idx, "w": p.detach()[idx].clone(), "g": p.grad.detach()[idx].clone(),
                "state": {k: (v.detach()[idx].clone() if torch.is_tensor(v) and v.dim() > 0 else (v.detach().clone() if torch.is_tensor(v) else v)) for k, v in st.items()}}

    def _watch_after(self, tr: dict[str, Any], lr: float) -> None:
        p, idx = tr["p"], tr["idx"]
        st_after = self.opt.state.get(p, {})
        predicted, st_pred, steps = manual_update(self.spec.optimizer, lr, tr["w"], tr["g"], tr["state"])
        actual = p.detach()[idx]
        self.emit("optimizer_trace", step=self.opt_step, param=self.spec.watch.param, index=list(idx), lr=lr, optimizer=self.spec.optimizer.kind,
                  w_before=float(tr["w"]), grad=float(tr["g"]), w_after=float(actual), w_predicted=float(predicted), abs_diff=abs(float(actual) - float(predicted)),
                  state_before={k: (float(v) if torch.is_tensor(v) or isinstance(v, (int, float)) else None) for k, v in tr["state"].items()},
                  state_after={k: float(v.detach()[idx] if v.dim() > 0 else v.detach()) for k, v in st_after.items() if torch.is_tensor(v)},
                  formula=steps, grad_is_after_clipping=self.spec.clip.kind != "none")

    # ------------------------------------------------------------------ main loop
    def run(self, until_opt_step: int | None = None) -> dict[str, Any]:
        """Train until the epochs are done, max_optimizer_steps is reached, early stopping fires, a breakpoint hits, cancellation is
        requested, or `until_opt_step` optimizer steps have happened in total (a safe boundary, so a checkpoint can be taken there)."""
        spec = self.spec
        self.stopped_by = None
        t0 = time.perf_counter()
        with deterministic_cpu(spec.deterministic):
            self.model.train()
            try:
                while self.epoch < spec.epochs and self.stopped_by is None:
                    order = self._order(self.epoch)
                    n_micro, k = self._n_micro(), spec.accumulation.steps
                    while self.micro_in_epoch < n_micro:
                        if self.cancel():
                            self.stopped_by = "cancelled"
                            break
                        w_start = (self.micro_in_epoch // k) * k
                        L = min(k, n_micro - w_start)
                        with D.CaptureSession(observe=False, assertions=True, step=self.opt_step + 1) if spec.assertions else _null():
                            self._window(order, w_start, L)
                        self._after_window()
                        if self.stopped_by:
                            break
                        if spec.max_optimizer_steps is not None and self.opt_step >= spec.max_optimizer_steps:
                            self.stopped_by = "max_steps"
                            break
                        if until_opt_step is not None and self.opt_step >= until_opt_step:
                            self.stopped_by = "until_step"
                            break
                    if self.stopped_by:
                        break
                    self._end_epoch()
                if self.stopped_by is None:
                    self.stopped_by = "epochs_done"
            except BreakpointHit as bp:
                self.stopped_by = f"breakpoint:{bp.bp_id}"
                self.emit("breakpoint_hit", step=self.opt_step + 1, breakpoint=bp.bp_id, detail=bp.detail, execution_changing=True)
                if self._breakpoint_snapshot is not None:
                    self.save_checkpoint("breakpoint", self._breakpoint_snapshot)
            except D.AssertionFailed as a:
                self.stopped_by = "assertion"
                self.emit("assertion_failed", a.node, detail=a.detail, step=self.opt_step + 1, execution_changing=True)
        return {"stopped_by": self.stopped_by, "opt_step": self.opt_step, "epoch": self.epoch, "seconds": time.perf_counter() - t0}

    def _after_window(self) -> None:
        for bp in self.spec.breakpoints:
            if bp.kind == "step_reached" and bp.step is not None and self.opt_step >= bp.step:
                self.stopped_by = f"breakpoint:{bp.id}"
                self.emit("breakpoint_hit", step=self.opt_step, breakpoint=bp.id, detail=f"optimizer step {self.opt_step} reached", execution_changing=True)
        sp = self.spec
        if self._cadence_hit(sp.validation.every, unit="optimizer_step"):
            self._after_validation(self.validate())
        if self._cadence_hit(sp.checkpoint.every, unit="optimizer_step"):
            self.save_checkpoint("periodic")
        if self.stopped_by and self.stopped_by.startswith("breakpoint"):
            self.save_checkpoint("breakpoint")

    def _end_epoch(self) -> None:
        sp = self.spec
        done = self.epoch + 1
        if self._cadence_hit(sp.validation.every, unit="epoch", epoch_done=done):
            self._after_validation(self.validate())
        if self.sched is not None and sp.scheduler.timing == "epoch":
            before = self.opt.param_groups[0]["lr"]
            with warnings.catch_warnings():   # a trainer rebuilt from a checkpoint has not called opt.step() in THIS process; the restored state is what matters
                warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`")
                self.sched.step()
            self.emit("scheduler_step", step=self.opt_step, timing="epoch", lr_before=before, lr_after=self.opt.param_groups[0]["lr"])
        loss = sum(self._epoch_losses) / max(1, len(self._epoch_losses))
        self._epoch_losses = []
        self.emit("epoch_end", epoch=self.epoch, step=self.opt_step, train_loss=loss, val_loss=self.last_metrics.get("val_loss"), val_acc=self.last_metrics.get("val_acc"))
        self.epoch += 1
        self.micro_in_epoch = 0
        if self._cadence_hit(sp.checkpoint.every, unit="epoch", epoch_done=done):
            self.save_checkpoint("periodic")


@contextmanager
def _null():
    yield
