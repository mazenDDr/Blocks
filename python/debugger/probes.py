"""Probes (observation-only), the invariants they must keep, and their measured overhead (VISION 13.5, A35).

A probe is a forward hook on a node's module. Documented mode: observation only. It records statistics (and optionally the first values) of a
detached copy under torch.no_grad(). It does not replace outputs or gradients, draws no random numbers, does not change the train/eval mode or the
order in which nodes run, and does not touch parameters. `verify_invariants` runs the same forward (and backward) with and without the probes and
compares every one of those things; `measure_overhead` times both."""
from __future__ import annotations

import copy
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from graph_core.lower import GraphModule


@dataclass
class ProbeSet:
    model: GraphModule
    nodes: list[str] | None = None  # None = every node
    values: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    _handles: list[Any] = field(default_factory=list)

    def __enter__(self) -> "ProbeSet":
        for nid, mod in self.model.named_children():
            if self.nodes is None or nid in self.nodes:
                self._handles.append(mod.register_forward_hook(self._hook(nid)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _hook(self, nid: str):
        def hook(m, a, out):
            self.order.append(nid)
            ts = [out] if isinstance(out, torch.Tensor) else [o for o in out if isinstance(o, torch.Tensor)]
            for t in ts:
                with torch.no_grad():
                    d = t.detach().double()
                    fin = torch.isfinite(d)
                    rec: dict[str, Any] = {"node": nid, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "nonFinite": int((~fin).sum())}
                    if fin.any():
                        rec.update(min=float(d[fin].min()), max=float(d[fin].max()), mean=float(d[fin].mean()))
                    if self.values:
                        rec["values"] = d.flatten()[:self.values].tolist()
                    self.records.append(rec)
            return None

        return hook


def _fingerprint(model: GraphModule) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def verify_invariants(model: GraphModule, inputs: list[torch.Tensor], nodes: list[str] | None = None, seed: int = 0, train: bool = True) -> dict[str, Any]:
    """Run forward+backward twice from identical state, without and with probes on `nodes`. Everything below must be equal."""
    results: dict[str, Any] = {}
    snaps = {}
    for label in ("without", "with"):
        m = copy.deepcopy(model)
        m.train(train)
        torch.manual_seed(seed)
        ctx = ProbeSet(m, nodes) if label == "with" else None
        if ctx:
            ctx.__enter__()
        out = m(*inputs)
        out = out if isinstance(out, tuple) else (out,)
        loss = sum(o.double().sum() for o in out if o.dtype.is_floating_point)
        loss.backward()
        after_rng = torch.get_rng_state().clone()
        if ctx:
            ctx.__exit__(None, None, None)
        snaps[label] = {"out": [o.detach().clone() for o in out], "grads": {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None},
                        "params": _fingerprint(m), "rng": after_rng, "training": [mod.training for mod in m.modules()], "probe": ctx}
    a, b = snaps["without"], snaps["with"]
    results["outputs_identical"] = all(torch.equal(x, y) for x, y in zip(a["out"], b["out"]))
    results["gradients_identical"] = a["grads"].keys() == b["grads"].keys() and all(torch.equal(a["grads"][k], b["grads"][k]) for k in a["grads"])
    results["parameters_and_buffers_identical"] = all(torch.equal(a["params"][k], b["params"][k]) for k in a["params"])
    results["rng_state_identical"] = torch.equal(a["rng"], b["rng"])
    results["train_eval_mode_identical"] = a["training"] == b["training"]
    planned = [nid for nid, _ in model._plan if nodes is None or nid in nodes]
    results["execution_order_identical"] = b["probe"].order == planned
    results["probed_nodes"] = len(set(b["probe"].order))
    results["ok"] = all(v for k, v in results.items() if k.endswith("identical"))
    return results


def measure_overhead(model: GraphModule, inputs: list[torch.Tensor], nodes: list[str] | None = None, repeats: int = 30, warmup: int = 3, backward: bool = True) -> dict[str, Any]:
    """Wall-clock cost of the probes, median over `repeats` runs, same machine, same process. The numbers describe THIS model and input only."""
    m = copy.deepcopy(model)
    m.train()

    def step():
        out = m(*inputs)
        out = out if isinstance(out, tuple) else (out,)
        if backward:
            sum(o.double().sum() for o in out if o.dtype.is_floating_point).backward()

    def timed(with_probes: bool) -> list[float]:
        ts = []
        for i in range(warmup + repeats):
            ps = ProbeSet(m, nodes) if with_probes else None
            if ps:
                ps.__enter__()
            t0 = time.perf_counter()
            step()
            dt = time.perf_counter() - t0
            if ps:
                ps.__exit__(None, None, None)
            if i >= warmup:
                ts.append(dt * 1000)
        return ts

    base, probed = [], []
    for _ in range(2):  # interleave to reduce drift
        base += timed(False)
        probed += timed(True)
    b, p = statistics.median(base), statistics.median(probed)
    return {"baselineMs": b, "probedMs": p, "overheadMs": p - b, "overheadPercent": 100 * (p - b) / b if b else None, "repeats": 2 * repeats,
            "probedNodes": len(m._modules) if nodes is None else len(nodes), "includesBackward": backward,
            "note": "median wall-clock per forward(+backward) with a hook on every probed node; measured on this machine for this model and batch only"}
