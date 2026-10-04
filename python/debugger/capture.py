"""Observation-only capture of node outputs, output gradients and parameter gradients for one forward/backward pass.

Mechanism: forward hooks on the lowered per-node modules and Tensor.register_hook on their outputs. A hook never modifies a value, draws
no random numbers and runs in the same order as the nodes themselves; copies are detached clones. tests/test_debugger.py checks that a
run with capture enabled produces bit-identical outputs, gradients, parameters and RNG state (VISION A35)."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from graph_core.lower import GraphModule


def _flat(nid: str, out) -> list[tuple[str, torch.Tensor]]:
    if isinstance(out, torch.Tensor):
        return [(nid, out)]
    if isinstance(out, (tuple, list)):
        return [(f"{nid}#{i}", o) for i, o in enumerate(out) if isinstance(o, torch.Tensor)]
    return []


@dataclass
class StepCapture:
    model: GraphModule
    nodes: list[str] | None = None  # None = every node
    max_elements: int = 400_000
    with_grads: bool = True
    activations: dict[str, torch.Tensor] = field(default_factory=dict)
    gradients: dict[str, torch.Tensor] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)  # execution order of the forward hooks
    truncated: dict[str, int] = field(default_factory=dict)
    _handles: list[Any] = field(default_factory=list)
    _used: int = 0

    def _keep(self, key: str, t: torch.Tensor) -> torch.Tensor:
        t = t.detach()
        n = t.numel()
        if self._used + n > self.max_elements:
            self.truncated[key] = n
            return torch.empty(0)
        self._used += n
        return t.clone()

    def attach(self) -> "StepCapture":
        for nid, mod in self.model.named_children():
            if self.nodes is not None and nid not in self.nodes:
                continue
            self._handles.append(mod.register_forward_hook(self._hook(nid)))
        return self

    def _hook(self, nid: str):
        def hook(m, args, out):
            self.order.append(nid)
            for key, t in _flat(nid, out):
                if key not in self.activations:
                    self.activations[key] = self._keep(key, t)
                if self.with_grads and t.requires_grad:
                    t.register_hook(self._grad_hook(key))
            return None  # never replaces the output

        return hook

    def _grad_hook(self, key: str):
        def hook(g):
            if key not in self.gradients:
                self.gradients[key] = self._keep("grad:" + key, g)
            return None  # observation only: the gradient is not replaced

        return hook

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def param_grads(self) -> dict[str, torch.Tensor]:
        return {n: p.grad.detach().clone() for n, p in self.model.named_parameters() if p.grad is not None}

    def payload(self, step: int, scale_note: str) -> bytes:
        buf = io.BytesIO()
        torch.save({"step": step, "activations": self.activations, "gradients": self.gradients, "param_grads": self.param_grads(),
                    "order": self.order, "truncated": self.truncated, "note": scale_note}, buf)
        return buf.getvalue()


def load_payload(data: bytes) -> dict[str, Any]:
    return torch.load(io.BytesIO(data), weights_only=True)


def tensor_summary(t: torch.Tensor, limit: int = 64) -> dict[str, Any]:
    f = t.detach().double()
    fin = torch.isfinite(f)
    out: dict[str, Any] = {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "count": t.numel(), "nonFinite": int((~fin).sum())}
    g = f[fin]
    if g.numel():
        out.update(min=float(g.min()), max=float(g.max()), mean=float(g.mean()), std=float(g.std()) if g.numel() > 1 else 0.0, norm=float(g.norm()))
    flat = f.flatten()[:limit]
    out["values"] = [None if not torch.isfinite(v) else float(v) for v in flat]
    out["truncatedValues"] = t.numel() > limit
    return out
