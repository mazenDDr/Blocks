"""Runtime for diagnostic blocks (probe, histogram, timer, assert) placed anywhere in a graph, also inside composite modules.

Observation-only blocks (probe, histogram, timer) pass their input through as the very same tensor object. They record only while a
CaptureSession is active and the block is inside its scope; recording runs on detached copies under torch.no_grad(), consumes no random
numbers and never alters the data path. With no session active a diagnostic block is an identity function.

An assert block is NOT observation-only: when assertions are enabled and the check fails it stops the execution by raising
AssertionFailed. It is labelled execution-changing everywhere it is shown. With assertions disabled it is an identity function."""
from __future__ import annotations

import contextvars
import fnmatch
import time
from dataclasses import dataclass, field
from typing import Any

import torch

_current: contextvars.ContextVar["CaptureSession | None"] = contextvars.ContextVar("void_capture_session", default=None)


class AssertionFailed(Exception):
    def __init__(self, node: str, message: str):
        super().__init__(f"assertion at {node} failed: {message}")
        self.node, self.detail = node, message


@dataclass
class CaptureSession:
    """Scope of a capture: node path patterns (fnmatch), optional step set, optional batch-sample indices, element budget."""

    nodes: list[str] | None = None  # None = every diagnostic block; patterns like "enc/*" or "enc/attn/probe_scores"
    steps: set[int] | None = None
    samples: list[int] | None = None
    max_elements: int = 200_000
    assertions: bool = False
    observe: bool = True  # False: only assertions are active (probes/histograms/timers stay identity)
    step: int | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    dropped: int = 0
    used: int = 0
    _t0: float = field(default_factory=time.perf_counter)
    _token: Any = None

    def __enter__(self) -> "CaptureSession":
        self._token = _current.set(self)
        return self

    def __exit__(self, *exc) -> None:
        _current.reset(self._token)

    def in_scope(self, node: str) -> bool:
        if self.steps is not None and self.step not in self.steps:
            return False
        return self.nodes is None or any(fnmatch.fnmatchcase(node, p) for p in self.nodes)

    def add(self, rec: dict[str, Any], cost: int) -> None:
        if self.used + cost > self.max_elements:
            self.dropped += 1
            return
        self.used += cost
        rec["order"] = len(self.records)
        rec["step"] = self.step
        self.records.append(rec)


def current() -> CaptureSession | None:
    return _current.get()


def _stats(x: torch.Tensor) -> dict[str, Any]:
    f = x.detach().double() if x.dtype != torch.bool else x.detach().double()
    n = f.numel()
    fin = torch.isfinite(f)
    out: dict[str, Any] = {"count": n, "nonFinite": int((~fin).sum())}
    g = f[fin]
    if g.numel():
        out.update(min=float(g.min()), max=float(g.max()), mean=float(g.mean()), std=float(g.std()) if g.numel() > 1 else 0.0)
    return out


def _sel(x: torch.Tensor, s: CaptureSession) -> torch.Tensor:
    if s.samples is not None and x.dim() > 0:
        idx = [i for i in s.samples if 0 <= i < x.shape[0]]
        return x[idx]
    return x


def observe(node: str, kind: str, x: torch.Tensor, cfg: dict[str, Any]) -> None:
    s = _current.get()
    if s is None or not s.observe or not s.in_scope(node):
        return
    with torch.no_grad():
        v = _sel(x.detach(), s)
        rec: dict[str, Any] = {"node": node, "kind": kind, "shape": list(x.shape), "dtype": str(x.dtype).replace("torch.", ""),
                               "sampled": s.samples is not None, "label": cfg.get("label", "")}
        cost = 1
        if kind == "probe":
            rec["stats"] = _stats(v)
            n = int(cfg.get("values", 0))
            if n:
                vals = v.flatten()[:n]
                rec["values"] = vals.double().tolist() if vals.dtype != torch.bool else vals.tolist()
                cost += vals.numel()
        elif kind == "histogram":
            f = v.double()[torch.isfinite(v.double())] if v.numel() else v.double()
            lo, hi = cfg.get("range") or (None, None)
            if f.numel():
                lo = float(f.min()) if lo is None else lo
                hi = float(f.max()) if hi is None else hi
                if hi <= lo:
                    hi = lo + 1.0
                counts = torch.histc(f, bins=int(cfg.get("bins", 16)), min=lo, max=hi)
                rec["hist"] = {"lo": lo, "hi": hi, "counts": counts.tolist()}
                cost += int(cfg.get("bins", 16))
            else:
                rec["hist"] = {"lo": 0.0, "hi": 1.0, "counts": [0.0] * int(cfg.get("bins", 16))}
        elif kind == "timer":
            rec["elapsedMs"] = (time.perf_counter() - s._t0) * 1000.0
            rec["note"] = "milliseconds since the capture session started; differences between timers give the time between those points"
        s.add(rec, cost)


def check(node: str, x: torch.Tensor, cfg: dict[str, Any]) -> None:
    s = _current.get()
    if s is None or not s.assertions or not cfg.get("enabled", True) or not s.in_scope(node):
        return
    with torch.no_grad():
        kind = cfg.get("check", "finite")
        v = x.detach()
        if kind == "finite":
            if v.dtype.is_floating_point and not bool(torch.isfinite(v).all()):
                bad = int((~torch.isfinite(v)).sum())
                raise AssertionFailed(node, f"{bad} of {v.numel()} values are NaN or infinite")
        elif kind == "nonnegative":
            if bool((v < 0).any()):
                raise AssertionFailed(node, f"minimum {float(v.min())} is negative")
        elif kind == "bounded":
            lo, hi = cfg.get("lo"), cfg.get("hi")
            if (lo is not None and bool((v < lo).any())) or (hi is not None and bool((v > hi).any())):
                raise AssertionFailed(node, f"values range over [{float(v.min())}, {float(v.max())}], outside [{lo}, {hi}]")
