"""diag.* diagnostic blocks (VISION 15.6) and the structural block catalog entries (core.composite / core.repeat / core.select).

Diagnostic blocks pass their input through unchanged (the same tensor object). See graph_core/diagnostics.py for the runtime contract."""
from __future__ import annotations

from typing import Literal

import torch.nn as nn
from pydantic import Field, model_validator

from graph_core import diagnostics as D
from graph_core.composite import CompositeConfig, RepeatConfig, SelectConfig
from graph_core.registry import Operation, register
from graph_core.types import OpError

from ._common import StrictConfig
from .layers import NO_PARAMS


class DiagModule(nn.Module):
    kind = "probe"

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg, self.node_id = cfg, "?"

    def bind_node(self, nid: str) -> None:
        self.node_id = nid

    def forward(self, x):
        D.observe(self.node_id, self.kind, x, self.cfg)
        return x  # the very same tensor: nothing in the differentiable path changes


class AssertModule(DiagModule):
    kind = "assert"

    def forward(self, x):
        D.check(self.node_id, x, self.cfg)
        return x


class DiagConfig(StrictConfig):
    label: str = ""
    enabled: bool = True


class _Diag(Operation):
    """Pass-through blocks. observation_only is shown in the UI; an assert is execution-changing."""

    observation_only = True
    module = DiagModule
    kind = "probe"

    def infer_shape(self, cfg, inputs):
        return {"output": inputs["input"]}

    def lower(self, cfg):
        m = self.module(cfg.model_dump())
        m.kind = self.kind
        return m

    def codegen(self, cfg):
        return None, "{0}"


class ProbeConfig(DiagConfig):
    values: int = Field(0, ge=0, le=4096)  # also keep the first n values (0 = summary statistics only)


@register
class ProbeOp(_Diag):
    type = "diag.probe"
    Config = ProbeConfig

    def explain(self, cfg, inputs, outputs):
        return {"equation": "out = in (identity). While a capture session covers this block it records min/max/mean/std/non-finite count of the value.",
                "executionEffect": "observation-only: the tensor is passed through unchanged; no random numbers are drawn; recording works on detached copies",
                "parameters": NO_PARAMS}


class HistConfig(DiagConfig):
    bins: int = Field(16, ge=2, le=256)
    range: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _r(self):
        if self.range is not None and self.range[1] <= self.range[0]:
            raise ValueError("range must be (lo, hi) with hi > lo")
        return self


@register
class HistogramOp(_Diag):
    type = "diag.histogram"
    kind = "histogram"
    Config = HistConfig

    def explain(self, cfg, inputs, outputs):
        return {"equation": f"out = in (identity). While captured, records a {cfg.bins}-bin histogram of the finite values.",
                "executionEffect": "observation-only; the histogram is computed on a detached copy, so no host conversion enters the differentiable path",
                "parameters": NO_PARAMS}


@register
class TimerOp(_Diag):
    type = "diag.timer"
    kind = "timer"
    Config = DiagConfig

    def explain(self, cfg, inputs, outputs):
        return {"equation": "out = in (identity). While captured, records the wall-clock time at which this point is reached.",
                "executionEffect": "observation-only", "parameters": NO_PARAMS}


class AssertConfig(DiagConfig):
    check: Literal["finite", "nonnegative", "bounded"] = "finite"
    lo: float | None = None
    hi: float | None = None

    @model_validator(mode="after")
    def _b(self):
        if self.check == "bounded" and self.lo is None and self.hi is None:
            raise ValueError("bounded needs lo, hi, or both")
        return self


@register
class AssertOp(_Diag):
    type = "diag.assert"
    kind = "assert"
    observation_only = False
    module = AssertModule
    Config = AssertConfig

    def explain(self, cfg, inputs, outputs):
        return {"equation": f"out = in (identity) if the check '{cfg.check}' holds; otherwise execution STOPS with an assertion failure naming this block.",
                "executionEffect": "EXECUTION-CHANGING: a failed assertion raises and ends the run. Only active when assertions are enabled for the run.",
                "parameters": NO_PARAMS}


# ---------------------------------------------------------------- structural blocks (expanded before validation, see graph_core/composite.py)
class _Structural(Operation):
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()

    def infer_shape(self, cfg, inputs):
        raise OpError("E_NOT_EXPANDED", f"{self.type} must be expanded before shape inference.")

    def lower(self, cfg):
        raise RuntimeError(f"{self.type} is expanded to its inner nodes before lowering")

    def explain(self, cfg, inputs, outputs):
        return {"equation": "see the expanded inner nodes", "parameters": NO_PARAMS}


@register
class CompositeOp(_Structural):
    type = "core.composite"
    Config = CompositeConfig


@register
class RepeatOp(_Structural):
    type = "core.repeat"
    Config = RepeatConfig


@register
class SelectOp(_Structural):
    type = "core.select"
    Config = SelectConfig
