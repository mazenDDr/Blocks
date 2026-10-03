"""core.* operations: tensor input and arithmetic primitives."""
from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
from pydantic import field_validator

from graph_core.registry import Operation, register
from graph_core.types import BATCH, OpError, TensorType

from ._common import EmptyConfig, StrictConfig, require_dtype

FLOATS = ("float32", "float64")


# ---------------------------------------------------------------- modules
class InputNode(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Sub(nn.Module):
    def forward(self, a, b):
        return a - b


class Add(nn.Module):
    def forward(self, a, b):
        return a + b


class Square(nn.Module):
    def forward(self, x):
        return torch.square(x)


class ScalarMul(nn.Module):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor


class Reduce(nn.Module):
    def __init__(self, kind: str, dims: list[int] | None, keepdim: bool):
        super().__init__()
        self.kind, self.dims, self.keepdim = kind, dims, keepdim

    def forward(self, x):
        fn = torch.sum if self.kind == "sum" else torch.mean
        dims = tuple(self.dims) if self.dims is not None else tuple(range(x.dim()))
        return fn(x, dim=dims, keepdim=self.keepdim)


# ---------------------------------------------------------------- tensor_input
class InputConfig(StrictConfig):
    shape: list[int | str] = ["N", 3, 64, 64]
    dtype: Literal["float32", "float64", "int64"] = "float32"
    layout: str = "NCHW"

    @field_validator("shape")
    @classmethod
    def _shape(cls, v):
        for i, d in enumerate(v):
            if isinstance(d, str):
                if d != BATCH or i != 0:
                    raise ValueError('the only symbolic dimension is the batch dim "N", and only at index 0')
            elif d <= 0:
                raise ValueError("dimensions must be positive")
        return v


@register
class TensorInput(Operation):
    type = "core.tensor_input"
    inputs = ()
    outputs = ("value",)
    Config = InputConfig

    def infer_shape(self, cfg, inputs):
        return {"value": TensorType(tuple(cfg.shape), cfg.dtype)}

    def lower(self, cfg):
        return InputNode()

    def explain(self, cfg, inputs, outputs):
        return {"equation": "x = input tensor", "parameters": {"formula": "0", "terms": [], "total": 0},
                "note": f"shape {list(cfg.shape)}, dtype {cfg.dtype}, layout {cfg.layout}; N is the batch size."}

    def codegen(self, cfg):
        return None, "{0}"


# ---------------------------------------------------------------- elementwise
def _binary_infer(inputs: dict[str, TensorType], what: str) -> dict[str, TensorType]:
    a, b = inputs["a"], inputs["b"]
    require_dtype(a, FLOATS, "a", what)
    if a.dtype != b.dtype:
        raise OpError("E_PORT_TYPE", f"{what} needs equal dtypes, got {a.dtype} and {b.dtype}.", "b")
    if a.shape != b.shape:
        raise OpError("E_SHAPE_MISMATCH", f"{what} needs equal shapes (no broadcasting), got {list(a.shape)} and {list(b.shape)}.", "b")
    return {"output": a}


class _Binary(Operation):
    inputs = ("a", "b")
    Config = EmptyConfig
    symbol = "-"
    cls: type[nn.Module]

    def infer_shape(self, cfg, inputs):
        return _binary_infer(inputs, self.type)

    def lower(self, cfg):
        return self.cls()

    def explain(self, cfg, inputs, outputs):
        return {"equation": f"out = a {self.symbol} b (elementwise)",
                "parameters": {"formula": "0", "terms": [], "total": 0}}

    def codegen(self, cfg):
        return None, "{0} " + self.symbol + " {1}"


@register
class SubOp(_Binary):
    type = "core.sub"
    symbol = "-"
    cls = Sub


@register
class AddOp(_Binary):
    type = "core.add"
    symbol = "+"
    cls = Add


@register
class SquareOp(Operation):
    type = "core.square"
    Config = EmptyConfig

    def infer_shape(self, cfg, inputs):
        require_dtype(inputs["input"], FLOATS, "input", self.type)
        return {"output": inputs["input"]}

    def lower(self, cfg):
        return Square()

    def explain(self, cfg, inputs, outputs):
        return {"equation": "out = x^2 (elementwise)", "parameters": {"formula": "0", "terms": [], "total": 0}}

    def codegen(self, cfg):
        return None, "torch.square({0})"


class ScalarMulConfig(StrictConfig):
    factor: float = 1.0


@register
class ScalarMulOp(Operation):
    type = "core.scalar_mul"
    Config = ScalarMulConfig

    def infer_shape(self, cfg, inputs):
        require_dtype(inputs["input"], FLOATS, "input", self.type)
        return {"output": inputs["input"]}

    def lower(self, cfg):
        return ScalarMul(cfg.factor)

    def explain(self, cfg, inputs, outputs):
        return {"equation": f"out = {cfg.factor!r} * x (elementwise)", "parameters": {"formula": "0", "terms": [], "total": 0}}

    def codegen(self, cfg):
        return None, "{0} * " + repr(cfg.factor)


# ---------------------------------------------------------------- reductions
class ReduceConfig(StrictConfig):
    dims: list[int] | None = None  # None = all dims
    keepdim: bool = False


class MeanConfig(ReduceConfig):
    # Explicit divisor semantics: the mean divides by the number of reduced elements.
    divisor: Literal["element_count"] = "element_count"


class _Reduce(Operation):
    kind: str
    Config = ReduceConfig

    def _dims(self, cfg, t: TensorType) -> list[int]:
        rank = len(t.shape)
        if cfg.dims is None:
            return list(range(rank))
        out = []
        for d in cfg.dims:
            if not -rank <= d < rank:
                raise OpError("E_CONFIG", f"dim {d} is out of range for rank {rank}.", "input")
            out.append(d % rank)
        if len(set(out)) != len(out):
            raise OpError("E_CONFIG", "dims contains duplicates.", "input")
        return sorted(out)

    def infer_shape(self, cfg, inputs):
        t = inputs["input"]
        require_dtype(t, FLOATS, "input", self.type)
        dims = self._dims(cfg, t)
        if cfg.keepdim:
            shape = tuple(1 if i in dims else d for i, d in enumerate(t.shape))
        else:
            shape = tuple(d for i, d in enumerate(t.shape) if i not in dims)
        return {"output": TensorType(shape, t.dtype)}

    def lower(self, cfg):
        return Reduce(self.kind, cfg.dims, cfg.keepdim)

    def codegen(self, cfg):
        dims = "tuple(range({0}.dim()))" if cfg.dims is None else repr(tuple(cfg.dims))
        return None, f"torch.{self.kind}({{0}}, dim={dims}, keepdim={cfg.keepdim})"


@register
class SumOp(_Reduce):
    type = "core.sum"
    kind = "sum"

    def explain(self, cfg, inputs, outputs):
        return {"equation": f"out = sum of x over dims {cfg.dims if cfg.dims is not None else 'all'}; no divisor",
                "parameters": {"formula": "0", "terms": [], "total": 0}}


@register
class MeanOp(_Reduce):
    type = "core.mean"
    kind = "mean"
    Config = MeanConfig

    def explain(self, cfg, inputs, outputs):
        t = inputs["input"]
        dims = self._dims(cfg, t)
        count = "*".join(str(t.shape[i]) for i in dims) or "1"
        return {"equation": f"out = (sum of x over dims {dims}) / ({count})",
                "divisor": {"semantics": cfg.divisor, "expression": count},
                "parameters": {"formula": "0", "terms": [], "total": 0}}
