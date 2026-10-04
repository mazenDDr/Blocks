"""tensor.* primitives: the arithmetic, shape, reduction-adjacent, masking and normalisation operations a scientist composes into
new blocks (VISION 9.1). Broadcasting follows PyTorch/NumPy and is declared in each op's explain data:

    shapes are right-aligned; a pair of dimensions is compatible if equal or one of them is 1; a missing leading dimension is 1;
    the symbolic batch dimension N is compatible with N or 1 (never with another number, because N is not known).

No implicit dtype promotion: operands of arithmetic must have the same dtype (cast explicitly with tensor.cast)."""
from __future__ import annotations

import math
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field, field_validator, model_validator

from graph_core.registry import Operation, register
from graph_core.types import BATCH, Dim, Fix, OpError, TensorType

from ._common import EmptyConfig, StrictConfig, prod_dims
from .layers import NO_PARAMS, _terms

FLOATS = ("float32", "float64")
NUMERIC = ("float32", "float64", "int64")
DTYPES = {"float32": torch.float32, "float64": torch.float64, "int64": torch.int64, "bool": torch.bool}
BROADCAST_RULE = ("Shapes are right-aligned; two dimensions are compatible if they are equal or one of them is 1 (a missing leading "
                  "dimension counts as 1); the symbolic batch dimension N is compatible with N or 1. No implicit dtype promotion.")


def broadcast_shapes(shapes: list[tuple[Dim, ...]], ports: list[str], what: str) -> tuple[Dim, ...]:
    rank = max(len(s) for s in shapes)
    out: list[Dim] = []
    for i in range(rank):
        dims = [(s[i - (rank - len(s))] if i - (rank - len(s)) >= 0 else 1) for s in shapes]
        known = [d for d in dims if d != 1]
        if not known:
            out.append(1)
            continue
        if any(d != known[0] for d in known):
            detail = ", ".join(f"{p}: {list(s)}" for p, s in zip(ports, shapes))
            raise OpError("E_BROADCAST", f"{what}: shapes {detail} do not broadcast at output axis {i} (sizes {dims}). {BROADCAST_RULE}", ports[-1])
        out.append(known[0])
    return tuple(out)


def _bcast_explain(shapes: dict[str, TensorType], out: TensorType) -> dict[str, Any]:
    rank = len(out.shape)
    rows = []
    for i in range(rank):
        row = {"axis": i, "out": out.shape[i]}
        for p, t in shapes.items():
            j = i - (rank - len(t.shape))
            row[p] = t.shape[j] if j >= 0 else "(missing = 1)"
        rows.append(row)
    return {"rule": BROADCAST_RULE, "alignment": rows}


def _need(t: TensorType, dtypes: tuple[str, ...], port: str, what: str) -> None:
    if t.dtype not in dtypes:
        raise OpError("E_PORT_TYPE", f"{what} needs dtype in {list(dtypes)} on '{port}', got {t.dtype}. Use tensor.cast to convert explicitly.", port)


def _norm_dim(d: int, rank: int, port: str = "input") -> int:
    if not -rank <= d < rank:
        raise OpError("E_CONFIG", f"dim {d} is out of range for a rank-{rank} tensor.", port)
    return d % rank


def _plain_dim(t: TensorType, i: int, what: str, port: str = "input") -> int:
    d = t.shape[i]
    if not isinstance(d, int):
        raise OpError("E_CONFIG", f"{what}: axis {i} is the symbolic batch dimension N, which cannot be used here.", port)
    return d


class _Op(Operation):
    """Common defaults for the primitives below."""

    def explain(self, cfg, inputs, outputs):  # pragma: no cover - every concrete op overrides
        raise NotImplementedError

    def _x(self, equation: str, **extra) -> dict[str, Any]:
        return {"equation": equation, "parameters": NO_PARAMS, **extra}


# ============================================================================================ constants
class Constant(nn.Module):
    def __init__(self, value, dtype: str):
        super().__init__()
        self.register_buffer("value", torch.tensor(value, dtype=DTYPES[dtype]), persistent=False)

    def forward(self):
        return self.value


class ConstantConfig(StrictConfig):
    value: Any = 0.0  # scalar or rectangular nested list
    dtype: Literal["float32", "float64", "int64", "bool"] = "float32"
    label: str = ""

    @model_validator(mode="after")
    def _check(self):
        try:
            torch.tensor(self.value, dtype=DTYPES[self.dtype])
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"value is not a valid {self.dtype} tensor literal: {e}") from e
        return self


@register
class ConstantOp(_Op):
    type = "tensor.constant"
    inputs = ()
    outputs = ("value",)
    Config = ConstantConfig

    def infer_shape(self, cfg, inputs):
        return {"value": TensorType(tuple(torch.tensor(cfg.value).shape), cfg.dtype)}

    def lower(self, cfg):
        return Constant(cfg.value, cfg.dtype)

    def explain(self, cfg, inputs, outputs):
        return self._x(f"out = {cfg.value!r}" if len(repr(cfg.value)) < 60 else "out = constant tensor",
                       note=f"A typed constant {cfg.dtype}{list(outputs['value'].shape)}{(' — ' + cfg.label) if cfg.label else ''}. It is not a parameter and is never trained.")

    def codegen(self, cfg):
        return None, f"torch.tensor({cfg.value!r}, dtype=torch.{cfg.dtype})"


class Arange(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self):
        return torch.arange(self.n, dtype=torch.int64)


class ArangeConfig(StrictConfig):
    n: int = Field(8, gt=0)


@register
class ArangeOp(_Op):
    type = "tensor.arange"
    inputs = ()
    outputs = ("output",)
    Config = ArangeConfig

    def infer_shape(self, cfg, inputs):
        return {"output": TensorType((cfg.n,), "int64")}

    def lower(self, cfg):
        return Arange(cfg.n)

    def explain(self, cfg, inputs, outputs):
        return self._x(f"out = [0, 1, ..., {cfg.n - 1}] (int64), e.g. position indices")

    def codegen(self, cfg):
        return None, f"torch.arange({cfg.n}, dtype=torch.int64)"


# ============================================================================================ elementwise binary (broadcasting)
class Bin(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, a, b):
        return self.fn(a, b)


class _Binary(_Op):
    inputs = ("a", "b")
    Config = EmptyConfig
    symbol = "+"
    fn: Any
    dtypes: tuple[str, ...] = NUMERIC
    spoken = ""

    def infer_shape(self, cfg, inputs):
        a, b = inputs["a"], inputs["b"]
        _need(a, self.dtypes, "a", self.type)
        if a.dtype != b.dtype:
            raise OpError("E_PORT_TYPE", f"{self.type} needs equal dtypes (no implicit promotion), got {a.dtype} and {b.dtype}.", "b",
                          [Fix(f"Cast b to {a.dtype} with tensor.cast")])
        return {"output": TensorType(broadcast_shapes([a.shape, b.shape], ["a", "b"], self.type), a.dtype)}

    def lower(self, cfg):
        return Bin(type(self).fn)

    def explain(self, cfg, inputs, outputs):
        return self._x(f"out = a {self.symbol} b (elementwise, broadcast)", shapeRule=BROADCAST_RULE,
                       broadcast=_bcast_explain(inputs, outputs["output"]))

    def codegen(self, cfg):
        return None, "{0} " + self.symbol + " {1}"


def _mk_binary(name, symbol, fn, dtypes=NUMERIC):
    cls = type(f"T{name.title()}", (_Binary,), {"type": f"tensor.{name}", "symbol": symbol, "fn": staticmethod(fn), "dtypes": dtypes})
    return register(cls)


_mk_binary("add", "+", torch.add)
_mk_binary("sub", "-", torch.sub)
_mk_binary("mul", "*", torch.mul)
_mk_binary("div", "/", torch.div, FLOATS)


# ============================================================================================ unary
class Un(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class _Unary(_Op):
    Config = EmptyConfig
    fn: Any
    eq = ""
    cg = ""

    def infer_shape(self, cfg, inputs):
        _need(inputs["input"], FLOATS, "input", self.type)
        return {"output": inputs["input"]}

    def lower(self, cfg):
        return Un(type(self).fn)

    def explain(self, cfg, inputs, outputs):
        return self._x(self.eq)

    def codegen(self, cfg):
        return None, self.cg


def _mk_unary(name, fn, eq, cg):
    register(type(f"U{name.title()}", (_Unary,), {"type": f"tensor.{name}", "fn": staticmethod(fn), "eq": eq, "cg": cg}))


_mk_unary("exp", torch.exp, "out = e^x (elementwise)", "torch.exp({0})")
_mk_unary("log", torch.log, "out = ln(x) (elementwise; -inf at 0 and NaN for x < 0 — clamp first if that can happen)", "torch.log({0})")
_mk_unary("sqrt", torch.sqrt, "out = sqrt(x) (elementwise; NaN for x < 0)", "torch.sqrt({0})")
_mk_unary("abs", torch.abs, "out = |x| (elementwise)", "torch.abs({0})")
_mk_unary("neg", torch.neg, "out = -x (elementwise)", "(-{0})")
_mk_unary("tanh", torch.tanh, "out = tanh(x) (elementwise)", "torch.tanh({0})")
_mk_unary("sigmoid", torch.sigmoid, "out = 1 / (1 + e^-x) (elementwise)", "torch.sigmoid({0})")
_mk_unary("gelu", F.gelu, "out = x * Phi(x), Phi = standard normal CDF (exact erf form, PyTorch default)", "torch.nn.functional.gelu({0})")


class ClampConfig(StrictConfig):
    min: float | None = 0.0
    max: float | None = None

    @model_validator(mode="after")
    def _c(self):
        if self.min is None and self.max is None:
            raise ValueError("give min, max, or both")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min must not exceed max")
        return self


class Clamp(nn.Module):
    def __init__(self, lo, hi):
        super().__init__()
        self.lo, self.hi = lo, hi

    def forward(self, x):
        return torch.clamp(x, self.lo, self.hi)


@register
class ClampOp(_Op):
    type = "tensor.clamp"
    Config = ClampConfig

    def infer_shape(self, cfg, inputs):
        _need(inputs["input"], FLOATS, "input", self.type)
        return {"output": inputs["input"]}

    def lower(self, cfg):
        return Clamp(cfg.min, cfg.max)

    def explain(self, cfg, inputs, outputs):
        return self._x(f"out = min(max(x, {cfg.min}), {cfg.max}) (None = unbounded); gradient is 0 where the value was clamped")

    def codegen(self, cfg):
        return None, f"torch.clamp({{0}}, {cfg.min!r}, {cfg.max!r})"


class DimConfig(StrictConfig):
    dim: int = -1


class Sm(nn.Module):
    def __init__(self, dim, log):
        super().__init__()
        self.dim, self.log = dim, log

    def forward(self, x):
        return F.log_softmax(x, dim=self.dim) if self.log else F.softmax(x, dim=self.dim)


class _Softmax(_Op):
    Config = DimConfig
    log = False

    def infer_shape(self, cfg, inputs):
        t = inputs["input"]
        _need(t, FLOATS, "input", self.type)
        _norm_dim(cfg.dim, len(t.shape))
        return {"output": t}

    def lower(self, cfg):
        return Sm(cfg.dim, self.log)

    def codegen(self, cfg):
        fn = "log_softmax" if self.log else "softmax"
        return None, f"torch.nn.functional.{fn}({{0}}, dim={cfg.dim})"


@register
class SoftmaxOp(_Softmax):
    type = "tensor.softmax"

    def explain(self, cfg, inputs, outputs):
        return self._x(f"out_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x)) along dim {cfg.dim}",
                       note="The max is subtracted before exponentiating (PyTorch's stable implementation); rows sum to 1. A row that is entirely -inf gives NaN.")


@register
class LogSoftmaxOp(_Softmax):
    type = "tensor.log_softmax"
    log = True

    def explain(self, cfg, inputs, outputs):
        return self._x(f"out_i = x_i - logsumexp(x) along dim {cfg.dim}",
                       note="Fused stable log-softmax; do not replace by log(softmax(x)), which underflows for large negative logits.")


# ============================================================================================ where / compare / logic
class Where(nn.Module):
    def forward(self, cond, a, b):
        return torch.where(cond, a, b)


@register
class WhereOp(_Op):
    type = "tensor.where"
    inputs = ("cond", "a", "b")
    Config = EmptyConfig

    def infer_shape(self, cfg, inputs):
        c, a, b = inputs["cond"], inputs["a"], inputs["b"]
        _need(c, ("bool",), "cond", self.type)
        if a.dtype != b.dtype:
            raise OpError("E_PORT_TYPE", f"tensor.where needs equal dtypes for a and b, got {a.dtype} and {b.dtype}.", "b")
        return {"output": TensorType(broadcast_shapes([c.shape, a.shape, b.shape], ["cond", "a", "b"], self.type), a.dtype)}

    def lower(self, cfg):
        return Where()

    def explain(self, cfg, inputs, outputs):
        return self._x("out = a where cond is true, else b (elementwise, broadcast)", shapeRule=BROADCAST_RULE,
                       broadcast=_bcast_explain(inputs, outputs["output"]),
                       note="Both a and b are computed; the gradient to the unselected side is zero. With a scalar cond this is the select of a conditional block.")

    def codegen(self, cfg):
        return None, "torch.where({0}, {1}, {2})"


class CompareConfig(StrictConfig):
    op: Literal["gt", "ge", "lt", "le", "eq", "ne"] = "gt"


_CMP = {"gt": (torch.gt, ">"), "ge": (torch.ge, ">="), "lt": (torch.lt, "<"), "le": (torch.le, "<="), "eq": (torch.eq, "=="), "ne": (torch.ne, "!=")}


@register
class CompareOp(_Op):
    type = "tensor.compare"
    inputs = ("a", "b")
    Config = CompareConfig

    def infer_shape(self, cfg, inputs):
        a, b = inputs["a"], inputs["b"]
        if a.dtype != b.dtype:
            raise OpError("E_PORT_TYPE", f"tensor.compare needs equal dtypes, got {a.dtype} and {b.dtype}.", "b")
        return {"output": TensorType(broadcast_shapes([a.shape, b.shape], ["a", "b"], self.type), "bool")}

    def lower(self, cfg):
        return Bin(_CMP[cfg.op][0])

    def explain(self, cfg, inputs, outputs):
        return self._x(f"out = (a {_CMP[cfg.op][1]} b), a bool tensor (elementwise, broadcast)", shapeRule=BROADCAST_RULE,
                       broadcast=_bcast_explain(inputs, outputs["output"]))

    def codegen(self, cfg):
        return None, "(({0}) " + _CMP[cfg.op][1] + " ({1}))"


class LogicConfig(StrictConfig):
    op: Literal["and", "or", "xor"] = "and"


@register
class LogicalOp(_Op):
    type = "tensor.logical"
    inputs = ("a", "b")
    Config = LogicConfig

    def infer_shape(self, cfg, inputs):
        for p in ("a", "b"):
            _need(inputs[p], ("bool",), p, self.type)
        return {"output": TensorType(broadcast_shapes([inputs["a"].shape, inputs["b"].shape], ["a", "b"], self.type), "bool")}

    def lower(self, cfg):
        return Bin({"and": torch.logical_and, "or": torch.logical_or, "xor": torch.logical_xor}[cfg.op])

    def explain(self, cfg, inputs, outputs):
        return self._x(f"out = a {cfg.op} b (bool, elementwise, broadcast)", shapeRule=BROADCAST_RULE)

    def codegen(self, cfg):
        return None, "torch.logical_" + cfg.op + "({0}, {1})"


@register
class LogicalNotOp(_Op):
    type = "tensor.logical_not"
    Config = EmptyConfig

    def infer_shape(self, cfg, inputs):
        _need(inputs["input"], ("bool",), "input", self.type)
        return {"output": inputs["input"]}

    def lower(self, cfg):
        return Un(torch.logical_not)

    def explain(self, cfg, inputs, outputs):
        return self._x("out = not x (bool, elementwise)")

    def codegen(self, cfg):
        return None, "torch.logical_not({0})"


class AnyAllConfig(StrictConfig):
    kind: Literal["any", "all"] = "any"


class AnyAll(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, x):
        return x.any() if self.kind == "any" else x.all()


@register
class AnyAllOp(_Op):
    type = "tensor.any_all"
    Config = AnyAllConfig

    def infer_shape(self, cfg, inputs):
        _need(inputs["input"], ("bool",), "input", self.type)
        return {"output": TensorType((), "bool")}

    def lower(self, cfg):
        return AnyAll(cfg.kind)

    def explain(self, cfg, inputs, outputs):
        return self._x(f"out = {cfg.kind}(x) over every element: one scalar bool (shape []), usable as the predicate of a select block")

    def codegen(self, cfg):
        return None, "{0}." + cfg.kind + "()"


class CastConfig(StrictConfig):
    dtype: Literal["float32", "float64", "int64", "bool"] = "float32"


@register
class CastOp(_Op):
    type = "tensor.cast"
    Config = CastConfig

    def infer_shape(self, cfg, inputs):
        return {"output": TensorType(inputs["input"].shape, cfg.dtype)}

    def lower(self, cfg):
        return Un(lambda x, dt=DTYPES[cfg.dtype]: x.to(dt))

    def explain(self, cfg, inputs, outputs):
        return self._x(f"out = x converted to {cfg.dtype}", note="An explicit dtype conversion. Casting to int64 or bool cuts the gradient path.")

    def codegen(self, cfg):
        return None, f"{{0}}.to(torch.{cfg.dtype})"


# ============================================================================================ matmul & shape ops
@register
class MatmulOp(_Op):
    type = "tensor.matmul"
    inputs = ("a", "b")
    Config = EmptyConfig

    def infer_shape(self, cfg, inputs):
        a, b = inputs["a"], inputs["b"]
        _need(a, FLOATS, "a", self.type)
        if a.dtype != b.dtype:
            raise OpError("E_PORT_TYPE", f"tensor.matmul needs equal dtypes, got {a.dtype} and {b.dtype}.", "b")
        if len(a.shape) < 2 or len(b.shape) < 2:
            raise OpError("E_RANK_MISMATCH", f"tensor.matmul needs rank >= 2 operands, got {list(a.shape)} and {list(b.shape)}.", "b")
        if a.shape[-1] != b.shape[-2]:
            raise OpError("E_SHAPE_MISMATCH", f"tensor.matmul: inner dimensions differ: a {list(a.shape)} has k={a.shape[-1]}, b {list(b.shape)} has k={b.shape[-2]}.", "b")
        batch = broadcast_shapes([a.shape[:-2] or (1,), b.shape[:-2] or (1,)], ["a", "b"], "tensor.matmul batch dimensions")
        if not a.shape[:-2] and not b.shape[:-2]:
            batch = ()
        return {"output": TensorType(tuple(batch) + (a.shape[-2], b.shape[-1]), a.dtype)}

    def lower(self, cfg):
        return Bin(torch.matmul)

    def explain(self, cfg, inputs, outputs):
        return self._x("out[..., i, j] = sum_k a[..., i, k] * b[..., k, j]", shapeRule="[..., m, k] @ [..., k, n] -> [..., m, n]; leading (batch) dimensions broadcast",
                       note="Operands of rank >= 2 only (PyTorch also accepts vectors; this block does not, to keep axis meaning explicit).")

    def codegen(self, cfg):
        return None, "torch.matmul({0}, {1})"


class Reshape(nn.Module):
    def __init__(self, target, n_index):
        super().__init__()
        self.target, self.n_index = target, n_index

    def forward(self, x):
        shape = [x.shape[self.n_index] if d == BATCH else d for d in self.target]
        return x.reshape(shape)


class ReshapeConfig(StrictConfig):
    shape: list[int | str] = ["N", -1]
    axes: list[str] | None = None  # optional axis names of the OUTPUT, e.g. ["N","T","H","dh"]

    @field_validator("shape")
    @classmethod
    def _s(cls, v):
        if sum(1 for d in v if d == -1) > 1:
            raise ValueError("at most one -1")
        for i, d in enumerate(v):
            if isinstance(d, str) and d != BATCH:
                raise ValueError('the only symbolic dimension is "N"')
            if isinstance(d, int) and d != -1 and d <= 0:
                raise ValueError("dimensions must be positive (or -1 once)")
        return v

    @model_validator(mode="after")
    def _axes(self):
        if self.axes is not None and len(self.axes) != len(self.shape):
            raise ValueError("axes must have one name per output dimension")
        return self


class ReshapeResolved(ReshapeConfig):
    n_index: int = 0  # position of the batch axis N in the INPUT (0 when the input has no N)


@register
class ReshapeOp(_Op):
    type = "tensor.reshape"
    Config = ReshapeConfig

    def resolve(self, cfg, inputs):
        n = [i for i, d in enumerate(inputs["input"].shape) if d == BATCH]
        return ReshapeResolved.model_validate({**cfg.model_dump(), "n_index": n[0] if n else 0})

    def infer_shape(self, cfg, inputs):
        t = inputs["input"]
        n_in = [i for i, d in enumerate(t.shape) if d == BATCH]
        n_out = [i for i, d in enumerate(cfg.shape) if d == BATCH]
        if len(n_in) != len(n_out):
            raise OpError("E_SHAPE_MISMATCH", f"reshape: the input {list(t.shape)} {'has' if n_in else 'has no'} batch dimension N but the target {cfg.shape} "
                          f"{'has' if n_out else 'has none'}. Keep N as an explicit 'N' entry.", "input")
        in_rest = prod_dims([d for d in t.shape if d != BATCH]) if t.shape else 1
        out = list(cfg.shape)
        known = 1
        for d in out:
            if isinstance(d, int) and d != -1:
                known *= d
        if -1 in out:
            if known == 0 or in_rest % known:
                raise OpError("E_SHAPE_MISMATCH", f"reshape: cannot infer -1: {in_rest} elements per batch item do not divide by {known}.", "input")
            out[out.index(-1)] = in_rest // known
        elif known != in_rest:
            raise OpError("E_SHAPE_MISMATCH", f"reshape: input {list(t.shape)} has {in_rest} elements per batch item, target {cfg.shape} has {known}.", "input",
                          [Fix("Use -1 for one dimension", key="shape", value=[("N" if d == BATCH else d) for d in cfg.shape[:-1]] + [-1])])
        return {"output": TensorType(tuple(out), t.dtype)}

    def lower(self, cfg):
        return Reshape(list(cfg.shape), cfg.n_index)

    def explain(self, cfg, inputs, outputs):
        return self._x(f"out = x.reshape({cfg.shape}) — the same elements in row-major order, new axes {cfg.axes or '(unnamed)'}",
                       shapeRule="element count per batch item must be preserved; N stays the leading batch axis",
                       axes={"in": None, "out": cfg.axes})

    def codegen(self, cfg):
        sh = ", ".join(("{0}.shape[%d]" % cfg.n_index) if d == BATCH else str(d) for d in cfg.shape)
        return None, "{0}.reshape(" + sh + ")"


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = tuple(dims)

    def forward(self, x):
        return x.permute(self.dims)


class PermuteConfig(StrictConfig):
    dims: list[int] = [0, 2, 1]
    axes: list[str] | None = None

    @model_validator(mode="after")
    def _a(self):
        if self.axes is not None and len(self.axes) != len(self.dims):
            raise ValueError("axes must have one name per output dimension")
        return self


@register
class PermuteOp(_Op):
    type = "tensor.permute"
    Config = PermuteConfig

    def infer_shape(self, cfg, inputs):
        t = inputs["input"]
        r = len(t.shape)
        if sorted(d % r if -r <= d < r else d for d in cfg.dims) != list(range(r)):
            raise OpError("E_CONFIG", f"permute dims {cfg.dims} must be a permutation of 0..{r - 1} for the rank-{r} input {list(t.shape)}.", "input")
        return {"output": TensorType(tuple(t.shape[d] for d in cfg.dims), t.dtype)}

    def lower(self, cfg):
        return Permute(cfg.dims)

    def explain(self, cfg, inputs, outputs):
        return self._x(f"out[i_0..i_k] = x with axes reordered by {cfg.dims}: out axis j is input axis {cfg.dims}[j]",
                       axes={"in": None, "out": cfg.axes}, note="A view: no data is copied until a later op needs contiguous memory.")

    def codegen(self, cfg):
        return None, f"{{0}}.permute({tuple(cfg.dims)!r})"


class Cat(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, a, b):
        return torch.cat([a, b], dim=self.dim)


@register
class ConcatOp(_Op):
    type = "tensor.concat"
    inputs = ("a", "b")
    Config = DimConfig

    def infer_shape(self, cfg, inputs):
        a, b = inputs["a"], inputs["b"]
        if a.dtype != b.dtype:
            raise OpError("E_PORT_TYPE", f"tensor.concat needs equal dtypes, got {a.dtype} and {b.dtype}.", "b")
        if len(a.shape) != len(b.shape):
            raise OpError("E_RANK_MISMATCH", f"tensor.concat needs equal ranks, got {list(a.shape)} and {list(b.shape)}.", "b")
        d = _norm_dim(cfg.dim, len(a.shape), "a")
        for i, (x, y) in enumerate(zip(a.shape, b.shape)):
            if i != d and x != y:
                raise OpError("E_SHAPE_MISMATCH", f"tensor.concat along dim {d}: axis {i} differs ({x} vs {y}).", "b")
        sa, sb = _plain_dim(a, d, "concat"), _plain_dim(b, d, "concat", "b")
        out = list(a.shape)
        out[d] = sa + sb
        return {"output": TensorType(tuple(out), a.dtype)}

    def lower(self, cfg):
        return Cat(cfg.dim)

    def explain(self, cfg, inputs, outputs):
        return self._x(f"out = [a ; b] joined along dim {cfg.dim}", shapeRule="all other axes must be equal; the joined axis sizes add")

    def codegen(self, cfg):
        return None, f"torch.cat([{{0}}, {{1}}], dim={cfg.dim})"


class Slice(nn.Module):
    def __init__(self, dim, start, end, step):
        super().__init__()
        self.sl = slice(start, end, step)
        self.dim = dim

    def forward(self, x):
        idx = [slice(None)] * x.dim()
        idx[self.dim] = self.sl
        return x[tuple(idx)]


class SliceConfig(StrictConfig):
    dim: int = 1
    start: int | None = 0
    end: int | None = None
    step: int = Field(1, gt=0)


@register
class SliceOp(_Op):
    type = "tensor.slice"
    Config = SliceConfig

    def infer_shape(self, cfg, inputs):
        t = inputs["input"]
        d = _norm_dim(cfg.dim, len(t.shape))
        size = _plain_dim(t, d, "slice")
        n = len(range(size)[cfg.start:cfg.end:cfg.step])
        if n == 0:
            raise OpError("E_SHAPE_INVALID", f"slice [{cfg.start}:{cfg.end}:{cfg.step}] of axis {d} (size {size}) is empty.", "input")
        out = list(t.shape)
        out[d] = n
        return {"output": TensorType(tuple(out), t.dtype)}

    def lower(self, cfg):
        return Slice(cfg.dim, cfg.start, cfg.end, cfg.step)

    def explain(self, cfg, inputs, outputs):
        return self._x(f"out = x[..., {cfg.start}:{cfg.end}:{cfg.step}, ...] along dim {cfg.dim} (Python slice semantics, end exclusive)")

    def codegen(self, cfg):
        return None, f"{{0}}[(slice(None),) * ({cfg.dim} % {{0}}.dim()) + (slice({cfg.start}, {cfg.end}, {cfg.step}),)]"


# ============================================================================================ norms, dropout, embedding, dense
class LayerNorm(nn.Module):
    def __init__(self, shape, eps, affine):
        super().__init__()
        self.ln = nn.LayerNorm(shape, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        return self.ln(x)


class LayerNormConfig(StrictConfig):
    normalized_dims: int = Field(1, gt=0)  # normalise over the last k dimensions
    eps: float = Field(1e-5, gt=0)
    elementwise_affine: bool = True


class LayerNormResolved(LayerNormConfig):
    normalized_shape: list[int] = []  # concrete shape, resolved from the input type


@register
class LayerNormOp(_Op):
    type = "tensor.layernorm"
    Config = LayerNormConfig

    def _shape(self, cfg, t):
        if cfg.normalized_dims >= len(t.shape):
            raise OpError("E_RANK_MISMATCH", f"layernorm over the last {cfg.normalized_dims} dims needs rank > {cfg.normalized_dims} (the batch axis stays), got {list(t.shape)}.", "input")
        tail = t.shape[-cfg.normalized_dims:]
        if not all(isinstance(d, int) for d in tail):
            raise OpError("E_CONFIG", "layernorm cannot normalise over the symbolic batch axis.", "input")
        return tuple(tail)

    def resolve(self, cfg, inputs):
        t = inputs["input"]
        if cfg.normalized_dims < len(t.shape) and all(isinstance(d, int) for d in t.shape[-cfg.normalized_dims:]):
            return LayerNormResolved.model_validate({**cfg.model_dump(), "normalized_shape": list(t.shape[-cfg.normalized_dims:])})
        return cfg

    def infer_shape(self, cfg, inputs):
        _need(inputs["input"], ("float32",), "input", self.type)
        self._shape(cfg, inputs["input"])
        return {"output": inputs["input"]}

    def param_count(self, cfg, inputs):
        return 2 * math.prod(self._shape(cfg, inputs["input"])) if cfg.elementwise_affine else 0

    def lower(self, cfg):
        return LayerNorm(list(cfg.normalized_shape), cfg.eps, cfg.elementwise_affine)

    def explain(self, cfg, inputs, outputs):
        shape = list(self._shape(cfg, inputs["input"]))
        terms, total = _terms(("weight (gamma)", shape), ("bias (beta)", shape)) if cfg.elementwise_affine else ([], 0)
        return {"equation": f"y = (x - mean) / sqrt(var + {cfg.eps}) * gamma + beta; mean and (biased) variance over the last {cfg.normalized_dims} dim(s), per item, no batch statistics",
                "parameters": {"formula": "2 * prod(normalized shape)" if cfg.elementwise_affine else "0", "terms": terms, "total": total},
                "note": "Layer normalisation uses no running statistics and behaves identically in train and eval mode."}

    def codegen(self, cfg):
        return f"nn.LayerNorm({list(cfg.normalized_shape)}, eps={cfg.eps}, elementwise_affine={cfg.elementwise_affine})", "{m}({0})"


class BatchNorm(nn.Module):
    def __init__(self, rank, features, eps, momentum, affine, track, mode):
        super().__init__()
        cls = nn.BatchNorm2d if rank == 4 else nn.BatchNorm1d
        self.bn = cls(features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track)
        self.mode = mode

    def forward(self, x):
        if self.mode != "follow":
            self.bn.training = self.mode == "train"
        else:
            self.bn.training = self.training
        return self.bn(x)


class BatchNormConfig(StrictConfig):
    num_features: int | Literal["infer"] = "infer"
    eps: float = Field(1e-5, gt=0)
    momentum: float = Field(0.1, gt=0, le=1)
    affine: bool = True
    track_running_stats: bool = True
    # train/eval behaviour made explicit: "follow" uses the model's mode (model.train() / model.eval()); "train" always normalises with
    # batch statistics and updates the running statistics; "eval" always uses the running statistics.
    mode: Literal["follow", "train", "eval"] = "follow"


class BatchNormResolved(BatchNormConfig):
    rank: int = 2


@register
class BatchNormOp(_Op):
    type = "tensor.batchnorm"
    Config = BatchNormConfig

    def resolve(self, cfg, inputs):
        t = inputs["input"]
        if len(t.shape) in (2, 3, 4):
            nf = t.shape[1] if cfg.num_features == "infer" and isinstance(t.shape[1], int) else cfg.num_features
            return BatchNormResolved.model_validate({**cfg.model_dump(), "num_features": nf, "rank": len(t.shape)})
        return cfg

    def infer_shape(self, cfg, inputs):
        t = inputs["input"]
        _need(t, ("float32",), "input", self.type)
        if len(t.shape) not in (2, 3, 4):
            raise OpError("E_RANK_MISMATCH", f"batchnorm expects rank 2 [N,C], 3 [N,C,L] or 4 [N,C,H,W], got {list(t.shape)}.", "input")
        if cfg.num_features == "infer" or cfg.num_features != t.shape[1]:
            raise OpError("E_CHANNEL_MISMATCH", f"batchnorm expects {cfg.num_features} features (axis 1) but the input has {t.shape[1]}.", "input",
                          [Fix('Set num_features to "infer"', key="num_features", value="infer")])
        return {"output": t}

    def param_count(self, cfg, inputs):
        return 2 * cfg.num_features if cfg.affine else 0

    def has_state(self, cfg, params):
        return True  # running statistics are persistent buffers; sharing the call site shares them too

    def lower(self, cfg):
        return BatchNorm(cfg.rank, cfg.num_features, cfg.eps, cfg.momentum, cfg.affine, cfg.track_running_stats, cfg.mode)

    def explain(self, cfg, inputs, outputs):
        c = cfg.num_features
        terms, total = _terms(("weight (gamma)", [c]), ("bias (beta)", [c])) if cfg.affine else ([], 0)
        eff = {"follow": "train mode: batch statistics, running statistics updated; eval mode: running statistics",
               "train": "always batch statistics (running statistics updated)", "eval": "always running statistics (never updated)"}[cfg.mode]
        return {"equation": f"y = (x - mu) / sqrt(var + {cfg.eps}) * gamma + beta per channel; mu, var over batch and spatial axes (train) or running mean/var (eval)",
                "parameters": {"formula": "2 * num_features" if cfg.affine else "0", "terms": terms, "total": total},
                "note": f"Mode '{cfg.mode}': {eff}. Persistent buffers (not parameters): running_mean, running_var, num_batches_tracked."}

    def codegen(self, cfg):
        cls = "BatchNorm2d" if cfg.rank == 4 else "BatchNorm1d"
        return f"nn.{cls}({cfg.num_features}, eps={cfg.eps}, momentum={cfg.momentum}, affine={cfg.affine}, track_running_stats={cfg.track_running_stats})", "{m}({0})"


class Dropout(nn.Module):
    def __init__(self, p, mode):
        super().__init__()
        self.p, self.mode = p, mode

    def forward(self, x):
        training = self.training if self.mode == "follow" else self.mode == "train"
        return F.dropout(x, self.p, training)


class DropoutConfig(StrictConfig):
    p: float = Field(0.1, ge=0, lt=1)
    mode: Literal["follow", "train", "eval"] = "follow"  # explicit train/eval behaviour, see BatchNormConfig


@register
class DropoutOp(_Op):
    type = "tensor.dropout"
    Config = DropoutConfig

    def infer_shape(self, cfg, inputs):
        _need(inputs["input"], FLOATS, "input", self.type)
        return {"output": inputs["input"]}

    def lower(self, cfg):
        return Dropout(cfg.p, cfg.mode)

    def explain(self, cfg, inputs, outputs):
        eff = {"follow": "active only when the model is in train mode", "train": "always active (even in eval)", "eval": "never active (identity)"}[cfg.mode]
        return self._x(f"train: y = x * m / (1 - {cfg.p}), m ~ Bernoulli(1 - {cfg.p}) per element; eval: y = x",
                       note=f"Mode '{cfg.mode}': {eff}. Consumes the global torch random stream when active, so seeds and the position in the stream matter for reproducibility.")

    def codegen(self, cfg):
        return f"nn.Dropout({cfg.p})", "{m}({0})"


class EmbeddingConfig(StrictConfig):
    num_embeddings: int = Field(16, gt=0)
    embedding_dim: int = Field(8, gt=0)
    padding_idx: int | None = None


@register
class EmbeddingOp(_Op):
    type = "tensor.embedding"
    Config = EmbeddingConfig

    def infer_shape(self, cfg, inputs):
        t = inputs["input"]
        _need(t, ("int64",), "input", self.type)
        return {"output": TensorType(tuple(t.shape) + (cfg.embedding_dim,), "float32")}

    def param_count(self, cfg, inputs):
        return cfg.num_embeddings * cfg.embedding_dim

    def lower(self, cfg):
        return nn.Embedding(cfg.num_embeddings, cfg.embedding_dim, padding_idx=cfg.padding_idx)

    def explain(self, cfg, inputs, outputs):
        terms, total = _terms(("weight", [cfg.num_embeddings, cfg.embedding_dim]))
        return {"equation": "out[..., :] = W[index[...], :] (row lookup; ids must be in [0, num_embeddings))",
                "shapeRule": "[...] int64 -> [..., embedding_dim]",
                "parameters": {"formula": "num_embeddings * embedding_dim", "terms": terms, "total": total},
                "note": "Only the looked-up rows receive gradient." + (f" Row {cfg.padding_idx} is the padding row: it is zero-initialised and gets no gradient." if cfg.padding_idx is not None else "")}

    def codegen(self, cfg):
        return f"nn.Embedding({cfg.num_embeddings}, {cfg.embedding_dim}, padding_idx={cfg.padding_idx})", "{m}({0})"


class DenseConfig(StrictConfig):
    in_features: int | Literal["infer"] = "infer"
    out_features: int = Field(8, gt=0)
    bias: bool = True


@register
class DenseOp(_Op):
    type = "tensor.dense"
    Config = DenseConfig

    def resolve(self, cfg, inputs):
        t = inputs["input"]
        if cfg.in_features == "infer" and len(t.shape) >= 2 and isinstance(t.shape[-1], int):
            return cfg.model_copy(update={"in_features": t.shape[-1]})
        return cfg

    def infer_shape(self, cfg, inputs):
        t = inputs["input"]
        _need(t, ("float32",), "input", self.type)
        if len(t.shape) < 2:
            raise OpError("E_RANK_MISMATCH", f"dense expects rank >= 2 [..., features], got {list(t.shape)}.", "input")
        if cfg.in_features != t.shape[-1]:
            raise OpError("E_CHANNEL_MISMATCH", f"dense expects {cfg.in_features} input features (last axis) but the input has {t.shape[-1]}.", "input",
                          [Fix('Set in_features to "infer"', key="in_features", value="infer")])
        return {"output": TensorType(tuple(t.shape[:-1]) + (cfg.out_features,), t.dtype)}

    def param_count(self, cfg, inputs):
        return cfg.in_features * cfg.out_features + (cfg.out_features if cfg.bias else 0)

    def lower(self, cfg):
        return nn.Linear(cfg.in_features, cfg.out_features, cfg.bias)

    def explain(self, cfg, inputs, outputs):
        terms, total = _terms(("weight", [cfg.out_features, cfg.in_features]), *([("bias", [cfg.out_features])] if cfg.bias else []))
        return {"equation": "y[..., o] = b[o] + sum_i W[o, i] * x[..., i] (applied to the last axis; all leading axes are batch-like)",
                "shapeRule": "[..., in] -> [..., out]",
                "parameters": {"formula": "in_features * out_features" + (" + out_features" if cfg.bias else ""), "terms": terms, "total": total}}

    def codegen(self, cfg):
        return f"nn.Linear({cfg.in_features}, {cfg.out_features}, bias={cfg.bias})", "{m}({0})"
