"""Backend-specific operations (VISION 14.2): nodes with no sound common equivalent. Each one is bound to ONE backend; a graph on any other
backend is rejected before execution (E_BACKEND_OP). They are shape-inferable everywhere (so the editor can show shapes) but only lowered
by the backend named in `backend`. Their semantics are the backend library's own and are NOT claimed to match any other framework.

  keras.layers.separable_conv2d   Keras SeparableConv2D: a depthwise conv then a 1x1 pointwise conv, with Keras padding semantics
  jax.lax.cumsum                  jax.lax.cumsum: an inclusive cumulative sum along one axis, optionally reversed
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import Field

from graph_core.registry import Operation, register
from graph_core.types import Fix, OpError, TensorType

from ._common import StrictConfig, require_dtype, require_rank
from .layers import F32, NO_PARAMS, _terms


class SeparableConv2dConfig(StrictConfig):
    in_channels: int | Literal["infer"] = "infer"
    out_channels: int = Field(8, gt=0)
    kernel_size: tuple[int, int] = (3, 3)
    stride: tuple[int, int] = (1, 1)
    padding: Literal["valid", "same"] = "valid"  # Keras semantics: 'same' pads so that out = ceil(in / stride), extra padding at the end
    depth_multiplier: int = Field(1, gt=0)
    bias: bool = True


@register
class KerasSeparableConv2d(Operation):
    type = "keras.layers.separable_conv2d"
    backend = "keras"
    Config = SeparableConv2dConfig

    def resolve(self, cfg, inputs):
        t = inputs["input"]
        if cfg.in_channels == "infer" and len(t.shape) == 4 and isinstance(t.shape[1], int):
            return cfg.model_copy(update={"in_channels": t.shape[1]})
        return cfg

    def infer_shape(self, cfg, inputs):
        t = inputs["input"]
        require_dtype(t, F32, "input", "SeparableConv2D")
        require_rank(t, 4, "input", "SeparableConv2D")
        n, c, h, w = t.shape
        if cfg.in_channels == "infer":
            raise OpError("E_CHANNEL_MISMATCH", "in_channels cannot be inferred from the input.", "input")
        if cfg.in_channels != c:
            raise OpError("E_CHANNEL_MISMATCH", f"SeparableConv2D expects {cfg.in_channels} input channels (locked) but the connected tensor has {c}.", "input",
                          [Fix('Set in_channels to "infer"', key="in_channels", value="infer"), Fix(f"Set in_channels to {c}", key="in_channels", value=c)])
        if min(cfg.kernel_size) < 1 or min(cfg.stride) < 1:
            raise OpError("E_CONFIG", "kernel_size and stride must be positive.", "input")
        dims = []
        for size, k, s in zip((h, w), cfg.kernel_size, cfg.stride):
            dims.append(math.ceil(size / s) if cfg.padding == "same" else (size - k) // s + 1)
        if min(dims) <= 0:
            raise OpError("E_SHAPE_INVALID", f"SeparableConv2D output would be {dims[0]}x{dims[1]}; the kernel does not fit the input {h}x{w}.", "input")
        return {"output": TensorType((n, cfg.out_channels, dims[0], dims[1]), t.dtype)}

    def _count(self, cfg):
        kh, kw = cfg.kernel_size
        cm = cfg.in_channels * cfg.depth_multiplier
        return _terms(("depthwise_weight", [cm, 1, kh, kw]), ("pointwise_weight", [cfg.out_channels, cm, 1, 1]),
                      *([("bias", [cfg.out_channels])] if cfg.bias else []))

    def param_count(self, cfg, inputs):
        return self._count(cfg)[1]

    def lower(self, cfg):
        raise NotImplementedError("keras.layers.separable_conv2d is a Keras-specific node; it has no PyTorch lowering")

    def explain(self, cfg, inputs, outputs):
        terms, total = self._count(cfg)
        return {"equation": "d[n,c*m+q,i,j] = sum_{u,v} Wd[c*m+q,0,u,v] * x[n,c,i*s_h + u, j*s_w + v];  y[n,o,i,j] = b[o] + sum_{k} Wp[o,k,0,0] * d[n,k,i,j]",
                "shapeRule": "valid: floor((in - kernel) / stride) + 1;  same (Keras): ceil(in / stride)",
                "parameters": {"formula": "in_channels*depth_multiplier*k_h*k_w + out_channels*in_channels*depth_multiplier" + (" + out_channels" if cfg.bias else ""),
                               "terms": terms, "total": total},
                "note": "Keras-specific: depthwise then pointwise convolution in one layer. Not a PyTorch module; no cross-framework equivalence is claimed."}

    def codegen(self, cfg):
        raise NotImplementedError("keras.layers.separable_conv2d has no PyTorch export")


class CumsumConfig(StrictConfig):
    axis: int = 1
    reverse: bool = False


@register
class JaxCumsum(Operation):
    type = "jax.lax.cumsum"
    backend = "jax"
    Config = CumsumConfig

    def infer_shape(self, cfg, inputs):
        t = inputs["input"]
        require_dtype(t, F32 + ("float64",), "input", "jax.lax.cumsum")
        rank = len(t.shape)
        if not 0 <= cfg.axis < rank:
            raise OpError("E_CONFIG", f"axis {cfg.axis} is out of range for rank {rank} (jax.lax.cumsum takes a non-negative axis).", "input")
        return {"output": t}

    def lower(self, cfg):
        raise NotImplementedError("jax.lax.cumsum is a JAX-specific node; it has no PyTorch lowering")

    def explain(self, cfg, inputs, outputs):
        return {"equation": f"y[..., i, ...] = sum_{{j <= i}} x[..., j, ...] along axis {cfg.axis}" + (" (reversed: j >= i)" if cfg.reverse else ""),
                "parameters": NO_PARAMS, "note": "jax.lax.cumsum: inclusive scan; a JAX-specific node with no common equivalent claimed."}

    def codegen(self, cfg):
        raise NotImplementedError("jax.lax.cumsum has no PyTorch export")
