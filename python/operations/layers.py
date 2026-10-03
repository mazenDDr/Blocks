"""pytorch.nn.* and pytorch.loss.* operations."""
from __future__ import annotations

from typing import Literal

import torch.nn as nn
from pydantic import Field, model_validator

from graph_core.registry import Operation, register
from graph_core.types import Fix, OpError, TensorType

from ._common import EmptyConfig, NonNegPair, PosPair, StrictConfig, prod_dims, require_dtype, require_rank, spatial_out

F32 = ("float32",)
NO_PARAMS = {"formula": "0", "terms": [], "total": 0}


def _terms(*items: tuple[str, list[int]]) -> tuple[list[dict], int]:
    terms, total = [], 0
    for name, shape in items:
        n = 1
        for s in shape:
            n *= s
        terms.append({"name": name, "shape": shape, "count": n})
        total += n
    return terms, total


# ---------------------------------------------------------------- conv2d
class Conv2dConfig(StrictConfig):
    in_channels: int | Literal["infer"] = "infer"
    out_channels: int = Field(8, gt=0)
    kernel_size: PosPair = (3, 3)
    stride: PosPair = (1, 1)
    padding: NonNegPair | Literal["same", "valid"] = (0, 0)
    dilation: PosPair = (1, 1)
    groups: int = Field(1, gt=0)
    bias: bool = True
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros"

    @model_validator(mode="after")
    def _check(self):
        if isinstance(self.in_channels, int) and self.in_channels <= 0:
            raise ValueError("in_channels must be positive or 'infer'")
        return self


@register
class Conv2d(Operation):
    type = "pytorch.nn.conv2d"
    Config = Conv2dConfig

    def resolve(self, cfg, inputs):
        t = inputs["input"]
        if cfg.in_channels == "infer" and len(t.shape) == 4 and isinstance(t.shape[1], int):
            return cfg.model_copy(update={"in_channels": t.shape[1]})
        return cfg

    def _pads(self, cfg):
        if cfg.padding == "valid":
            return (0, 0)
        if cfg.padding == "same":
            return None
        return cfg.padding

    def infer_shape(self, cfg, inputs):
        t = inputs["input"]
        require_dtype(t, F32, "input", "Conv2d")
        require_rank(t, 4, "input", "Conv2d")
        n, c, h, w = t.shape
        if cfg.in_channels == "infer":
            raise OpError("E_CHANNEL_MISMATCH", "in_channels cannot be inferred from the input.", "input")
        if cfg.in_channels != c:
            raise OpError(
                "E_CHANNEL_MISMATCH",
                f"Conv2d expects {cfg.in_channels} input channels (locked) but the connected tensor has {c}.",
                "input",
                [Fix('Set in_channels to "infer"', key="in_channels", value="infer"),
                 Fix(f"Set in_channels to {c}", key="in_channels", value=c)],
            )
        if cfg.in_channels % cfg.groups or cfg.out_channels % cfg.groups:
            raise OpError("E_GROUPS_INVALID", f"in_channels ({cfg.in_channels}) and out_channels ({cfg.out_channels}) must both be divisible by groups ({cfg.groups}).", "input",
                          [Fix("Set groups to 1", key="groups", value=1)])
        pads = self._pads(cfg)
        if pads is None:
            if cfg.stride != (1, 1):
                raise OpError("E_CONFIG", "padding='same' requires stride 1.", "input", [Fix("Set stride to 1", key="stride", value=[1, 1])])
            oh, ow = h, w
        else:
            oh = spatial_out(h, cfg.kernel_size[0], cfg.stride[0], pads[0], cfg.dilation[0])
            ow = spatial_out(w, cfg.kernel_size[1], cfg.stride[1], pads[1], cfg.dilation[1])
        if oh <= 0 or ow <= 0:
            raise OpError("E_SHAPE_INVALID", f"Conv2d output would be {oh}x{ow}; the kernel does not fit the input {h}x{w}.", "input",
                          [Fix("Increase padding", key="padding", value=[cfg.kernel_size[0] // 2, cfg.kernel_size[1] // 2])])
        return {"output": TensorType((n, cfg.out_channels, oh, ow), t.dtype)}

    def _count(self, cfg):
        kh, kw = cfg.kernel_size
        return _terms(("weight", [cfg.out_channels, cfg.in_channels // cfg.groups, kh, kw]),
                      *([("bias", [cfg.out_channels])] if cfg.bias else []))

    def param_count(self, cfg, inputs):
        return self._count(cfg)[1]

    def lower(self, cfg):
        return nn.Conv2d(cfg.in_channels, cfg.out_channels, cfg.kernel_size, cfg.stride, cfg.padding, cfg.dilation,
                         cfg.groups, cfg.bias, cfg.padding_mode)

    def explain(self, cfg, inputs, outputs):
        terms, total = self._count(cfg)
        return {
            "equation": "y[n,co,i,j] = b[co] + sum_{ci in group(co), u, v} W[co,ci,u,v] * x[n,ci,i*s_h - p_h + d_h*u, j*s_w - p_w + d_w*v]",
            "shapeRule": "out = floor((in + 2*pad - dilation*(kernel-1) - 1) / stride) + 1",
            "parameters": {"formula": "out_channels * (in_channels / groups) * k_h * k_w" + (" + out_channels" if cfg.bias else ""),
                           "terms": terms, "total": total},
        }

    def codegen(self, cfg):
        return (f"nn.Conv2d({cfg.in_channels}, {cfg.out_channels}, kernel_size={cfg.kernel_size!r}, stride={cfg.stride!r}, "
                f"padding={cfg.padding!r}, dilation={cfg.dilation!r}, groups={cfg.groups}, bias={cfg.bias}, "
                f"padding_mode={cfg.padding_mode!r})"), "{m}({0})"


# ---------------------------------------------------------------- relu
@register
class ReLU(Operation):
    type = "pytorch.nn.relu"
    Config = EmptyConfig

    def infer_shape(self, cfg, inputs):
        return {"output": inputs["input"]}

    def lower(self, cfg):
        return nn.ReLU()

    def explain(self, cfg, inputs, outputs):
        return {"equation": "y = max(0, x) (elementwise)", "parameters": NO_PARAMS}

    def codegen(self, cfg):
        return "nn.ReLU()", "{m}({0})"


# ---------------------------------------------------------------- max_pool2d
class MaxPoolConfig(StrictConfig):
    kernel_size: PosPair = (2, 2)
    stride: PosPair | None = None  # None means = kernel_size, as in PyTorch
    padding: NonNegPair = (0, 0)
    dilation: PosPair = (1, 1)
    ceil_mode: bool = False


@register
class MaxPool2d(Operation):
    type = "pytorch.nn.max_pool2d"
    Config = MaxPoolConfig

    def _stride(self, cfg):
        return cfg.stride or cfg.kernel_size

    def infer_shape(self, cfg, inputs):
        t = inputs["input"]
        require_rank(t, 4, "input", "MaxPool2d")
        n, c, h, w = t.shape
        s = self._stride(cfg)
        if any(p > k // 2 for p, k in zip(cfg.padding, cfg.kernel_size)):
            raise OpError("E_CONFIG", "MaxPool2d padding must be at most half the kernel size.", "input",
                          [Fix("Set padding to 0", key="padding", value=[0, 0])])
        oh = spatial_out(h, cfg.kernel_size[0], s[0], cfg.padding[0], cfg.dilation[0], cfg.ceil_mode)
        ow = spatial_out(w, cfg.kernel_size[1], s[1], cfg.padding[1], cfg.dilation[1], cfg.ceil_mode)
        if oh <= 0 or ow <= 0:
            raise OpError("E_SHAPE_INVALID", f"MaxPool2d output would be {oh}x{ow} for input {h}x{w}.", "input")
        return {"output": TensorType((n, c, oh, ow), t.dtype)}

    def lower(self, cfg):
        return nn.MaxPool2d(cfg.kernel_size, self._stride(cfg), cfg.padding, cfg.dilation, ceil_mode=cfg.ceil_mode)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "y[n,c,i,j] = max_{u,v in window} x[n,c,i*s_h - p_h + d_h*u, j*s_w - p_w + d_w*v]",
                "shapeRule": "out = floor((in + 2*pad - dilation*(kernel-1) - 1) / stride) + 1 (ceil if ceil_mode)",
                "parameters": NO_PARAMS}

    def codegen(self, cfg):
        return (f"nn.MaxPool2d(kernel_size={cfg.kernel_size!r}, stride={self._stride(cfg)!r}, padding={cfg.padding!r}, "
                f"dilation={cfg.dilation!r}, ceil_mode={cfg.ceil_mode})"), "{m}({0})"


# ---------------------------------------------------------------- adaptive_avg_pool2d
class AdaptivePoolConfig(StrictConfig):
    output_size: PosPair = (1, 1)


@register
class AdaptiveAvgPool2d(Operation):
    type = "pytorch.nn.adaptive_avg_pool2d"
    Config = AdaptivePoolConfig

    def infer_shape(self, cfg, inputs):
        t = inputs["input"]
        require_rank(t, 4, "input", "AdaptiveAvgPool2d")
        return {"output": TensorType((t.shape[0], t.shape[1], *cfg.output_size), t.dtype)}

    def lower(self, cfg):
        return nn.AdaptiveAvgPool2d(cfg.output_size)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "y[n,c,i,j] = mean of x[n,c] over the input bin mapped to output cell (i,j)"
                            + ("; with output 1x1 this is the mean over all H*W positions" if cfg.output_size == (1, 1) else ""),
                "parameters": NO_PARAMS}

    def codegen(self, cfg):
        return f"nn.AdaptiveAvgPool2d({cfg.output_size!r})", "{m}({0})"


# ---------------------------------------------------------------- flatten
class FlattenConfig(StrictConfig):
    start_dim: int = Field(1, ge=1)  # the batch axis (0) is always preserved
    end_dim: int = -1


@register
class Flatten(Operation):
    type = "pytorch.nn.flatten"
    Config = FlattenConfig

    def infer_shape(self, cfg, inputs):
        t = inputs["input"]
        rank = len(t.shape)
        end = cfg.end_dim % rank if -rank <= cfg.end_dim < rank else None
        if end is None or cfg.start_dim >= rank or cfg.start_dim > end:
            raise OpError("E_CONFIG", f"Flatten dims ({cfg.start_dim}, {cfg.end_dim}) are invalid for rank {rank}.", "input")
        flat = prod_dims(list(t.shape[cfg.start_dim:end + 1]))
        return {"output": TensorType((*t.shape[:cfg.start_dim], flat, *t.shape[end + 1:]), t.dtype)}

    def lower(self, cfg):
        return nn.Flatten(cfg.start_dim, cfg.end_dim)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "y = reshape(x) merging dims start_dim..end_dim into one, keeping the batch axis", "parameters": NO_PARAMS}

    def codegen(self, cfg):
        return f"nn.Flatten(start_dim={cfg.start_dim}, end_dim={cfg.end_dim})", "{m}({0})"


# ---------------------------------------------------------------- linear
class LinearConfig(StrictConfig):
    in_features: int | Literal["infer"] = "infer"
    out_features: int = Field(10, gt=0)
    bias: bool = True


@register
class Linear(Operation):
    type = "pytorch.nn.linear"
    Config = LinearConfig

    def resolve(self, cfg, inputs):
        t = inputs["input"]
        if cfg.in_features == "infer" and len(t.shape) >= 2 and isinstance(t.shape[-1], int):
            return cfg.model_copy(update={"in_features": t.shape[-1]})
        return cfg

    def infer_shape(self, cfg, inputs):
        t = inputs["input"]
        require_dtype(t, F32, "input", "Linear")
        if len(t.shape) < 2:
            raise OpError("E_RANK_MISMATCH", f"Linear expects rank >= 2 input, got {list(t.shape)}.", "input")
        last = t.shape[-1]
        if cfg.in_features == "infer" or not isinstance(last, int):
            raise OpError("E_FEATURE_MISMATCH", "in_features cannot be inferred from the input.", "input")
        if cfg.in_features != last:
            raise OpError("E_FEATURE_MISMATCH",
                          f"Linear expects {cfg.in_features} input features (locked) but the connected tensor has {last}.", "input",
                          [Fix('Set in_features to "infer"', key="in_features", value="infer"),
                           Fix(f"Set in_features to {last}", key="in_features", value=last)])
        return {"output": TensorType((*t.shape[:-1], cfg.out_features), t.dtype)}

    def _count(self, cfg):
        return _terms(("weight", [cfg.out_features, cfg.in_features]), *([("bias", [cfg.out_features])] if cfg.bias else []))

    def param_count(self, cfg, inputs):
        return self._count(cfg)[1]

    def lower(self, cfg):
        return nn.Linear(cfg.in_features, cfg.out_features, cfg.bias)

    def explain(self, cfg, inputs, outputs):
        terms, total = self._count(cfg)
        return {"equation": "y = x W^T + b" if cfg.bias else "y = x W^T",
                "parameters": {"formula": "out_features * in_features" + (" + out_features" if cfg.bias else ""),
                               "terms": terms, "total": total}}

    def codegen(self, cfg):
        return f"nn.Linear({cfg.in_features}, {cfg.out_features}, bias={cfg.bias})", "{m}({0})"


# ---------------------------------------------------------------- cross_entropy
class CrossEntropyConfig(StrictConfig):
    reduction: Literal["mean", "sum", "none"] = "mean"


@register
class CrossEntropy(Operation):
    type = "pytorch.loss.cross_entropy"
    inputs = ("logits", "target")
    Config = CrossEntropyConfig

    def infer_shape(self, cfg, inputs):
        z, y = inputs["logits"], inputs["target"]
        require_dtype(z, F32, "logits", "CrossEntropy")
        require_rank(z, 2, "logits", "CrossEntropy")
        require_dtype(y, ("int64",), "target", "CrossEntropy")
        require_rank(y, 1, "target", "CrossEntropy")
        if y.shape[0] != z.shape[0]:
            raise OpError("E_SHAPE_MISMATCH", f"target batch {y.shape[0]} != logits batch {z.shape[0]}.", "target")
        return {"output": TensorType((z.shape[0],) if cfg.reduction == "none" else (), z.dtype)}

    def lower(self, cfg):
        return nn.CrossEntropyLoss(reduction=cfg.reduction)

    def explain(self, cfg, inputs, outputs):
        red = {"mean": "mean over the N samples (divisor N)", "sum": "sum over samples", "none": "per-sample losses, unreduced"}[cfg.reduction]
        return {"equation": "l_n = -z_n[y_n] + log(sum_k exp(z_n[k]))  (fused, numerically stable log-softmax)",
                "reduction": red, "parameters": NO_PARAMS}

    def codegen(self, cfg):
        return f"nn.CrossEntropyLoss(reduction={cfg.reduction!r})", "{m}({0}, {1})"
