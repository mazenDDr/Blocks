"""Single-operation conformance cases: ONE table drives the cross-backend tests (tests/test_backend_conformance.py) and the coverage ledger
(a ledger entry's `tests` column is counted from these cases, so it cannot claim a case that does not exist).

A case builds a one-operation graph (inputs `x` or `a`/`b` or `logits`/`target`, node `op`), and a handwritten NATIVE PyTorch reference.
Every backend runs the graph with the reference's own weights copied in (graph layout) and must match it within backends/tolerances.py.
`expect` lists backends that must REFUSE the case before execution, with the stable error code; absent = must run."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Case:
    id: str
    op: str
    config: dict
    inputs: dict  # input node id -> (shape (ints, batch concrete), dtype)
    native: Callable  # () -> (fn(*input tensors) -> tensor, {param name: torch tensor})
    expect: dict = field(default_factory=dict)  # backend -> refusal code
    classes: int = 0  # for int targets: values in [0, classes)

    def shapes(self):
        return {k: (["N", *s[1:]] if s else [], d) for k, (s, d) in self.inputs.items()}


def _mod(m: nn.Module):
    return (lambda *a: m(*a)), {k: v for k, v in m.named_parameters()}


def _f(fn):
    return lambda: (fn, {})


def conv(cid, cin, cout, k, hw, config, mod_kwargs, expect=None, h=None):
    x = (2, cin, *hw)
    return Case(cid, "pytorch.nn.conv2d", {"out_channels": cout, "kernel_size": k, **config}, {"x": (x, "float32")},
                lambda: _mod(nn.Conv2d(cin, cout, k, **mod_kwargs)), expect or {})


PM = "E_BACKEND_UNSUPPORTED_PADDING_MODE"
DIL = "E_BACKEND_UNSUPPORTED_DILATION"
CFG = "E_BACKEND_UNSUPPORTED_CONFIG"
DT = "E_BACKEND_UNSUPPORTED_DTYPE"

CASES: list[Case] = [
    conv("conv_pad1", 3, 8, 3, (9, 9), {"padding": [1, 1]}, {"padding": 1}),
    conv("conv_valid", 3, 5, 3, (9, 8), {"padding": "valid"}, {"padding": "valid"}),
    conv("conv_same_odd", 3, 5, 3, (8, 8), {"padding": "same"}, {"padding": "same"}),
    conv("conv_same_even_kernel", 2, 4, 4, (8, 7), {"padding": "same"}, {"padding": "same"}),
    conv("conv_stride2_pad", 3, 6, 3, (11, 10), {"stride": [2, 2], "padding": [1, 1]}, {"stride": 2, "padding": 1}),
    conv("conv_rect_kernel_asym_pad", 2, 4, (3, 5), (10, 12), {"kernel_size": [3, 5], "padding": [1, 2], "stride": [1, 2]}, {"padding": (1, 2), "stride": (1, 2)}),
    conv("conv_dilated", 3, 4, 3, (12, 12), {"dilation": [2, 2], "padding": [2, 2]}, {"dilation": 2, "padding": 2}),
    conv("conv_dilated_strided", 3, 4, 3, (12, 12), {"dilation": [2, 2], "stride": [2, 2]}, {"dilation": 2, "stride": 2}, {"keras": DIL}),
    conv("conv_groups", 4, 6, 3, (8, 8), {"groups": 2, "padding": [1, 1]}, {"groups": 2, "padding": 1}),
    conv("conv_depthwise", 4, 4, 3, (8, 8), {"groups": 4, "padding": [1, 1]}, {"groups": 4, "padding": 1}),
    conv("conv_no_bias", 3, 4, 3, (8, 8), {"bias": False, "padding": [1, 1]}, {"bias": False, "padding": 1}),
    conv("conv_reflect", 3, 4, 3, (8, 8), {"padding": [1, 1], "padding_mode": "reflect"}, {"padding": 1, "padding_mode": "reflect"}, {"keras": PM}),
    conv("conv_replicate", 3, 4, 3, (8, 8), {"padding": [2, 1], "padding_mode": "replicate"}, {"padding": (2, 1), "padding_mode": "replicate"}, {"keras": PM}),
    conv("conv_circular", 3, 4, 3, (8, 8), {"padding": [1, 1], "padding_mode": "circular"}, {"padding": 1, "padding_mode": "circular"}, {"keras": PM}),
    conv("conv_same_reflect", 2, 3, 3, (8, 8), {"padding": "same", "padding_mode": "reflect"}, {"padding": "same", "padding_mode": "reflect"}, {"keras": PM}),
    Case("relu", "pytorch.nn.relu", {}, {"x": ((3, 4, 5, 5), "float32")}, lambda: _mod(nn.ReLU())),
    Case("max_pool_2x2", "pytorch.nn.max_pool2d", {"kernel_size": [2, 2]}, {"x": ((2, 3, 8, 8), "float32")}, lambda: _mod(nn.MaxPool2d(2))),
    Case("max_pool_k3_s2", "pytorch.nn.max_pool2d", {"kernel_size": [3, 3], "stride": [2, 2]}, {"x": ((2, 3, 9, 10), "float32")}, lambda: _mod(nn.MaxPool2d(3, 2))),
    Case("max_pool_padding", "pytorch.nn.max_pool2d", {"kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1]}, {"x": ((2, 3, 9, 9), "float32")},
         lambda: _mod(nn.MaxPool2d(3, 2, 1)), {"keras": CFG}),
    Case("max_pool_ceil_mode", "pytorch.nn.max_pool2d", {"kernel_size": [2, 2], "ceil_mode": True}, {"x": ((2, 3, 7, 7), "float32")},
         lambda: _mod(nn.MaxPool2d(2, ceil_mode=True)), {"keras": CFG}),
    Case("max_pool_dilation", "pytorch.nn.max_pool2d", {"kernel_size": [2, 2], "stride": [1, 1], "dilation": [2, 2]}, {"x": ((2, 3, 8, 8), "float32")},
         lambda: _mod(nn.MaxPool2d(2, 1, 0, 2)), {"keras": DIL, "jax": DIL}),
    Case("global_avg_pool", "pytorch.nn.adaptive_avg_pool2d", {"output_size": [1, 1]}, {"x": ((2, 5, 6, 7), "float32")}, lambda: _mod(nn.AdaptiveAvgPool2d(1))),
    Case("adaptive_avg_divisible", "pytorch.nn.adaptive_avg_pool2d", {"output_size": [2, 3]}, {"x": ((2, 3, 6, 9), "float32")}, lambda: _mod(nn.AdaptiveAvgPool2d((2, 3)))),
    Case("adaptive_avg_not_divisible", "pytorch.nn.adaptive_avg_pool2d", {"output_size": [2, 2]}, {"x": ((2, 3, 5, 5), "float32")},
         lambda: _mod(nn.AdaptiveAvgPool2d(2)), {"keras": CFG, "jax": CFG}),
    Case("flatten_4d", "pytorch.nn.flatten", {}, {"x": ((2, 3, 4, 5), "float32")}, lambda: _mod(nn.Flatten())),
    Case("flatten_from_dim2", "pytorch.nn.flatten", {"start_dim": 2}, {"x": ((2, 3, 4, 5), "float32")}, lambda: _mod(nn.Flatten(2))),
    Case("linear", "pytorch.nn.linear", {"out_features": 7}, {"x": ((4, 6), "float32")}, lambda: _mod(nn.Linear(6, 7))),
    Case("linear_no_bias_rank3", "pytorch.nn.linear", {"out_features": 5, "bias": False}, {"x": ((3, 4, 6), "float32")}, lambda: _mod(nn.Linear(6, 5, bias=False))),
    Case("cross_entropy_mean", "pytorch.loss.cross_entropy", {"reduction": "mean"}, {"logits": ((5, 7), "float32"), "target": ((5,), "int64")},
         _f(lambda z, y: F.cross_entropy(z, y)), classes=7),
    Case("cross_entropy_sum", "pytorch.loss.cross_entropy", {"reduction": "sum"}, {"logits": ((5, 7), "float32"), "target": ((5,), "int64")},
         _f(lambda z, y: F.cross_entropy(z, y, reduction="sum")), classes=7),
    Case("cross_entropy_none", "pytorch.loss.cross_entropy", {"reduction": "none"}, {"logits": ((5, 7), "float32"), "target": ((5,), "int64")},
         _f(lambda z, y: F.cross_entropy(z, y, reduction="none")), classes=7),
    Case("sub", "core.sub", {}, {"a": ((3, 4), "float32"), "b": ((3, 4), "float32")}, _f(lambda a, b: a - b)),
    Case("add", "core.add", {}, {"a": ((3, 4), "float32"), "b": ((3, 4), "float32")}, _f(lambda a, b: a + b)),
    Case("square", "core.square", {}, {"x": ((3, 4), "float32")}, _f(lambda x: torch.square(x))),
    Case("scalar_mul", "core.scalar_mul", {"factor": -2.5}, {"x": ((3, 4), "float32")}, _f(lambda x: x * -2.5)),
    Case("sum_all", "core.sum", {}, {"x": ((3, 4, 5), "float32")}, _f(lambda x: x.sum())),
    Case("sum_dims_keepdim", "core.sum", {"dims": [0, 2], "keepdim": True}, {"x": ((3, 4, 5), "float32")}, _f(lambda x: x.sum(dim=(0, 2), keepdim=True))),
    Case("mean_all", "core.mean", {}, {"x": ((3, 4, 5), "float32")}, _f(lambda x: x.mean())),
    Case("mean_dim_negative", "core.mean", {"dims": [-1]}, {"x": ((3, 4, 5), "float32")}, _f(lambda x: x.mean(dim=-1))),
    Case("sub_float64", "core.sub", {}, {"a": ((3, 4), "float64"), "b": ((3, 4), "float64")}, _f(lambda a, b: a - b), {"jax": DT}),
    Case("mean_float64", "core.mean", {}, {"x": ((3, 4, 5), "float64")}, _f(lambda x: x.mean()), {"jax": DT}),
    Case("square_float64", "core.square", {}, {"x": ((3, 4), "float64")}, _f(lambda x: torch.square(x)), {"jax": DT}),
]

BY_ID = {c.id: c for c in CASES}
assert len(BY_ID) == len(CASES)

# backend-specific nodes: reference is the backend's own native layer / function, not PyTorch (tests/test_backend_specific_nodes.py)
SPECIFIC_CASES = {
    "keras.layers.separable_conv2d": ["separable_valid", "separable_same_stride2_multiplier2", "separable_no_bias"],
    "jax.lax.cumsum": ["cumsum_axis1", "cumsum_reverse_axis2"],
}


# ------------------------------------------------------------------------------------------------ graphs and data for a case
_PORT = {"x": "input", "a": "a", "b": "b", "logits": "logits", "target": "target"}


def case_graph(c: Case, with_loss: bool = True):
    """The one-op graph. With `with_loss`, a non-scalar result is reduced by square -> sum into node `loss` (a scalar to differentiate);
    a scalar result is the loss itself (node `op`). Returns (graph, loss node id)."""
    from graph_core.build import GraphBuilder

    b = GraphBuilder()
    for k, (s, d) in c.inputs.items():
        b.input(k, ["N", *s[1:]] if s else [], d)
    b.node("op", c.op, **c.config)
    for k in c.inputs:
        b.wire(k, f"op.{_PORT[k]}")
    scalar = c.op in ("core.sum", "core.mean") and not c.config.get("dims") and not c.config.get("keepdim") or (
        c.op == "pytorch.loss.cross_entropy" and c.config.get("reduction") != "none")
    loss = "op"
    if with_loss and not scalar:
        b.node("sq", "core.square")
        b.node("loss", "core.sum")
        b.chain("op", "sq", "loss")
        loss = "loss"
    return b.build(), loss


def case_inputs(c: Case, rng) -> dict:
    import numpy as np

    out = {}
    for k, (s, d) in c.inputs.items():
        out[k] = rng.integers(0, c.classes, s).astype("int64") if d == "int64" else rng.standard_normal(s).astype(d)
    return out


def case_dtype(c: Case) -> str:
    return "float64" if any(d == "float64" for _, d in c.inputs.values()) else "float32"
