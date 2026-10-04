"""Parameter names and shapes in GRAPH layout (PyTorch layout): the one layout in which weights cross a backend boundary."""
from __future__ import annotations

import math

from .plan import ModelPlan, Step


def param_shapes(step: Step) -> dict[str, list[int]]:
    """name -> shape (graph layout) of the tensors owned by this step's module; {} for parameter-free operations."""
    c = step.cfg
    if step.type == "pytorch.nn.conv2d":
        kh, kw = c.kernel_size
        return {"weight": [c.out_channels, c.in_channels // c.groups, kh, kw], **({"bias": [c.out_channels]} if c.bias else {})}
    if step.type == "pytorch.nn.linear":
        return {"weight": [c.out_features, c.in_features], **({"bias": [c.out_features]} if c.bias else {})}
    if step.type == "keras.layers.separable_conv2d":
        kh, kw = c.kernel_size
        cm = c.in_channels * c.depth_multiplier
        return {"depthwise_weight": [cm, 1, kh, kw], "pointwise_weight": [c.out_channels, cm, 1, 1], **({"bias": [c.out_channels]} if c.bias else {})}
    return {}


def fan_in(step: Step) -> int:
    c = step.cfg
    if step.type == "pytorch.nn.conv2d":
        return (c.in_channels // c.groups) * c.kernel_size[0] * c.kernel_size[1]
    if step.type == "pytorch.nn.linear":
        return c.in_features
    return 1


def param_spec(plan: ModelPlan) -> dict[str, dict[str, list[int]]]:
    """owner node -> {param name: shape}; only nodes that own tensors."""
    return {s.nid: ps for s in plan.owners() if (ps := param_shapes(s))}


def torch_like_bound(step: Step) -> float:
    return 1.0 / math.sqrt(fan_in(step))
