"""JAX backend: rules, parameter initialization and exported source. Pure Python (nothing here imports jax).

Design (ADR 0010): plain `jax.numpy` / `jax.lax` functions over an explicit parameter pytree `{node id: {name: array}}`; no Flax. The graph
layout is already JAX-expressible (lax.conv_general_dilated accepts NCHW / OIHW), so there are NO layout conversions: weights and
activations stay in graph layout. State and randomness are explicit: parameters are an argument of `apply`, initialization takes a PRNG seed."""
from __future__ import annotations

import math

from .check import Check, Conversion, native, refuse
from .params import fan_in, param_shapes
from .plan import ModelPlan, Step

BACKEND = "jax"

INIT = ("jax.random with an explicit PRNG key: every weight and bias ~ Uniform(+-1/sqrt(fan_in)). This is the bound PyTorch uses for biases and the "
        "limit of its default Linear/Conv kernels (Kaiming-uniform with a=sqrt(5)), but the random stream is different, so the initial VALUES differ "
        "from PyTorch for any seed; compare backends only with copied weights.")

SUPPORTED_TYPES = {"core.tensor_input", "pytorch.nn.conv2d", "pytorch.nn.relu", "pytorch.nn.max_pool2d", "pytorch.nn.adaptive_avg_pool2d",
                   "pytorch.nn.flatten", "pytorch.nn.linear", "pytorch.loss.cross_entropy", "core.sub", "core.add", "core.square",
                   "core.sum", "core.mean", "core.scalar_mul", "jax.lax.cumsum"}
RESERVED_NAMES = {"jax", "jnp", "lax", "np", "params", "init_params", "apply", "key", "sub", "math", "k_weight", "k_bias"}
PAD_MODES = {"reflect": "reflect", "replicate": "edge", "circular": "wrap"}


def conv_pads(c) -> tuple[tuple[int, int], tuple[int, int]]:
    """(lo, hi) per spatial dim. 'same' uses PyTorch's split: the lower side gets the floor of the total padding."""
    if c.padding == "same":
        out = []
        for k, d in zip(c.kernel_size, c.dilation):
            total = d * (k - 1)
            out.append((total // 2, total - total // 2))
        return tuple(out)  # type: ignore[return-value]
    if c.padding == "valid":
        return ((0, 0), (0, 0))
    return ((c.padding[0], c.padding[0]), (c.padding[1], c.padding[1]))


def pool_pads(c, h: int, w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """(lo, hi) per spatial dim; hi includes the extra right padding ceil_mode needs (windows may run off the end, as in PyTorch)."""
    s = c.stride or c.kernel_size
    out = []
    for size, k, st, p, d in zip((h, w), c.kernel_size, s, c.padding, c.dilation):
        num = size + 2 * p - d * (k - 1) - 1
        o = (math.ceil(num / st) if c.ceil_mode else num // st) + 1
        if c.ceil_mode and (o - 1) * st >= size + p:
            o -= 1
        need = (o - 1) * st + d * (k - 1) + 1
        out.append((p, max(p, need - size - p)))
    return tuple(out)  # type: ignore[return-value]


def check_step(step: Step) -> Check:
    t = step.type
    if t not in SUPPORTED_TYPES:
        return refuse("E_BACKEND_UNSUPPORTED_OP", f"'{t}' has no JAX implementation in this adapter (portable subset only). It is not replaced by a similar operation.")
    conv: list[Conversion] = []
    seen_int64 = False
    for ty in list(step.in_types.values()) + [step.out_type]:
        if ty.dtype == "float64":
            return refuse("E_BACKEND_UNSUPPORTED_DTYPE", "float64: JAX silently computes in float32 unless jax_enable_x64 is set globally; this adapter refuses rather than downcast or change global JAX state.")
        if ty.dtype not in ("float32", "int64", "bool"):
            return refuse("E_BACKEND_UNSUPPORTED_DTYPE", f"dtype {ty.dtype} is not supported")
        if ty.dtype == "int64" and not seen_int64:
            seen_int64 = True
            conv.append(Conversion("dtype", "int64 -> int32 (JAX has no int64 without jax_enable_x64); values outside the int32 range are rejected when fed"))
    if t == "pytorch.nn.conv2d":
        c = step.cfg
        pads = conv_pads(c)
        if c.padding_mode != "zeros" and pads != ((0, 0), (0, 0)):
            conv.append(Conversion("padding", f"padding_mode='{c.padding_mode}' -> jnp.pad(mode='{PAD_MODES[c.padding_mode]}') with pads {[list(p) for p in pads]}, then an unpadded convolution"))
        elif c.padding == "same":
            conv.append(Conversion("padding", f"padding='same' -> explicit (lo, hi) pairs {[list(p) for p in pads]} (PyTorch's split: lower side gets the floor)"))
    elif t == "pytorch.nn.max_pool2d":
        c = step.cfg
        if tuple(c.dilation) != (1, 1):
            return refuse("E_BACKEND_UNSUPPORTED_DILATION", f"max_pool2d dilation={list(c.dilation)}: lax.reduce_window(max) with window dilation has no gradient rule in JAX "
                          "(NotImplementedError: VJP not implemented for select_and_gather with window dilation), so training/gradients would fail. Not substituted.")
        if tuple(c.padding) != (0, 0) or c.ceil_mode:
            conv.append(Conversion("padding", "lax.reduce_window with -inf padding (explicit lo/hi, including the extra right padding ceil_mode needs)"))
    elif t == "pytorch.nn.adaptive_avg_pool2d":
        h, w = step.in_types["input"].shape[2:]
        oh, ow = step.cfg.output_size
        if (oh, ow) == (1, 1):
            conv.append(Conversion("op_mapping", "AdaptiveAvgPool2d(1) -> jnp.mean over axes (2, 3), keepdims"))
        elif h % oh == 0 and w % ow == 0:
            conv.append(Conversion("op_mapping", f"AdaptiveAvgPool2d({oh},{ow}) on {h}x{w} -> lax.reduce_window average, window {h // oh}x{w // ow}, non-overlapping"))
        else:
            return refuse("E_BACKEND_UNSUPPORTED_CONFIG", f"AdaptiveAvgPool2d output {oh}x{ow} does not divide the input {h}x{w}; PyTorch's overlapping adaptive bins are not reproduced. Not approximated.")
    elif t == "pytorch.loss.cross_entropy":
        conv.append(Conversion("op_mapping", "CrossEntropyLoss -> jax.nn.log_softmax + take_along_axis (+ explicit mean/sum/none reduction)"))
    return native(*conv)


def analyze(plan: ModelPlan) -> dict[str, Check]:
    return {s.nid: check_step(s) for s in plan.steps}


# ------------------------------------------------------------------------------------------------ code export
def _py(nid: str) -> str:
    return nid.replace("/", "__")


def step_expr(s: Step, ins: dict[str, str]) -> str:
    """The JAX expression computing one node (the exported source; the runtime in jax_runtime.py implements the same semantics)."""
    c, t = s.cfg, s.type
    pn = f"params[{s.owner!r}]"
    x = ins.get("input")
    if t == "pytorch.nn.conv2d":
        pads = conv_pads(c)
        if c.padding_mode != "zeros" and pads != ((0, 0), (0, 0)):
            x = f"jnp.pad({x}, ((0, 0), (0, 0), {tuple(pads[0])}, {tuple(pads[1])}), mode={PAD_MODES[c.padding_mode]!r})"
            pads = ((0, 0), (0, 0))
        e = (f"lax.conv_general_dilated({x}, {pn}['weight'], window_strides={tuple(c.stride)}, padding={tuple(tuple(p) for p in pads)}, "
             f"rhs_dilation={tuple(c.dilation)}, dimension_numbers=('NCHW', 'OIHW', 'NCHW'), feature_group_count={c.groups})")
        return e + (f" + {pn}['bias'].reshape(1, -1, 1, 1)" if c.bias else "")
    if t == "pytorch.nn.relu":
        return f"jnp.maximum({x}, 0)"
    if t == "pytorch.nn.max_pool2d":
        h, w = s.in_types["input"].shape[2:]
        pads = pool_pads(c, h, w)
        st = c.stride or c.kernel_size
        return (f"lax.reduce_window({x}, -jnp.inf, lax.max, window_dimensions=(1, 1, {c.kernel_size[0]}, {c.kernel_size[1]}), window_strides=(1, 1, {st[0]}, {st[1]}), "
                f"padding=((0, 0), (0, 0), {tuple(pads[0])}, {tuple(pads[1])}), window_dilation=(1, 1, {c.dilation[0]}, {c.dilation[1]}))")
    if t == "pytorch.nn.adaptive_avg_pool2d":
        h, w = s.in_types["input"].shape[2:]
        oh, ow = c.output_size
        if (oh, ow) == (1, 1):
            return f"jnp.mean({x}, axis=(2, 3), keepdims=True)"
        kh, kw = h // oh, w // ow
        return f"lax.reduce_window({x}, 0.0, lax.add, (1, 1, {kh}, {kw}), (1, 1, {kh}, {kw}), 'VALID') / {kh * kw}"
    if t == "pytorch.nn.flatten":
        tail = [str(d) for d in s.out_type.shape[c.start_dim + 1:]]
        shape = ", ".join([f"{x}.shape[{i}]" for i in range(c.start_dim)] + ["-1"] + tail)
        return f"{x}.reshape({shape})"
    if t == "pytorch.nn.linear":
        return f"{x} @ {pn}['weight'].T" + (f" + {pn}['bias']" if c.bias else "")
    if t == "pytorch.loss.cross_entropy":
        per = f"-jnp.take_along_axis(jax.nn.log_softmax({ins['logits']}, axis=-1), {ins['target']}[:, None], axis=-1)[:, 0]"
        return {"mean": f"jnp.mean({per})", "sum": f"jnp.sum({per})", "none": per}[c.reduction]
    if t == "core.sub":
        return f"{ins['a']} - {ins['b']}"
    if t == "core.add":
        return f"{ins['a']} + {ins['b']}"
    if t == "core.square":
        return f"jnp.square({x})"
    if t == "core.scalar_mul":
        return f"{x} * {c.factor!r}"
    if t in ("core.sum", "core.mean"):
        r = len(s.in_types["input"].shape)
        dims = tuple(range(r)) if c.dims is None else tuple(d % r for d in c.dims)
        return f"jnp.{'sum' if t == 'core.sum' else 'mean'}({x}, axis={dims!r}, keepdims={c.keepdim})"
    if t == "jax.lax.cumsum":
        return f"lax.cumsum({x}, axis={c.axis}, reverse={c.reverse})"
    raise AssertionError(t)


def generate_jax(plan: ModelPlan, an: dict[str, Check]) -> str:
    bad = sorted(s.nid for s in plan.steps if _py(s.nid) in RESERVED_NAMES)
    if bad:
        from .base import BackendError
        raise BackendError("E_BACKEND_NAME_COLLISION", f"node id(s) {bad} collide with names used by the generated JAX code; rename them")
    var = {s.nid: _py(s.nid) for s in plan.steps}
    body: list[str] = []
    for s in plan.steps:
        if s.type == "core.tensor_input":
            body.append(f"    # node: {s.nid} (argument)")
            continue
        ins = {p: var[s.srcs[p][0]] for p in s.ports}
        note = f" (shares parameters with {s.owner})" if s.shared else ""
        body.append(f"    {var[s.nid]} = {step_expr(s, ins)}  # node: {s.nid}{note}")
    init = []
    for s in (s for s in plan.owners() if param_shapes(s)):
        bound = f"1.0 / math.sqrt({fan_in(s)})"
        items = []
        init.append(f"    key, sub = jax.random.split(key)  # node: {s.nid}")
        init.append(f"    k_weight, k_bias = jax.random.split(sub)  # node: {s.nid}")
        for name, shape in param_shapes(s).items():
            k = "k_bias" if name == "bias" else "k_weight"
            items.append(f"{name!r}: jax.random.uniform({k}, {tuple(shape)}, jnp.float32, -{bound}, {bound})")
        init.append(f"    params[{s.nid!r}] = {{{', '.join(items)}}}  # node: {s.nid}")
    args = ", ".join(_py(i) for i in plan.input_ids)
    lines = [
        f"# Generated from graph {plan.graph_hash} for backend 'jax' (jax.numpy / jax.lax, explicit parameter pytree). Regenerate rather than edit.",
        "# Layout: graph layout throughout (NCHW activations, OIHW weights); no conversions are needed.",
        f"# Initialization: {INIT}",
        "import math",
        "",
        "import jax",
        "import jax.numpy as jnp",
        "from jax import lax",
        "",
        "",
        "def init_params(seed=0):",
        "    key = jax.random.PRNGKey(seed)",
        "    params = {}",
        *init,
        "    return params",
        "",
        "",
        f"def apply(params, {args}):",
        *body,
        f"    return {', '.join(var[o] for o in plan.output_ids)}",
        "",
    ]
    return "\n".join(lines)
