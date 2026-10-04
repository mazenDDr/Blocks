"""TensorFlow/Keras 3 backend: rules (what is supported, which conversions are needed), the layout plan and the exported source.
Pure Python: nothing here imports TensorFlow, so compatibility reports and code export work without the framework installed.

Native layout is NHWC; the graph is NCHW (graph convention for every rank-4 tensor). Each rank-4 value carries a layout tag, "graph" or
"native"; every transpose is decided here, recorded as a `layout` conversion on the node that needs it, and emitted in the exported code
with a `# layout:` marker. Nothing is transposed silently."""
from __future__ import annotations

from dataclasses import dataclass, field

from graph_core.hashing import semantic_hash
from graph_core.types import TensorType

from .check import Check, Conversion, native, refuse
from .params import param_shapes
from .plan import ModelPlan, Step

BACKEND = "keras"
GRAPH, NATIVE = "graph", "native"
FLOATS = ("float32", "float64")

INIT = ("Keras defaults: Glorot-uniform kernels, zero biases, drawn from Keras's seeded RNG. PyTorch uses Kaiming-uniform(a=sqrt(5)) kernels and "
        "uniform(+-1/sqrt(fan_in)) biases. The initial VALUES therefore differ from PyTorch for any seed; compare backends only with copied weights.")

NEEDS_NATIVE = {"pytorch.nn.conv2d", "pytorch.nn.max_pool2d", "pytorch.nn.adaptive_avg_pool2d", "keras.layers.separable_conv2d"}
NEEDS_GRAPH = {"pytorch.nn.linear", "pytorch.loss.cross_entropy", "core.sum", "core.mean"}
PASS_THROUGH = {"pytorch.nn.relu", "core.square", "core.scalar_mul"}
BINARY = {"core.sub", "core.add"}
SUPPORTED_TYPES = {"core.tensor_input", "pytorch.nn.conv2d", "pytorch.nn.relu", "pytorch.nn.max_pool2d", "pytorch.nn.adaptive_avg_pool2d",
                   "pytorch.nn.flatten", "pytorch.nn.linear", "pytorch.loss.cross_entropy", "core.sub", "core.add", "core.square",
                   "core.sum", "core.mean", "core.scalar_mul", "keras.layers.separable_conv2d"}
RESERVED_NAMES = {"keras", "tf", "np", "os", "self", "trainable_variables", "set_graph_weights"}


@dataclass
class Analysis:
    check: Check
    in_actions: dict[str, str | None] = field(default_factory=dict)  # port -> "to_native" | "to_graph" | None
    out_layout: str = GRAPH


def _rank4(t: TensorType) -> bool:
    return len(t.shape) == 4


def _conv_check(step: Step) -> Check:
    c = step.cfg
    conv: list[Conversion] = []
    if c.padding_mode != "zeros":
        return refuse("E_BACKEND_UNSUPPORTED_PADDING_MODE",
                      f"padding_mode='{c.padding_mode}' has no equivalent in Keras Conv2D (zero padding only). It is not substituted or dropped.")
    if c.dilation != (1, 1) and c.stride != (1, 1):
        return refuse("E_BACKEND_UNSUPPORTED_DILATION",
                      f"dilation={list(c.dilation)} with stride={list(c.stride)}: TensorFlow convolutions support dilation > 1 only with stride 1.")
    if c.padding == "same":
        conv.append(Conversion("padding", "padding='same' (stride 1) -> Keras padding='same' (total padding split floor/ceil: lower side gets the floor, as in PyTorch)"))
    elif c.padding != "valid" and tuple(c.padding) != (0, 0):
        conv.append(Conversion("padding", f"explicit padding {list(c.padding)} -> tf.pad with zeros, then a 'valid' convolution (Keras has no explicit-padding argument)"))
    conv.append(Conversion("weight_layout", f"weight OIHW {param_shapes(step)['weight']} -> Keras kernel HWIO (transpose 2,3,1,0); bias unchanged"))
    return native(*conv)


def _maxpool_check(step: Step) -> Check:
    c = step.cfg
    if tuple(c.padding) != (0, 0):
        return refuse("E_BACKEND_UNSUPPORTED_CONFIG", f"max_pool2d padding={list(c.padding)}: Keras MaxPooling2D pads only with 'same' (different semantics, not -inf padding). Not substituted.")
    if tuple(c.dilation) != (1, 1):
        return refuse("E_BACKEND_UNSUPPORTED_DILATION", f"max_pool2d dilation={list(c.dilation)} has no Keras MaxPooling2D equivalent.")
    if c.ceil_mode:
        return refuse("E_BACKEND_UNSUPPORTED_CONFIG", "max_pool2d ceil_mode=True has no Keras MaxPooling2D equivalent (it always floors).")
    return native()


def _adaptive_check(step: Step) -> Check:
    t = step.in_types["input"]
    oh, ow = step.cfg.output_size
    h, w = t.shape[2], t.shape[3]
    if (oh, ow) == (1, 1):
        return native(Conversion("op_mapping", "AdaptiveAvgPool2d(1) -> GlobalAveragePooling2D(keepdims=True)"))
    if h % oh == 0 and w % ow == 0:
        return native(Conversion("op_mapping", f"AdaptiveAvgPool2d({oh},{ow}) on {h}x{w} -> AveragePooling2D(pool={h // oh}x{w // ow}, stride equal): bins are non-overlapping because the sizes divide"))
    return refuse("E_BACKEND_UNSUPPORTED_CONFIG", f"AdaptiveAvgPool2d output {oh}x{ow} does not divide the input {h}x{w}; PyTorch's overlapping adaptive bins have no Keras layer. Not approximated.")


def check_step(step: Step) -> Check:
    t = step.type
    if t not in SUPPORTED_TYPES:
        return refuse("E_BACKEND_UNSUPPORTED_OP", f"'{t}' has no Keras implementation in this adapter (portable subset only). It is not replaced by a similar operation.")
    for ty in list(step.in_types.values()) + [step.out_type]:
        if ty.dtype not in ("float32", "float64", "int64", "bool"):
            return refuse("E_BACKEND_UNSUPPORTED_DTYPE", f"dtype {ty.dtype} is not supported")
    if t == "pytorch.nn.conv2d":
        return _conv_check(step)
    if t == "pytorch.nn.max_pool2d":
        return _maxpool_check(step)
    if t == "pytorch.nn.adaptive_avg_pool2d":
        return _adaptive_check(step)
    if t == "pytorch.nn.linear":
        return native(Conversion("weight_layout", f"weight (out,in) {param_shapes(step)['weight']} -> Keras Dense kernel (in,out) (transpose); bias unchanged"))
    if t == "keras.layers.separable_conv2d":
        return native(Conversion("weight_layout", "depthwise (C*m,1,kh,kw) -> (kh,kw,C,m); pointwise (O,C*m,1,1) -> (1,1,C*m,O)"))
    if t == "pytorch.loss.cross_entropy":
        return native(Conversion("op_mapping", "CrossEntropyLoss -> tf.nn.sparse_softmax_cross_entropy_with_logits (+ explicit mean/sum/none reduction)"))
    return native()


def analyze(plan: ModelPlan) -> dict[str, Analysis]:
    """Verdict and layout actions per node, in plan order. A node whose layout needs are met without a transpose has no layout conversion."""
    layout: dict[str, str] = {}
    out: dict[str, Analysis] = {}
    for s in plan.steps:
        chk = check_step(s)
        a = Analysis(chk)
        ins = {p: layout.get(src[0], GRAPH) if _rank4(s.in_types[p]) else GRAPH for p, src in s.srcs.items()}
        rank4_in = {p for p in ins if _rank4(s.in_types[p])}
        out_layout = GRAPH
        convs: list[Conversion] = []
        if s.type in NEEDS_NATIVE:
            for p in rank4_in:
                if ins[p] == GRAPH:
                    a.in_actions[p] = "to_native"
                    convs.append(Conversion("layout", f"input '{p}' NCHW -> NHWC (tf.transpose [0,2,3,1])"))
            out_layout = NATIVE
        elif s.type in NEEDS_GRAPH or s.type.startswith("jax."):
            for p in rank4_in:
                if ins[p] == NATIVE:
                    a.in_actions[p] = "to_graph"
                    convs.append(Conversion("layout", f"input '{p}' NHWC -> NCHW (tf.transpose [0,3,1,2]) so axes/ordering match the graph"))
        elif s.type == "pytorch.nn.flatten":
            for p in rank4_in:
                if ins[p] == NATIVE:
                    t = s.in_types[p]
                    if s.cfg.start_dim == 1 and s.cfg.end_dim in (-1, 3) and t.shape[2] == 1 and t.shape[3] == 1:
                        convs.append(Conversion("layout", "no transpose needed: H = W = 1, so NHWC and NCHW flatten to the same element order"))
                    else:
                        a.in_actions[p] = "to_graph"
                        convs.append(Conversion("layout", f"input '{p}' NHWC -> NCHW before reshape: flatten order must follow the graph's CHW order"))
        elif s.type in PASS_THROUGH:
            out_layout = ins.get("input", GRAPH)
        elif s.type in BINARY:
            ls = set(ins.values())
            if len(ls) == 2:
                for p, l in ins.items():
                    if l == GRAPH and p in rank4_in:
                        a.in_actions[p] = "to_native"
                        convs.append(Conversion("layout", f"input '{p}' NCHW -> NHWC to match the other operand"))
                out_layout = NATIVE
            else:
                out_layout = next(iter(ls)) if ls else GRAPH
        if out_layout == NATIVE and _rank4(s.out_type):
            convs.append(Conversion("layout", "result is kept NHWC; outputs and captured activations of this node are transposed NHWC -> NCHW when read"))
        else:
            out_layout = GRAPH if not _rank4(s.out_type) else out_layout
        if convs and chk.status != "unsupported":
            chk = Check("converted", chk.conversions + convs, None, None)
        a.check, a.out_layout = chk, out_layout
        layout[s.nid] = out_layout
        out[s.nid] = a
    return out


# ------------------------------------------------------------------------------------------------ code export
def _py(nid: str) -> str:
    return nid.replace("/", "__")


def _tuple(v) -> str:
    return repr(tuple(v))


def _layer_ctor(s: Step) -> str | None:
    c = s.cfg
    if s.type == "pytorch.nn.conv2d":
        pad = "same" if c.padding == "same" else "valid"
        return (f"keras.layers.Conv2D({c.out_channels}, {_tuple(c.kernel_size)}, strides={_tuple(c.stride)}, padding={pad!r}, "
                f"dilation_rate={_tuple(c.dilation)}, groups={c.groups}, use_bias={c.bias})")
    if s.type == "keras.layers.separable_conv2d":
        return (f"keras.layers.SeparableConv2D({c.out_channels}, {_tuple(c.kernel_size)}, strides={_tuple(c.stride)}, padding={c.padding!r}, "
                f"depth_multiplier={c.depth_multiplier}, use_bias={c.bias})")
    if s.type == "pytorch.nn.linear":
        return f"keras.layers.Dense({c.out_features}, use_bias={c.bias})"
    if s.type == "pytorch.nn.max_pool2d":
        return f"keras.layers.MaxPooling2D({_tuple(c.kernel_size)}, strides={_tuple(c.stride or c.kernel_size)}, padding='valid')"
    if s.type == "pytorch.nn.adaptive_avg_pool2d":
        return None
    return None


def _build_shape(s: Step, layout_in: str) -> str | None:
    """Static input shape used to build a layer's weights eagerly, in the layer's native layout."""
    t = s.in_types["input"]
    if s.type in ("pytorch.nn.conv2d", "keras.layers.separable_conv2d"):
        _, ch, h, w = t.shape
        return f"(None, {h}, {w}, {ch})"
    if s.type == "pytorch.nn.linear":
        return f"(None, {t.shape[-1]})" if len(t.shape) == 2 else f"({', '.join('None' if isinstance(d, str) else str(d) for d in t.shape)})"
    return None


def generate_keras(plan: ModelPlan, an: dict[str, Analysis]) -> str:
    py_ids = {_py(s.nid) for s in plan.steps}
    bad = sorted(s.nid for s in plan.steps if _py(s.nid) in RESERVED_NAMES)
    if bad:
        from .base import BackendError
        raise BackendError("E_BACKEND_NAME_COLLISION", f"node id(s) {bad} collide with names used by the generated Keras code; rename them")
    init, body, loads = [], [], []
    var: dict[str, str] = {}  # node -> python variable holding its value in its own layout
    layout: dict[str, str] = {}
    for s in plan.steps:
        a = an[s.nid]
        py = _py(s.nid)
        ins = {}
        for p in s.ports:
            src = s.srcs[p][0]
            v = var[src]
            act = a.in_actions.get(p)
            if act:
                nv = f"{_py(src)}__{'nhwc' if act == 'to_native' else 'nchw'}"
                assert nv not in py_ids, nv
                perm = "[0, 2, 3, 1]" if act == "to_native" else "[0, 3, 1, 2]"
                lay = "NCHW -> NHWC" if act == "to_native" else "NHWC -> NCHW"
                body.append(f"        {nv} = tf.transpose({v}, {perm})  # node: {s.nid} (layout: {lay})")
                v = nv
            ins[p] = v
        c = s.cfg
        t = s.type
        if t == "core.tensor_input":
            var[s.nid] = py
            body.append(f"        # node: {s.nid} (forward argument)")
            continue
        if not s.shared and (ctor := _layer_ctor(s)):
            init.append(f"        self.{py} = {ctor}  # node: {s.nid}")
            bs = _build_shape(s, GRAPH)
            if bs:
                init.append(f"        self.{py}.build({bs})  # node: {s.nid} (create the weights now, in Keras layout)")
        owner = _py(s.owner)
        x = ins.get("input")
        note = f" (shares parameters with {s.owner})" if s.shared else ""
        if t == "pytorch.nn.conv2d":
            if c.padding != "same" and c.padding != "valid" and tuple(c.padding) != (0, 0):
                ph, pw = c.padding
                pv = f"{py}__padded"
                body.append(f"        {pv} = tf.pad({x}, [[0, 0], [{ph}, {ph}], [{pw}, {pw}], [0, 0]])  # node: {s.nid} (padding: explicit zeros)")
                x = pv
            expr = f"self.{owner}({x})"
        elif t in ("keras.layers.separable_conv2d", "pytorch.nn.linear", "pytorch.nn.max_pool2d"):
            expr = f"self.{owner}({x})"
        elif t == "pytorch.nn.relu":
            expr = f"keras.ops.relu({x})"
        elif t == "pytorch.nn.adaptive_avg_pool2d":
            h, w = s.in_types["input"].shape[2:]
            oh, ow = c.output_size
            expr = (f"keras.layers.GlobalAveragePooling2D(keepdims=True)({x})" if (oh, ow) == (1, 1)
                    else f"keras.layers.AveragePooling2D(({h // oh}, {w // ow}), strides=({h // oh}, {w // ow}))({x})")
        elif t == "pytorch.nn.flatten":
            shape = ", ".join(str(d) for d in s.out_type.shape[1:])
            expr = f"keras.ops.reshape({x}, (-1, {shape}))"
        elif t == "pytorch.loss.cross_entropy":
            per = f"tf.nn.sparse_softmax_cross_entropy_with_logits(labels={ins['target']}, logits={ins['logits']})"
            expr = {"mean": f"tf.reduce_mean({per})", "sum": f"tf.reduce_sum({per})", "none": per}[c.reduction]
        elif t == "core.sub":
            expr = f"{ins['a']} - {ins['b']}"
        elif t == "core.add":
            expr = f"{ins['a']} + {ins['b']}"
        elif t == "core.square":
            expr = f"keras.ops.square({x})"
        elif t == "core.scalar_mul":
            expr = f"{x} * {c.factor!r}"
        elif t in ("core.sum", "core.mean"):
            dims = tuple(range(len(s.in_types["input"].shape))) if c.dims is None else tuple(d % len(s.in_types["input"].shape) for d in c.dims)
            fn = "reduce_sum" if t == "core.sum" else "reduce_mean"
            expr = f"tf.{fn}({x}, axis={list(dims)!r}, keepdims={c.keepdim})"
        else:  # unreachable for a compatible graph
            raise AssertionError(t)
        var[s.nid] = py
        body.append(f"        {py} = {expr}  # node: {s.nid}{note}")
    inputs = plan.input_ids
    outs = plan.output_ids
    out_exprs = []
    for o in outs:
        v = var[o]
        if an[o].out_layout == NATIVE and _rank4(plan.by_id[o].out_type):
            body.append(f"        {_py(o)}__out = tf.transpose({v}, [0, 3, 1, 2])  # node: {o} (layout: NHWC -> NCHW, output boundary)")
            v = f"{_py(o)}__out"
        out_exprs.append(v)
    # weights in graph layout in / out
    for s in plan.owners():
        ps = param_shapes(s)
        if not ps:
            continue
        py = _py(s.nid)
        if s.type == "pytorch.nn.conv2d":
            loads.append(f"        self.{py}.kernel.assign(np.transpose(params[{s.nid!r}]['weight'], (2, 3, 1, 0)))  # node: {s.nid} (weight_layout: OIHW -> HWIO)")
        elif s.type == "pytorch.nn.linear":
            loads.append(f"        self.{py}.kernel.assign(np.transpose(params[{s.nid!r}]['weight']))  # node: {s.nid} (weight_layout: (out,in) -> (in,out))")
        elif s.type == "keras.layers.separable_conv2d":
            m, cin = s.cfg.depth_multiplier, s.cfg.in_channels
            kh, kw = s.cfg.kernel_size
            loads.append(f"        self.{py}.depthwise_kernel.assign(np.transpose(params[{s.nid!r}]['depthwise_weight'], (2, 3, 0, 1)).reshape({kh}, {kw}, {cin}, {m}))  # node: {s.nid} (weight_layout)")
            loads.append(f"        self.{py}.pointwise_kernel.assign(np.transpose(params[{s.nid!r}]['pointwise_weight'], (2, 3, 1, 0)))  # node: {s.nid} (weight_layout)")
        if "bias" in ps:
            loads.append(f"        self.{py}.bias.assign(params[{s.nid!r}]['bias'])  # node: {s.nid}")
    owners = [s for s in plan.owners() if param_shapes(s)]
    tv = " + ".join(f"self.{_py(s.nid)}.trainable_variables" for s in owners) or "[]"
    lines = [
        f"# Generated from graph {plan.graph_hash} for backend 'keras' (Keras 3 on TensorFlow). Regenerate rather than edit.",
        "# The graph is NCHW; Keras convolutions run NHWC. Every transpose below is marked 'layout:' and is the only place data changes layout.",
        f"# Initialization: {INIT}",
        "# Weights cross the boundary in graph (PyTorch) layout through set_graph_weights().",
        "import os",
        "",
        'os.environ.setdefault("KERAS_BACKEND", "tensorflow")',
        "",
        "import keras",
        "import numpy as np",
        "import tensorflow as tf",
        "",
        "",
        "class Model:",
        "    def __init__(self, seed=0):",
        "        keras.utils.set_random_seed(seed)",
        *(init or ["        pass"]),
        "",
        "    @property",
        "    def trainable_variables(self):",
        f"        return {tv}",
        "",
        "    def set_graph_weights(self, params):",
        '        """params: {node id: {name: ndarray}} in graph (PyTorch) layout."""',
        *(loads or ["        pass"]),
        "",
        f"    def __call__(self, {', '.join(_py(i) for i in inputs)}):",
        *body,
        f"        return {', '.join(out_exprs)}",
        "",
    ]
    return "\n".join(lines)
