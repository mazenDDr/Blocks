"""JAX executable: a pure `apply(params, inputs) -> {node: value}` over a parameter pytree, jit-compiled. Graph layout throughout."""
from __future__ import annotations

import math

import numpy as np

from .base import BackendError, Executable, ForwardResult, Params
from .jax_spec import PAD_MODES, conv_pads, pool_pads
from .params import fan_in, param_shapes, param_spec


class JaxExecutable(Executable):
    backend = "jax"

    def __init__(self, plan, seed: int = 0, jit: bool = True):
        import jax
        import jax.numpy as jnp

        self.jax, self.jnp = jax, jnp
        self.plan = plan
        self.input_ids, self.output_ids = list(plan.input_ids), list(plan.output_ids)
        self.spec = param_spec(plan)
        self.use_jit = jit
        self.params = self._init(seed)
        self._fwd = jax.jit(self._apply_all) if jit else self._apply_all  # every node's value (capture)
        self._fwd_out = jax.jit(self._apply_outputs) if jit else self._apply_outputs  # terminal nodes only
        self._lg: dict = {}

    # ---- init: identical key splitting to the exported init_params(seed)
    def _init(self, seed):
        jax, jnp = self.jax, self.jnp
        key = jax.random.PRNGKey(seed)
        params = {}
        for s in self.plan.owners():
            ps = param_shapes(s)
            if not ps:
                continue
            key, sub = jax.random.split(key)
            k_weight, k_bias = jax.random.split(sub)
            bound = 1.0 / math.sqrt(fan_in(s))
            params[s.nid] = {name: jax.random.uniform(k_bias if name == "bias" else k_weight, tuple(shape), jnp.float32, -bound, bound) for name, shape in ps.items()}
        return params

    def init_info(self):
        from .jax_spec import INIT
        return INIT

    def get_params(self) -> Params:
        return {n: {k: np.asarray(v).copy() for k, v in ps.items()} for n, ps in self.params.items()}

    def set_params(self, params: Params) -> None:
        jnp = self.jnp
        for nid, ps in self.spec.items():
            for name, shape in ps.items():
                a = np.asarray(params[nid][name], dtype=np.float32)
                if list(a.shape) != shape:
                    raise BackendError("E_BACKEND_PARAM_SHAPE", f"{nid}.{name}: expected {shape}, got {list(a.shape)}")
                self.params[nid][name] = jnp.asarray(a)

    # ---- the computation
    def _node(self, s, ins, params):
        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        c, t = s.cfg, s.type
        x = ins.get("input")
        pn = params.get(s.owner)
        if t == "pytorch.nn.conv2d":
            pads = conv_pads(c)
            if c.padding_mode != "zeros" and pads != ((0, 0), (0, 0)):
                x = jnp.pad(x, ((0, 0), (0, 0), pads[0], pads[1]), mode=PAD_MODES[c.padding_mode])
                pads = ((0, 0), (0, 0))
            y = lax.conv_general_dilated(x, pn["weight"], window_strides=tuple(c.stride), padding=tuple(tuple(p) for p in pads), rhs_dilation=tuple(c.dilation),
                                         dimension_numbers=("NCHW", "OIHW", "NCHW"), feature_group_count=c.groups)
            return y + pn["bias"].reshape(1, -1, 1, 1) if c.bias else y
        if t == "pytorch.nn.relu":
            return jnp.maximum(x, 0)
        if t == "pytorch.nn.max_pool2d":
            h, w = s.in_types["input"].shape[2:]
            pads = pool_pads(c, h, w)
            st = c.stride or c.kernel_size
            return lax.reduce_window(x, -jnp.inf, lax.max, (1, 1, *c.kernel_size), (1, 1, *st), ((0, 0), (0, 0), pads[0], pads[1]), window_dilation=(1, 1, *c.dilation))
        if t == "pytorch.nn.adaptive_avg_pool2d":
            h, w = s.in_types["input"].shape[2:]
            oh, ow = c.output_size
            if (oh, ow) == (1, 1):
                return jnp.mean(x, axis=(2, 3), keepdims=True)
            kh, kw = h // oh, w // ow
            return lax.reduce_window(x, 0.0, lax.add, (1, 1, kh, kw), (1, 1, kh, kw), "VALID") / (kh * kw)
        if t == "pytorch.nn.flatten":
            return x.reshape(*x.shape[:c.start_dim], -1, *s.out_type.shape[c.start_dim + 1:])
        if t == "pytorch.nn.linear":
            y = x @ pn["weight"].T
            return y + pn["bias"] if c.bias else y
        if t == "pytorch.loss.cross_entropy":
            per = -jnp.take_along_axis(jax.nn.log_softmax(ins["logits"], axis=-1), ins["target"][:, None], axis=-1)[:, 0]
            return {"mean": jnp.mean, "sum": jnp.sum, "none": lambda v: v}[c.reduction](per)
        if t == "core.sub":
            return ins["a"] - ins["b"]
        if t == "core.add":
            return ins["a"] + ins["b"]
        if t == "core.square":
            return jnp.square(x)
        if t == "core.scalar_mul":
            return x * c.factor
        if t in ("core.sum", "core.mean"):
            r = len(s.in_types["input"].shape)
            dims = tuple(range(r)) if c.dims is None else tuple(d % r for d in c.dims)
            return (jnp.sum if t == "core.sum" else jnp.mean)(x, axis=dims, keepdims=c.keepdim)
        if t == "jax.lax.cumsum":
            return lax.cumsum(x, axis=c.axis, reverse=c.reverse)
        raise BackendError("E_BACKEND_UNSUPPORTED_OP", t)

    def _apply_all(self, params, inputs):
        values = {}
        for s in self.plan.steps:
            if s.type == "core.tensor_input":
                values[s.nid] = inputs[s.nid]
            else:
                values[s.nid] = self._node(s, {p: values[src[0]] for p, src in s.srcs.items()}, params)
        return values

    def _apply_outputs(self, params, inputs):
        vals = self._apply_all(params, inputs)
        return {o: vals[o] for o in self.output_ids}

    def _feed(self, inputs, names=None):
        jnp = self.jnp
        out = {}
        for i in names or self.input_ids:
            a = np.asarray(inputs[i])
            if a.dtype == np.int64:
                if a.size and (a.max() > np.iinfo(np.int32).max or a.min() < np.iinfo(np.int32).min):
                    raise BackendError("E_BACKEND_DTYPE_RANGE", f"input '{i}' has int64 values outside the int32 range; JAX without x64 cannot hold them")
                a = a.astype(np.int32)
            elif a.dtype == np.float64:
                raise BackendError("E_BACKEND_UNSUPPORTED_DTYPE", f"input '{i}' is float64; the JAX adapter refuses to downcast silently")
            out[i] = jnp.asarray(a)
        return out

    def forward(self, inputs, capture=False) -> ForwardResult:
        if not capture:
            return ForwardResult({o: np.asarray(v) for o, v in self._fwd_out(self.params, self._feed(inputs)).items()})
        vals = self._fwd(self.params, self._feed(inputs))
        return ForwardResult({o: np.asarray(vals[o]) for o in self.output_ids}, {k: np.asarray(v) for k, v in vals.items()})

    def _vg(self, ln, wrt, inputs):
        jax = self.jax
        key = (ln, wrt)
        if key not in self._lg:
            def loss_fn(params, diff_inputs, fixed_inputs):
                return self._apply_all(params, {**fixed_inputs, **diff_inputs})[ln]
            f = jax.value_and_grad(loss_fn, argnums=(0, 1))
            self._lg[key] = jax.jit(f) if self.use_jit else f
        feed = self._feed(inputs)
        return self._lg[key](self.params, {k: feed[k] for k in wrt}, {k: v for k, v in feed.items() if k not in wrt})

    def loss_and_grads(self, inputs, loss_node=None, wrt_inputs=()):
        loss, (gp, gi) = self._vg(self._loss_node(loss_node), tuple(wrt_inputs), inputs)
        grads = {n: {k: np.asarray(v) for k, v in ps.items()} for n, ps in gp.items()}
        return np.asarray(loss), grads, {k: np.asarray(v) for k, v in gi.items()}

    def _step_fn(self, ln):
        key = ("step", ln)
        if key not in self._lg:
            jax = self.jax

            def step(params, feed, lr):
                loss, gp = jax.value_and_grad(lambda p: self._apply_all(p, feed)[ln])(params)
                return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, gp), loss  # plain SGD: p <- p - lr * grad

            self._lg[key] = jax.jit(step) if self.use_jit else step
        return self._lg[key]

    def sgd_step(self, inputs, lr, loss_node=None):
        self.params, loss = self._step_fn(self._loss_node(loss_node))(self.params, self._feed(inputs), self.jnp.float32(lr))
        return np.asarray(loss)

    def bench_closures(self, inputs, lr, loss_node=None, grad_inputs=()):
        """(forward, train step) closures over inputs converted once (see KerasExecutable.bench_closures). They block until the result is ready."""
        feed, ln, lr_t = self._feed(inputs), self._loss_node(loss_node), self.jnp.float32(lr)
        step = self._step_fn(ln)

        def fwd():
            vals = self._fwd_out(self.params, feed)
            return [self.jax.block_until_ready(vals[o]) for o in self.output_ids]

        if grad_inputs:  # loss + gradient w.r.t. the named inputs, no parameter update
            wrt = tuple(grad_inputs)
            self._vg(ln, wrt, inputs)  # builds the jitted function
            vg = self._lg[(ln, wrt)]
            diff, fixed = {k: feed[k] for k in wrt}, {k: v for k, v in feed.items() if k not in wrt}
            return fwd, lambda: self.jax.block_until_ready(vg(self.params, diff, fixed)[1][1])

        def train():
            self.params, loss = step(self.params, feed, lr_t)
            return self.jax.block_until_ready(loss)

        return fwd, train
