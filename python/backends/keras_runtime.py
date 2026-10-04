"""Keras 3 (TensorFlow backend) executable. Layers hold the variables; the layout plan from keras_spec decides every transpose."""
from __future__ import annotations

import os

import numpy as np

from .base import BackendError, Executable, ForwardResult, Params
from .keras_spec import GRAPH, NATIVE, analyze
from .params import param_shapes, param_spec


class KerasExecutable(Executable):
    backend = "keras"

    def __init__(self, plan, seed: int = 0, compiled: bool = False):
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        import keras
        import tensorflow as tf

        self.keras, self.tf = keras, tf
        if keras.backend.backend() != "tensorflow":
            raise BackendError("E_BACKEND_UNAVAILABLE", f"Keras runs on '{keras.backend.backend()}', not TensorFlow")
        self.plan = plan
        self.an = analyze(plan)
        self.input_ids, self.output_ids = list(plan.input_ids), list(plan.output_ids)
        self.spec = param_spec(plan)
        keras.utils.set_random_seed(seed)
        self.layers: dict[str, object] = {}
        for i, s in enumerate(plan.owners()):
            self._make_layer(s, seed * 1000 + i)
        self.compiled = compiled
        self._opt: dict = {}
        self._fwd = tf.function(self._eval) if compiled else self._eval  # every node's value (capture)
        self._fwd_out = tf.function(self._eval_outputs) if compiled else self._eval_outputs  # terminal nodes only
        self._vgs: dict = {}

    # ---- layers
    def _make_layer(self, s, seed):
        k, tf = self.keras, self.tf
        c, L = s.cfg, k.layers
        ki = k.initializers.GlorotUniform(seed=seed)
        t = s.type
        if t == "pytorch.nn.conv2d":
            lay = L.Conv2D(c.out_channels, c.kernel_size, strides=c.stride, padding="same" if c.padding == "same" else "valid", dilation_rate=c.dilation,
                           groups=c.groups, use_bias=c.bias, kernel_initializer=ki)
            ci, h, w = s.in_types["input"].shape[1:]
            lay.build((None, h, w, ci))
        elif t == "keras.layers.separable_conv2d":
            lay = L.SeparableConv2D(c.out_channels, c.kernel_size, strides=c.stride, padding=c.padding, depth_multiplier=c.depth_multiplier, use_bias=c.bias,
                                    depthwise_initializer=ki, pointwise_initializer=k.initializers.GlorotUniform(seed=seed + 1))
            ci, h, w = s.in_types["input"].shape[1:]
            lay.build((None, h, w, ci))
        elif t == "pytorch.nn.linear":
            lay = L.Dense(c.out_features, use_bias=c.bias, kernel_initializer=ki)
            lay.build(tuple(None if isinstance(d, str) else d for d in s.in_types["input"].shape))
        elif t == "pytorch.nn.max_pool2d":
            lay = L.MaxPooling2D(c.kernel_size, strides=c.stride or c.kernel_size, padding="valid")
        elif t == "pytorch.nn.adaptive_avg_pool2d":
            h, w = s.in_types["input"].shape[2:]
            oh, ow = c.output_size
            lay = L.GlobalAveragePooling2D(keepdims=True) if (oh, ow) == (1, 1) else L.AveragePooling2D((h // oh, w // ow), strides=(h // oh, w // ow))
        else:
            return
        self.layers[s.nid] = lay

    def init_info(self):
        from .keras_spec import INIT
        return INIT

    # ---- weights: graph layout <-> Keras layout
    def _slots(self, nid):
        """[(name, variable, to_graph(np), from_graph(np))] for the owner node."""
        s, lay = self.plan.by_id[nid], self.layers[nid]
        if s.type == "pytorch.nn.conv2d":
            out = [("weight", lay.kernel, lambda a: a.transpose(3, 2, 0, 1), lambda a: a.transpose(2, 3, 1, 0))]
        elif s.type == "pytorch.nn.linear":
            out = [("weight", lay.kernel, lambda a: a.T, lambda a: a.T)]
        elif s.type == "keras.layers.separable_conv2d":
            m, ci = s.cfg.depth_multiplier, s.cfg.in_channels
            kh, kw = s.cfg.kernel_size
            out = [("depthwise_weight", lay.depthwise_kernel, lambda a: a.reshape(kh, kw, ci * m).transpose(2, 0, 1)[:, None], lambda a: a[:, 0].transpose(1, 2, 0).reshape(kh, kw, ci, m)),
                   ("pointwise_weight", lay.pointwise_kernel, lambda a: a.transpose(3, 2, 0, 1), lambda a: a.transpose(2, 3, 1, 0))]
        else:
            return []
        if s.cfg.bias:
            out.append(("bias", lay.bias, lambda a: a, lambda a: a))
        return out

    def get_params(self) -> Params:
        return {nid: {name: to_g(np.asarray(v.numpy())).copy() for name, v, to_g, _ in self._slots(nid)} for nid in self.spec}

    def set_params(self, params: Params) -> None:
        for nid, ps in self.spec.items():
            for name, v, _, from_g in self._slots(nid):
                a = np.asarray(params[nid][name], dtype=np.float32)
                if list(a.shape) != ps[name]:
                    raise BackendError("E_BACKEND_PARAM_SHAPE", f"{nid}.{name}: expected {ps[name]}, got {list(a.shape)}")
                v.assign(np.ascontiguousarray(from_g(a)))

    def _variables(self):
        return [(nid, name, v, to_g) for nid in self.spec for name, v, to_g, _ in self._slots(nid)]

    # ---- the computation
    def _node(self, s, ins):
        tf, k = self.tf, self.keras
        c, t = s.cfg, s.type
        x = ins.get("input")
        lay = self.layers.get(s.owner)
        if t == "pytorch.nn.conv2d":
            if c.padding not in ("same", "valid") and tuple(c.padding) != (0, 0):
                ph, pw = c.padding
                x = tf.pad(x, [[0, 0], [ph, ph], [pw, pw], [0, 0]])
            return lay(x)
        if t in ("keras.layers.separable_conv2d", "pytorch.nn.linear", "pytorch.nn.max_pool2d", "pytorch.nn.adaptive_avg_pool2d"):
            return lay(x)
        if t == "pytorch.nn.relu":
            return k.ops.relu(x)
        if t == "pytorch.nn.flatten":
            return k.ops.reshape(x, (-1, *s.out_type.shape[1:]))
        if t == "pytorch.loss.cross_entropy":
            per = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ins["target"], logits=ins["logits"])
            return {"mean": tf.reduce_mean, "sum": tf.reduce_sum, "none": lambda v: v}[c.reduction](per)
        if t == "core.sub":
            return ins["a"] - ins["b"]
        if t == "core.add":
            return ins["a"] + ins["b"]
        if t == "core.square":
            return k.ops.square(x)
        if t == "core.scalar_mul":
            return x * c.factor
        if t in ("core.sum", "core.mean"):
            r = len(s.in_types["input"].shape)
            dims = list(range(r)) if c.dims is None else [d % r for d in c.dims]
            return (tf.reduce_sum if t == "core.sum" else tf.reduce_mean)(x, axis=dims, keepdims=c.keepdim)
        raise BackendError("E_BACKEND_UNSUPPORTED_OP", t)

    def _eval(self, inputs):
        """-> {node: value}; each value in its own (native or graph) layout."""
        tf = self.tf
        values = {}
        for s in self.plan.steps:
            if s.type == "core.tensor_input":
                values[s.nid] = inputs[s.nid]
                continue
            a = self.an[s.nid]
            ins = {}
            for p, (src, _) in s.srcs.items():
                v = values[src]
                act = a.in_actions.get(p)
                if act == "to_native":
                    v = tf.transpose(v, [0, 2, 3, 1])
                elif act == "to_graph":
                    v = tf.transpose(v, [0, 3, 1, 2])
                ins[p] = v
            values[s.nid] = self._node(s, ins)
        return values

    def _eval_outputs(self, inputs):
        vals = self._eval(inputs)
        return {o: self._graph_value(o, vals[o]) for o in self.output_ids}

    def _graph_value(self, nid, v):
        s = self.plan.by_id[nid]
        if self.an[nid].out_layout == NATIVE and len(s.out_type.shape) == 4:
            v = self.tf.transpose(v, [0, 3, 1, 2])
        return v

    def _feed(self, inputs, names=None):
        tf = self.tf
        return {i: tf.constant(np.asarray(inputs[i])) for i in (names or self.input_ids)}

    def forward(self, inputs, capture=False) -> ForwardResult:
        feed = self._feed(inputs)
        if not capture:
            return ForwardResult({o: v.numpy() for o, v in self._fwd_out(feed).items()})
        vals = self._fwd(feed)
        outs = {o: self._graph_value(o, vals[o]).numpy() for o in self.output_ids}
        return ForwardResult(outs, {n: self._graph_value(n, v).numpy() for n, v in vals.items()})

    def _vg_impl(self, ln, wrt, feed):
        tf = self.tf
        diff = {k: feed[k] for k in wrt}
        with tf.GradientTape() as tape:
            for v in diff.values():
                tape.watch(v)
            vals = self._eval({**feed, **diff})
            loss = vals[ln]
        vars_ = self._variables()
        grads = tape.gradient(loss, [v for _, _, v, _ in vars_] + list(diff.values()))
        gp, gi = grads[:len(vars_)], grads[len(vars_):]
        gp = [tf.zeros_like(v) if g is None else g for (_, _, v, _), g in zip(vars_, gp)]
        gi = [tf.zeros_like(v) if g is None else g for v, g in zip(diff.values(), gi)]
        return loss, gp, gi

    def _vg(self, ln, wrt, inputs):
        key = (ln, wrt)
        if key not in self._vgs:
            f = lambda feed: self._vg_impl(ln, wrt, feed)  # noqa: E731
            self._vgs[key] = self.tf.function(f) if self.compiled else f
        return self._vgs[key](self._feed(inputs))

    def loss_and_grads(self, inputs, loss_node=None, wrt_inputs=()):
        wrt = tuple(wrt_inputs)
        loss, gp, gi = self._vg(self._loss_node(loss_node), wrt, inputs)
        grads: Params = {}
        for (nid, name, _, to_g), g in zip(self._variables(), gp):
            grads.setdefault(nid, {})[name] = to_g(g.numpy()).copy()
        return loss.numpy(), grads, {k: g.numpy() for k, g in zip(wrt, gi)}

    def _step_impl(self, ln, feed, lr):
        loss, gp, _ = self._vg_impl(ln, (), feed)
        for (_, _, v, _), g in zip(self._variables(), gp):
            v.assign_sub(lr * g)  # plain SGD: p <- p - lr * grad
        return loss

    def _step_fn(self, ln):
        key = ("step", ln)
        if key not in self._vgs:
            f = lambda feed, lr: self._step_impl(ln, feed, lr)  # noqa: E731
            self._vgs[key] = self.tf.function(f) if self.compiled else f
        return self._vgs[key]

    def sgd_step(self, inputs, lr, loss_node=None):
        return self._step_fn(self._loss_node(loss_node))(self._feed(inputs), self.tf.constant(lr, "float32")).numpy()

    def bench_closures(self, inputs, lr, loss_node=None, grad_inputs=()):
        """(forward, train step) closures over inputs converted once, for the benchmark: the same compute path as forward()/sgd_step()
        without the NumPy <-> tensor conversion at the boundary. Each returns after the result is materialized."""
        feed, ln, lr_t = self._feed(inputs), self._loss_node(loss_node), self.tf.constant(lr, "float32")

        def fwd():
            vals = self._fwd_out(feed)
            return [np.asarray(vals[o]) for o in self.output_ids]

        if grad_inputs:  # loss + gradient w.r.t. the named inputs, no parameter update
            key = (ln, tuple(grad_inputs))
            if key not in self._vgs:
                f = lambda fd: self._vg_impl(ln, key[1], fd)  # noqa: E731
                self._vgs[key] = self.tf.function(f) if self.compiled else f
            vg = self._vgs[key]
            return fwd, lambda: vg(feed)[2][0].numpy()
        step = self._step_fn(ln)
        return fwd, lambda: step(feed, lr_t).numpy()
