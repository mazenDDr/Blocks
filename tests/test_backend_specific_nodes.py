"""Explicit backend-specific nodes (VISION 14.2): `keras.layers.separable_conv2d` and `jax.lax.cumsum` run on their own backend against that backend's
native layer/function, and are REJECTED, with a clear message and a stable code, on every other backend. No equivalence to other frameworks is claimed."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import backends
from backend_helpers import cmp, skip_unless_available
from backends import BackendError
from backends.tolerances import tol
from graph_core.build import GraphBuilder
from graph_core.validate import validate


def sep_graph(backend="keras", **cfg):
    b = GraphBuilder()
    b.input("x", ["N", 3, 9, 9])
    b.node("op", "keras.layers.separable_conv2d", **cfg)
    b.chain("x", "op")
    b.node("sq", "core.square")
    b.node("loss", "core.sum")
    b.chain("op", "sq", "loss")
    g = b.build()
    g.backend = backend
    return g


SEP = {"separable_valid": dict(out_channels=5, kernel_size=[3, 3]),
       "separable_same_stride2_multiplier2": dict(out_channels=4, kernel_size=[3, 3], stride=[2, 2], padding="same", depth_multiplier=2),
       "separable_no_bias": dict(out_channels=4, kernel_size=[2, 3], bias=False)}


@pytest.mark.parametrize("name", list(SEP))
def test_keras_separable_conv_matches_the_native_keras_layer(name):
    skip_unless_available("keras")
    import keras

    cfg = SEP[name]
    g = sep_graph(**cfg)
    rep = backends.compat_report(g, "keras")
    assert rep.ok and rep.nodes["op"]["status"] == "converted"
    ex = backends.compile_graph(g, "keras", compiled=False)
    rng = np.random.default_rng(0)
    p = ex.get_params()["op"]
    p = {k: rng.standard_normal(v.shape).astype(np.float32) * 0.3 for k, v in p.items()}
    ex.set_params({"op": p})
    x = rng.standard_normal((2, 3, 9, 9)).astype(np.float32)
    res = ex.forward({"x": x}, capture=True)
    # native Keras, weights converted independently here (not through the adapter)
    m, kh, kw = cfg.get("depth_multiplier", 1), *cfg["kernel_size"]
    lay = keras.layers.SeparableConv2D(cfg["out_channels"], cfg["kernel_size"], strides=cfg.get("stride", (1, 1)), padding=cfg.get("padding", "valid"),
                                       depth_multiplier=m, use_bias=cfg.get("bias", True))
    nhwc = np.transpose(x, (0, 2, 3, 1))
    lay.build(nhwc.shape)
    dw = np.zeros((kh, kw, 3, m), np.float32)
    for c in range(3):
        for q in range(m):
            dw[:, :, c, q] = p["depthwise_weight"][c * m + q, 0]
    lay.depthwise_kernel.assign(dw)
    lay.pointwise_kernel.assign(np.transpose(p["pointwise_weight"], (2, 3, 1, 0)))
    if "bias" in p:
        lay.bias.assign(p["bias"])
    native = np.transpose(lay(nhwc).numpy(), (0, 3, 1, 2))
    assert res.activations["op"].shape == native.shape
    cmp(res.activations["op"], native, tol("keras", "float32", "forward"), "separable output")
    # shape inference and parameter count agree with the real layer
    r = validate(g)
    assert list(r.output_types["op"]["output"].shape[1:]) == list(native.shape[1:])
    assert r.params["op"] == sum(int(np.prod(w.shape)) for w in lay.trainable_variables) == sum(v.size for v in p.values())
    # gradients through the adapter vs keras autograd on the native layer
    import tensorflow as tf
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(tf.square(lay(tf.constant(nhwc))))
    gn = tape.gradient(loss, lay.pointwise_kernel).numpy()
    _, grads, _ = ex.loss_and_grads({"x": x}, "loss")
    cmp(grads["op"]["pointwise_weight"], np.transpose(gn, (3, 2, 0, 1)), tol("keras", "float32", "grad"), "pointwise gradient")


def test_keras_separable_equals_pytorch_grouped_convs_for_valid_stride1():
    """For 'valid' / stride 1 the Keras layer is mathematically depthwise-then-pointwise; checked against torch only where the semantics coincide."""
    skip_unless_available("keras")
    g = sep_graph(out_channels=5, kernel_size=[3, 3], depth_multiplier=2)
    ex = backends.compile_graph(g, "keras", compiled=False)
    rng = np.random.default_rng(1)
    p = {k: rng.standard_normal(v.shape).astype(np.float32) * 0.3 for k, v in ex.get_params()["op"].items()}
    ex.set_params({"op": p})
    x = rng.standard_normal((2, 3, 9, 9)).astype(np.float32)
    t = F.conv2d(F.conv2d(torch.as_tensor(x), torch.as_tensor(p["depthwise_weight"]), groups=3), torch.as_tensor(p["pointwise_weight"]), torch.as_tensor(p["bias"]))
    cmp(ex.forward({"x": x}).outputs["loss"], (t ** 2).sum().numpy(), tol("keras", "float32", "loss"), "loss")


def test_keras_specific_node_is_rejected_elsewhere_with_a_clear_message():
    for target in ("pytorch", "jax"):
        g = sep_graph(target, out_channels=4)
        r = validate(g)
        [d] = [d for d in r.errors if d.code == "E_BACKEND_OP" and d.nodeId == "op"]
        assert "keras-specific" in d.message and f"targets '{target}'" in d.message and "not replaced" in d.message
        rep = backends.compat_report(g, target)
        v = rep.nodes["op"]
        assert v["status"] == "unsupported" and v["code"] == "E_BACKEND_OP" and "TensorFlow / Keras 3-specific" in v["reason"] and not rep.ok
        with pytest.raises(BackendError) as ei:
            backends.compile_graph(g, target)
        assert ei.value.code == "E_BACKEND_INCOMPATIBLE" and "E_BACKEND_OP" in str(ei.value)
        with pytest.raises(BackendError):
            backends.export_code(g, target)


def cumsum_graph(backend="jax", **cfg):
    b = GraphBuilder()
    b.input("x", ["N", 4, 5])
    b.node("op", "jax.lax.cumsum", **cfg)
    b.node("sq", "core.square")
    b.node("loss", "core.sum")
    b.chain("x", "op", "sq", "loss")
    g = b.build()
    g.backend = backend
    return g


@pytest.mark.parametrize("cfg", [{"axis": 1}, {"axis": 2, "reverse": True}], ids=["cumsum_axis1", "cumsum_reverse_axis2"])
def test_jax_cumsum_matches_native_jax_and_torch(cfg):
    skip_unless_available("jax")
    import jax
    import jax.numpy as jnp

    g = cumsum_graph(**cfg)
    ex = backends.compile_graph(g, "jax", compiled=False)
    x = np.random.default_rng(2).standard_normal((3, 4, 5)).astype(np.float32)
    res = ex.forward({"x": x}, capture=True)
    ax, rev = cfg["axis"], cfg.get("reverse", False)
    native = jnp.flip(jnp.cumsum(jnp.flip(jnp.asarray(x), ax), ax), ax) if rev else jnp.cumsum(jnp.asarray(x), ax)
    cmp(res.activations["op"], np.asarray(native), tol("jax", "float32", "forward"), "vs jnp.cumsum")
    tx = torch.as_tensor(x).requires_grad_(True)
    tref = torch.cumsum(tx.flip(ax), ax).flip(ax) if rev else torch.cumsum(tx, ax)
    cmp(res.activations["op"], tref.detach().numpy(), tol("jax", "float32", "forward"), "vs torch.cumsum")
    (tref ** 2).sum().backward()
    _, _, gi = ex.loss_and_grads({"x": x}, "loss", wrt_inputs=("x",))
    cmp(gi["x"], tx.grad.numpy(), tol("jax", "float32", "grad"), "gradient vs torch autograd")
    assert "lax.cumsum" in backends.export_code(g, "jax")


def test_jax_specific_node_is_rejected_elsewhere_with_a_clear_message():
    for target in ("pytorch", "keras"):
        g = cumsum_graph(target)
        r = validate(g)
        assert any(d.code == "E_BACKEND_OP" and d.nodeId == "op" and "jax-specific" in d.message for d in r.errors)
        v = backends.compat_report(g, target).nodes["op"]
        assert v["code"] == "E_BACKEND_OP" and "JAX-specific" in v["reason"]
        with pytest.raises(BackendError):
            backends.compile_graph(g, target)
    with pytest.raises(Exception):
        from graph_core.codegen import generate_pytorch
        generate_pytorch(cumsum_graph("pytorch"))  # PyTorch export refuses it too


def test_switching_a_graph_with_a_specific_node_back_to_pytorch_is_refused_not_rewritten():
    g = cumsum_graph("jax")
    assert validate(g).ok
    g.backend = "pytorch"
    r = validate(g)
    assert not r.ok and [d.code for d in r.errors] == ["E_BACKEND_OP"]
    assert g.node("op").type == "jax.lax.cumsum"  # still there: never replaced by torch.cumsum
