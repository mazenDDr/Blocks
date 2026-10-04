"""Handwritten NATIVE baselines for the benchmark (what a user of each framework would write without the workbench), with the same optimizations
as the graph-lowered executable: PyTorch eager; Keras under tf.function; JAX under jit with a fused train step.

Every builder returns (forward, train) closures over inputs prepared once, and takes the SAME weights in graph layout (OIHW / (out, in)) that the
lowered executable receives, converting layout by hand (that conversion is part of building the model, not of the timed region).
Keras native data is NHWC: a Keras user feeds NHWC arrays, so the transposes the graph executable performs inside its compute path are NOT paid here;
that is a real difference between the two and is reported as such."""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------------------------------------- reference CNN (VISION 8.1)


def torch_cnn(params, x, y, lr):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    m = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
                      nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))
    with torch.no_grad():
        for idx, nid in ((0, "conv_1"), (3, "conv_2"), (8, "fc")):
            m[idx].weight.copy_(torch.as_tensor(params[nid]["weight"]))
            m[idx].bias.copy_(torch.as_tensor(params[nid]["bias"]))
    tx, ty = torch.as_tensor(x), torch.as_tensor(y)
    opt = torch.optim.SGD(m.parameters(), lr=lr)

    def fwd():
        with torch.no_grad():
            return F.cross_entropy(m(tx), ty)

    def train():
        opt.zero_grad()
        loss = F.cross_entropy(m(tx), ty)
        loss.backward()
        opt.step()
        return loss.detach()

    return fwd, train


def keras_cnn(params, x, y, lr):
    import keras
    import tensorflow as tf

    m = keras.Sequential([keras.layers.Input((64, 64, 3)), keras.layers.Conv2D(32, 3, padding="same"), keras.layers.ReLU(), keras.layers.MaxPooling2D(2),
                          keras.layers.Conv2D(64, 3, padding="same"), keras.layers.ReLU(), keras.layers.MaxPooling2D(2),
                          keras.layers.GlobalAveragePooling2D(), keras.layers.Dense(10)])
    for lay, nid in ((m.layers[0], "conv_1"), (m.layers[3], "conv_2")):
        lay.kernel.assign(np.transpose(params[nid]["weight"], (2, 3, 1, 0)))
        lay.bias.assign(params[nid]["bias"])
    m.layers[-1].kernel.assign(params["fc"]["weight"].T)
    m.layers[-1].bias.assign(params["fc"]["bias"])
    tx, ty = tf.constant(np.transpose(x, (0, 2, 3, 1))), tf.constant(y)
    opt = keras.optimizers.SGD(learning_rate=lr)

    def loss_fn():
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ty, logits=m(tx)))

    @tf.function
    def fwd_tf():
        return loss_fn()

    @tf.function
    def train_tf():
        with tf.GradientTape() as tape:
            loss = loss_fn()
        opt.apply(tape.gradient(loss, m.trainable_variables), m.trainable_variables)
        return loss

    return (lambda: fwd_tf().numpy()), (lambda: train_tf().numpy())


def jax_cnn(params, x, y, lr):
    import jax
    import jax.numpy as jnp
    from jax import lax

    p = {k: {n: jnp.asarray(v) for n, v in d.items()} for k, d in params.items()}
    tx, ty = jnp.asarray(x), jnp.asarray(y.astype(np.int32))
    dn = ("NCHW", "OIHW", "NCHW")

    def conv(h, d):
        return lax.conv_general_dilated(h, d["weight"], (1, 1), ((1, 1), (1, 1)), dimension_numbers=dn) + d["bias"].reshape(1, -1, 1, 1)

    def pool(h):
        return lax.reduce_window(h, -jnp.inf, lax.max, (1, 1, 2, 2), (1, 1, 2, 2), "VALID")

    def loss_fn(p):
        h = pool(jax.nn.relu(conv(tx, p["conv_1"])))
        h = pool(jax.nn.relu(conv(h, p["conv_2"])))
        h = h.mean(axis=(2, 3))
        logits = h @ p["fc"]["weight"].T + p["fc"]["bias"]
        return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), ty[:, None], axis=-1))

    fwd_j = jax.jit(loss_fn)

    @jax.jit
    def step(p, lr):
        loss, g = jax.value_and_grad(loss_fn)(p)
        return jax.tree_util.tree_map(lambda a, b: a - lr * b, p, g), loss

    state = {"p": p}
    lr_a = jnp.float32(lr)

    def train():
        state["p"], loss = step(state["p"], lr_a)
        return jax.block_until_ready(loss)

    return (lambda: jax.block_until_ready(fwd_j(p))), train


# ---------------------------------------------------------------------------------------------------------- tiny MSE fixture (absolute overhead of a 4-node graph)
def torch_mse(pred, target):
    import torch

    p, t = torch.as_tensor(pred).requires_grad_(True), torch.as_tensor(target)

    def fwd():
        with torch.no_grad():
            return ((p - t) ** 2).sum() * (1.0 / 3.0)

    def grad():
        p.grad = None
        loss = ((p - t) ** 2).sum() * (1.0 / 3.0)
        loss.backward()
        return p.grad

    return fwd, grad


def keras_mse(pred, target):
    import tensorflow as tf

    p, t = tf.constant(pred), tf.constant(target)

    @tf.function
    def f():
        return tf.reduce_sum(tf.square(p - t)) * (1.0 / 3.0)

    @tf.function
    def g():
        with tf.GradientTape() as tape:
            tape.watch(p)
            loss = tf.reduce_sum(tf.square(p - t)) * (1.0 / 3.0)
        return tape.gradient(loss, p)

    return (lambda: f().numpy()), (lambda: g().numpy())


def jax_mse(pred, target):
    import jax
    import jax.numpy as jnp

    p, t = jnp.asarray(pred), jnp.asarray(target)
    f = jax.jit(lambda p: jnp.sum(jnp.square(p - t)) * (1.0 / 3.0))
    g = jax.jit(jax.grad(lambda p: jnp.sum(jnp.square(p - t)) * (1.0 / 3.0)))
    return (lambda: jax.block_until_ready(f(p))), (lambda: jax.block_until_ready(g(p)))


NATIVE = {("pytorch", "cnn"): torch_cnn, ("keras", "cnn"): keras_cnn, ("jax", "cnn"): jax_cnn,
          ("pytorch", "mse"): torch_mse, ("keras", "mse"): keras_mse, ("jax", "mse"): jax_mse}
