"""Milestone 6a exit evidence: the shared reference workloads (reference CNN, MSE fixture, SGD teaching step, residual CNN, shared parameters) run on
every available backend with IDENTICAL copied weights and match PyTorch within the declared tolerances; layout, weight layout, padding and
initialization are explicit; unsupported semantics are refused before execution; a backend switch never drops a setting or changes sharing."""
import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import backends
from backend_helpers import EXAMPLES, cmp, cnn_with_loss, skip_unless_available
from backends import BackendError
from backends.registry import BACKEND_IDS, availability
from backends.tolerances import tol
from backends.workloads import conv_flatten_linear_graph, sgd_teaching_graph, residual_core_add_graph, shared_graph
from graph_core.build import GraphBuilder
from graph_core.hashing import semantic_hash
from graph_core.lower import lower_graph
from graph_core.project_io import load_project
from graph_core.validate import ExecutionBlocked, validate

ALL = list(BACKEND_IDS)
LR = 0.1


def native_reference():
    return nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))


NATIVE_NODE = {"conv_1": 0, "conv_2": 3, "fc": 8}


@pytest.mark.parametrize("backend", ALL)
def test_reference_cnn_matches_native_pytorch(backend):
    """VISION 8.1 CNN: logits, loss, parameter gradients and one SGD update match handwritten native PyTorch with copied weights."""
    skip_unless_available(backend)
    g = cnn_with_loss()
    rep = backends.compat_report(g, backend)
    assert rep.ok and rep.counts["unsupported"] == 0, rep.nodes
    torch.manual_seed(3)
    ref = native_reference()
    rng = np.random.default_rng(3)
    x = rng.standard_normal((5, 3, 64, 64)).astype(np.float32)
    y = rng.integers(0, 10, (5,)).astype(np.int64)
    ex = backends.compile_graph(g, backend, compiled=False)
    ex.set_params({nid: {n: p.detach().numpy().copy() for n, p in ref[i].named_parameters()} for nid, i in NATIVE_NODE.items()})
    t = {q: tol(backend, "float32", q) for q in ("forward", "loss", "grad", "update")}

    inp = {"images": x, "labels": y}
    res = ex.forward(inp, capture=True)
    tx, ty = torch.as_tensor(x), torch.as_tensor(y)
    logits = ref(tx)
    assert res.activations["fc"].shape == (5, 10) == tuple(logits.shape)  # shapes
    cmp(res.activations["fc"], logits.detach().numpy(), t["forward"], "logits")
    loss_ref = F.cross_entropy(logits, ty)
    cmp(res.outputs["ce"], loss_ref.detach().numpy(), t["loss"], "loss")
    # every node's activation is readable in GRAPH layout (NCHW) whatever the native layout
    assert res.activations["conv_1"].shape == (5, 32, 64, 64) and res.activations["pool_2"].shape == (5, 64, 16, 16)

    loss, grads, _ = ex.loss_and_grads(inp, "ce")
    loss_ref.backward()
    for nid, i in NATIVE_NODE.items():
        for n, p in ref[i].named_parameters():
            cmp(grads[nid][n], p.grad.numpy(), t["grad"], f"grad {nid}.{n}")
            assert np.abs(grads[nid][n]).sum() > 0
    ex.sgd_step(inp, LR, "ce")
    torch.optim.SGD(ref.parameters(), lr=LR).step()
    after = ex.get_params()
    for nid, i in NATIVE_NODE.items():
        for n, p in ref[i].named_parameters():
            cmp(after[nid][n], p.detach().numpy(), t["update"], f"updated {nid}.{n}")


@pytest.mark.parametrize("backend", ALL)
def test_mse_fixture_a22_on_every_backend(backend):
    """A22: loss 5/12, gradient [0, 1/3, -2/3], each intermediate addressable by node id."""
    skip_unless_available(backend)
    g = load_project(EXAMPLES / "mse_teaching.project.json").graph
    assert backends.compat_report(g, backend).ok
    ex = backends.compile_graph(g, backend, compiled=False)
    pred, target = np.array([1.0, 2.5, 2.0], np.float32), np.array([1.0, 2.0, 3.0], np.float32)
    res = ex.forward({"pred": pred, "target": target}, capture=True)
    t = tol(backend, "float32", "forward")
    cmp(res.activations["diff"], [0.0, 0.5, -1.0], t, "diff")
    cmp(res.activations["squared"], [0.0, 0.25, 1.0], t, "squared")
    cmp(res.activations["total"], 1.25, t, "total")
    cmp(res.outputs["loss"], 5 / 12, tol(backend, "float32", "loss"), "loss")
    loss, _, gi = ex.loss_and_grads({"pred": pred, "target": target}, "loss", wrt_inputs=("pred",))
    cmp(gi["pred"], [0.0, 1 / 3, -2 / 3], tol(backend, "float32", "grad"), "d loss / d pred")


@pytest.mark.parametrize("backend", ["pytorch", "keras"])
def test_mse_fixture_float64_where_supported(backend):
    skip_unless_available(backend)
    g = copy.deepcopy(load_project(EXAMPLES / "mse_teaching.project.json").graph)
    for n in g.nodes:
        if n.type == "core.tensor_input":
            n.config["dtype"] = "float64"
    ex = backends.compile_graph(g, backend, compiled=False)
    p, tg = np.array([1.0, 2.5, 2.0]), np.array([1.0, 2.0, 3.0])
    loss, _, gi = ex.loss_and_grads({"pred": p, "target": tg}, "loss", wrt_inputs=("pred",))
    cmp(loss, 5 / 12, tol(backend, "float64", "loss"), "float64 loss")
    cmp(gi["pred"], [0.0, 1 / 3, -2 / 3], tol(backend, "float64", "grad"), "float64 grad")


@pytest.mark.parametrize("backend", ALL)
def test_a34_sgd_teaching_step_from_a_graph_gradient(backend):
    """w = 2, gradient 0.6 from the graph, lr = 0.1 -> 1.94 (an input updated by the gradient the backend computed)."""
    skip_unless_available(backend)
    ex = backends.compile_graph(sgd_teaching_graph(), backend, compiled=False)
    w = np.array([2.0], np.float32)
    loss, _, gi = ex.loss_and_grads({"w": w}, "loss", wrt_inputs=("w",))
    cmp(gi["w"], [0.6], tol(backend, "float32", "grad"), "gradient")
    cmp(w - 0.1 * gi["w"], [1.94], tol(backend, "float32", "update"), "w after the step")


@pytest.mark.parametrize("backend", ALL)
def test_residual_cnn_with_expanded_modules_matches_pytorch(backend):
    """Composite instances expand to flat nodes (res1/conv_a ...); skip connections join NHWC values on Keras without a silent layout mismatch."""
    skip_unless_available(backend)
    g = load_project(EXAMPLES / "residual_cnn.project.json").graph
    if backend != "pytorch":  # the shipped example joins with tensor.add (outside the portable subset): reported, not substituted
        rep0 = backends.compat_report(g, backend)
        assert {n: v["code"] for n, v in rep0.nodes.items() if v["status"] == "unsupported"} == {"res1/skip_add": "E_BACKEND_UNSUPPORTED_OP", "res2/skip_add": "E_BACKEND_UNSUPPORTED_OP"}
    g = residual_core_add_graph()  # the same network with the portable core.add as the skip join
    rep = backends.compat_report(g, backend)
    assert rep.ok, rep.nodes
    assert any(n.startswith("res1/") for n in rep.nodes)
    rng = np.random.default_rng(5)
    shape = [d if isinstance(d, int) else 2 for d in g.node("images").config["shape"]]
    x = rng.standard_normal(shape).astype(np.float32)
    ref = backends.compile_graph(g, "pytorch")
    ex = backends.compile_graph(g, backend, compiled=False)
    ex.set_params(ref.get_params())
    a, b = ref.forward({"images": x}, capture=True), ex.forward({"images": x}, capture=True)
    t = tol(backend, "float32", "forward")
    for node in a.activations:
        cmp(b.activations[node], a.activations[node], t, node)


@pytest.mark.parametrize("backend", ["keras", "jax"])
def test_flatten_after_spatial_maps_keeps_graph_order(backend):
    """NHWC flattens H,W,C and NCHW flattens C,H,W: Keras must transpose back before the reshape (recorded); a silent skip would permute the Linear inputs."""
    skip_unless_available(backend)
    g = conv_flatten_linear_graph()
    rep = backends.compat_report(g, backend)
    if backend == "keras":
        assert any("NHWC -> NCHW" in c["detail"] and c["kind"] == "layout" for c in rep.nodes["f"]["conversions"]), rep.nodes["f"]
    else:
        assert all(c["kind"] != "layout" for n in rep.nodes.values() for c in n["conversions"])  # JAX: no layout conversion anywhere
    ref = backends.compile_graph(g, "pytorch")
    ex = backends.compile_graph(g, backend, compiled=False)
    ex.set_params(ref.get_params())
    x = np.random.default_rng(0).standard_normal((3, 2, 8, 8)).astype(np.float32)
    cmp(ex.forward({"x": x}).outputs["fc"], ref.forward({"x": x}).outputs["fc"], tol(backend, "float32", "forward"), "logits")


def test_keras_layout_and_weight_layout_conversions_are_listed_per_node():
    rep = backends.compat_report(cnn_with_loss(), "keras")
    conv_1 = rep.nodes["conv_1"]
    kinds = {c["kind"] for c in conv_1["conversions"]}
    assert conv_1["status"] == "converted" and {"layout", "weight_layout", "padding"} <= kinds
    assert any("OIHW" in c["detail"] and "HWIO" in c["detail"] for c in conv_1["conversions"])
    assert any("NCHW -> NHWC" in c["detail"] for c in conv_1["conversions"])  # recorded where the transpose happens
    assert any("(out,in)" in c["detail"] for c in rep.nodes["fc"]["conversions"])
    assert any("H = W = 1" in c["detail"] for c in rep.nodes["flatten"]["conversions"])  # the one place a transpose is provably unnecessary
    assert any("padding" == c["kind"] and "tf.pad" in c["detail"] for c in conv_1["conversions"])
    jx = backends.compat_report(cnn_with_loss(), "jax")
    assert jx.nodes["conv_1"]["status"] == "supported"  # NCHW/OIHW are native to lax.conv_general_dilated
    assert jx.nodes["images"]["conversions"] == []  # float images are not converted...
    assert any(c["kind"] == "dtype" and "int32" in c["detail"] for c in jx.nodes["labels"]["conversions"])  # ...but int64 labels become int32, listed
    assert any(c["kind"] == "dtype" and "int32" in c["detail"] for c in jx.nodes["ce"]["conversions"])


def test_init_differences_are_declared_and_real():
    for b in ("keras", "jax"):
        rep = backends.compat_report(cnn_with_loss(), b).to_json()
        assert "differ from PyTorch" in rep["init"] or "differ" in rep["init"]
    skip_unless_available("keras")
    skip_unless_available("jax")
    g = cnn_with_loss()
    pt = backends.compile_graph(g, "pytorch", seed=0).get_params()
    ke = backends.compile_graph(g, "keras", seed=0, compiled=False).get_params()
    jx = backends.compile_graph(g, "jax", seed=0, compiled=False).get_params()
    # Keras: Glorot-uniform kernels (fan_in + fan_out), zero biases
    w = ke["conv_1"]["weight"]
    limit = (6 / (3 * 9 + 32 * 9)) ** 0.5
    assert np.abs(w).max() <= limit + 1e-6 and np.abs(w).max() > 0.5 * limit
    assert not ke["conv_1"]["bias"].any()
    # JAX: uniform(+-1/sqrt(fan_in)) for kernel and bias
    assert np.abs(jx["conv_1"]["weight"]).max() <= 1 / 27 ** 0.5 + 1e-6 and np.abs(jx["conv_1"]["bias"]).max() > 0
    # same seed, different backend -> different values (declared, never claimed equal)
    assert not np.allclose(pt["conv_1"]["weight"], jx["conv_1"]["weight"]) and not np.allclose(pt["conv_1"]["weight"], ke["conv_1"]["weight"])
    # determinism within a backend
    assert np.array_equal(jx["fc"]["weight"], backends.compile_graph(g, "jax", seed=0, compiled=False).get_params()["fc"]["weight"])


# ---------------------------------------------------------------------------------------------------- A18: unsupported, before execution
def reflect_graph(backend="pytorch"):
    b = GraphBuilder()
    b.input("x", ["N", 3, 8, 8])
    b.node("c", "pytorch.nn.conv2d", out_channels=4, kernel_size=[3, 3], padding=[1, 1], padding_mode="reflect")
    b.chain("x", "c")
    g = b.build()
    g.backend = backend
    return g


def test_a18_selecting_a_backend_with_unsupported_features_fails_clearly_before_execution():
    g = reflect_graph("keras")
    before = semantic_hash(g)
    r = validate(g)
    [d] = [d for d in r.errors if d.code == "E_BACKEND_UNSUPPORTED_PADDING_MODE"]
    assert d.nodeId == "c" and "reflect" in d.message and d.path == "/nodes/c" and "not substituted" in d.message
    assert any(f.key == "backend" and f.value == "pytorch" for f in d.fixes)
    assert not r.ok
    # nothing may be executed or exported from it
    with pytest.raises(BackendError) as ei:
        backends.compile_graph(g, "keras")
    assert ei.value.code == "E_BACKEND_INCOMPATIBLE" and "padding_mode" in str(ei.value) and ei.value.report.nodes["c"]["status"] == "unsupported"
    with pytest.raises(BackendError) as ei:
        backends.export_code(g, "keras")
    assert ei.value.code == "E_BACKEND_INCOMPATIBLE"
    with pytest.raises(ExecutionBlocked):
        lower_graph(g)  # PyTorch lowering refuses a graph that targets another backend instead of silently running it
    assert semantic_hash(g) == before and g.node("c").config["padding_mode"] == "reflect"  # no setting was dropped or rewritten
    # the same graph is fine on PyTorch and JAX
    assert backends.compat_report(g, "pytorch").ok and backends.compat_report(g, "jax").ok


@pytest.mark.parametrize("config,code", [
    ({"dilation": [2, 2], "stride": [2, 2]}, "E_BACKEND_UNSUPPORTED_DILATION"),
    ({"padding_mode": "circular", "padding": [1, 1]}, "E_BACKEND_UNSUPPORTED_PADDING_MODE"),
])
def test_unsupported_conv_settings_report_stable_codes_on_keras(config, code):
    b = GraphBuilder()
    b.input("x", ["N", 3, 12, 12])
    b.node("c", "pytorch.nn.conv2d", out_channels=4, kernel_size=[3, 3], **config)
    b.chain("x", "c")
    rep = backends.compat_report(b.build(), "keras")
    assert rep.nodes["c"]["code"] == code and not rep.ok and rep.executable is False


def test_unsupported_operations_are_listed_per_node_never_substituted():
    g = load_project(EXAMPLES / "shared_encoder.project.json").graph  # uses tensor.* ops
    for b in ("keras", "jax"):
        rep = backends.compat_report(g, b)
        bad = {n: v for n, v in rep.nodes.items() if v["status"] == "unsupported"}
        assert bad and all(v["code"] == "E_BACKEND_UNSUPPORTED_OP" and "not replaced" in v["reason"] for v in bad.values()), (b, rep.nodes)
        assert not rep.ok
        with pytest.raises(BackendError):
            backends.compile_graph(g, b)
    assert backends.compat_report(g, "pytorch").ok


def test_unavailable_backend_is_reported_with_its_reason_and_refuses(monkeypatch):
    from backends import compat as compat_mod
    from backends.registry import Availability

    monkeypatch.setattr(compat_mod, "availability", lambda b: Availability(False, {}, "no wheel for this platform (simulated)") if b == "jax" else availability(b))
    rep = backends.compat_report(cnn_with_loss(), "jax")
    assert rep.ok and not rep.available and not rep.executable and "simulated" in rep.availability_reason
    with pytest.raises(BackendError) as ei:
        backends.compile_graph(cnn_with_loss(), "jax")
    assert ei.value.code == "E_BACKEND_UNAVAILABLE" and "no wheel" in ei.value.message


def test_jax_refuses_to_feed_out_of_range_labels_and_float64():
    skip_unless_available("jax")
    ex = backends.compile_graph(cnn_with_loss(), "jax", compiled=False)
    x = np.zeros((1, 3, 64, 64), np.float32)
    with pytest.raises(BackendError) as ei:
        ex.forward({"images": x, "labels": np.array([2 ** 40], np.int64)})
    assert ei.value.code == "E_BACKEND_DTYPE_RANGE"
    with pytest.raises(BackendError) as ei:
        ex.forward({"images": x.astype(np.float64), "labels": np.array([1], np.int64)})
    assert ei.value.code == "E_BACKEND_UNSUPPORTED_DTYPE"


# ---------------------------------------------------------------------------------------------------- parameter sharing
@pytest.mark.parametrize("backend", ALL)
def test_shared_parameters_stay_shared_on_every_backend(backend):
    skip_unless_available(backend)
    g = shared_graph()
    rep = backends.compat_report(g, backend)
    assert rep.ok and rep.sharing == [{"owner": "enc_u", "users": ["enc_v"]}]
    ex = backends.compile_graph(g, backend, compiled=False)
    params = ex.get_params()
    assert set(params) == {"enc_u"}  # one set of tensors, owned once; the call site has none of its own
    torch.manual_seed(0)
    conv = nn.Conv2d(2, 3, 3, padding=1)
    ex.set_params({"enc_u": {k: p.detach().numpy().copy() for k, p in conv.named_parameters()}})
    rng = np.random.default_rng(2)
    u, v = (rng.standard_normal((2, 2, 6, 6)).astype(np.float32) for _ in range(2))
    tu, tv = torch.as_tensor(u), torch.as_tensor(v)
    loss_ref = ((conv(tu) + conv(tv)) ** 2).mean()
    loss_ref.backward()  # gradients accumulate over BOTH call sites
    loss, grads, _ = ex.loss_and_grads({"u": u, "v": v}, "loss")
    t = tol(backend, "float32", "grad")
    cmp(loss, loss_ref.detach().numpy(), tol(backend, "float32", "loss"), "loss")
    cmp(grads["enc_u"]["weight"], conv.weight.grad.numpy(), t, "shared weight gradient")
    cmp(grads["enc_u"]["bias"], conv.bias.grad.numpy(), t, "shared bias gradient")
    # one update moves both call sites: the outputs of the two sites stay equal for equal inputs
    ex.sgd_step({"u": u, "v": v}, 0.1, "loss")
    res = ex.forward({"u": u, "v": u}, capture=True)
    cmp(res.activations["enc_v"], res.activations["enc_u"], tol(backend, "float32", "forward"), "call sites after an update")


def test_exports_keep_the_sharing_visible():
    g = shared_graph()
    k, j = backends.export_code(g, "keras"), backends.export_code(g, "jax")
    assert "# node: enc_v (shares parameters with enc_u)" in k and "self.enc_v =" not in k  # one layer object, called twice
    assert "params['enc_u']" in j and "params['enc_v']" not in j and "(shares parameters with enc_u)" in j


@pytest.mark.parametrize("backend", ["keras", "jax"])
def test_compiled_tf_function_and_jit_agree_with_eager(backend):
    """The benchmark runs the compiled path (tf.function / jax.jit); it must compute what the eager path computes: outputs, activations, gradients, SGD update."""
    skip_unless_available(backend)
    g = cnn_with_loss()
    ref = backends.compile_graph(g, "pytorch", seed=6)
    rng = np.random.default_rng(6)
    inp = {"images": rng.standard_normal((4, 3, 64, 64)).astype(np.float32), "labels": rng.integers(0, 10, (4,)).astype(np.int64)}
    eager, comp = backends.compile_graph(g, backend, compiled=False), backends.compile_graph(g, backend, compiled=True)
    for e in (eager, comp):
        e.set_params(ref.get_params())
    t = tol(backend, "float32", "forward")
    a, b = eager.forward(inp, capture=True), comp.forward(inp, capture=True)
    for n in a.activations:
        cmp(b.activations[n], a.activations[n], t, n)
    cmp(comp.forward(inp).outputs["ce"], a.outputs["ce"], t, "outputs-only forward")
    (la, ga, _), (lb, gb, _) = eager.loss_and_grads(inp, "ce"), comp.loss_and_grads(inp, "ce")
    for n in ga:
        for k in ga[n]:
            cmp(gb[n][k], ga[n][k], tol(backend, "float32", "grad"), f"grad {n}.{k}")
    eager.sgd_step(inp, LR, "ce")
    comp.sgd_step(inp, LR, "ce")
    pa, pb = eager.get_params(), comp.get_params()
    for n in pa:
        for k in pa[n]:
            cmp(pb[n][k], pa[n][k], tol(backend, "float32", "update"), f"update {n}.{k}")
    fwd, step = comp.bench_closures(inp, LR, "ce")  # the benchmark's closures run the same step
    fwd()
    l0 = float(step())
    l1 = float(step())
    assert l1 < l0 + 1e-6 and np.isfinite(l0)
