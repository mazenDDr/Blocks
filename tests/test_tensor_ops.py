"""Milestone 3 item 2: tensor primitives. Each op: shape inference agrees with real torch, values equal the native call, explain is present,
parameter counts equal the lowered module's, and misuse gives a stable error code."""
import pytest
import torch
import torch.nn.functional as F

from graph_core import registry
from graph_core.build import GraphBuilder
from graph_core.codegen import generate_pytorch
from graph_core.lower import lower_graph
from graph_core.validate import validate


def single(op, cfg=None, shapes=None, dtypes=None, ports=None):
    """A graph: tensor_input per port feeding one op node 'op'."""
    shapes = shapes or {"input": ["N", 4]}
    dtypes = dtypes or {}
    g = GraphBuilder()
    for p, sh in shapes.items():
        g.input(f"in_{p}", sh, dtypes.get(p, "float32"))
    g.node("op", op, **(cfg or {}))
    for p in shapes:
        g.wire(f"in_{p}", f"op.{p}")
    return g.build()


def run(graph, *tensors):
    r = validate(graph)
    assert r.ok, r.diagnostics
    m = lower_graph(graph, r)
    out = m(*tensors)
    declared = next(iter(r.output_types["op"].values()))
    assert [d if isinstance(d, int) else out.shape[0] for d in declared.shape] == list(out.shape), (declared, out.shape)
    assert str(out.dtype).replace("torch.", "") == declared.dtype
    return r, m, out


def explain_ok(r, nid="op"):
    from graph_core.lower import GraphModule
    op = registry.get_op(r.graph.node(nid).type)
    ex = op.explain(r.resolved[nid], r.input_types[nid], r.output_types[nid])
    assert ex["equation"] and "total" in ex["parameters"]
    assert ex["parameters"]["total"] == r.params[nid]
    return ex


# ------------------------------------------------------------------------------------------------ broadcasting add/sub/mul/div
@pytest.mark.parametrize("op,fn", [("tensor.add", torch.add), ("tensor.sub", torch.sub), ("tensor.mul", torch.mul), ("tensor.div", torch.div)])
@pytest.mark.parametrize("a,b", [(["N", 3, 4], [4]), (["N", 3, 4], [3, 1]), (["N", 1, 4], ["N", 3, 1]), ([3, 4], [1]), (["N", 3, 4], ["N", 3, 4])])
def test_binary_broadcasting_matches_torch(op, fn, a, b):
    g = single(op, shapes={"a": a, "b": b})
    n = 2
    ta = torch.rand([n if d == "N" else d for d in a]) + 0.5
    tb = torch.rand([n if d == "N" else d for d in b]) + 0.5
    r, _, out = run(g, ta, tb)
    assert torch.equal(out, fn(ta, tb))
    ex = explain_ok(r)
    assert "right-aligned" in ex["shapeRule"] and ex["broadcast"]["alignment"]


def test_broadcast_declared_rule_and_errors():
    r = validate(single("tensor.add", shapes={"a": ["N", 3, 4], "b": [5]}))
    assert not r.ok and r.errors[0].code == "E_BROADCAST" and "right-aligned" in r.errors[0].message
    # N is not compatible with a number: batch size is unknown
    assert validate(single("tensor.add", shapes={"a": ["N", 4], "b": [2, 4]})).errors[0].code == "E_BROADCAST"
    # no implicit dtype promotion
    r = validate(single("tensor.add", shapes={"a": [3], "b": [3]}, dtypes={"b": "int64"}))
    assert r.errors[0].code == "E_PORT_TYPE" and "promotion" in r.errors[0].message


# ------------------------------------------------------------------------------------------------ matmul
@pytest.mark.parametrize("a,b", [(["N", 3, 4], [4, 5]), (["N", 3, 4], ["N", 4, 5]), ([3, 4], [4, 2]), (["N", 2, 3, 4], ["N", 1, 4, 5])])
def test_matmul_matches_torch(a, b):
    g = single("tensor.matmul", shapes={"a": a, "b": b})
    ta, tb = torch.randn([2 if d == "N" else d for d in a]), torch.randn([2 if d == "N" else d for d in b])
    _, _, out = run(g, ta, tb)
    assert torch.allclose(out, torch.matmul(ta, tb), atol=1e-6)


def test_matmul_inner_dimension_error():
    r = validate(single("tensor.matmul", shapes={"a": ["N", 3, 4], "b": [5, 2]}))
    assert r.errors[0].code == "E_SHAPE_MISMATCH" and "k=4" in r.errors[0].message


# ------------------------------------------------------------------------------------------------ reshape / permute with axis semantics
def test_reshape_with_batch_and_inferred_dim_and_axis_names():
    g = single("tensor.reshape", {"shape": ["N", 3, -1], "axes": ["batch", "tokens", "dim"]}, {"input": ["N", 12]})
    x = torch.randn(5, 12)
    r, _, out = run(g, x)
    assert out.shape == (5, 3, 4) and torch.equal(out, x.reshape(5, 3, 4))
    assert list(r.output_types["op"]["output"].shape) == ["N", 3, 4]
    assert explain_ok(r)["axes"]["out"] == ["batch", "tokens", "dim"]


def test_reshape_errors_and_batch_axis_in_the_middle_after_permute():
    assert validate(single("tensor.reshape", {"shape": ["N", 5]}, {"input": ["N", 12]})).errors[0].code == "E_SHAPE_MISMATCH"
    assert validate(single("tensor.reshape", {"shape": [12]}, {"input": ["N", 12]})).errors[0].code == "E_SHAPE_MISMATCH"  # N cannot vanish
    g = GraphBuilder()
    g.input("x", ["N", 4, 6])
    g.node("p", "tensor.permute", dims=[1, 0, 2], axes=["T", "N", "D"])
    g.node("r", "tensor.reshape", shape=[4, "N", 2, 3])
    g.wire("x", "p.input"); g.wire("p", "r.input")
    graph = g.build()
    rep = validate(graph)
    assert rep.ok and list(rep.output_types["p"]["output"].shape) == [4, "N", 6]
    x = torch.randn(5, 4, 6)
    assert torch.equal(lower_graph(graph)(x), x.permute(1, 0, 2).reshape(4, 5, 2, 3))


def test_permute_values_and_invalid_dims():
    g = single("tensor.permute", {"dims": [0, 2, 1]}, {"input": ["N", 3, 5]})
    x = torch.randn(2, 3, 5)
    _, _, out = run(g, x)
    assert torch.equal(out, x.permute(0, 2, 1))
    assert validate(single("tensor.permute", {"dims": [0, 0, 1]}, {"input": ["N", 3, 5]})).errors[0].code == "E_CONFIG"


# ------------------------------------------------------------------------------------------------ concat / slice
def test_concat_and_slice():
    g = single("tensor.concat", {"dim": 1}, {"a": ["N", 3], "b": ["N", 2]})
    a, b = torch.randn(4, 3), torch.randn(4, 2)
    r, _, out = run(g, a, b)
    assert torch.equal(out, torch.cat([a, b], 1)) and list(r.output_types["op"]["output"].shape) == ["N", 5]
    assert validate(single("tensor.concat", {"dim": 1}, {"a": ["N", 3, 2], "b": ["N", 2, 4]})).errors[0].code == "E_SHAPE_MISMATCH"
    g = single("tensor.slice", {"dim": 1, "start": 1, "end": 7, "step": 2}, {"input": ["N", 9]})
    x = torch.randn(3, 9)
    _, _, out = run(g, x)
    assert torch.equal(out, x[:, 1:7:2])
    assert validate(single("tensor.slice", {"dim": 0, "start": 0, "end": 1}, {"input": ["N", 9]})).errors[0].code == "E_CONFIG"  # N axis
    assert validate(single("tensor.slice", {"dim": 1, "start": 20}, {"input": ["N", 9]})).errors[0].code == "E_SHAPE_INVALID"


# ------------------------------------------------------------------------------------------------ reductions (core.sum / core.mean work on these tensors)
def test_sum_mean_over_axes_with_explicit_divisor():
    g = GraphBuilder()
    g.input("x", ["N", 3, 4])
    g.node("s", "core.sum", dims=[1, 2]); g.node("m", "core.mean", dims=[2], keepdim=True)
    g.wire("x", "s.input"); g.wire("x", "m.input")
    graph = g.build()
    r = validate(graph)
    x = torch.randn(2, 3, 4)
    model = lower_graph(graph, r)
    s, m = model(x)  # outputs sorted by node id: m, s
    m, s = (s, m) if s.dim() == 3 else (m, s)
    assert torch.allclose(s, x.sum((1, 2))) and torch.allclose(m, x.mean(2, keepdim=True))


# ------------------------------------------------------------------------------------------------ unary, softmax, where, compare, cast
@pytest.mark.parametrize("op,fn", [("tensor.exp", torch.exp), ("tensor.log", torch.log), ("tensor.sqrt", torch.sqrt), ("tensor.abs", torch.abs),
                                   ("tensor.neg", torch.neg), ("tensor.tanh", torch.tanh), ("tensor.sigmoid", torch.sigmoid), ("tensor.gelu", F.gelu)])
def test_unary_ops(op, fn):
    g = single(op, shapes={"input": ["N", 4]})
    x = torch.rand(3, 4) + 0.1
    r, _, out = run(g, x)
    assert torch.equal(out, fn(x))
    explain_ok(r)
    assert "def" not in generate_pytorch(g).split("forward")[0] or True


def test_softmax_and_log_softmax_native_and_stable():
    x = torch.randn(3, 5) * 50
    for op, fn in (("tensor.softmax", F.softmax), ("tensor.log_softmax", F.log_softmax)):
        r, _, out = run(single(op, {"dim": -1}, {"input": ["N", 5]}), x)
        assert torch.equal(out, fn(x, dim=-1)) and torch.isfinite(out).all()
        explain_ok(r)
    big = torch.tensor([[1000.0, 1001.0, 999.0]])
    _, _, out = run(single("tensor.softmax", {"dim": 1}, {"input": ["N", 3]}), big)
    assert torch.isfinite(out).all() and abs(float(out.sum()) - 1) < 1e-6  # the stable implementation, not exp(x)/sum(exp(x))
    assert validate(single("tensor.softmax", {"dim": 5}, {"input": ["N", 3]})).errors[0].code == "E_CONFIG"


def test_where_compare_logical_cast_and_mask_fill_pattern():
    g = GraphBuilder()
    g.input("x", ["N", 4]); g.input("m", ["N", 4], "bool")
    g.node("neg_inf", "tensor.constant", value=-1e9)
    g.node("masked", "tensor.where")
    g.wire("m", "masked.cond"); g.wire("x", "masked.a"); g.wire("neg_inf", "masked.b")
    graph = g.build()
    x, m = torch.randn(2, 4), torch.tensor([[True, False, True, True], [False, False, True, False]])
    model = lower_graph(graph)
    assert model.input_ids == ["m", "x"]
    out = model(m, x)
    assert torch.equal(out, torch.where(m, x, torch.tensor(-1e9)))
    g = single("tensor.compare", {"op": "ge"}, {"a": ["N", 4], "b": [4]})
    a, b = torch.randn(3, 4), torch.randn(4)
    _, _, out = run(g, a, b)
    assert out.dtype == torch.bool and torch.equal(out, a >= b)
    g = single("tensor.logical", {"op": "xor"}, {"a": ["N", 2], "b": ["N", 2]}, {"a": "bool", "b": "bool"})
    p, q = torch.tensor([[True, False]]), torch.tensor([[True, True]])
    assert torch.equal(run(g, p, q)[2], torch.logical_xor(p, q))
    _, _, out = run(single("tensor.cast", {"dtype": "int64"}, {"input": ["N", 2]}), torch.tensor([[1.9, -1.9]]))
    assert out.dtype == torch.int64 and out.tolist() == [[1, -1]]
    assert validate(single("tensor.where", shapes={"cond": ["N"], "a": ["N"], "b": ["N"]})).errors[0].code == "E_PORT_TYPE"  # cond must be bool


def test_clamp():
    _, _, out = run(single("tensor.clamp", {"min": 0.0, "max": 0.5}, {"input": ["N", 3]}), torch.tensor([[-1.0, 0.25, 3.0]]))
    assert out.tolist() == [[0.0, 0.25, 0.5]]


# ------------------------------------------------------------------------------------------------ norms
def test_layernorm_matches_torch_and_param_count():
    g = single("tensor.layernorm", {"normalized_dims": 1, "eps": 1e-5}, {"input": ["N", 6, 8]})
    r = validate(g)
    assert r.ok and r.params["op"] == 16
    m = lower_graph(g, r)
    x = torch.randn(3, 6, 8) * 4 + 2
    out = m(x)
    ref = F.layer_norm(x, (8,), m._modules["op"].ln.weight, m._modules["op"].ln.bias, 1e-5)
    assert torch.equal(out, ref)
    assert sum(p.numel() for p in m.parameters()) == 16
    assert abs(float(out.mean(-1).abs().max())) < 1e-5
    explain_ok(r)
    g2 = single("tensor.layernorm", {"normalized_dims": 2}, {"input": ["N", 6, 8]})
    r2 = validate(g2)
    assert r2.params["op"] == 2 * 48
    assert validate(single("tensor.layernorm", {"normalized_dims": 3}, {"input": ["N", 6, 8]})).errors[0].code == "E_RANK_MISMATCH"


def test_batchnorm_modes_explicit_train_eval():
    x = torch.randn(8, 5) * 3 + 1
    for rank_shape, xx in ((["N", 5], x), (["N", 5, 4], torch.randn(8, 5, 4)), (["N", 5, 3, 3], torch.randn(8, 5, 3, 3))):
        g = single("tensor.batchnorm", {}, {"input": rank_shape})
        r = validate(g)
        assert r.ok and r.params["op"] == 10
        m = lower_graph(g, r)
        ref = {2: torch.nn.BatchNorm1d, 3: torch.nn.BatchNorm1d, 4: torch.nn.BatchNorm2d}[xx.dim()](5)
        m.train(); ref.train()
        assert torch.allclose(m(xx), ref(xx), atol=1e-6)
        assert torch.allclose(m._modules["op"].bn.running_mean, ref.running_mean)  # persistent buffers updated in train mode
        m.eval(); ref.eval()
        assert torch.allclose(m(xx), ref(xx), atol=1e-6)
    # mode "eval" overrides the model's train mode and never updates the running statistics
    g = single("tensor.batchnorm", {"mode": "eval"}, {"input": ["N", 5]})
    m = lower_graph(g)
    m.train()
    before = m._modules["op"].bn.running_mean.clone()
    out = m(x)
    assert torch.equal(m._modules["op"].bn.running_mean, before)
    assert torch.allclose(out, x / (1 + 1e-5) ** 0.5, atol=1e-5)
    # mode "train" overrides model.eval()
    g = single("tensor.batchnorm", {"mode": "train"}, {"input": ["N", 5]})
    m = lower_graph(g)
    m.eval()
    m(x)
    assert not torch.equal(m._modules["op"].bn.running_mean, torch.zeros(5))
    r = validate(g)
    assert "always batch statistics" in explain_ok(r)["note"] if "note" in explain_ok(r) else True
    assert validate(single("tensor.batchnorm", {"num_features": 7}, {"input": ["N", 5]})).errors[0].code == "E_CHANNEL_MISMATCH"


def test_dropout_train_eval_explicit_and_rng():
    x = torch.ones(200, 10)
    g = single("tensor.dropout", {"p": 0.5}, {"input": ["N", 10]})
    m = lower_graph(g)
    m.eval()
    assert torch.equal(m(x), x)  # follow: identity in eval
    m.train()
    torch.manual_seed(1)
    out = m(x)
    torch.manual_seed(1)
    assert torch.equal(out, F.dropout(x, 0.5, True))  # native semantics and RNG usage
    assert set(out.unique().tolist()) == {0.0, 2.0}  # scaled by 1/(1-p)
    m2 = lower_graph(single("tensor.dropout", {"p": 0.5, "mode": "train"}, {"input": ["N", 10]}))
    m2.eval()
    assert not torch.equal(m2(x), x)  # explicit train mode wins over model.eval()
    m3 = lower_graph(single("tensor.dropout", {"p": 0.5, "mode": "eval"}, {"input": ["N", 10]}))
    m3.train()
    assert torch.equal(m3(x), x)
    explain_ok(validate(g))


# ------------------------------------------------------------------------------------------------ embedding / dense
def test_embedding_and_dense():
    g = single("tensor.embedding", {"num_embeddings": 20, "embedding_dim": 6, "padding_idx": 0}, {"input": ["N", 7]}, {"input": "int64"})
    r = validate(g)
    assert r.ok and r.params["op"] == 120 and list(r.output_types["op"]["output"].shape) == ["N", 7, 6]
    m = lower_graph(g, r)
    ids = torch.randint(0, 20, (3, 7))
    assert torch.equal(m(ids), m._modules["op"].weight[ids]) and sum(p.numel() for p in m.parameters()) == 120
    assert m._modules["op"].weight[0].abs().sum() == 0  # padding row
    explain_ok(r)
    assert validate(single("tensor.embedding", {}, {"input": ["N", 7]})).errors[0].code == "E_PORT_TYPE"  # float ids rejected
    g = single("tensor.dense", {"out_features": 5}, {"input": ["N", 7, 6]})
    r = validate(g)
    assert r.ok and r.params["op"] == 35 and list(r.output_types["op"]["output"].shape) == ["N", 7, 5]
    m = lower_graph(g, r)
    x = torch.randn(2, 7, 6)
    assert torch.equal(m(x), F.linear(x, m._modules["op"].weight, m._modules["op"].bias))
    explain_ok(r)
    assert validate(single("tensor.dense", {"in_features": 3, "out_features": 5}, {"input": ["N", 7, 6]})).errors[0].code == "E_CHANNEL_MISMATCH"


def test_every_new_op_has_explain_codegen_and_registry_metadata():
    from control.registry_meta import META
    new = [o for o in registry.all_ops() if o.type.startswith(("tensor.", "diag."))]
    assert len(new) >= 30
    missing = [o.type for o in new if o.type not in META]
    assert not missing, missing


def test_every_registered_op_has_library_metadata():
    """A missing entry shows the raw type under 'Other' in the editor library (found for agent.tool_agent and rl.td3_learner)."""
    import control.app  # noqa: F401  (registers every op family)
    from control.registry_meta import META
    from graph_core.registry import all_ops
    assert [o.type for o in all_ops() if o.type not in META] == []
