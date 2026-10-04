"""Milestone 3 item 1/3/6: reusable composite modules, nested identity, versioning, A24, A13, A37."""
import copy

import pytest
import torch

from graph_core.build import GraphBuilder, ModuleBuilder, residual_block
from graph_core.codegen import generate_pytorch, source_map
from graph_core.composite import expand
from graph_core.hashing import semantic_hash
from graph_core.lower import capture_activations, lower_graph
from graph_core.project_io import Project, load_project, save_project
from graph_core.schema import Graph
from graph_core.validate import ExecutionBlocked, validate
from m3_helpers import residual_net


def codes(r):
    return {d.code for d in r.diagnostics}


def test_residual_block_without_code_two_instances_nested_identity():
    g = residual_net()
    r = validate(g)
    assert r.ok, r.diagnostics
    # every inner node has its full identity path
    assert {"res1/conv_a", "res1/conv_b", "res2/conv_a", "res2/skip_add", "res2/relu_out"} <= set(r.order)
    model = lower_graph(g)
    assert "res1/conv_a" in dict(model.named_children())
    x = torch.randn(2, 3, 16, 16)
    assert model(x).shape == (2, 8, 16, 16)
    # default instantiation clones: separate parameter tensors, counted separately
    p1, p2 = model._modules["res1/conv_a"].weight, model._modules["res2/conv_a"].weight
    assert p1 is not p2 and not torch.equal(p1, p2)
    assert r.total_params == (3 * 8 * 9 + 8) + 2 * 2 * (8 * 8 * 9 + 8)
    assert sum(p.numel() for p in model.parameters()) == r.total_params
    # the packaged skip really is a skip: zero both convs' weights and bias => y = relu(x)
    with torch.no_grad():
        for n in ("res1/conv_a", "res1/conv_b", "res2/conv_a", "res2/conv_b"):
            model._modules[n].weight.zero_(); model._modules[n].bias.zero_()
    stem_out = model._modules["stem"](x)
    assert torch.allclose(model(x), torch.relu(torch.relu(stem_out)))


def test_expansion_reports_instances_and_members():
    ex = expand(residual_net())
    inst = ex.instances["res1"]
    assert inst.kind == "core.composite" and inst.module == "residual_block" and inst.inputs == ["x"] and inst.outputs == ["y"]
    assert set(inst.members) == {"res1/conv_a", "res1/relu_a", "res1/conv_b", "res1/skip_add", "res1/relu_out"}
    assert inst.out_map["y"] == ("res1/relu_out", "output")
    assert "own parameters" in ex.instances["res2"].note  # the clone default is stated


def test_a24_modifying_internal_op_updates_equation_graph_runtime_hash_and_leaves_no_stale_behaviour():
    g1 = residual_net()
    h1, code1 = semantic_hash(g1), generate_pytorch(g1)
    m1 = lower_graph(g1)
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16, 16)
    # edit one internal op of the module definition: the final activation becomes gelu (tensor.gelu)
    g2 = Graph.model_validate(copy.deepcopy(g1.to_json()))
    mod = g2.modules[0]
    for n in mod.nodes:
        if n.id == "relu_out":
            n.type, n.config = "tensor.gelu", {}
    r2 = validate(g2)
    assert r2.ok, r2.diagnostics
    assert semantic_hash(g2) != h1  # semantic hash changes
    code2 = generate_pytorch(g2)
    assert code2 != code1 and "gelu" in code2 and "gelu" not in code1  # exported graph agrees
    ops = [type(m).__name__ for m in lower_graph(g2).modules()]
    assert "Un" in ops  # runtime: a different module is built (the gelu wrapper), no cached relu
    # equation panel text (explain) of the edited node is the new op's
    from graph_core import registry
    ex = registry.get_op("tensor.gelu").explain(r2.resolved["res1/relu_out"], r2.input_types["res1/relu_out"], r2.output_types["res1/relu_out"])
    assert "Phi" in ex["equation"]
    # runtime values differ: same weights, relu vs gelu
    m2 = lower_graph(g2)
    m2.load_state_dict(m1.state_dict())
    assert not torch.allclose(m1(x), m2(x))
    # the original is untouched and re-validating it still gives the original behaviour (no stale cache in either direction)
    assert semantic_hash(g1) == h1 and generate_pytorch(g1) == code1
    m1b = lower_graph(g1)
    m1b.load_state_dict(m1.state_dict())
    assert torch.equal(m1b(x), m1(x))


def test_both_instances_follow_the_definition():
    g = residual_net()
    g.modules[0].nodes[-1].type, g.modules[0].nodes[-1].config = "tensor.gelu", {}
    r = validate(g)
    assert r.ok
    assert {r.expansion.graph.node(n).type for n in ("res1/relu_out", "res2/relu_out")} == {"tensor.gelu"}


def test_module_signature_is_checked_with_clear_errors():
    g = residual_net()
    g.nodes[1].config["out_channels"] = 5  # stem now outputs 5 channels but residual_block declares 8 input channels
    r = validate(g)
    assert "E_PORT_TYPE" in codes(r)
    msg = next(d.message for d in r.diagnostics if d.code == "E_PORT_TYPE")
    assert "res1" in msg and "expects dimension 1 = 8" in msg


def test_module_version_pinning_no_silent_upgrade():
    g = residual_net()
    g.modules.append(residual_block(8, version="2.0.0"))
    r = validate(g)
    assert r.ok  # both versions exist; instances pin 1.0.0
    for n in g.nodes:
        if n.id == "res1":
            n.config["version"] = "3.0.0"
    r = validate(g)
    assert "E_MODULE_VERSION" in codes(r) and "No silent upgrade" in next(d.message for d in r.diagnostics if d.code == "E_MODULE_VERSION")
    with pytest.raises(ExecutionBlocked):
        lower_graph(g)


def test_missing_module_and_bad_args_and_param_type():
    g = residual_net()
    g.nodes[2].config["module"] = "nope"
    r = validate(g)
    assert "E_MODULE_MISSING" in codes(r) and not r.ok
    g = residual_net()
    g.nodes[2].config["args"] = {"typo": 1}
    assert "E_CONFIG" in codes(validate(g))
    g = residual_net()
    g.nodes[2].config["args"] = {"channels": "eight"}
    assert "E_CONFIG" in codes(validate(g))


def test_nested_composites_keep_full_paths_and_codegen_source_map():
    outer = ModuleBuilder("two_res", "1.0.0")
    x = outer.in_("x", [None, 8, None, None])
    outer.instance("ra", "residual_block", channels=8)
    outer.instance("rb", "residual_block", channels=8)
    outer.wire(x, "ra.x")
    outer.wire("ra.y", "rb.x")
    outer.out("y", "rb.y")
    g = GraphBuilder()
    g.modules += [residual_block(8), outer.build()]
    g.input("img", ["N", 8, 8, 8])
    g.instance("block", "two_res")
    g.wire("img", "block.x")
    r = validate(g.build())
    assert r.ok, r.diagnostics
    assert "block/ra/conv_a" in r.order and "block/rb/relu_out" in r.order
    assert set(r.expansion.instances["block"].children) == {"block/ra", "block/rb"}
    assert {"block/ra", "block/rb", "block"} <= set(r.expansion.instances)
    smap = source_map(generate_pytorch(g.build()))
    assert "block/rb/conv_b" in smap


def test_recursive_module_is_rejected():
    m = ModuleBuilder("loopy")
    x = m.in_("x")
    m.instance("again", "loopy")
    m.wire(x, "again.x")
    m.out("y", "again.y")
    g = GraphBuilder()
    g.modules.append(m.build())
    g.input("i", ["N", 4])
    g.instance("a", "loopy")
    g.wire("i", "a.x")
    r = validate(g.build())
    assert "E_MODULE_RECURSION" in codes(r)


def test_cycle_through_composite_still_rejected():
    g = GraphBuilder()
    g.modules.append(residual_block(8))
    g.instance("a", "residual_block", channels=8)
    g.instance("b", "residual_block", channels=8)
    g.wire("a.y", "b.x")
    g.wire("b.y", "a.x")
    r = validate(g.build())
    assert "E_CYCLE" in codes(r) and not r.ok


def test_project_roundtrip_keeps_modules_and_hash(tmp_path):
    g = residual_net(share_second=True)
    save_project(Project(g), tmp_path / "p.project.json")
    g2 = load_project(tmp_path / "p.project.json").graph
    assert semantic_hash(g2) == semantic_hash(g) and g2.modules[0].id == "residual_block"
    # module definition order and node order inside a definition are not semantic
    g3 = Graph.model_validate(copy.deepcopy(g.to_json()))
    g3.modules[0].nodes.reverse()
    assert semantic_hash(g3) == semantic_hash(g)


def test_graphs_without_modules_keep_their_exact_json_and_hash(cnn_graph):
    assert "modules" not in cnn_graph.to_json() and all("sharedWith" not in n for n in cnn_graph.to_json()["nodes"])


# -------------------------------------------------------------------------------------------------------- A13 sharing
def test_a13_shared_instance_uses_identical_parameter_tensors():
    g = residual_net(share_second=True)
    r = validate(g)
    assert r.ok, r.diagnostics
    model = lower_graph(g)
    for n in ("conv_a", "conv_b"):
        a, b = model._modules[f"res1/{n}"], model._modules[f"res2/{n}"]
        assert b.inner is a  # the very same module object
        assert b.inner.weight is a.weight and b.inner.weight.data_ptr() == a.weight.data_ptr()
    assert len(list(model.parameters())) == 2 + 2 * 2  # stem + one set of block weights, not two
    assert r.total_params == (3 * 8 * 9 + 8) + 2 * (8 * 8 * 9 + 8)
    assert "shares parameters with 'res1'" in r.expansion.instances["res2"].note
    # state_dict aliases are the same storage
    sd = model.state_dict()
    assert sd["res1/conv_a.weight"].data_ptr() == sd["res2/conv_a.inner.weight"].data_ptr()


def test_a13_gradients_accumulate_from_both_call_sites_and_match_torch():
    enc = ModuleBuilder("enc")
    x = enc.in_("x", [None, 5])
    enc.node("fc", "tensor.dense", out_features=3)
    enc.node("act", "tensor.tanh")
    enc.wire(x, "fc.input")
    enc.chain("fc", "act")
    enc.out("z", "act.output")
    g = GraphBuilder()
    g.modules.append(enc.build())
    g.input("xa", ["N", 5]); g.input("xb", ["N", 5])
    g.instance("left", "enc"); g.instance("right", "enc", share="left")
    g.node("both", "tensor.add")
    g.wire("xa", "left.x"); g.wire("xb", "right.x")
    g.wire("left.z", "both.a"); g.wire("right.z", "both.b")
    graph = g.build()
    model = lower_graph(graph)
    xa, xb = torch.randn(4, 5), torch.randn(4, 5)
    out = model(xa, xb)
    out.sum().backward()
    W, b = model._modules["left/fc"].weight, model._modules["left/fc"].bias
    # hand-written reference with the same tensors, used at two call sites
    W2, b2 = W.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
    ref = (torch.tanh(xa @ W2.T + b2) + torch.tanh(xb @ W2.T + b2)).sum()
    ref.backward()
    assert torch.allclose(W.grad, W2.grad, atol=1e-6) and torch.allclose(b.grad, b2.grad, atol=1e-6)
    # and the gradient really is the SUM of the two sites' contributions
    only_a = torch.autograd.grad(torch.tanh(xa @ W2.T + b2).sum(), W2)[0]
    only_b = torch.autograd.grad(torch.tanh(xb @ W2.T + b2).sum(), W2)[0]
    assert torch.allclose(W.grad, only_a + only_b, atol=1e-6)
    assert not torch.allclose(W.grad, only_a, atol=1e-3)


def test_clone_default_gives_independent_gradients():
    g = residual_net()
    model = lower_graph(g)
    model(torch.randn(2, 3, 16, 16)).sum().backward()
    assert model._modules["res1/conv_a"].weight.grad is not model._modules["res2/conv_a"].weight.grad


def test_sharing_errors():
    g = residual_net(share_second=True)
    g.nodes[3].config["share"] = "ghost"
    assert "E_SHARE_TARGET" in codes(validate(g))
    # node-level sharing between different configs
    gb = GraphBuilder()
    gb.input("x", ["N", 4])
    gb.node("a", "tensor.dense", out_features=3)
    gb.node("b", "tensor.dense", shared_with="a", out_features=5)
    gb.wire("x", "a.input"); gb.wire("x", "b.input")
    assert "E_SHARE_MISMATCH" in codes(validate(gb.build()))


def test_node_level_shared_with_between_plain_nodes():
    gb = GraphBuilder()
    gb.input("x", ["N", 4])
    gb.node("a", "tensor.dense", out_features=4)
    gb.node("b", "tensor.dense", shared_with="a", out_features=4)
    gb.wire("x", "a.input"); gb.wire("a", "b.input")
    m = lower_graph(gb.build())
    assert m._modules["b"].inner is m._modules["a"]
    assert generate_pytorch(gb.build()).count("nn.Linear(") == 1


# -------------------------------------------------------------------------------------------------------- A37 repeat / select
def _step_module():
    m = ModuleBuilder("refine")
    h = m.in_("h", [None, 6])
    c = m.in_("context", [None, 6])
    m.node("mix", "tensor.add")
    m.node("fc", "tensor.dense", out_features=6)
    m.node("act", "tensor.tanh")
    m.wire(h, "mix.a"); m.wire(c, "mix.b")
    m.chain("mix", "fc", "act")
    m.out("h", "act.output", [None, 6])
    return m.build()


def test_a37_repeat_static_count_loop_carried_state_tied_parameters():
    g = GraphBuilder()
    g.modules.append(_step_module())
    g.input("h0", ["N", 6]); g.input("ctx", ["N", 6])
    g.node("loop", "core.repeat", module="refine", version="1.0.0", count=3, carry=[{"input": "h", "output": "h"}],
           termination={"kind": "fixed_count"})
    g.wire("h0", "loop.h"); g.wire("ctx", "loop.context")
    graph = g.build()
    r = validate(graph)
    assert r.ok, r.diagnostics
    assert [n for n in r.order if n.startswith("loop/it")].count("loop/it2/fc") == 1
    assert r.expansion.instances["loop"].iterations == 3 and r.expansion.instances["loop"].termination == "fixed_count"
    model = lower_graph(graph)
    assert len(list(model.parameters())) == 2  # tied: ONE dense layer for the three iterations
    h0, ctx = torch.randn(4, 6), torch.randn(4, 6)
    out = model(h0, ctx) if model.input_ids == ["h0", "ctx"] else model(ctx, h0)
    fc = model._modules["loop/it0/fc"]
    h = h0
    for _ in range(3):  # hand-written torch reference of the same loop
        h = torch.tanh(fc(h + ctx))
    assert torch.allclose(out, h, atol=1e-6)
    # loop-carried state: every iteration used the previous iteration's output (not h0)
    assert not torch.allclose(out, torch.tanh(fc(h0 + ctx)), atol=1e-3)
    out.sum().backward()  # gradients accumulate from the three uses of the tied layer
    assert fc.weight.grad is not None


def test_repeat_clone_has_separate_parameters_and_rejects_dynamic_termination():
    g = GraphBuilder()
    g.modules.append(_step_module())
    g.input("h0", ["N", 6]); g.input("ctx", ["N", 6])
    g.node("loop", "core.repeat", module="refine", count=2, carry=[{"input": "h", "output": "h"}], share="clone")
    g.wire("h0", "loop.h"); g.wire("ctx", "loop.context")
    assert len(list(lower_graph(g.build()).parameters())) == 4
    g2 = GraphBuilder()
    g2.modules.append(_step_module())
    g2.input("h0", ["N", 6]); g2.input("ctx", ["N", 6])
    g2.node("loop", "core.repeat", module="refine", count=2, carry=[{"input": "h", "output": "h"}], termination={"kind": "until_converged"})
    g2.wire("h0", "loop.h"); g2.wire("ctx", "loop.context")
    r = validate(g2.build())
    assert "E_UNSUPPORTED_TERMINATION" in codes(r) and not r.ok
    # count is bounded
    g3 = GraphBuilder()
    g3.modules.append(_step_module())
    g3.input("h0", ["N", 6]); g3.input("ctx", ["N", 6])
    g3.node("loop", "core.repeat", module="refine", count=1000, carry=[{"input": "h", "output": "h"}])
    g3.wire("h0", "loop.h"); g3.wire("ctx", "loop.context")
    assert "E_CONFIG" in codes(validate(g3.build()))


def test_repeat_carried_state_must_keep_its_type():
    m = ModuleBuilder("grow")
    h = m.in_("h", [None, 4])
    m.node("fc", "tensor.dense", out_features=7)
    m.wire(h, "fc.input")
    m.out("h", "fc.output")
    g = GraphBuilder()
    g.modules.append(m.build())
    g.input("h0", ["N", 4])
    g.node("loop", "core.repeat", module="grow", count=2, carry=[{"input": "h", "output": "h"}])
    g.wire("h0", "loop.h")
    r = validate(g.build())
    assert not r.ok and ({"E_CARRY_TYPE", "E_PORT_TYPE", "E_CHANNEL_MISMATCH"} & codes(r))


def _branch(name, op, **cfg):
    m = ModuleBuilder(name)
    x = m.in_("x", [None, 4])
    m.node("f", op, **cfg)
    m.wire(x, "f.input")
    m.out("y", "f.output", [None, 4])
    return m.build()


def _select_graph(otherwise_module=None):
    g = GraphBuilder()
    g.modules += [_branch("pos", "tensor.exp"), otherwise_module or _branch("neg", "tensor.neg")]
    g.input("x", ["N", 4])
    g.node("flag", "tensor.constant", value=True, dtype="bool")
    g.node("pick", "core.select", then={"module": "pos", "version": "1.0.0"}, otherwise={"module": (otherwise_module.id if otherwise_module else "neg"), "version": "1.0.0"})
    g.wire("flag", "pick.pred"); g.wire("x", "pick.x")
    return g


def test_a37_select_typed_branches_scalar_predicate_and_gradient_only_through_chosen_branch():
    graph = _select_graph().build()
    r = validate(graph)
    assert r.ok, r.diagnostics
    assert r.expansion.instances["pick"].kind == "core.select" and "both branches are evaluated" in r.expansion.instances["pick"].note
    model = lower_graph(graph)
    x = torch.randn(3, 4, requires_grad=True)
    out = model(x)
    assert torch.allclose(out, torch.exp(x))  # predicate true => then-branch
    out.sum().backward()
    assert torch.allclose(x.grad, torch.exp(x))  # not exp(x) + (-1)
    # flip the predicate constant: the otherwise branch is chosen
    g2 = _select_graph().build()
    for n in g2.nodes:
        if n.id == "flag":
            n.config["value"] = False
    m2 = lower_graph(g2)
    x2 = torch.randn(3, 4, requires_grad=True)
    out2 = m2(x2)
    assert torch.allclose(out2, -x2)
    out2.sum().backward()
    assert torch.allclose(x2.grad, -torch.ones_like(x2))


def test_select_rejects_mismatched_branch_types_and_non_scalar_predicate():
    # the otherwise branch changes the width => branch types differ
    m = ModuleBuilder("wide")
    x = m.in_("x", [None, 4])
    m.node("f", "tensor.dense", out_features=9)
    m.wire(x, "f.input")
    m.out("y", "f.output")
    r = validate(_select_graph(m.build()).build())
    assert "E_BRANCH_TYPE" in codes(r)
    # non-scalar predicate
    g = _select_graph()
    for n in g.nodes:
        if n.id == "flag":
            n.config["value"] = [True, False]
    assert "E_BRANCH_PRED" in codes(validate(g.build()))
    # branches with different port names
    m2 = ModuleBuilder("odd")
    x2 = m2.in_("z", [None, 4])
    m2.node("f", "tensor.neg")
    m2.wire(x2, "f.input")
    m2.out("y", "f.output")
    assert "E_BRANCH_MISMATCH" in codes(validate(_select_graph(m2.build()).build()))


def test_predicate_from_data_with_any_all():
    g = GraphBuilder()
    g.modules += [_branch("pos", "tensor.exp"), _branch("neg", "tensor.neg")]
    g.input("x", ["N", 4])
    g.node("zero", "tensor.constant", value=0.0)
    g.node("gt", "tensor.compare", op="gt")
    g.node("anyp", "tensor.any_all", kind="all")
    g.node("pick", "core.select", then={"module": "pos"}, otherwise={"module": "neg"})
    g.wire("x", "gt.a"); g.wire("zero", "gt.b"); g.wire("gt", "anyp.input")
    g.wire("anyp", "pick.pred"); g.wire("x", "pick.x")
    model = lower_graph(g.build())
    assert torch.allclose(model(torch.ones(2, 4)), torch.exp(torch.ones(2, 4)))   # all positive => exp
    assert torch.allclose(model(-torch.ones(2, 4)), torch.ones(2, 4))              # not all positive => neg
