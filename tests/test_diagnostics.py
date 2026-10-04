"""Milestone 3 item 8 / A46: probe, histogram, timer and assert blocks inside composite modules keep their nested identity through lowering, respect the
capture scope, and (except assert) leave the scientific computation untouched."""
import pytest
import torch

from graph_core import diagnostics as D
from graph_core import registry
from graph_core.build import GraphBuilder, ModuleBuilder
from graph_core.lower import lower_graph
from graph_core.validate import validate


def module_with_diagnostics():
    m = ModuleBuilder("encoder_stage")
    x = m.in_("x", [None, 6])
    m.node("fc", "tensor.dense", out_features=6)
    m.node("probe_pre", "diag.probe", label="pre-activation", values=3)
    m.node("act", "tensor.tanh")
    m.node("hist_post", "diag.histogram", bins=4, range=[-1.0, 1.0])
    m.node("clock", "diag.timer")
    m.node("check", "diag.assert", check="finite")
    m.wire(x, "fc.input")
    m.chain("fc", "probe_pre", "act", "hist_post", "clock", "check")
    m.out("y", "check.output", [None, 6])
    return m.build()


def graph(with_diag=True):
    g = GraphBuilder()
    g.modules.append(module_with_diagnostics())
    g.input("i", ["N", 6])
    g.instance("stage1", "encoder_stage")
    g.instance("stage2", "encoder_stage")
    g.node("head", "tensor.dense", out_features=2)
    g.wire("i", "stage1.x"); g.wire("stage1.y", "stage2.x"); g.wire("stage2.y", "head.input")
    return g.build()


def plain_graph():
    """The same network with the diagnostic blocks removed (wired straight through)."""
    g = graph()
    mod = g.modules[0]
    mod.nodes = [n for n in mod.nodes if not n.type.startswith("diag.")]
    mod.edges = []
    from graph_core.build import Body
    b = Body()
    b.nodes = mod.nodes
    b.wire("$in.x", "fc.input"); b.chain("fc", "act")
    mod.edges = b.edges
    mod.outputs[0].from_.node, mod.outputs[0].from_.port = "act", "output"
    return g


def test_nested_identity_survives_lowering():
    r = validate(graph())
    assert r.ok, r.diagnostics
    ids = {"stage1/probe_pre", "stage1/hist_post", "stage1/clock", "stage1/check", "stage2/probe_pre"}
    assert ids <= set(r.order)
    m = lower_graph(graph())
    assert m._modules["stage2/hist_post"].node_id == "stage2/hist_post" and m._modules["stage1/check"].node_id == "stage1/check"


def test_identity_without_a_capture_session_and_same_values_as_the_graph_without_diagnostics():
    torch.manual_seed(0)
    a = lower_graph(graph())
    torch.manual_seed(0)
    b = lower_graph(plain_graph())
    b.load_state_dict(a.state_dict())
    x = torch.randn(5, 6)
    assert torch.equal(a(x), b(x))
    # with a full capture session (observation only) the values are still bit-identical, and so are the gradients
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    with D.CaptureSession() as s:
        oa = a(xa)
    ob = b(xb)
    oa.sum().backward(); ob.sum().backward()
    assert torch.equal(oa, ob) and torch.equal(xa.grad, xb.grad)
    assert {k: v.grad for k, v in a.named_parameters()}.keys() == {k: v.grad for k, v in b.named_parameters()}.keys()
    assert all(torch.equal(pa.grad, pb.grad) for pa, pb in zip(a.parameters(), b.parameters()))
    assert s.records, "the session captured nothing"


def test_capture_records_nested_paths_stats_values_histograms_and_timers():
    m = lower_graph(graph())
    x = torch.randn(4, 6)
    with D.CaptureSession() as s:
        m(x)
    by_node = {}
    for r in s.records:
        by_node.setdefault(r["node"], []).append(r)
    assert set(by_node) == {"stage1/probe_pre", "stage1/hist_post", "stage1/clock", "stage2/probe_pre", "stage2/hist_post", "stage2/clock"}
    p = by_node["stage1/probe_pre"][0]
    assert p["kind"] == "probe" and p["shape"] == [4, 6] and p["label"] == "pre-activation" and len(p["values"]) == 3 and p["stats"]["nonFinite"] == 0
    h = by_node["stage2/hist_post"][0]["hist"]
    assert len(h["counts"]) == 4 and sum(h["counts"]) == 24 and (h["lo"], h["hi"]) == (-1.0, 1.0)
    t1, t2 = by_node["stage1/clock"][0]["elapsedMs"], by_node["stage2/clock"][0]["elapsedMs"]
    assert 0 <= t1 <= t2                                                 # execution order is visible in the timers
    assert [r["node"] for r in s.records if r["kind"] == "probe"] == ["stage1/probe_pre", "stage2/probe_pre"]


def test_capture_scope_nodes_steps_samples_budget():
    m = lower_graph(graph())
    x = torch.randn(4, 6)
    with D.CaptureSession(nodes=["stage2/*"]) as s:
        m(x)
    assert {r["node"] for r in s.records} == {"stage2/probe_pre", "stage2/hist_post", "stage2/clock"}
    with D.CaptureSession(nodes=["stage1/probe_pre"], samples=[1, 3]) as s:
        m(x)
    assert s.records[0]["sampled"] is True and s.records[0]["stats"]["count"] == 12        # 2 of 4 samples x 6 values
    sess = D.CaptureSession(nodes=["stage1/probe_pre"], steps={2})
    with sess:
        sess.step = 1
        m(x)
        sess.step = 2
        m(x)
        sess.step = 3
        m(x)
    assert [r["step"] for r in sess.records] == [2]
    with D.CaptureSession(nodes=["*/probe_pre"], max_elements=3) as s:
        m(x)
    assert s.dropped == 2 and not s.records                                  # budget too small for even one record (1 + 3 values): dropped, and counted
    with D.CaptureSession(nodes=["*/probe_pre"], max_elements=10) as s:
        m(x)
    assert len(s.records) == 2 and s.dropped == 0


def test_probe_does_not_consume_random_numbers_or_change_mode_or_order():
    g = GraphBuilder()
    g.input("i", ["N", 6])
    g.node("fc", "tensor.dense", out_features=6)
    g.node("pr", "diag.probe")
    g.node("dr", "tensor.dropout", p=0.5)
    g.wire("i", "fc.input"); g.chain("fc", "pr", "dr")
    m = lower_graph(g.build())
    m.train()
    x = torch.randn(8, 6)
    torch.manual_seed(5)
    a = m(x)
    ra = torch.get_rng_state()
    torch.manual_seed(5)
    with D.CaptureSession():
        b = m(x)
    assert torch.equal(a, b) and torch.equal(ra, torch.get_rng_state()) and m.training


def test_assert_is_execution_changing_labelled_and_reports_the_nested_block():
    g = graph()
    m = lower_graph(g)
    bad = torch.full((2, 6), float("nan"))
    assert torch.isnan(m(bad)).all()                                         # assertions are off: the computation runs exactly as without the block
    with D.CaptureSession(assertions=True, observe=False):
        with pytest.raises(D.AssertionFailed) as e:
            m(bad)
    assert e.value.node == "stage1/check" and "NaN or infinite" in e.value.detail
    # scoped assertion: only stage2 is checked, but stage1's check is out of scope so stage2 raises
    with D.CaptureSession(assertions=True, observe=False, nodes=["stage2/*"]):
        with pytest.raises(D.AssertionFailed) as e:
            m(bad)
    assert e.value.node == "stage2/check"
    # explain labels the two classes of blocks differently
    r = validate(g)
    ex_a = registry.get_op("diag.assert").explain(r.resolved["stage1/check"], {}, {})
    ex_p = registry.get_op("diag.probe").explain(r.resolved["stage1/probe_pre"], {}, {})
    assert "EXECUTION-CHANGING" in ex_a["executionEffect"] and "observation-only" in ex_p["executionEffect"]
    assert registry.get_op("diag.assert").observation_only is False and registry.get_op("diag.probe").observation_only is True


def test_bounded_and_nonnegative_assertions():
    for cfg, x, ok in (({"check": "nonnegative"}, torch.tensor([[1.0, 2.0]]), True), ({"check": "nonnegative"}, torch.tensor([[1.0, -2.0]]), False),
                       ({"check": "bounded", "lo": 0.0, "hi": 1.0}, torch.tensor([[0.5]]), True), ({"check": "bounded", "lo": 0.0, "hi": 1.0}, torch.tensor([[1.5]]), False)):
        g = GraphBuilder()
        g.input("i", ["N", x.shape[1]])
        g.node("a", "diag.assert", **cfg)
        g.wire("i", "a.input")
        m = lower_graph(g.build())
        with D.CaptureSession(assertions=True, observe=False):
            if ok:
                m(x)
            else:
                with pytest.raises(D.AssertionFailed):
                    m(x)


def test_a_tap_on_a_wire_is_not_a_model_output():
    g = GraphBuilder()
    g.input("i", ["N", 4])
    g.node("fc", "tensor.dense", out_features=2)
    g.node("tap", "diag.probe")
    g.wire("i", "fc.input"); g.wire("fc", "tap.input")
    m = lower_graph(g.build())
    assert m.output_ids == ["fc"] and isinstance(m(torch.randn(3, 4)), torch.Tensor)
