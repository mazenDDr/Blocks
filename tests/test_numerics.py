"""A06, A22, A34: the graph is compared with handwritten native PyTorch."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import EXAMPLES
from graph_core.lower import capture_activations, lower_graph
from graph_core.schema import Node
from graph_core.validate import validate

ATOL = 1e-6


def native_reference():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))


def copy_weights(graph_model, ref):
    pairs = {"conv_1": ref[0], "conv_2": ref[3], "fc": ref[8]}
    for nid, mod in pairs.items():
        mod.load_state_dict(getattr(graph_model, nid).state_dict())


def test_a06_graph_matches_native_sequential(cnn_graph):
    torch.manual_seed(0)
    gm = lower_graph(cnn_graph)
    ref = native_reference()
    copy_weights(gm, ref)
    x = torch.randn(5, 3, 64, 64)
    y = torch.randint(0, 10, (5,))

    out_g, out_r = gm(x), ref(x)
    assert out_g.shape == (5, 10)
    assert torch.allclose(out_g, out_r, atol=ATOL)

    loss_g, loss_r = F.cross_entropy(out_g, y), F.cross_entropy(out_r, y)
    assert abs(loss_g.item() - loss_r.item()) < ATOL
    loss_g.backward()
    loss_r.backward()
    for nid, rmod in {"conv_1": ref[0], "conv_2": ref[3], "fc": ref[8]}.items():
        gmod = getattr(gm, nid)
        assert torch.allclose(gmod.weight.grad, rmod.weight.grad, atol=ATOL), nid
        assert torch.allclose(gmod.bias.grad, rmod.bias.grad, atol=ATOL), nid
        assert gmod.weight.grad.abs().sum() > 0

    og, orr = torch.optim.SGD(gm.parameters(), lr=0.1), torch.optim.SGD(ref.parameters(), lr=0.1)
    og.step()
    orr.step()
    for nid, rmod in {"conv_1": ref[0], "conv_2": ref[3], "fc": ref[8]}.items():
        assert torch.allclose(getattr(gm, nid).weight, rmod.weight, atol=ATOL)
        assert torch.allclose(getattr(gm, nid).bias, rmod.bias, atol=ATOL)


def test_a06_cross_entropy_node_in_graph_matches_native(cnn_graph):
    g = copy.deepcopy(cnn_graph)
    g.nodes.append(Node.model_validate({"id": "labels", "type": "core.tensor_input", "config": {"shape": ["N"], "dtype": "int64", "layout": "N"}}))
    g.nodes.append(Node.model_validate({"id": "ce", "type": "pytorch.loss.cross_entropy", "config": {"reduction": "mean"}}))
    g.edges += [type(g.edges[0]).model_validate(d) for d in (
        {"id": "fc_ce", "from": {"node": "fc", "port": "output"}, "to": {"node": "ce", "port": "logits"}},
        {"id": "labels_ce", "from": {"node": "labels", "port": "value"}, "to": {"node": "ce", "port": "target"}})]
    r = validate(g)
    assert r.ok, r.diagnostics
    assert r.output_types["ce"]["output"].shape == ()
    torch.manual_seed(1)
    gm = lower_graph(g, r)
    assert gm.input_ids == ["images", "labels"] and gm.output_ids == ["ce"]
    ref = native_reference()
    copy_weights(gm, ref)
    x, y = torch.randn(4, 3, 64, 64), torch.randint(0, 10, (4,))
    loss_g = gm(x, y)
    loss_r = F.cross_entropy(ref(x), y)
    assert abs(loss_g.item() - loss_r.item()) < ATOL
    loss_g.backward()
    loss_r.backward()
    assert torch.allclose(gm.conv_1.weight.grad, ref[0].weight.grad, atol=ATOL)


def test_activation_capture_is_opt_in_and_side_effect_free(cnn_graph):
    torch.manual_seed(0)
    gm = lower_graph(cnn_graph)
    x = torch.randn(2, 3, 64, 64)
    plain = gm(x)
    with capture_activations(gm, ["conv_1", "pool_2"]) as acts:
        probed = gm(x)
    assert set(acts) == {"conv_1", "pool_2"}
    assert acts["conv_1"].shape == (2, 32, 64, 64) and acts["pool_2"].shape == (2, 64, 16, 16)
    assert torch.equal(plain, probed)
    assert not gm.conv_1._forward_hooks  # hooks removed
    assert torch.equal(acts["conv_1"], gm.conv_1(x))
    with capture_activations(gm) as all_acts:
        gm(x)
    assert len(all_acts) == 10


def test_a22_mse_teaching_fixture(mse_graph):
    r = validate(mse_graph)
    assert r.ok, r.diagnostics
    gm = lower_graph(mse_graph, r)
    assert gm.input_ids == ["pred", "target"] and gm.output_ids == ["loss"]
    pred = torch.tensor([1.0, 2.5, 2.0], requires_grad=True)
    target = torch.tensor([1.0, 2.0, 3.0])
    with capture_activations(gm) as acts:
        loss = gm(pred, target)
    assert abs(loss.item() - 5 / 12) < 1e-6
    assert torch.allclose(acts["diff"], torch.tensor([0.0, 0.5, -1.0]))
    assert torch.allclose(acts["squared"], torch.tensor([0.0, 0.25, 1.0]))
    assert abs(acts["total"].item() - 1.25) < 1e-6
    assert abs(acts["loss"].item() - 5 / 12) < 1e-6
    loss.backward()
    assert torch.allclose(pred.grad, torch.tensor([0.0, 1 / 3, -2 / 3]), atol=1e-6)
    # native autograd reference
    p2 = pred.detach().clone().requires_grad_(True)
    ((p2 - target) ** 2).sum().div(3).backward()
    assert torch.allclose(pred.grad, p2.grad, atol=1e-6)


def test_mean_divisor_semantics_equal_sum_over_count(mse_graph):
    g = copy.deepcopy(mse_graph)
    g.node("total").type = "core.mean"
    g.node("total").config = {"dims": None, "keepdim": False, "divisor": "element_count"}
    g.node("loss").config = {"factor": 1.0}
    gm = lower_graph(g)
    assert abs(gm(torch.tensor([1.0, 2.5, 2.0]), torch.tensor([1.0, 2.0, 3.0])).item() - 5 / 12) < 1e-6


def test_a34_sgd_teaching_step():
    """w = 2, gradient 0.6 (from a real graph), lr = 0.1 -> w_next = 1.94."""
    from graph_core.schema import Graph
    g = Graph.model_validate({
        "nodes": [{"id": "w", "type": "core.tensor_input", "config": {"shape": [1], "layout": "scalar"}},
                  {"id": "scaled", "type": "core.scalar_mul", "config": {"factor": 0.6}},
                  {"id": "loss", "type": "core.sum", "config": {}}],
        "edges": [{"id": "e1", "from": {"node": "w", "port": "value"}, "to": {"node": "scaled", "port": "input"}},
                  {"id": "e2", "from": {"node": "scaled", "port": "output"}, "to": {"node": "loss", "port": "input"}}]})
    gm = lower_graph(g)
    w = torch.nn.Parameter(torch.tensor([2.0]))
    opt = torch.optim.SGD([w], lr=0.1)
    gm(w).backward()
    assert abs(w.grad.item() - 0.6) < 1e-7
    opt.step()
    assert abs(w.item() - 1.94) < 1e-6
