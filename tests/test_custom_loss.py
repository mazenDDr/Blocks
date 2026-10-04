"""Milestone 3 item 4 / A23: a masked, weighted loss composed from primitives, packaged as a module with declared reduction semantics."""
import pytest
import torch

from graph_core.build import DIVISORS, masked_weighted_loss
from graph_core.modtest import run_module
from graph_core.schema import Graph
from graph_core.validate import validate


def project(divisor):
    return Graph(modules=[masked_weighted_loss(divisor)])


def ref_loss(p, t, m, w, divisor, eps=1e-8):
    terms = w * m * (p - t) ** 2
    num = terms.sum()
    den = {"element_count": None, "valid_count": m.sum().clamp(min=eps), "weight_sum": (w * m).sum().clamp(min=eps)}[divisor]
    return terms.mean() if den is None else num / den


def data(seed=0, n=6, k=3):
    g = torch.Generator().manual_seed(seed)
    p, t = torch.randn(n, k, generator=g), torch.randn(n, k, generator=g)
    m = (torch.rand(n, k, generator=g) > 0.3).float()
    w = torch.rand(n, k, generator=g) * 2 + 0.1
    return p, t, m, w


@pytest.mark.parametrize("divisor", list(DIVISORS))
def test_a23_values_and_gradients_match_hand_written_torch(divisor):
    p, t, m, w = data()
    res = run_module(project(divisor), f"masked_weighted_mse_{divisor}", {"pred": p, "target": t, "mask": m, "weights": w}, grads=True)
    loss = res["outputs"]["loss"]
    pr = p.clone().requires_grad_()
    ref = ref_loss(pr, t, m, w, divisor)
    ref.backward()
    assert loss.shape == () and torch.allclose(loss, ref, atol=1e-7)
    assert torch.allclose(res["grads"]["pred"], pr.grad, atol=1e-7)
    # masked-out elements get exactly zero gradient
    assert (res["grads"]["pred"][m == 0] == 0).all()


def test_the_three_reductions_are_really_different_computations():
    p, t, m, w = data(1)
    vals = {d: float(run_module(project(d), f"masked_weighted_mse_{d}", {"pred": p, "target": t, "mask": m, "weights": w})["outputs"]["loss"]) for d in DIVISORS}
    assert len({round(v, 6) for v in vals.values()}) == 3
    terms = (w * m * (p - t) ** 2).sum()
    assert vals["element_count"] == pytest.approx(float(terms) / p.numel(), rel=1e-5)
    assert vals["valid_count"] == pytest.approx(float(terms) / float(m.sum()), rel=1e-5)
    assert vals["weight_sum"] == pytest.approx(float(terms) / float((w * m).sum()), rel=1e-5)


def test_teaching_fixture_from_the_vision_mse_example():
    # targets [1,2,3], predictions [1,2.5,2] => sum of squared errors 1.25, mean over 3 elements 0.4166666667, gradient [0, 1/3, -2/3]
    p, t = torch.tensor([[1.0, 2.5, 2.0]]), torch.tensor([[1.0, 2.0, 3.0]])
    ones = torch.ones(1, 3)
    res = run_module(project("element_count"), "masked_weighted_mse_element_count", {"pred": p, "target": t, "mask": ones, "weights": ones}, grads=True)
    assert float(res["outputs"]["loss"].detach()) == pytest.approx(1.25 / 3, rel=1e-6)
    assert res["grads"]["pred"].tolist()[0] == pytest.approx([0.0, 1 / 3, -2 / 3], abs=1e-6)


def test_empty_mask_policy_is_declared_and_holds():
    p, t = torch.randn(4, 3), torch.randn(4, 3)
    zeros, w = torch.zeros(4, 3), torch.ones(4, 3)
    for d in ("valid_count", "weight_sum"):
        mod = masked_weighted_loss(d)
        assert "clamped" in mod.reduction["empty_policy"] and mod.reduction["divisor"] == d
        res = run_module(project(d), mod.id, {"pred": p, "target": t, "mask": zeros, "weights": w}, grads=True)
        assert float(res["outputs"]["loss"].detach()) == 0.0 and torch.isfinite(res["grads"]["pred"]).all() and (res["grads"]["pred"] == 0).all()


def test_packaged_module_has_signature_description_and_primitives_only():
    mod = masked_weighted_loss("valid_count")
    assert [p.name for p in mod.inputs] == ["pred", "target", "mask", "weights"] and [o.name for o in mod.outputs] == ["loss"]
    assert "valid" in mod.description.lower() and mod.reduction["formula"].startswith("sum(w*m*(p-t)^2)")
    assert all(n.type.startswith(("tensor.", "core.")) for n in mod.nodes)  # no code anywhere
    # editing an internal op changes the computation (here: absolute error instead of squared error)
    p, t, m, w = data(2)
    g = project("valid_count")
    for n in g.modules[0].nodes:
        if n.id == "sq":
            n.type = "tensor.abs"
    res = run_module(g, mod.id, {"pred": p, "target": t, "mask": m, "weights": w})
    ref = (w * m * (p - t).abs()).sum() / m.sum()
    assert torch.allclose(res["outputs"]["loss"], ref, atol=1e-6)
