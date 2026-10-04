"""Milestone 6a: every portable operation, on every backend, against a handwritten native PyTorch reference (outputs, input and parameter
gradients, one SGD update) within the declared tolerances; unsupported combinations are refused BEFORE execution with a stable code."""
import numpy as np
import pytest
import torch

import backends
from backend_helpers import cmp, skip_unless_available
from backends import BackendError
from backends.conformance import CASES, case_dtype, case_graph, case_inputs
from backends.registry import BACKEND_IDS
from backends.tolerances import tol

LR = 0.05
PARAMS = [pytest.param(c, b, id=f"{b}-{c.id}") for c in CASES for b in BACKEND_IDS]


def _reference(c, xs):
    """Handwritten native PyTorch: (inputs requiring grad, its own parameters, output)."""
    fn, tp = c.native()
    tin = {k: torch.as_tensor(v) for k, v in xs.items()}
    for v in tin.values():
        if v.is_floating_point():
            v.requires_grad_(True)
    return tin, tp, fn(*tin.values())


@pytest.mark.parametrize("case,backend", PARAMS)
def test_op_matches_native_pytorch(case, backend):
    g, loss_node = case_graph(case)
    rep = backends.compat_report(g, backend)
    verdict = rep.nodes["op"]
    expected = case.expect.get(backend)
    if expected:
        assert verdict["status"] == "unsupported" and verdict["code"] == expected, verdict
        assert verdict["reason"]
        assert not rep.ok
        with pytest.raises(BackendError) as ei:  # refused before any execution, with the report attached
            backends.compile_graph(g, backend)
        assert ei.value.code == "E_BACKEND_INCOMPATIBLE" or ei.value.code == "E_BACKEND_UNAVAILABLE"
        return
    assert verdict["status"] in ("supported", "converted"), verdict
    skip_unless_available(backend)
    rng = np.random.default_rng(abs(hash(case.id)) % 1000)
    xs = case_inputs(case, rng)
    dtype = case_dtype(case)
    t = {q: tol(backend, dtype, q) for q in ("forward", "loss", "grad", "update")}

    tin, tp, out = _reference(case, xs)
    ref_loss = out if loss_node == "op" else (out ** 2).sum()
    ref_loss.backward()

    ex = backends.compile_graph(g, backend, compiled=False)
    ex.set_params({"op": {k: v.detach().numpy() for k, v in tp.items()}} if tp else {})
    res = ex.forward(xs, capture=True)
    cmp(res.activations["op"], out.detach().numpy(), t["forward"], "op output")
    cmp(res.outputs[loss_node], ref_loss.detach().numpy(), t["loss"], "loss")

    wrt = tuple(k for k, v in xs.items() if v.dtype.kind == "f")
    loss, gp, gi = ex.loss_and_grads(xs, loss_node, wrt_inputs=wrt)
    cmp(loss, ref_loss.detach().numpy(), t["loss"], "loss (grad pass)")
    for k in wrt:
        cmp(gi[k], tin[k].grad.numpy(), t["grad"], f"d loss / d {k}")
    for k, p in tp.items():
        cmp(gp["op"][k], p.grad.numpy(), t["grad"], f"d loss / d {k}")
        assert np.abs(gp["op"][k]).sum() > 0, f"vacuous gradient for {k}"
    if not tp:
        return
    before = ex.get_params()
    ex.sgd_step(xs, LR, loss_node)
    after = ex.get_params()
    for k, p in tp.items():
        cmp(after["op"][k], p.detach().numpy() - LR * p.grad.numpy(), t["update"], f"SGD update of {k}")
        assert not np.allclose(after["op"][k], before["op"][k]), "the update must change the parameter"
