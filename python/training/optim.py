"""Optimizer construction with real torch semantics, plus a hand-written mirror of the update rule used to explain one element's update.

The mirror follows the order and conventions of torch's single-tensor implementations (torch.optim.SGD / Adam / AdamW) and is checked
against them in tests; the inspector shows both the actual new value and the value the formula predicts, so any difference is visible."""
from __future__ import annotations

import math
from typing import Any

import torch

from .spec import OptimizerSpec, SchedulerSpec


def build_optimizer(params, spec: OptimizerSpec) -> torch.optim.Optimizer:
    if spec.kind == "sgd":
        return torch.optim.SGD(params, lr=spec.lr, momentum=spec.momentum, dampening=spec.dampening, weight_decay=spec.weight_decay, nesterov=spec.nesterov)
    cls = torch.optim.Adam if spec.kind == "adam" else torch.optim.AdamW
    return cls(params, lr=spec.lr, betas=spec.betas, eps=spec.eps, weight_decay=spec.weight_decay, amsgrad=spec.amsgrad)


def build_scheduler(opt: torch.optim.Optimizer, spec: SchedulerSpec):
    if spec.kind == "none":
        return None
    sch = torch.optim.lr_scheduler
    if spec.kind == "step":
        return sch.StepLR(opt, spec.step_size, spec.gamma)
    if spec.kind == "exponential":
        return sch.ExponentialLR(opt, spec.gamma)
    if spec.kind == "cosine":
        return sch.CosineAnnealingLR(opt, spec.t_max)
    if spec.kind == "linear_warmup":
        w = max(1, spec.warmup_steps)
        return sch.LambdaLR(opt, lambda s: min(1.0, (s + 1) / w))
    return sch.ReduceLROnPlateau(opt, mode=spec.mode, factor=spec.factor, patience=spec.patience)


def manual_update(spec: OptimizerSpec, lr: float, w: torch.Tensor, g: torch.Tensor, state: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any], list[str]]:
    """One update of a tensor element (or tensor) from its value, gradient and optimizer state BEFORE the step.
    Returns (new value, new state, the formula steps with numbers as text)."""
    steps: list[str] = []
    w, g = w.clone().double(), g.clone().double()
    st = {k: (v.clone().double() if torch.is_tensor(v) else v) for k, v in state.items()}
    wd = spec.weight_decay
    if spec.kind == "sgd":
        d = g
        if wd:
            d = d + wd * w
            steps.append(f"g' = g + weight_decay*w = {float(d.flatten()[0]):.6g}")
        if spec.momentum:
            buf = st.get("momentum_buffer")
            buf = d.clone() if buf is None else spec.momentum * buf + (1 - spec.dampening) * d
            st["momentum_buffer"] = buf
            steps.append(f"buf = {'g' if state.get('momentum_buffer') is None else 'momentum*buf + (1-dampening)*g'} = {float(buf.flatten()[0]):.6g}")
            d = d + spec.momentum * buf if spec.nesterov else buf
        wn = w - lr * d
        steps.append(f"w = w - lr*d = {float(w.flatten()[0]):.6g} - {lr:.6g}*{float(d.flatten()[0]):.6g} = {float(wn.flatten()[0]):.6g}")
        return wn, st, steps
    b1, b2 = spec.betas
    t = float(st["step"]) + 1.0 if "step" in st else 1.0
    st["step"] = torch.tensor(t)
    m = st.get("exp_avg", torch.zeros_like(w))
    v = st.get("exp_avg_sq", torch.zeros_like(w))
    if spec.kind == "adamw":
        w = w * (1 - lr * wd)
        steps.append(f"decoupled decay: w = w*(1 - lr*weight_decay) = {float(w.flatten()[0]):.6g}")
    elif wd:
        g = g + wd * w
        steps.append(f"g' = g + weight_decay*w = {float(g.flatten()[0]):.6g}")
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    st["exp_avg"], st["exp_avg_sq"] = m, v
    bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
    if spec.amsgrad:
        mx = torch.maximum(st.get("max_exp_avg_sq", torch.zeros_like(w)), v)
        st["max_exp_avg_sq"] = mx
        denom = mx.sqrt() / math.sqrt(bc2) + spec.eps
    else:
        denom = v.sqrt() / math.sqrt(bc2) + spec.eps
    wn = w - (lr / bc1) * m / denom
    steps += [f"step t = {t:g}; m = beta1*m + (1-beta1)*g = {float(m.flatten()[0]):.6g}; v = beta2*v + (1-beta2)*g^2 = {float(v.flatten()[0]):.6g}",
              f"bias corrections: 1-beta1^t = {bc1:.6g}, 1-beta2^t = {bc2:.6g}",
              f"w = w - lr/(1-beta1^t) * m / (sqrt(v)/sqrt(1-beta2^t) + eps) = {float(wn.flatten()[0]):.6g}"]
    return wn, st, steps
