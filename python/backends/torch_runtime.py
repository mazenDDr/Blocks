"""PyTorch executable behind the common interface: the graph-lowered GraphModule (the reference implementation of the graph)."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from graph_core.lower import capture_activations, lower_graph

from .base import Executable, ForwardResult, Params
from .params import param_spec
from .plan import make_plan


class TorchExecutable(Executable):
    backend = "pytorch"

    def __init__(self, graph, report, seed: int = 0):
        torch.manual_seed(seed)
        self.model = lower_graph(graph, report)
        self.plan = make_plan(report)
        self.input_ids, self.output_ids = list(self.model.input_ids), list(self.model.output_ids)
        self.spec = param_spec(self.plan)

    def init_info(self) -> str:
        from .registry import INFO
        return INFO["pytorch"].init

    def _mod(self, nid):
        return self.model._modules[nid]

    def get_params(self) -> Params:
        return {nid: {name: getattr(self._mod(nid), name).detach().numpy().copy() for name in ps} for nid, ps in self.spec.items()}

    def set_params(self, params: Params) -> None:
        with torch.no_grad():
            for nid, ps in self.spec.items():
                for name in ps:
                    getattr(self._mod(nid), name).copy_(torch.as_tensor(np.asarray(params[nid][name])))

    def _tensors(self, inputs, wrt=()):
        out = []
        for i in self.input_ids:
            t = torch.as_tensor(np.asarray(inputs[i]))
            if i in wrt:
                t = t.clone().requires_grad_(True)
            out.append(t)
        return out

    def forward(self, inputs, capture=False) -> ForwardResult:
        with torch.no_grad():
            args = self._tensors(inputs)
            if capture:
                with capture_activations(self.model) as acts:
                    y = self.model(*args)
                act = {k: v.numpy().copy() for k, v in acts.items()}
            else:
                y, act = self.model(*args), {}
        ys = (y,) if torch.is_tensor(y) else y
        return ForwardResult({o: t.numpy().copy() for o, t in zip(self.output_ids, ys)}, act)

    def loss_and_grads(self, inputs, loss_node=None, wrt_inputs=()):
        ln = self._loss_node(loss_node)
        args = self._tensors(inputs, wrt_inputs)
        for p in self.model.parameters():
            p.grad = None
        y = self.model(*args)
        ys = (y,) if torch.is_tensor(y) else y
        loss = dict(zip(self.output_ids, ys))[ln]
        if loss.requires_grad:  # a graph without parameters or differentiated inputs has nothing to differentiate
            loss.backward()
        grads = {nid: {name: getattr(self._mod(nid), name).grad.numpy().copy() for name in ps} for nid, ps in self.spec.items() if loss.requires_grad}
        ig = {i: a.grad.numpy().copy() for i, a in zip(self.input_ids, args) if i in wrt_inputs and a.grad is not None}
        return loss.detach().numpy().copy(), grads, ig

    def sgd_step(self, inputs, lr, loss_node=None):
        loss, _, _ = self.loss_and_grads(inputs, loss_node)
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    p -= lr * p.grad
        return loss

    def bench_closures(self, inputs, lr, loss_node=None, grad_inputs=()):
        """(forward, train step) closures over inputs converted once (see KerasExecutable.bench_closures)."""
        ln = self._loss_node(loss_node)
        args = self._tensors(inputs, tuple(grad_inputs))
        params = list(self.model.parameters())
        opt = torch.optim.SGD(params, lr=lr) if params else None

        def fwd():
            with torch.no_grad():
                return self.model(*args)

        def grad():
            for a in args:
                a.grad = None
            y = self.model(*args)
            dict(zip(self.output_ids, (y,) if torch.is_tensor(y) else y))[ln].backward()
            return [a.grad for a in args if a.grad is not None]

        def train():
            opt.zero_grad()
            y = self.model(*args)
            loss = dict(zip(self.output_ids, (y,) if torch.is_tensor(y) else y))[ln]
            loss.backward()
            opt.step()
            return loss.detach()

        return fwd, (grad if grad_inputs else train)
