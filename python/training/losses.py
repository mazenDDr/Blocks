"""The loss of a procedure: a built-in (cross-entropy, MSE) or a visual loss module from the project's module library.

A module loss is lowered through exactly the same path as any other graph (composite expansion, shape inference, torch modules), so its
value and its gradient are those of the primitives it is made of."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_core.composite import COMPOSITE, Expansion
from graph_core.lower import GraphModule, lower_graph
from graph_core.schema import Edge, Endpoint, Graph, Node
from graph_core.types import TensorType
from graph_core.validate import ExecutionBlocked, Report, validate

from .spec import LossSpec


class BuiltinLoss(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind

    def forward(self, pred, target, **_):
        return F.cross_entropy(pred, target) if self.kind == "cross_entropy" else F.mse_loss(pred, target)


class ModuleLoss(nn.Module):
    """A visual loss module wrapped for the trainer: forward(**named tensors) -> scalar tensor."""

    def __init__(self, gm: GraphModule, report: Report, out_node: tuple[str, str], in_ports: dict[str, str]):
        super().__init__()
        self.gm, self.report, self.out_node, self.in_ports = gm, report, out_node, in_ports  # in_ports: module input name -> tensor_input node id

    def forward(self, **tensors):
        given = {self.in_ports[k]: v for k, v in tensors.items() if k in self.in_ports}
        values = self.gm.evaluate(given)
        return self.gm.value_of(values, *self.out_node)


def loss_wrapper_graph(project: Graph, spec: LossSpec, port_types: dict[str, TensorType]) -> Graph:
    """tensor_input nodes (one per module input, typed from the data) feeding one instance of the loss module."""
    mod = project.module(spec.module, spec.version)
    if mod is None:
        raise ExecutionBlocked([_diag("E_MODULE_MISSING", f"loss module '{spec.module}' version {spec.version} is not defined in the project")])
    nodes = [Node(id="loss_fn", type=COMPOSITE, config={"module": mod.id, "version": mod.version, "args": spec.args})]
    edges = []
    for p in mod.inputs:
        t = port_types.get(p.name)
        if t is None:
            raise ExecutionBlocked([_diag("E_PROC_LOSS", f"the loss module input '{p.name}' has no source (declare it in loss.ports)")])
        nodes.append(Node(id=f"in_{p.name}", type="core.tensor_input", config={"shape": list(t.shape), "dtype": t.dtype}))
        edges.append(Edge(id=f"w_{p.name}", **{"from": Endpoint(node=f"in_{p.name}", port="value")}, to=Endpoint(node="loss_fn", port=p.name)))
    return project.model_copy(update={"nodes": nodes, "edges": edges})


def _diag(code, msg):
    from graph_core.types import Diagnostic

    return Diagnostic(code, msg)


def build_loss(project: Graph, spec: LossSpec, port_types: dict[str, TensorType]) -> nn.Module:
    if spec.kind != "module":
        return BuiltinLoss(spec.kind)
    g = loss_wrapper_graph(project, spec, port_types)
    report = validate(g)
    if not report.ok:
        raise ExecutionBlocked(report.errors)
    gm = lower_graph(g, report)
    inst = report.expansion.instances["loss_fn"]
    if spec.output not in inst.out_map:
        raise ExecutionBlocked([_diag("E_PROC_LOSS", f"the loss module has no output '{spec.output}' (has {list(inst.out_map)})")])
    t = report.output_types[inst.out_map[spec.output][0]][inst.out_map[spec.output][1]]
    if t.shape != ():
        raise ExecutionBlocked([_diag("E_PROC_LOSS", f"the loss output '{spec.output}' must be a scalar (shape []), got {list(t.shape)}")])
    return ModuleLoss(gm, report, inst.out_map[spec.output], {p: f"in_{p}" for p in inst.inputs})
