"""The shared reference workloads (Milestone 6a exit evidence). One definition each, used by the tests, the benchmarks and the coverage ledger
(which derives the "workload" evidence of an operation from the nodes of these graphs)."""
from __future__ import annotations

import copy
from pathlib import Path

from graph_core.build import GraphBuilder
from graph_core.project_io import load_project
from graph_core.schema import Edge, Graph, Node

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def cnn_with_loss() -> Graph:
    """The reference CNN (VISION 8.1) plus an int64 label input and a cross-entropy node `ce` (the A06 construction)."""
    g = copy.deepcopy(load_project(EXAMPLES / "reference_cnn.project.json").graph)
    g.nodes.append(Node.model_validate({"id": "labels", "type": "core.tensor_input", "config": {"shape": ["N"], "dtype": "int64", "layout": "N"}}))
    g.nodes.append(Node.model_validate({"id": "ce", "type": "pytorch.loss.cross_entropy", "config": {"reduction": "mean"}}))
    g.edges += [Edge.model_validate(d) for d in (
        {"id": "fc_ce", "from": {"node": "fc", "port": "output"}, "to": {"node": "ce", "port": "logits"}},
        {"id": "labels_ce", "from": {"node": "labels", "port": "value"}, "to": {"node": "ce", "port": "target"}})]
    return g


def mse_graph() -> Graph:
    return load_project(EXAMPLES / "mse_teaching.project.json").graph


def sgd_teaching_graph() -> Graph:
    """A34: w = 2, gradient 0.6, lr = 0.1 -> 1.94."""
    b = GraphBuilder()
    b.input("w", [1])
    b.node("scaled", "core.scalar_mul", factor=0.6)
    b.node("loss", "core.sum")
    b.chain("w", "scaled", "loss")
    return b.build()


def shared_graph() -> Graph:
    """Two call sites of one convolution (explicit parameter sharing), summed, squared, averaged."""
    b = GraphBuilder()
    b.input("u", ["N", 2, 6, 6])
    b.input("v", ["N", 2, 6, 6])
    b.node("enc_u", "pytorch.nn.conv2d", out_channels=3, kernel_size=[3, 3], padding=[1, 1])
    b.node("enc_v", "pytorch.nn.conv2d", shared_with="enc_u", out_channels=3, kernel_size=[3, 3], padding=[1, 1])
    b.node("join", "core.add")
    b.node("sq", "core.square")
    b.node("loss", "core.mean")
    b.wire("u", "enc_u.input")
    b.wire("v", "enc_v.input")
    b.wire("enc_u.output", "join.a")
    b.wire("enc_v.output", "join.b")
    b.chain("join", "sq", "loss")
    return b.build()


def conv_flatten_linear_graph() -> Graph:
    """Spatial maps flattened into a Linear: exercises the NHWC -> NCHW transpose before the reshape."""
    b = GraphBuilder()
    b.input("x", ["N", 2, 8, 8])
    b.node("c", "pytorch.nn.conv2d", out_channels=3, kernel_size=[3, 3], padding=[1, 1])
    b.node("r", "pytorch.nn.relu")
    b.node("p", "pytorch.nn.max_pool2d", kernel_size=[2, 2])
    b.node("f", "pytorch.nn.flatten")
    b.node("fc", "pytorch.nn.linear", out_features=4)
    b.chain("x", "c", "r", "p", "f", "fc")
    return b.build()


def residual_core_add_graph() -> Graph:
    """The residual_cnn example with the portable core.add as the skip join (the shipped example uses tensor.add, outside the portable subset)."""
    g = copy.deepcopy(load_project(EXAMPLES / "residual_cnn.project.json").graph)
    for m in g.modules:
        for n in m.nodes:
            if n.type == "tensor.add":
                n.type, n.config = "core.add", {}
    return g


WORKLOADS = {
    "reference_cnn_with_cross_entropy": cnn_with_loss,
    "mse_teaching": mse_graph,
    "sgd_teaching_step": sgd_teaching_graph,
    "shared_parameters": shared_graph,
    "conv_flatten_linear": conv_flatten_linear_graph,
    "residual_cnn_core_add": residual_core_add_graph,
}
