"""Shared builders for the Milestone 3 tests."""
import torch

from graph_core.build import GraphBuilder, ModuleBuilder, masked_weighted_loss, residual_block  # noqa: F401
from graph_core.lower import lower_graph
from graph_core.validate import validate


def residual_net(share_second: bool = False, channels: int = 8):
    g = GraphBuilder()
    g.modules.append(residual_block(channels))
    g.input("img", ["N", 3, 16, 16])
    g.node("stem", "pytorch.nn.conv2d", out_channels=channels, padding=[1, 1])
    g.instance("res1", "residual_block", channels=channels)
    g.instance("res2", "residual_block", channels=channels, share="res1" if share_second else "clone")
    g.chain("img", "stem")
    g.wire("stem", "res1.x")
    g.wire("res1.y", "res2.x")
    return g.build()
