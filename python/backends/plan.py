"""A backend-neutral, flat execution plan taken from a validated model graph (composites already expanded, ids are full paths)."""
from __future__ import annotations

from dataclasses import dataclass, field

from graph_core.types import TensorType


@dataclass
class Step:
    nid: str
    type: str
    cfg: object  # resolved pydantic config
    srcs: dict[str, tuple[str, str]]  # input port -> (source node, source port)
    in_types: dict[str, TensorType]
    out_type: TensorType  # the (first) output port type
    ports: tuple[str, ...]  # input ports in order
    owner: str  # node that owns the parameters this step uses (itself unless it shares)

    @property
    def shared(self) -> bool:
        return self.owner != self.nid


@dataclass
class ModelPlan:
    steps: list[Step]
    input_ids: list[str]
    output_ids: list[str]
    graph_hash: str = ""
    by_id: dict[str, Step] = field(default_factory=dict)

    def __post_init__(self):
        self.by_id = {s.nid: s for s in self.steps}

    def owners(self) -> list[Step]:
        return [s for s in self.steps if not s.shared]


def make_plan(report, graph_hash: str = "", partial: bool = False) -> ModelPlan:
    """report: a graph_core.validate.Report. Without `partial` it must be ok; with it, nodes that could not be typed are left out."""
    from graph_core import registry

    g = report.graph
    types = {n.id: n.type for n in g.nodes}
    src_of = {(e.to.node, e.to.port): (e.from_.node, e.from_.port) for e in g.edges}
    consumers: dict[str, int] = {}
    steps: list[Step] = []
    for nid in report.order:
        if nid not in report.resolved:
            if partial:
                continue
            raise KeyError(nid)
        op = registry.get_op(types[nid])
        cfg = report.resolved[nid]
        ports = tuple(op.input_ports(cfg))
        srcs = {p: src_of[(nid, p)] for p in ports}
        for s, _ in srcs.values():
            consumers[s] = consumers.get(s, 0) + 1
        outs = report.output_types[nid]
        steps.append(Step(nid, types[nid], cfg, srcs, dict(report.input_types[nid]), next(iter(outs.values())), ports, report.shared.get(nid, nid)))
    inputs = sorted(s.nid for s in steps if s.type == "core.tensor_input")
    outputs = sorted(s.nid for s in steps if not consumers.get(s.nid))
    return ModelPlan(steps, inputs, outputs, graph_hash)
