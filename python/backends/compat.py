"""Backend compatibility report (VISION 14.2, A18): for a model graph and a target backend, every node is `supported`,
`converted` (supported with the listed conversions) or `unsupported` (stable error code and reason). Computed BEFORE anything executes.
Nothing is substituted or dropped: an unsupported node makes the graph unexecutable on that backend."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import jax_spec, keras_spec
from .base import BackendError
from .check import Check, refuse
from .plan import ModelPlan, make_plan
from .registry import BACKEND_IDS, INFO, availability

__all__ = ["CompatReport", "analyze_plan", "node_checks", "compat_report", "require_compatible"]


def analyze_plan(plan: ModelPlan, backend: str) -> dict[str, Check]:
    from graph_core import registry as ops

    if backend == "pytorch":
        return {s.nid: _pytorch_check(s) for s in plan.steps}
    spec = {"keras": keras_spec, "jax": jax_spec}[backend]
    checks = spec.analyze(plan)
    if backend == "keras":
        checks = {k: v.check for k, v in checks.items()}
    for s in plan.steps:
        op = ops.get_op(s.type)
        if op is not None and op.backend in ("keras", "jax") and op.backend != backend:
            checks[s.nid] = _specific_mismatch(s.type, op.backend, backend)
    return checks


def _specific_mismatch(op_type: str, owner: str, target: str) -> Check:
    return refuse("E_BACKEND_OP", f"'{op_type}' is a {INFO[owner].title}-specific node (backend '{owner}') and cannot run on '{target}'. It is not replaced by a similar operation.")


def _pytorch_check(s) -> Check:
    from graph_core import registry as ops
    from .check import native

    op = ops.get_op(s.type)
    if op is not None and op.backend in ("keras", "jax"):
        return _specific_mismatch(s.type, op.backend, "pytorch")
    return native()


def node_checks(report, backend: str) -> dict[str, Check]:
    """Verdicts for every node of a validated Report that could be typed (unresolved nodes already carry a validation error)."""
    plan = make_plan(report, partial=True)
    return analyze_plan(plan, backend)


@dataclass
class CompatReport:
    backend: str
    graph_hash: str
    nodes: dict[str, dict]  # flat node id -> {type, status, conversions, code, reason}
    unchecked: list[str]  # nodes that could not be typed (see diagnostics)
    structural: list[dict]  # validation errors that are not backend errors
    sharing: list[dict]  # {owner, users}: parameter-sharing groups, preserved by every supported backend
    available: bool
    availability_reason: str | None
    versions: dict[str, str]

    @property
    def counts(self) -> dict[str, int]:
        c = {"supported": 0, "converted": 0, "unsupported": 0}
        for n in self.nodes.values():
            c[n["status"]] += 1
        return c

    @property
    def ok(self) -> bool:
        """Compatible: nothing unsupported, nothing untyped, no structural error."""
        return self.counts["unsupported"] == 0 and not self.unchecked and not self.structural

    @property
    def executable(self) -> bool:
        return self.ok and self.available

    def to_json(self) -> dict:
        info = INFO[self.backend]
        return {"backend": self.backend, "title": info.title, "graphHash": self.graph_hash, "ok": self.ok, "executable": self.executable,
                "counts": self.counts, "nodes": self.nodes, "unchecked": self.unchecked, "structural": self.structural, "sharing": self.sharing,
                "available": self.available, "availabilityReason": self.availability_reason, "versions": self.versions,
                "facets": info.facets, "init": info.init, "layout": info.layout, "training": info.training}


def compat_report(graph, backend: str) -> CompatReport:
    """The report for `graph` on `backend` (whatever backend the graph currently declares). Model graphs only."""
    from graph_core.hashing import semantic_hash
    from graph_core.validate import validate

    if backend not in BACKEND_IDS:
        raise BackendError("E_UNSUPPORTED_BACKEND", f"unknown backend '{backend}' (known: {list(BACKEND_IDS)})")
    g = graph.model_copy(deep=True)
    g.backend = backend
    r = validate(g)
    plan = make_plan(r, semantic_hash(graph), partial=True)
    checks = analyze_plan(plan, backend)
    nodes = {}
    for s in plan.steps:
        c = checks[s.nid]
        nodes[s.nid] = {"type": s.type, **c.to_json()}
    # unknown ops (never typed) are reported as unsupported nodes too, with the validator's code
    for d in r.errors:
        if d.code == "E_UNKNOWN_OP" and d.nodeId and d.nodeId not in nodes:
            nodes[d.nodeId] = {"type": g.node(d.nodeId).type, **refuse("E_UNKNOWN_OP", d.message).to_json()}
    backend_codes = ("E_BACKEND_", "E_UNKNOWN_OP")
    structural = [d.to_json() for d in r.errors if not d.code.startswith(backend_codes)]
    groups: dict[str, list[str]] = {}
    for u, o in r.shared.items():
        groups.setdefault(o, []).append(u)
    av = availability(backend)
    return CompatReport(backend, semantic_hash(graph), nodes, [n for n in r.unresolved_nodes], structural,
                        [{"owner": o, "users": sorted(us)} for o, us in sorted(groups.items())], av.available, av.reason, av.versions)


def require_compatible(graph, backend: str) -> CompatReport:
    """Raises BackendError(E_BACKEND_INCOMPATIBLE) with the report attached unless the graph is compatible AND the backend is available."""
    rep = compat_report(graph, backend)
    if not rep.available:
        raise BackendError("E_BACKEND_UNAVAILABLE", f"backend '{backend}' is not available here: {rep.availability_reason}", rep)
    if not rep.ok:
        bad = [f"{n} ({v['type']}): {v['code']} {v['reason']}" for n, v in rep.nodes.items() if v["status"] == "unsupported"]
        bad += [f"{d['nodeId']}: {d['code']} {d['message']}" for d in rep.structural]
        bad += [f"{n}: could not be typed" for n in rep.unchecked if not any(d.get("nodeId") == n for d in rep.structural)]
        raise BackendError("E_BACKEND_INCOMPATIBLE", f"graph cannot run on '{backend}':\n  " + "\n  ".join(bad), rep)
    return rep
