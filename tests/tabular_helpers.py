"""Helpers shared by the tabular / statistics tests."""
import copy
import json

import pandas as pd

from artifact_store import ArtifactStore
from conftest import EXAMPLES
from graph_core.hashing import semantic_hash
from graph_core.project_io import load_project
from graph_core.schema import Edge, Graph, Node
from worker.tabular_run import TabularRunConfig, run_tabular

FIX = EXAMPLES / "fixtures"


def example(name: str) -> Graph:
    return copy.deepcopy(load_project(EXAMPLES / f"{name}.project.json").graph)


def set_cfg(g: Graph, node: str, **kw) -> Graph:
    g.node(node).config.update(kw)
    return g


def rewire(g: Graph, to_node: str, to_port: str, from_node: str, from_port: str) -> Graph:
    """Replace the edge into (to_node, to_port) by one from (from_node, from_port); the edge kind follows the source port."""
    from graph_core import registry

    kind = registry.get_op(g.node(from_node).type).out_kinds[from_port]
    g.edges = [e for e in g.edges if not (e.to.node == to_node and e.to.port == to_port)]
    g.edges.append(Edge.model_validate({"id": f"{from_node}_{from_port}__{to_node}_{to_port}", "kind": kind,
                                        "from": {"node": from_node, "port": from_port}, "to": {"node": to_node, "port": to_port}}))
    return g


def run_inproc(g: Graph, tmp_path, run_id="r1"):
    store = ArtifactStore(tmp_path / "wb")
    cfg = TabularRunConfig()
    store.create_run(run_id, semantic_hash(g), cfg.model_dump())
    status = run_tabular(g, cfg, store, run_id)
    return store, status


def summary(store, run_id, node):
    a = [x for x in store.artifacts(run_id, "node_summary") if x["meta"]["node"] == node][-1]
    return json.loads(store.read_artifact(a["sha256"]))


def table(store, run_id, node, port="table") -> pd.DataFrame:
    a = [x for x in store.artifacts(run_id, "node_output") if x["meta"]["node"] == node and x["meta"]["port"] == port][-1]
    return pd.read_csv(store.path_of(a["sha256"]), index_col="row_id")


def codes(report, severity=None):
    return [d.code for d in report.diagnostics if severity is None or d.severity == severity]
