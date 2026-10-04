"""Helpers for the connected-data tests: a seeded workbench with real local services (PostgreSQL via pgserver, S3 API via moto, Git+DVC)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from artifact_store import ArtifactStore
from connectors import context as conn_context
from connectors import synthetic
from connectors.registry import ConnectionRegistry
from graph_core import registry as op_registry
from graph_core.hashing import semantic_hash
from graph_core.schema import Graph
from worker.tabular_run import TabularRunConfig, run_tabular

BUCKET, PLAIN_BUCKET = "labdata", "plainbkt"
S3_KEY_ENV, S3_SECRET_ENV = "VOID_TEST_S3_KEY", "VOID_TEST_S3_SECRET"


def mk(nodes: list[tuple[str, str, dict[str, Any]]], edges: list[tuple[str, str, str, str]]) -> Graph:
    es = []
    for a, ap, b, bp in edges:
        kind = op_registry.get_op(next(t for i, t, _ in nodes if i == a)).out_kinds[ap]
        es.append({"id": f"{a}_{ap}__{b}_{bp}", "kind": kind, "from": {"node": a, "port": ap}, "to": {"node": b, "port": bp}})
    return Graph.model_validate({"schemaVersion": "1.0.0", "graphKind": "tabular", "backend": "python",
                                 "nodes": [{"id": i, "type": t, "version": "1.0.0", "config": c} for i, t, c in nodes], "edges": es})


def run_graph(g: Graph, wb: Path, run_id: str, **cfg: Any):
    store = ArtifactStore(wb)
    c = TabularRunConfig(**cfg)
    store.create_run(run_id, semantic_hash(g), c.model_dump())
    store.add_artifact(run_id, "graph", json.dumps(g.to_json(), sort_keys=True).encode(), "complete", None, {"graph_hash": semantic_hash(g)})
    status = run_tabular(g, c, store, run_id)
    return store, status


def node_summary(store: ArtifactStore, run_id: str, node: str) -> dict[str, Any]:
    a = [x for x in store.artifacts(run_id, "node_summary") if x["meta"]["node"] == node][-1]
    return json.loads(store.read_artifact(a["sha256"]))


def node_table(store: ArtifactStore, run_id: str, node: str, port: str = "table") -> pd.DataFrame:
    a = [x for x in store.artifacts(run_id, "node_output") if x["meta"]["node"] == node and x["meta"]["port"] == port][-1]
    return pd.read_csv(store.path_of(a["sha256"]), index_col="row_id")


class Lab:
    """A workbench directory with connections to the local services."""

    def __init__(self, tmp: Path, pg, s3, monkeypatch):
        self.wb = tmp / "wb"
        self.pg, self.s3 = pg, s3
        monkeypatch.setenv(S3_KEY_ENV, s3.ACCESS)
        monkeypatch.setenv(S3_SECRET_ENV, s3.SECRET)
        conn_context.set_workbench(self.wb)
        self.reg = ConnectionRegistry(self.wb)
        self.reg.create("lab", "Lab database (admin role)", "postgres", pg.settings(), {})
        self.reg.create("reader", "Lab database (reader role)", "postgres", pg.settings("reader"), {})
        refs = {"access_key_id": {"kind": "env", "name": S3_KEY_ENV}, "secret_access_key": {"kind": "env", "name": S3_SECRET_ENV}}
        self.reg.create("files", "Lab files (versioned bucket)", "s3", {"bucket": BUCKET, "endpoint_url": s3.endpoint}, refs)
        self.reg.create("plain", "Unversioned bucket", "s3", {"bucket": PLAIN_BUCKET, "endpoint_url": s3.endpoint}, refs)
