"""Builders of the unsupervised example graphs (tabular graph kind), shared by the examples generator and the tests."""
from __future__ import annotations

from typing import Any


def _n(i: str, t: str, cfg: dict[str, Any]) -> dict[str, Any]:
    return {"id": i, "type": t, "version": "1.0.0", "config": cfg}


def _e(k: int, a: tuple[str, str], b: tuple[str, str], kind: str) -> dict[str, Any]:
    return {"id": f"e{k}", "kind": kind, "from": {"node": a[0], "port": a[1]}, "to": {"node": b[0], "port": b[1]}}


def graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schemaVersion": "1.0.0", "graphKind": "tabular", "backend": "python", "nodes": nodes, "edges": edges}


def cells_graph(path: str = "examples/fixtures/synthetic_cells.csv") -> dict[str, Any]:
    feats = ["area_um2", "intensity", "granularity", "elongation"]
    nodes = [
        _n("cells", "tabular.csv_source", {"path": path}),
        _n("kmeans", "sklearn.kmeans", {"features": feats, "n_clusters": 3, "seed": 0}),
        _n("km_diag", "sklearn.cluster_diagnostics", {"labels_column": "true_group", "k_min": 2, "k_max": 8, "stability_runs": 8, "resample": "subsample"}),
        _n("gmm", "sklearn.gaussian_mixture", {"features": feats, "n_components": 3, "covariance_type": "full", "seed": 0}),
        _n("gmm_diag", "sklearn.cluster_diagnostics", {"labels_column": "true_group", "k_min": 2, "k_max": 6, "stability_runs": 6}),
        _n("pca", "sklearn.pca", {"features": feats, "n_components": 2}),
        _n("pca_diag", "sklearn.cluster_diagnostics", {"stability_runs": 10}),
        _n("tsne", "sklearn.projection", {"features": feats, "perplexity": 30, "seed": 0}),
    ]
    edges, k = [], 0
    for m, d in (("kmeans", "km_diag"), ("gmm", "gmm_diag"), ("pca", "pca_diag")):
        edges.append(_e(k, ("cells", "table"), (m, "table"), "table")); k += 1
        edges.append(_e(k, (m, "model"), (d, "model"), "unsup_model")); k += 1
        edges.append(_e(k, ("cells", "table"), (d, "table"), "table")); k += 1
    edges.append(_e(k, ("cells", "table"), ("tsne", "table"), "table"))
    return graph(nodes, edges)


def moons_graph(path: str = "examples/fixtures/synthetic_moons.csv") -> dict[str, Any]:
    feats = ["x", "y"]
    nodes = [
        _n("moons", "tabular.csv_source", {"path": path}),
        _n("kmeans", "sklearn.kmeans", {"features": feats, "n_clusters": 2, "seed": 0}),
        _n("km_diag", "sklearn.cluster_diagnostics", {"labels_column": "true_shape", "k_min": 2, "k_max": 6, "stability_runs": 8}),
        _n("dbscan", "sklearn.dbscan", {"features": feats, "eps": 0.2, "min_samples": 5, "scale": "none"}),
        _n("db_diag", "sklearn.cluster_diagnostics", {"labels_column": "true_shape", "stability_runs": 8}),
    ]
    edges, k = [], 0
    for m, d in (("kmeans", "km_diag"), ("dbscan", "db_diag")):
        edges.append(_e(k, ("moons", "table"), (m, "table"), "table")); k += 1
        edges.append(_e(k, (m, "model"), (d, "model"), "unsup_model")); k += 1
        edges.append(_e(k, ("moons", "table"), (d, "table"), "table")); k += 1
    return graph(nodes, edges)
