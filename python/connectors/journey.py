"""The connected-data journey project (VISION 7.10): PostgreSQL assay rows + S3 spectra + S3 image-object metadata (size) -> joined, split by
specimen -> standardized -> linear regression. Built from code so the example script and the tests use the same graph.

Connection ids: `lab_db` (PostgreSQL) and `lab_files` (S3-compatible). DATA IS SYNTHETIC (see connectors/synthetic.py)."""
from __future__ import annotations

from typing import Any


def _edge(a: str, ap: str, b: str, bp: str, kind: str = "table") -> dict[str, Any]:
    return {"id": f"{a}_{ap}__{b}_{bp}", "kind": kind, "from": {"node": a, "port": ap}, "to": {"node": b, "port": bp}}


def graph(db: str = "lab_db", files: str = "lab_files", pins: dict[str, str] | None = None) -> dict[str, Any]:
    pins = pins or {}
    query = {"base": {"schema": "public", "name": "assays"}, "base_alias": "a",
             "columns": [{"table": "a", "column": "specimen_id"}, {"table": "a", "column": "response"}, {"table": "s", "column": "ph"}, {"table": "s", "column": "temp_c"}],
             "joins": [{"table": {"schema": "public", "name": "specimens"}, "alias": "s", "type": "inner",
                        "on": [{"left": {"table": "a", "column": "specimen_id"}, "right": {"table": "s", "column": "specimen_id"}}]}],
             "order_by": [{"by": "specimen_id", "direction": "asc"}, {"by": "response", "direction": "asc"}], "limit": 1000}
    n = lambda i, t, c: {"id": i, "type": t, "version": "1.0.0", "config": c, "stateRef": None}  # noqa: E731
    nodes = [
        n("assays", "postgres.query", {"connection": db, "mode": "visual", "query": query, **({"pin": pins["assays"]} if "assays" in pins else {})}),
        n("spectra", "s3.csv_source", {"connection": files, "key": "features/spectra.csv", **({"pin": pins["spectra"]} if "spectra" in pins else {})}),
        n("images", "s3.object_listing", {"connection": files, "prefix": "images/", "key_regex": r"images/(?P<id>SP\d+)\.png", **({"pin": pins["images"]} if "images" in pins else {})}),
        n("join_spectra", "tabular.join", {"left_on": ["specimen_id"], "right_on": ["specimen_id"], "how": "left", "expect": "many_to_one"}),
        n("join_images", "tabular.join", {"left_on": ["specimen_id"], "right_on": ["key_id"], "how": "left", "expect": "many_to_one"}),
        n("select", "tabular.select_columns", {"columns": [{"name": "specimen_id", "dtype": "string"}, {"name": "response", "dtype": "float"}, {"name": "ph", "dtype": "float"},
                                                          {"name": "temp_c", "dtype": "float"}, {"name": "absorbance_a", "dtype": "float"}, {"name": "absorbance_b", "dtype": "float"},
                                                          {"name": "size", "dtype": "float"}], "on_cast_failure": "error"}),
        n("complete", "tabular.drop_missing", {"columns": ["absorbance_a", "absorbance_b", "size"]}),
        n("split", "tabular.train_validation_split", {"seed": 7, "validation_fraction": 0.25, "group_by": "specimen_id"}),
        n("sc_fit", "tabular.fit_standardize", {"columns": ["ph", "temp_c", "absorbance_a", "absorbance_b", "size"]}),
        n("sc_train", "tabular.apply_transform", {}),
        n("sc_val", "tabular.apply_transform", {}),
        n("ols", "sklearn.linear_regression", {"target": "response", "features": ["ph", "temp_c", "absorbance_a", "absorbance_b", "size"], "fit_intercept": True}),
        n("metrics", "sklearn.metrics", {}),
        n("predictions", "tabular.predictions_export", {}),
    ]
    edges = [_edge("assays", "table", "join_spectra", "left"), _edge("spectra", "table", "join_spectra", "right"),
             _edge("join_spectra", "table", "join_images", "left"), _edge("images", "table", "join_images", "right"),
             _edge("join_images", "table", "select", "table"), _edge("select", "table", "complete", "table"), _edge("complete", "table", "split", "table"),
             _edge("split", "train", "sc_fit", "train"), _edge("split", "train", "sc_train", "table"), _edge("sc_fit", "fit", "sc_train", "fit", "fit_state"),
             _edge("split", "validation", "sc_val", "table"), _edge("sc_fit", "fit", "sc_val", "fit", "fit_state"),
             _edge("sc_train", "table", "ols", "train"), _edge("ols", "model", "metrics", "model", "model"), _edge("sc_val", "table", "metrics", "table"),
             _edge("ols", "model", "predictions", "model", "model"), _edge("sc_val", "table", "predictions", "table")]
    return {"schemaVersion": "1.0.0", "graphKind": "tabular", "backend": "python", "nodes": nodes, "edges": edges}


def ui(g: dict[str, Any]) -> dict[str, Any]:
    cols = {"assays": (0, 0), "spectra": (0, 160), "images": (0, 320), "join_spectra": (1, 60), "join_images": (2, 160), "select": (3, 160), "complete": (4, 160),
            "split": (5, 160), "sc_fit": (6, 80), "sc_train": (7, 80), "sc_val": (7, 240), "ols": (8, 80), "metrics": (9, 160), "predictions": (9, 0)}
    return {"schemaVersion": "1.0.0", "positions": {k: {"x": x * 290, "y": y} for k, (x, y) in cols.items()}, "synthetic": True,
            "description": "Connected-data journey: PostgreSQL assay rows joined with S3 spectra and image metadata (SYNTHETIC data). Needs the local services from examples/connected_journey.py."}
