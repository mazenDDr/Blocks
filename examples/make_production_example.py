"""Deterministic labelled SYNTHETIC sensors and a real native classifier pipeline."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def generate():
    rng = np.random.default_rng(71)
    rows = []
    for _ in range(160):
        x, y = rng.normal(size=2)
        rows.append({"signal": round(x, 6), "background": round(y, 6), "label": int(x+0.3*y>0)})
    path = ROOT / "fixtures" / "synthetic_serving_sensors.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["signal", "background", "label"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    nodes = []
    def node(id_, type_, config):
        nodes.append({"id": id_, "type": type_, "version": "1.0.0", "config": config, "stateRef": None})
    node("sensors", "tabular.csv_source", {"path": "examples/fixtures/synthetic_serving_sensors.csv"})
    node("split", "tabular.train_validation_split", {"seed": 71, "validation_fraction": 0.25})
    node("scale", "tabular.fit_standardize", {"columns": ["signal", "background"]})
    node("train_scaled", "tabular.apply_transform", {})
    node("val_scaled", "tabular.apply_transform", {})
    node("classifier", "sklearn.logistic_regression", {"target": "label", "features": ["signal", "background"], "C": 1.0, "max_iter": 200})
    node("metrics", "sklearn.metrics", {})
    edges = []
    def edge(src, port, dst, dp, kind="table"):
        edges.append({"id": f"{src}_{port}_{dst}_{dp}", "kind": kind, "from": {"node": src, "port": port}, "to": {"node": dst, "port": dp}})
    edge("sensors", "table", "split", "table")
    edge("split", "train", "scale", "train")
    for name, port in (("train_scaled", "train"), ("val_scaled", "validation")):
        edge("split", port, name, "table")
        edge("scale", "fit", name, "fit", "fit_state")
    edge("train_scaled", "table", "classifier", "train")
    edge("classifier", "model", "metrics", "model", "model")
    edge("val_scaled", "table", "metrics", "table")
    graph = {"schemaVersion": "1.0.0", "graphKind": "tabular", "backend": "python", "nodes": nodes, "edges": edges}
    (ROOT / "production_sensors.project.json").write_text(json.dumps(graph, indent=2)+"\n")
    ui = {"schemaVersion": "1.0.0", "positions": {n["id"]: {"x": (i%4)*260, "y": (i//4)*190} for i,n in enumerate(nodes)},
          "synthetic": True, "description": "Labelled SYNTHETIC sensors. Train, then open Production to register the native classifier and fitted scaler, preview a local/staging release, send requests and measure HTTP traffic."}
    (ROOT / "production_sensors.ui.json").write_text(json.dumps(ui, indent=2)+"\n")
    print(f"Generated {len(rows)} labelled SYNTHETIC sensor rows and production_sensors project.")


if __name__ == "__main__":
    generate()
