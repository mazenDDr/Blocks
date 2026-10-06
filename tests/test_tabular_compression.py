"""Large table outputs stored gzip-compressed, deterministic hashes, unchanged inspection (ADR 0079)."""
import gzip
import io
import json

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from control.app import create_app
from test_tabular_api import inspect, submit
from test_tabular_api import wait_for as wait_run


def graph_for(csv):
    from graph_core.schema import Graph
    doc = json.loads(open("examples/production_sensors.project.json").read())
    next(n for n in doc["nodes"] if n["id"] == "sensors")["config"]["path"] = str(csv)
    return Graph.model_validate(doc)


def test_large_tables_are_compressed_deterministically_and_inspect_unchanged(tmp_path):
    rng = np.random.default_rng(3)
    s = rng.normal(0, 1, 20000)
    df = pd.DataFrame({"signal": s.round(6), "background": rng.normal(0, 1, 20000).round(6), "label": (s > 0).astype(int)})
    csv = tmp_path / "synthetic_sensors.csv"
    df.to_csv(csv, index=False)
    with TestClient(create_app(tmp_path / "wb")) as c:
        store = c.app.state.services.store
        runs = []
        for _ in range(2):
            rid = submit(c, graph_for(csv)).json()["runId"]
            assert wait_run(c, rid)["status"] == "completed"
            runs.append(rid)
        outs = {(a["meta"]["node"], a["meta"]["port"]): a for a in store.artifacts(runs[0], "node_output")}
        src = outs[("sensors", "table")]
        assert src["meta"]["encoding"] == "gzip" and src["meta"]["rawBytes"] > src["size"] * 2
        raw = gzip.decompress(store.read_artifact(src["sha256"]))
        assert len(raw) == src["meta"]["rawBytes"] and pd.read_csv(io.BytesIO(raw))["row_id"].tolist() == list(range(20000))
        small = [a for a in outs.values() if a["meta"].get("valueKind") != "table"]
        assert small and all("encoding" not in a["meta"] for a in small)
        again = {(a["meta"]["node"], a["meta"]["port"]): a["sha256"] for a in store.artifacts(runs[1], "node_output")}
        assert again == {k: a["sha256"] for k, a in outs.items()}  # deterministic gzip: same table, same hash
        page = inspect(c, runs[0], kind="table", node="sensors", offset=19990, limit=10).json()
        assert page["available"] and page["rowIds"] == list(range(19990, 20000)) and page["total"] == 20000
        assert page["rows"][0][0] == df.loc[19990, "signal"]
