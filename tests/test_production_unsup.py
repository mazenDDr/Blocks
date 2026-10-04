"""Unsupervised models in the production registry (ADR 0020): k-means / GMM / PCA from the unsupervised_cells example are captured with
their native fitted scaler and estimator, served with outputs equal to those objects, monitored with external agreement (never accuracy)
or label-free reconstruction error, and t-SNE is refused because it cannot map new points."""
import pickle

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import adjusted_rand_score

from control.app import create_app
from tabular_helpers import example
from test_tabular_api import submit, wait_for

META = {"name": "cells", "owner": "tests", "intendedUse": "local CPU fixture clustering", "limitations": "SYNTHETIC cells; clusters are not classes"}


@pytest.fixture(scope="module")
def cells(tmp_path_factory):
    with TestClient(create_app(tmp_path_factory.mktemp("wb"))) as c:
        r = submit(c, example("unsupervised_cells"))
        assert r.status_code == 201, r.text
        rid = r.json()["runId"]
        assert wait_for(c, rid)["status"] == "completed"
        yield c, rid


def deploy(c, rid, node, ns):
    v = c.post("/api/production/versions", json={"runId": rid, "node": node, **META})
    assert v.status_code == 201, v.text
    rel = c.post("/api/production/releases", json={"versionId": v.json()["id"], "config": {"namespace": ns, "captureInputs": True}})
    assert rel.status_code == 201, rel.text
    assert c.post(f"/api/production/releases/{rel.json()['id']}/deploy", json={"expectedCurrent": None}).status_code == 200
    return v.json(), rel.json()


def native(c, rid, node):
    store = c.app.state.services.store
    blob = [a for a in store.artifacts(rid, "unsup_model") if a["meta"]["node"] == node][-1]
    return pickle.loads(store.read_artifact(blob["sha256"]))


def test_candidates_and_tsne_refusal(cells):
    c, rid = cells
    fams = {x["node"]: x["family"] for x in c.get("/api/production").json()["candidates"] if x["runId"] == rid and x["adapter"] == "unsup"}
    assert fams == {"kmeans": "kmeans", "gmm": "gmm", "pca": "pca"}
    r = c.post("/api/production/versions", json={"runId": rid, "node": "tsne", **META})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "E_PIPELINE_UNSUPPORTED" and "cannot map new points" in r.json()["detail"]["message"]


@pytest.mark.parametrize("node", ["kmeans", "gmm", "pca"])
def test_served_outputs_equal_the_native_fitted_objects(cells, node):
    c, rid = cells
    v, rel = deploy(c, rid, node, f"u-{node}")
    assert v["adapter"] == "unsup" and v["manifest"]["features"] == ["area_um2", "intensity", "granularity", "elongation"]
    rows = c.get(f"/api/production/versions/{v['id']}/reference-input").json()["records"]
    t = c.post(f"/api/serve/local/u-{node}/predict", json={"requestId": f"{node}-1", "records": rows}).json()
    assert t["status"] == 200, t["error"]
    m = native(c, rid, node)
    X = m.scaler.transform(pd.DataFrame(rows)[m.features].to_numpy(dtype="float64"))
    if node == "pca":
        np.testing.assert_allclose(t["result"]["predictions"], m.estimator.transform(X), atol=1e-10)
        recon = m.estimator.inverse_transform(m.estimator.transform(X))
        np.testing.assert_allclose(t["result"]["reconstructionError"], ((X - recon) ** 2).sum(1), atol=1e-10)
    else:
        assert t["result"]["predictions"] == m.estimator.predict(X).tolist()
        if node == "gmm":
            np.testing.assert_allclose(t["result"]["responsibilities"], m.estimator.predict_proba(X), atol=1e-10)
    bad = c.post(f"/api/serve/local/u-{node}/predict", json={"requestId": f"{node}-bad", "records": [{"area_um2": 1}]})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "E_REQUEST_SCHEMA"


def test_external_agreement_and_label_free_pca_monitoring(cells):
    c, rid = cells
    v, rel = deploy(c, rid, "kmeans", "agree")
    rows = c.get(f"/api/production/versions/{v['id']}/reference-input").json()["records"]
    t = c.post("/api/serve/local/agree/predict", json={"requestId": "a1", "records": rows}).json()
    labels = ["x" if p == 0 else "y" for p in t["result"]["predictions"]]
    assert c.post("/api/production/requests/a1/labels", json={"labels": [True] * len(rows)}).status_code == 422
    assert c.post("/api/production/requests/a1/labels", json={"labels": labels}).status_code == 200
    mon = c.get(f"/api/production/releases/{rel['id']}/monitor").json()
    q = mon["labelBasedQuality"]
    assert mon["family"] == "kmeans" and "accuracy" not in q["values"]
    assert q["values"]["adjustedRandIndex"] == pytest.approx(adjusted_rand_score(labels, t["result"]["predictions"]))
    assert mon["predictionDrift"]["available"] and mon["inputDrift"]["area_um2"]["available"]

    pv, prel = deploy(c, rid, "pca", "pca-mon")
    prow = c.get(f"/api/production/versions/{pv['id']}/reference-input").json()["records"]
    c.post("/api/serve/local/pca-mon/predict", json={"requestId": "p1", "records": prow})
    assert c.post("/api/production/requests/p1/labels", json={"labels": ["a"] * len(prow)}).json()["detail"]["code"] == "E_LABEL_SCHEMA"
    pm = c.get(f"/api/production/releases/{prel['id']}/monitor").json()
    assert pm["labelBasedQuality"]["labelFree"] and pm["labelBasedQuality"]["values"]["meanReconstructionError"] >= 0
    assert pm["predictionDrift"]["unit"] == "PC1 score"


def test_tabular_adapter_identity_is_unchanged():
    """The capture lives outside production/pipeline.py, so registered tabular versions keep their adapter identity."""
    from production.pipeline import implementation_sources

    assert set(implementation_sources()) == {"production/pipeline.py", "operations/tabular_ops.py", "tabular/core.py"}


def test_fitted_preprocessing_upstream_is_refused(tmp_path):
    """k-means fed by a fitted scaler: requests could not carry its raw inputs faithfully, so capture records a refusal."""
    from graph_core.schema import Edge, Node

    from tabular_helpers import run_inproc

    g = example("tabular_regression")
    g.nodes.append(Node.model_validate({"id": "km", "type": "sklearn.kmeans", "version": "1.0.0", "config": {"features": ["area_m2", "rooms"], "n_clusters": 2}}))
    g.edges.append(Edge.model_validate({"id": "sc_train_km", "kind": "table", "from": {"node": "sc_train", "port": "table"}, "to": {"node": "km", "port": "table"}}))
    store, status = run_inproc(g, tmp_path)
    assert status == "completed"
    import json
    refusals = [json.loads(store.read_artifact(a["sha256"])) for a in store.artifacts("r1", "unsup_refusal")]
    assert refusals == [{"node": "km", "code": "E_PIPELINE_UNSUPPORTED",
                         "message": "tabular.apply_transform upstream of the estimator is not served; requests must carry the raw feature columns."}]
    assert not store.artifacts("r1", "unsup_pipeline")
