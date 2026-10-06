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
    """The original adapter (ADR 0020) still refuses this path; ADR 0054 serves it from a separate module (tested below)."""
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


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """k-means after the regression example's fitted impute → one-hot → standardize path, run by the real tabular worker."""
    from graph_core.schema import Edge, Node

    g = example("tabular_regression")
    g.nodes.append(Node.model_validate({"id": "km", "type": "sklearn.kmeans", "version": "1.0.0",
                                        "config": {"features": ["area_m2", "rooms", "age_years", "neighborhood_north"], "n_clusters": 3}}))
    g.edges.append(Edge.model_validate({"id": "sc_train_km", "kind": "table", "from": {"node": "sc_train", "port": "table"}, "to": {"node": "km", "port": "table"}}))
    with TestClient(create_app(tmp_path_factory.mktemp("wb"))) as c:
        r = submit(c, g)
        assert r.status_code == 201, r.text
        rid = r.json()["runId"]
        assert wait_for(c, rid)["status"] == "completed"
        yield c, rid


def test_fitted_preprocessing_path_is_served_with_native_equal_outputs(fitted):
    """Raw rows replay the pinned impute/one-hot/standardize steps and give exactly the run's own predictions; nothing refits."""
    import json
    c, rid = fitted
    store = c.app.state.services.store
    manifest = json.loads(store.read_artifact([a for a in store.artifacts(rid, "unsup_fitted_pipeline") if a["meta"]["node"] == "km"][-1]["sha256"]))
    assert [s["type"] for s in manifest["steps"]] == ["tabular.select_columns"] + ["tabular.apply_transform"] * 3
    assert {x["name"]: x["nullable"] for x in manifest["inputSchema"]} == {"area_m2": True, "rooms": False, "age_years": True, "neighborhood": False}
    assert "price_k" not in manifest["features"] and [t["type"] for t in manifest["trainingOnly"]] == ["tabular.profile", "tabular.duplicates", "tabular.train_validation_split"]
    v, rel = deploy(c, rid, "km", "fitted-km")
    assert v["adapter"] == "unsup" and v["family"] == "kmeans"
    ref = c.get(f"/api/production/versions/{v['id']}/reference-input").json()
    assert "raw source columns" in ref["inputContract"]
    model = pickle.loads(store.read_artifact([a for a in store.artifacts(rid, "unsup_fitted_model") if a["meta"]["node"] == "km"][-1]["sha256"]))
    recorded = [a for a in store.artifacts(rid, "node_output") if a["meta"]["node"] == "sc_train" and a["meta"]["port"] == "table"][-1]
    transformed = pd.read_csv(store.path_of(recorded["sha256"]), index_col="row_id")
    raw = pd.read_csv(store.path_of(store.artifacts(rid, "unsup_fitted_reference")[-1]["sha256"]))
    records = json.loads(raw.head(40).to_json(orient="records"))
    assert any(r["age_years"] is None for r in records + json.loads(raw.to_json(orient="records"))[:200])  # nulls exercise the pinned imputer
    served, distances = [], []
    for i in range(0, 40, 8):  # the release's default batch limit
        out = c.post("/api/serve/local/fitted-km/predict", json={"requestId": f"fit-1-{i}", "records": records[i:i + 8]})
        assert out.status_code == 200, out.text
        served += out.json()["result"]["predictions"]
        distances += out.json()["result"]["distances"]
    X = transformed[model.features].to_numpy(dtype="float64")[:40]
    Xs = model.scaler.transform(X) if model.scaler is not None else X
    assert served == model.estimator.predict(Xs).tolist()
    assert np.allclose(distances, model.estimator.transform(Xs))
    bad = dict(records[0], price_k=1.0)
    assert c.post("/api/serve/local/fitted-km/predict", json={"requestId": "fit-2", "records": [bad]}).json()["error"]["code"] == "E_REQUEST_SCHEMA"
    wrong = dict(records[0], neighborhood=3)
    assert c.post("/api/serve/local/fitted-km/predict", json={"requestId": "fit-3", "records": [wrong]}).json()["error"]["code"] == "E_REQUEST_SCHEMA"
    mon = c.get(f"/api/production/releases/{rel['id']}/monitor").json()
    assert mon["inputDrift"]["neighborhood"]["available"] is not False and set(mon["inputDrift"]) == {"area_m2", "rooms", "age_years", "neighborhood"}


def test_existing_unsupervised_adapter_identity_is_unchanged():
    """Fitted paths live in their own module, so previously registered unsupervised versions keep their implementation hash."""
    from production import unsup_adapter
    assert unsup_adapter.IMPLEMENTATION_FILES == ("production/unsup_adapter.py", "operations/unsup_ops.py", "unsup/methods.py", "tabular/core.py")
