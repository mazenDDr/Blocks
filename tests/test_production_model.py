"""Model-graph image classifiers in the production registry (ADR 0018): register the reference CNN's latest complete checkpoint, freeze a
held-out reference only when the source image folder still matches its recorded identity, serve native predictions equal to the run's own
checkpoint inference, label/monitor, and refuse changed sources or implementations."""
import base64
import shutil

import pytest
from fastapi.testclient import TestClient

from control.app import create_app
from test_api import submit, wait_for

META = {"name": "reference-cnn", "owner": "tests", "intendedUse": "local CPU fixture classification", "limitations": "SYNTHETIC shapes; not a benchmark"}


@pytest.fixture(scope="module")
def cnn(tmp_path_factory, shapes_dir):
    data = tmp_path_factory.mktemp("shapes") / "shapes10"
    shutil.copytree(shapes_dir, data)  # a private copy so a test can change the source
    with TestClient(create_app(tmp_path_factory.mktemp("wb"))) as c:
        r = submit(c, data, epochs=2, seed=3)
        assert r.status_code == 201, r.text
        rid = r.json()["runId"]
        assert wait_for(c, rid)["status"] == "completed"
        yield c, rid, data


def register(c, rid):
    listed = [x for x in c.get("/api/production").json()["candidates"] if x["runId"] == rid]
    assert {x["adapter"] for x in listed} == {"model", "model_keras", "model_jax"}  # ADR 0061 adds the same checkpoint on Keras/JAX
    cand = [x for x in listed if x["adapter"] == "model"]
    assert len(cand) == 1 and cand[0]["family"] == "image_classifier"
    v = c.post("/api/production/versions", json={"runId": rid, "node": cand[0]["node"], **META})
    assert v.status_code == 201, v.text
    return v.json()


def test_register_serve_and_match_checkpoint_inference(cnn):
    c, rid, _ = cnn
    v = register(c, rid)
    m = v["manifest"]
    store = c.app.state.services.store
    latest = max((x for x in store.artifacts(rid, "checkpoint") if x["status"] == "complete"), key=lambda x: (x["step"], x["id"]))
    assert v["adapter"] == "model" and m["checkpointSha256"] == latest["sha256"] and m["inputSize"] == [64, 64]
    assert m["referenceRows"] > 0 and len(m["outputSchema"]["classes"]) == 10
    rel = c.post("/api/production/releases", json={"versionId": v["id"], "config": {"namespace": "cnn", "captureInputs": True}})
    assert rel.status_code == 201, rel.text
    assert c.post(f"/api/production/releases/{rel.json()['id']}/deploy", json={"expectedCurrent": None}).status_code == 200
    ref = c.get(f"/api/production/versions/{v['id']}/reference-input").json()
    assert ref["family"] == "image_classifier" and len(ref["records"]) == len(ref["observedLabels"]) == 3
    out = c.post("/api/serve/local/cnn/predict", json={"requestId": "c1", "records": ref["records"]})
    assert out.status_code == 200, out.text
    t = out.json()
    for i, rec in enumerate(ref["records"]):   # same image through the research inference path of the same checkpoint
        native = c.post("/api/infer", json={"runId": rid, "checkpointStep": latest["step"], "imageBase64": rec["imagePng"]}).json()
        assert t["result"]["predictions"][i] == native["predictedName"]
        assert t["result"]["probabilities"][i] == pytest.approx(native["probabilities"], abs=1e-6)
    assert t["lineage"]["modelSha256"] == latest["sha256"] and t["lineage"]["runId"] == rid
    assert c.post("/api/production/requests/c1/replay", json={}).json()["result"]["predictions"] == t["result"]["predictions"]

    bad = c.post("/api/serve/local/cnn/predict", json={"requestId": "c2", "records": [{"imagePng": base64.b64encode(b"not an image").decode()}]})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "E_REQUEST_SCHEMA"
    assert c.post("/api/serve/local/cnn/predict", json={"requestId": "c3", "records": ref["records"] * 11}).status_code == 413   # 33 > 32

    assert c.post("/api/production/requests/c1/labels", json={"labels": ["not-a-class"] * 3}).status_code == 422
    assert c.post("/api/production/requests/c1/labels", json={"labels": ref["observedLabels"]}).status_code == 200
    mon = c.get(f"/api/production/releases/{rel.json()['id']}/monitor").json()
    expect = sum(p == y for p, y in zip(t["result"]["predictions"], ref["observedLabels"])) / 3
    assert mon["family"] == "image_classifier" and mon["labelBasedQuality"]["values"]["accuracy"] == pytest.approx(expect)
    assert mon["inputDrift"]["meanIntensity"]["available"] and mon["predictionDrift"]["available"]
    assert 0.0 <= mon["reference"]["referenceAccuracy"] <= 1.0 and mon["health"]["errors"] == 2


def test_changed_source_folder_refuses_registration(cnn):
    c, rid, data = cnn
    first = next(p for p in sorted(data.rglob("*.png")))
    original = first.read_bytes()
    try:
        first.write_bytes(original + b"\0")       # one changed byte in the training folder
        node = next(x for x in c.get("/api/production").json()["candidates"] if x["runId"] == rid)["node"]
        r = c.post("/api/production/versions", json={"runId": rid, "node": node, **META, "name": "changed"})
        assert r.status_code == 409 and r.json()["detail"]["code"] == "E_REFERENCE_SOURCE"
    finally:
        first.write_bytes(original)


def test_serving_never_rereads_the_source_and_refuses_changed_implementation(cnn, monkeypatch, tmp_path):
    c, rid, data = cnn
    v = register(c, rid)
    rel = c.post("/api/production/releases", json={"versionId": v["id"], "config": {"namespace": "cnn2"}}).json()
    c.post(f"/api/production/releases/{rel['id']}/deploy", json={"expectedCurrent": None})
    rec = c.get(f"/api/production/versions/{v['id']}/reference-input").json()["records"][:1]
    moved = tmp_path / "moved"
    shutil.move(str(data), moved)
    try:
        assert c.post("/api/serve/local/cnn2/predict", json={"requestId": "s1", "records": rec}).status_code == 200   # source folder absent
    finally:
        shutil.move(str(moved), data)
    from production import model_adapter as MA
    real = MA.implementation
    monkeypatch.setattr(MA, "implementation", lambda: {**real(), "graph_core/lower.py": "0" * 64})
    h = c.get("/api/serve/local/cnn2/health")
    assert h.status_code == 409 and h.json()["detail"]["code"] == "E_SERVING_ENVIRONMENT"
