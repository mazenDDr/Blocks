"""PyTorch-trained image classifier served on Keras/JAX (ADR 0061): measured agreement gate, equal predictions, pinned versions."""
import pytest

from production import portable_model_adapter as portable
from production.pipeline import ProductionError
from test_production_model import META, cnn, register  # noqa: F401  (module fixture: a real trained reference CNN run)


def deploy(c, vid, ns):
    rel = c.post("/api/production/releases", json={"versionId": vid, "config": {"namespace": ns, "captureInputs": True}})
    assert rel.status_code == 201, rel.text
    assert c.post(f"/api/production/releases/{rel.json()['id']}/deploy", json={"expectedCurrent": None}).status_code == 200
    return rel.json()


@pytest.mark.parametrize("backend", ["keras", "jax"])
def test_registration_measures_agreement_and_serving_matches_pytorch(cnn, backend):  # noqa: F811
    c, rid, _ = cnn
    torch_v = register(c, rid)
    node = f"{backend}:{torch_v['node']}"
    v = c.post("/api/production/versions", json={"runId": rid, "node": node, **META})
    assert v.status_code == 201, v.text
    v = v.json()
    measured = v["manifest"]["portable"]["measuredAgainstPyTorch"]
    print("MEASURED", backend, measured)  # evidence for the ADR: actual differences on this machine
    assert v["adapter"] == f"model_{backend}" and v["family"] == "image_classifier"
    assert measured["images"] == v["manifest"]["referenceRows"] > 0 and measured["maxToleranceExcess"] <= 0 and measured["predictedClassAgreement"]
    assert v["manifest"]["checkpointSha256"] == torch_v["manifest"]["checkpointSha256"]
    deploy(c, torch_v["id"], f"torch-{backend}")
    rel = deploy(c, v["id"], f"portable-{backend}")
    ref = c.get(f"/api/production/versions/{v['id']}/reference-input").json()
    assert ref["family"] == "image_classifier" and len(ref["records"]) == 3
    a = c.post(f"/api/serve/local/torch-{backend}/predict", json={"requestId": f"t-{backend}", "records": ref["records"]}).json()
    b = c.post(f"/api/serve/local/portable-{backend}/predict", json={"requestId": f"p-{backend}", "records": ref["records"]}).json()
    assert b["status"] == 200, b["error"]
    assert b["result"]["backend"] == backend and b["result"]["predictions"] == a["result"]["predictions"]
    for pa, pb in zip(a["result"]["probabilities"], b["result"]["probabilities"]):
        assert pb == pytest.approx(pa, abs=1e-4)
    assert c.post(f"/api/production/requests/p-{backend}/labels", json={"labels": ref["observedLabels"]}).status_code == 200
    mon = c.get(f"/api/production/releases/{rel['id']}/monitor").json()
    assert mon["family"] == "image_classifier" and mon["labelBasedQuality"]["available"]


def test_agreement_gate_refuses_and_version_pins_are_checked(cnn, monkeypatch):  # noqa: F811
    c, rid, _ = cnn
    out = register(c, rid)["node"]
    from backends.tolerances import Tol
    monkeypatch.setattr(portable, "tol", lambda *a: Tol(0.0, 0.0))  # a zero tolerance: real float reassociation must now refuse
    r = c.post("/api/production/versions", json={"runId": rid, "node": f"keras:{out}", **META})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "E_PORTABLE_TOLERANCE"
    monkeypatch.undo()
    v = c.post("/api/production/versions", json={"runId": rid, "node": f"jax:{out}", **META}).json()
    monkeypatch.setattr(portable, "backend_versions", lambda: {**v["manifest"]["portable"]["versions"], "jax": "0.0.0"})
    with pytest.raises(ProductionError) as e:
        portable.verify(c.app.state.services.store, v["manifest"])
    assert e.value.code == "E_SERVING_ENVIRONMENT"
    assert c.post("/api/production/versions", json={"runId": rid, "node": f"onnx:{out}", **META}).status_code in (404, 409, 422)
