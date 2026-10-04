"""Domain models in the production registry (ADR 0017): register a recorded vision/NLP/speech model, warm up and deploy a pinned release,
serve real native predictions through the shared admission/trace path, replay, label, monitor, and refuse a changed implementation."""
import pytest
from fastapi.testclient import TestClient

from control.app import create_app
from domain import samples
from graph_core.schema import Graph
from test_tabular_api import submit, wait_for

BUILDERS = {"vision": samples.vision_graph, "nlp": samples.nlp_graph, "speech": samples.speech_graph}
META = {"name": "domain-model", "owner": "tests", "intendedUse": "local CPU fixture inference", "limitations": "SYNTHETIC fixture; not a benchmark"}


@pytest.fixture(params=list(BUILDERS), scope="module")
def served(request, tmp_path_factory, domain_fixtures):
    family = request.param
    g = Graph.model_validate(BUILDERS[family](epochs=2))
    g.nodes[0].config["n"] = 12
    if family == "vision":
        g.nodes[-1].config["width"] = 4
    if family == "nlp":
        g.nodes[-1].config.update(hidden=8, embedding=8)
    if family == "speech":
        g.nodes[-1].config["hidden"] = 8
    with TestClient(create_app(tmp_path_factory.mktemp(f"wb-{family}"))) as c:
        r = submit(c, g)
        assert r.status_code == 201, r.text
        rid = r.json()["runId"]
        assert wait_for(c, rid)["status"] == "completed"
        model = c.get("/api/domain/models", params={"runId": rid}).json()["models"][0]
        yield c, family, model


def deploy(c, model, namespace, **config):
    v = c.post("/api/production/versions", json={"runId": model["runId"], "node": model["node"], **META})
    assert v.status_code == 201, v.text
    rel = c.post("/api/production/releases", json={"versionId": v.json()["id"], "config": {"namespace": namespace, "maxBatch": 4, **config}})
    assert rel.status_code == 201, rel.text
    d = c.post(f"/api/production/releases/{rel.json()['id']}/deploy", json={"expectedCurrent": None})
    assert d.status_code == 200, d.text
    return v.json(), rel.json()


def test_register_release_serve_and_trace_match_native_inference(served):
    c, family, model = served
    v, rel = deploy(c, model, f"{family}-a", captureInputs=True)
    assert v["adapter"] == "domain" and v["family"] == family and v["modelId"] == model["modelId"]
    assert v["manifest"]["checkpointSha256"] == model["checkpointSha256"] and v["manifest"]["runId"] == model["runId"]
    assert rel["compatibility"]["ok"] and "PyTorch" in rel["adapter"]
    assert any(cand["adapter"] == "domain" and cand["family"] == family for cand in c.get("/api/production").json()["candidates"])
    ref = c.get(f"/api/production/versions/{v['id']}/reference-input").json()
    assert ref["family"] == family and ref["provenance"]["partition"] == "recorded held-out example"
    records = ref["records"]
    health = c.get(f"/api/serve/local/{family}-a/health").json()
    assert health["ready"] and health["releaseId"] == rel["id"]

    out = c.post(f"/api/serve/local/{family}-a/predict", json={"requestId": "r1", "records": records})
    assert out.status_code == 200, out.text
    t = out.json()
    native = c.post(f"/api/domain/models/{model['modelId']}/predict", json={"records": records}).json()["predictions"]
    assert t["result"]["predictions"] == native                      # the release serves exactly the pinned native model
    assert t["lineage"]["modelSha256"] == model["checkpointSha256"] and t["lineage"]["runId"] == model["runId"] and t["versionId"] == v["id"]
    assert c.post(f"/api/serve/local/{family}-a/predict", json={"requestId": "r1", "records": records}).json()["idempotentReplay"] is True
    rep = c.post("/api/production/requests/r1/replay", json={}).json()
    assert rep["result"]["predictions"] == native and "no route" in rep["mode"]


def test_bounds_schema_and_release_limits(served):
    c, family, model = served
    v, _ = deploy(c, model, f"{family}-b")
    assert c.post("/api/production/releases", json={"versionId": v["id"], "config": {"namespace": "x", "maxBatch": 8}}).status_code == 422
    rec = c.get(f"/api/production/versions/{v['id']}/reference-input").json()["records"][0]
    too_many = c.post(f"/api/serve/local/{family}-b/predict", json={"requestId": "big", "records": [rec] * 5})
    assert too_many.status_code == 413 and too_many.json()["error"]["code"] == "E_REQUEST_BOUNDS"
    bad = c.post(f"/api/serve/local/{family}-b/predict", json={"requestId": "bad", "records": [{"unexpected": 1}]})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "E_REQUEST_SCHEMA"
    no_replay = c.post(f"/api/serve/local/{family}-b/predict", json={"requestId": "ok", "records": [rec]})
    assert no_replay.status_code == 200 and no_replay.json()["records"] is None   # capture defaults off
    assert c.post("/api/production/requests/ok/replay", json={}).status_code == 409


def test_labels_drive_native_quality_and_monitoring(served):
    c, family, model = served
    v, rel = deploy(c, model, f"{family}-c", captureInputs=True)
    rec = c.get(f"/api/production/versions/{v['id']}/reference-input").json()["records"][0]
    t = c.post(f"/api/serve/local/{family}-c/predict", json={"requestId": "q1", "records": [rec, rec]}).json()
    preds = t["result"]["predictions"]
    if family == "nlp":
        good = [[w["label"] for w in p["words"]] for p in preds]
        bad = [["O"] * len(p["words"]) for p in preds]
    elif family == "speech":
        good = [p["text"] for p in preds]
        bad = [p["text"] + "zz" for p in preds]
    else:
        from production.domain_adapter import mask_of
        good = [mask_of(p).astype(int).tolist() for p in preds]
        bad = [[[1 if v == 0 else 0 for v in row] for row in g] for g in good]  # differs at every pixel
    assert c.post("/api/production/requests/q1/labels", json={"labels": [["wrong"]] * 2 if family != "speech" else [1, 2]}).status_code == 422
    assert c.post("/api/production/requests/q1/labels", json={"labels": good[:1]}).json()["detail"]["code"] == "E_LABEL_ALIGNMENT"
    assert c.post("/api/production/requests/q1/labels", json={"labels": good}).status_code == 200
    assert c.post("/api/production/requests/q1/labels", json={"labels": bad}).status_code == 409    # ground truth is immutable
    mon = c.get(f"/api/production/releases/{rel['id']}/monitor").json()
    q = mon["labelBasedQuality"]
    assert mon["family"] == family and q["available"] and q["labelledRecords"] == 2
    if family == "nlp":
        assert q["values"]["wordAccuracy"] == 1.0
        has_spans = any(v != "O" for g in good for v in g)
        assert (q["values"]["spanMicroF1"] == 1.0) if has_spans else (q["values"]["spanMicroF1"] is None and "undefined" in q["values"]["spanMicroF1Note"])
    elif family == "speech":
        assert q["values"]["cer"] == 0.0
    else:
        assert q["values"]["pixelAccuracy"] == 1.0
    assert all(d["available"] for d in mon["inputDrift"].values()) and mon["predictionDrift"]["available"]
    assert mon["predictionDrift"]["totalVariation"] == 0.0          # identical to the reference example's own predictions
    assert mon["health"]["requests"] == 1 and mon["health"]["errors"] == 0

    t2 = c.post(f"/api/serve/local/{family}-c/predict", json={"requestId": "q2", "records": [rec]}).json()
    assert c.post("/api/production/requests/q2/labels", json={"labels": bad[:1]}).status_code == 200
    q2 = c.get(f"/api/production/releases/{rel['id']}/monitor").json()["labelBasedQuality"]
    assert q2["labelledRecords"] == 3
    if family == "speech":
        from speech.ctc import corpus_rates
        expect = corpus_rates([(g, p["text"]) for g, p in zip(good + bad[:1], preds + t2["result"]["predictions"])])["cer"]["rate"]
        assert q2["values"]["cer"] == pytest.approx(expect) and expect > 0
    elif family == "nlp":
        p3 = t2["result"]["predictions"][0]
        hits = sum(len(g) for g in good) + sum(w["label"] == "O" for w in p3["words"])
        assert q2["values"]["wordAccuracy"] == pytest.approx(hits / (sum(len(g) for g in good) + len(p3["words"])))
    else:
        assert q2["values"]["pixelAccuracy"] == pytest.approx(2 / 3)   # two exact masks and one wrong at every pixel


def test_changed_implementation_refuses_serving(served, monkeypatch):
    c, family, model = served
    deploy(c, model, f"{family}-d")
    from domain import checkpoints as CP
    real = CP.implementation
    monkeypatch.setattr(CP, "implementation", lambda: {**real(), "domain/inference.py": "0" * 64})
    r = c.get(f"/api/serve/local/{family}-d/health")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "E_DOMAIN_CHECKPOINT_ENVIRONMENT"


def test_nlp_span_f1_is_native_with_spans_and_undefined_without():
    from seqeval.metrics import f1_score

    from production.domain_adapter import quality

    p = lambda labels: {"words": [{"label": v} for v in labels]}
    truth, pred = ["B-ORG", "I-ORG", "O", "B-LOC"], ["B-ORG", "I-ORG", "O", "O"]
    q = quality("nlp", [(p(pred), truth)], 0)
    assert q["spanMicroF1"] == pytest.approx(f1_score([truth], [pred])) and q["wordAccuracy"] == 0.75
    none = quality("nlp", [(p(["O", "O"]), ["O", "O"])], 0)
    assert none["spanMicroF1"] is None and none["wordAccuracy"] == 1.0
