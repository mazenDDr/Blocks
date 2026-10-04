"""Greedy DQN policies in the production registry (ADR 0019): register a completed CartPole run's final Q-network, serve observations with
the exact greedy action of the recorded network, enforce the environment's observation contract, measure agreement only against supplied
actions, monitor observation/action drift against the frozen replay-buffer reference, and refuse a changed implementation."""
import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from control.app import create_app
from rl.networks import build_network
from rl_helpers import tiny_cartpole
from test_rl_api import submit, wait_run

META = {"name": "cartpole-policy", "owner": "tests", "intendedUse": "local CPU greedy policy", "limitations": "tiny CPU training run; not a benchmark"}


@pytest.fixture(scope="module")
def policy(tmp_path_factory):
    with TestClient(create_app(tmp_path_factory.mktemp("wb"))) as c:
        rid = submit(c, tiny_cartpole(), "rl-prod", seed=5)
        assert wait_run(c, rid)["status"] == "completed"
        cand = [x for x in c.get("/api/production").json()["candidates"] if x["runId"] == rid]
        assert len(cand) == 1 and cand[0]["adapter"] == "rl"
        v = c.post("/api/production/versions", json={"runId": rid, "node": cand[0]["node"], **META})
        assert v.status_code == 201, v.text
        rel = c.post("/api/production/releases", json={"versionId": v.json()["id"], "config": {"namespace": "pole", "captureInputs": True, "maxBatch": 64}})
        assert rel.status_code == 201, rel.text
        assert c.post(f"/api/production/releases/{rel.json()['id']}/deploy", json={"expectedCurrent": None}).status_code == 200
        yield c, rid, v.json(), rel.json()


def test_greedy_actions_equal_the_recorded_final_q_network(policy):
    c, rid, v, _ = policy
    m = v["manifest"]
    store = c.app.state.services.store
    final = [a for a in store.artifacts(rid, "rl_checkpoint") if a["meta"].get("purpose") == "final"][-1]
    assert v["adapter"] == "rl" and m["checkpointSha256"] == final["sha256"] and m["observationDim"] == 4 and m["actionSpace"]["n"] == 2
    assert "greedy" in m["policy"] and m["referenceRows"] > 0
    net = build_network(m["network"], 4, 2)
    net.load_state_dict(torch.load(io.BytesIO(store.read_artifact(final["sha256"])), weights_only=True))
    ref = c.get(f"/api/production/versions/{v['id']}/reference-input").json()
    assert ref["family"] == "rl_policy" and ref["observedLabels"] is None and "behaviour policy" in ref["labelNote"]
    obs = [r["observation"] for r in ref["records"]] + [[0.0, 0.1, -0.02, 0.3]]
    t = c.post("/api/serve/local/pole/predict", json={"requestId": "p1", "records": [{"observation": o} for o in obs]}).json()
    with torch.no_grad():
        q = net(torch.tensor(obs, dtype=torch.float32))
    assert t["result"]["predictions"] == q.argmax(1).tolist()
    assert np.allclose(t["result"]["qValues"], q.numpy(), atol=1e-6) and t["lineage"]["modelSha256"] == final["sha256"]


@pytest.mark.parametrize("obs,why", [([0.0, 0.0, 0.0], "exactly"), ([0.0, 0.0, 9.0, 0.0], "outside"), ([0.0, float("nan"), 0.0, 0.0], None)])
def test_observation_contract(policy, obs, why):
    c, *_ = policy
    if why is None:   # NaN is not valid JSON for the request model at all
        r = c.post("/api/serve/local/pole/predict", content=b'{"requestId": "nan", "records": [{"observation": [0, NaN, 0, 0]}]}',
                   headers={"Content-Type": "application/json"})
        assert r.status_code == 422
        return
    r = c.post("/api/serve/local/pole/predict", json={"requestId": f"bad-{why}", "records": [{"observation": obs}]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "E_REQUEST_SCHEMA" and why in r.json()["error"]["message"]


def test_agreement_only_from_supplied_actions_and_monitoring(policy):
    c, _, v, rel = policy
    recs = c.get(f"/api/production/versions/{v['id']}/reference-input").json()["records"]
    t = c.post("/api/serve/local/pole/predict", json={"requestId": "m1", "records": recs}).json()
    mon = c.get(f"/api/production/releases/{rel['id']}/monitor").json()
    assert mon["labelBasedQuality"]["available"] is False and "No reference actions" in mon["labelBasedQuality"]["reason"]
    assert c.post("/api/production/requests/m1/labels", json={"labels": [True] * len(recs)}).status_code == 422   # booleans are not actions
    assert c.post("/api/production/requests/m1/labels", json={"labels": [5] * len(recs)}).status_code == 422
    supplied = [1 - a for a in t["result"]["predictions"][:1]] + t["result"]["predictions"][1:]   # first one disagrees
    assert c.post("/api/production/requests/m1/labels", json={"labels": supplied}).status_code == 200
    q = c.get(f"/api/production/releases/{rel['id']}/monitor").json()
    assert q["family"] == "rl_policy" and q["labelBasedQuality"]["values"]["actionAgreement"] == pytest.approx((len(recs) - 1) / len(recs))
    assert q["inputDrift"]["obs[0]"]["available"] and q["predictionDrift"]["available"] and q["reference"]["evaluation"] is not None


def test_changed_implementation_refuses_serving(policy, monkeypatch):
    c, *_ = policy
    from production import rl_adapter as RA
    real = RA.implementation
    monkeypatch.setattr(RA, "implementation", lambda: {**real(), "rl/networks.py": "0" * 64})
    h = c.get("/api/serve/local/pole/health")
    assert h.status_code == 409 and h.json()["detail"]["code"] == "E_SERVING_ENVIRONMENT"
