"""Continuous TD3 policies served from the registry (ADR 0069): real short Pendulum run, actions equal the run's own actor."""
import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from control.app import create_app
from rl.td3 import Actor
from test_rl_td3 import graph, run

META = {"name": "pendulum-td3", "owner": "tests", "intendedUse": "local continuous-control fixture", "limitations": "600 training steps; not a capable policy"}


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    wb = tmp_path_factory.mktemp("wb")
    store, status = run(wb, graph())
    assert status == "completed"
    with TestClient(create_app(wb)) as c:
        cand = [x for x in c.get("/api/production").json()["candidates"] if x["runId"] == "r1"]
        assert [x["adapter"] for x in cand] == ["rl_td3"]
        v = c.post("/api/production/versions", json={"runId": "r1", "node": cand[0]["node"], **META})
        assert v.status_code == 201, v.text
        rel = c.post("/api/production/releases", json={"versionId": v.json()["id"], "config": {"namespace": "pendulum", "captureInputs": True}})
        assert rel.status_code == 201, rel.text
        assert c.post(f"/api/production/releases/{rel.json()['id']}/deploy", json={"expectedCurrent": None}).status_code == 200
        yield c, store, v.json(), rel.json()


def test_served_actions_equal_the_runs_actor_and_stay_in_bounds(served):
    c, store, v, rel = served
    m = v["manifest"]
    assert v["adapter"] == "rl_td3" and m["actionDim"] == 1 and m["actionLow"] == [-2.0] and m["referenceRows"] > 0
    ref = c.get(f"/api/production/versions/{v['id']}/reference-input").json()
    out = c.post("/api/serve/local/pendulum/predict", json={"requestId": "p1", "records": ref["records"]})
    assert out.status_code == 200, out.text
    ck = torch.load(io.BytesIO(store.read_artifact(m["checkpointSha256"])), weights_only=True)
    actor = Actor(ck["obsDim"], ck["actDim"], ck["hidden"], np.array(ck["low"], np.float32), np.array(ck["high"], np.float32))
    actor.load_state_dict(ck["actor"])
    with torch.no_grad():
        expect = actor(torch.as_tensor([r["observation"] for r in ref["records"]], dtype=torch.float32)).numpy()
    got = np.asarray(out.json()["result"]["predictions"])
    assert np.allclose(got, expect, atol=1e-6) and (np.abs(got) <= 2.0).all()


def test_refusals_labels_and_monitoring(served):
    c, store, v, rel = served
    outside = c.post("/api/serve/local/pendulum/predict", json={"requestId": "p2", "records": [{"observation": [0.0, 0.0, 99.0]}]})
    assert outside.json()["error"]["code"] == "E_REQUEST_SCHEMA"  # angular velocity bound is 8
    wrong = c.post("/api/serve/local/pendulum/predict", json={"requestId": "p3", "records": [{"observation": [0.0, 1.0]}]})
    assert wrong.json()["error"]["code"] == "E_REQUEST_SCHEMA"
    ok = c.post("/api/serve/local/pendulum/predict", json={"requestId": "p4", "records": [{"observation": [1.0, 0.0, 0.0]}]}).json()
    action = ok["result"]["predictions"][0][0]
    assert c.post("/api/production/requests/p4/labels", json={"labels": [[3.0]]}).status_code == 422  # outside the torque bounds
    assert c.post("/api/production/requests/p4/labels", json={"labels": [[0.5]]}).status_code == 200
    mon = c.get(f"/api/production/releases/{rel['id']}/monitor").json()
    assert mon["family"] == "rl_continuous_policy" and set(mon["predictionDrift"]) == {"action[0]"}
    assert mon["labelBasedQuality"]["values"]["actionMeanAbsoluteError"] == pytest.approx(abs(action - 0.5), abs=1e-9)
    assert mon["health"]["schemaErrors"] == 2
