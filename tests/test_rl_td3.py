"""TD3 for continuous actions (ADR 0068): validation, a real short worker run on Pendulum-v1, artifacts and refusals."""
import io
import json

import pytest
import torch

from artifact_store import ArtifactStore
from graph_core.schema import Graph
from graph_core.validate import validate
from rl import samples
from rl.td3 import Actor
from worker.rl_run import RLRunConfig, run_rl

SHORT = {"total_steps": 600, "learning_starts": 200, "batch_size": 64, "eval_every": 300, "hidden": [32, 32]}


def graph(**kw):
    return Graph.model_validate(samples.pendulum_td3_graph(td3={**SHORT, **kw.pop("td3", {})}, evaluation={"seeds": [1000, 1001]}, **kw))


def run(tmp_path, g, **cfg):
    store = ArtifactStore(tmp_path)
    store.create_run("r1", "g", {"kind": "rl"})
    return store, run_rl(g, RLRunConfig(seed=0, **cfg), store, "r1")


def test_td3_graph_validates_and_refusals():
    assert not [d for d in validate(graph()).diagnostics if d.severity == "error"]
    cart = samples.pendulum_td3_graph(td3=SHORT)
    cart["nodes"][1]["config"]["env_id"] = "CartPole-v1"
    codes = {d.code for d in validate(Graph.model_validate(cart)).diagnostics if d.severity == "error"}
    assert "E_RL_ALGO_ACTION_SPACE" in codes
    budget = {d.code for d in validate(graph(td3={"total_steps": 100, "learning_starts": 200})).diagnostics if d.severity == "error"}
    assert "E_RL_BUDGET" in budget
    mixed = samples.pendulum_td3_graph(td3=SHORT)
    mixed["nodes"].append({"id": "replay", "type": "rl.replay_buffer", "version": "1.0.0", "config": {}})
    mixed["edges"].append({"id": "e9", "kind": "env", "from": {"node": "env", "port": "env"}, "to": {"node": "replay", "port": "env"}})
    assert "E_RL_NODE_COUNT" in {d.code for d in validate(Graph.model_validate(mixed)).diagnostics}


def test_short_real_td3_run_records_evaluations_checkpoints_and_bounded_actions(tmp_path):
    store, status = run(tmp_path, graph())
    assert status == "completed", store.get_run("r1")["error"]
    started = store.last_event("r1", "run_started")["data"]
    assert started["algorithm"] == "TD3" and started["totalSteps"] == 600
    report = json.loads(store.read_artifact(store.artifacts("r1", "rl_eval_report")[-1]["sha256"]))
    assert report["algorithm"] == "TD3" and report["final"]["final"] and len(report["final"]["episodes"]) == 2
    assert all(e["length"] == 200 and e["truncated"] and not e["terminated"] for e in report["final"]["episodes"])  # Pendulum: time limit only
    summary = json.loads(store.read_artifact(store.artifacts("r1", "rl_summary")[-1]["sha256"]))
    assert summary["gradientUpdates"] == 400 and summary["actorUpdates"] == 200 and summary["episodes"] == 3
    assert [e["data"]["tick"] for e in store.events("r1", types=("eval",))] == [300, 600]
    assert len(store.events("r1", types=("episode_end",))) == 3 and store.events("r1", types=("train_update",))
    ckpt = torch.load(io.BytesIO(store.read_artifact(store.artifacts("r1", "rl_checkpoint")[-1]["sha256"])), weights_only=True)
    actor = Actor(ckpt["obsDim"], ckpt["actDim"], ckpt["hidden"], torch.tensor(ckpt["low"]).numpy(), torch.tensor(ckpt["high"]).numpy())
    actor.load_state_dict(ckpt["actor"])
    with torch.no_grad():
        a = actor(torch.randn(64, 3) * 10)
    assert a.shape == (64, 1) and float(a.abs().max()) <= 2.0 + 1e-6  # Pendulum torque bounds [-2, 2]


def test_cuda_device_is_refused_without_cuda(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA")
    store, status = run(tmp_path, graph(), device="cuda")
    assert status == "failed" and "E_DEVICE_UNAVAILABLE" in store.get_run("r1")["error"]


def test_td3_run_is_readable_through_the_rl_api(tmp_path):
    from fastapi.testclient import TestClient
    from control.app import create_app
    store, status = run(tmp_path / "wb", graph())
    assert status == "completed"
    with TestClient(create_app(tmp_path / "wb")) as c:
        cur = c.get("/api/rl/runs/r1/curves")
        assert cur.status_code == 200, cur.text
        body = cur.json()
        assert len(body["episodes"]) == 3 and [e["tick"] for e in body["evals"]] == [300, 600] and body["updates"] and body["totalSteps"] == 600
