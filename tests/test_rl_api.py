"""RL over the control API: catalog, validation codes, runs through the worker process, rollout / buffer / trace inspection, studies with variants (A51, A54)."""
from __future__ import annotations

import copy
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from control.app import create_app
from rl import samples
from rl_helpers import tiny_cartpole, tiny_grid


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with TestClient(create_app(tmp_path_factory.mktemp("wb"))) as c:
        yield c


def wait_run(c, rid, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        r = c.get(f"/api/runs/{rid}").json()
        if r["status"] in ("completed", "failed", "cancelled"):
            return r
        time.sleep(0.2)
    raise AssertionError(f"run {rid} still {r['status']}")


def submit(c, graph, key, **cfg):
    r = c.post("/api/runs", json={"graph": graph.to_json(), "config": cfg}, headers={"Idempotency-Key": key})
    assert r.status_code == 201, r.text
    return r.json()["runId"]


@pytest.fixture(scope="module")
def done_run(client):
    g = tiny_grid(900)
    g.node("learner").config["trace"] = {"max_episodes": 3, "max_steps_per_episode": 25, "frame_width": 192}
    rid = submit(client, g, "rl-run-1", seed=3)
    r = wait_run(client, rid)
    assert r["status"] == "completed", r
    return rid


# ------------------------------------------------------------------------------------------------ catalog and validation
def test_catalog_lists_tested_environments_with_real_spaces(client):
    c = client.get("/api/rl/catalog").json()
    by = {e["id"]: e for e in c["entries"]}
    assert c["gymnasium"] == "1.3.0" and {"CartPole-v1", "Void/GridWorld-v0", "Pendulum-v1"} <= by.keys()
    cp = by["CartPole-v1"]
    assert cp["status"] == "verified" and cp["observationSpace"]["shape"] == [4] and cp["actionSpace"]["n"] == 2 and cp["timeLimit"] == 500
    assert cp["termination"] and cp["truncation"] and cp["license"] and cp["snapshot"] and cp["seeding"]
    assert by["Pendulum-v1"]["status"] == "space contract only"
    ops = {o["type"]: o for o in client.get("/api/registry").json()["ops"]}
    assert {"rl.environment", "rl.q_network", "rl.dqn_learner", "rl.replay_buffer", "rl.reward", "rl.evaluation"} <= ops.keys()
    assert ops["rl.dqn_learner"]["graphKind"] == "rl" and ops["rl.dqn_learner"]["inputKinds"] == {"network": "network", "buffer": "buffer"}


def test_validation_shows_typed_spaces_network_and_the_update_equation(client):
    v = client.post("/api/validate", json={"graph": tiny_cartpole().to_json()}).json()
    assert v["ok"] and v["graphKind"] == "rl"
    rl = v["rl"]
    assert rl["environment"]["observationSpace"]["type"] == "Box" and rl["environment"]["actionSpace"]["type"] == "Discrete"
    assert rl["network"]["nActions"] == 2 and rl["network"]["params"] == 4 * 64 + 64 + 64 * 64 + 64 + 64 * 2 + 2
    assert rl["learner"]["equation"]["targetEquation"].startswith("y_i = r_i + 0.99 * (1 - terminated_i)")
    assert "truncated does NOT" in rl["learner"]["equation"]["boundaries"]
    assert v["nodes"]["qnet"]["params"] == rl["network"]["params"]


@pytest.mark.parametrize("env_id,code", [("Pendulum-v1", "E_RL_ALGO_ACTION_SPACE"), ("FrozenLake-v1", "E_RL_OBS_SPACE"), ("CartPole-v9", "E_RL_ENV_VERSION"), ("Taxi-v4", "E_RL_ENV_NOT_CATALOGED")])
def test_incompatible_environment_is_rejected_before_execution_with_stable_codes(client, env_id, code):
    g = tiny_cartpole()
    g.node("env").config["env_id"] = env_id
    v = client.post("/api/validate", json={"graph": g.to_json()}).json()
    assert not v["ok"] and code in {d["code"] for d in v["diagnostics"]}
    r = client.post("/api/runs", json={"graph": g.to_json(), "config": {}}, headers={"Idempotency-Key": f"bad-{env_id}"})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "execution_blocked" and code in {d["code"] for d in r.json()["detail"]["diagnostics"]}


def test_network_must_match_the_spaces_and_offers_a_repair(client):
    g = tiny_cartpole()
    g.node("qnet").config["network"] = client.post("/api/rl/network/mlp", json={"obsDim": 5, "hidden": [8], "nActions": 2}).json()["network"]
    v = client.post("/api/validate", json={"graph": g.to_json()}).json()
    d = next(d for d in v["diagnostics"] if d["code"] == "E_RL_NETWORK_INPUT")
    assert d["nodeId"] == "qnet" and "expected [\"N\", 4]" in d["message"] and d["fixes"][0]["key"] == "network"
    g.node("qnet").config["network"] = client.post("/api/rl/network/mlp", json={"obsDim": 4, "hidden": [8], "nActions": 3}).json()["network"]
    assert "E_RL_NETWORK_OUTPUT" in {d["code"] for d in client.post("/api/validate", json={"graph": g.to_json()}).json()["diagnostics"]}


def test_buffer_budget_reward_component_and_node_count_rules(client):
    def codes(g):
        return {d["code"] for d in client.post("/api/validate", json={"graph": g.to_json()}).json()["diagnostics"]}
    g = tiny_cartpole(); g.node("replay").config["capacity"] = 10
    assert "E_RL_BUFFER_CAPACITY" in codes(g)
    g = tiny_cartpole(300); g.node("learner").config["learning_starts"] = 500
    assert "E_RL_BUDGET" in codes(g)
    g = tiny_grid(); g.node("reward").config["components"] = [{"name": "nonsense", "weight": 1}]
    assert "E_RL_REWARD_COMPONENT" in codes(g)
    g = tiny_cartpole(); g.nodes = [n for n in g.nodes if n.id != "evaluation"]; g.edges = [e for e in g.edges if e.to.node != "evaluation"]
    assert "E_RL_NODE_COUNT" in codes(g)
    g = tiny_grid(); g.node("env").config["kwargs"] = {**g.node("env").config["kwargs"], "start": [2, 0]}   # start on a wall
    assert "E_RL_ENV_CONFIG" in codes(g)


def test_modified_reward_is_flagged_as_a_warning(client):
    g = tiny_grid(); g.node("reward").config["components"] = [{"name": "step", "weight": 0.2}]
    w = [d for d in client.post("/api/validate", json={"graph": g.to_json()}).json()["diagnostics"] if d["severity"] == "warning"]
    assert any(d["code"] == "W_RL_REWARD_MODIFIED" for d in w)


def test_grid_builder_preview_returns_a_schematic_and_a_real_render(client):
    r = client.post("/api/rl/grid/preview", json={"kwargs": {"width": 4, "height": 3, "walls": [[1, 1]], "goal": [3, 2]}}).json()
    assert r["schematic"] == [["S", ".", ".", "."], [".", "#", ".", "."], [".", ".", ".", "G"]] and r["width"] == 4
    import base64
    assert base64.b64decode(r["framePng"])[:8] == b"\x89PNG\r\n\x1a\n" and "real render" in r["frameSource"]
    assert client.post("/api/rl/grid/preview", json={"kwargs": {"width": 3, "height": 3, "goal": [9, 9]}}).status_code == 422


# ------------------------------------------------------------------------------------------------ a real run through the worker
def test_run_records_unsmoothed_curves_setup_and_summary(client, done_run):
    s = client.get(f"/api/runs/{done_run}").json()
    assert s["kind"] == "rl" and s["status"] == "completed" and s["envSteps"] > 0 and s["updates"] > 0 and s["lastEval"]["final"]
    c = client.get(f"/api/rl/runs/{done_run}/curves").json()
    assert c["setup"]["autoresetMode"] == "NextStep" and c["setup"]["components"] == ["goal", "step", "bump"] and c["setup"]["effectiveWeights"]["step"] == 0.01
    assert [w["name"] for w in c["setup"]["wrapperChain"]][:3] == ["RewardComponents", "TimeLimit", "OrderEnforcing"]
    assert c["episodes"] and all(e["terminated"] != e["truncated"] for e in c["episodes"]) and c["updates"] and len(c["evals"]) == 2 and c["evals"][0]["tick"] == 0
    inc = client.get(f"/api/rl/runs/{done_run}/curves", params={"after": c["episodes"][2]["seq"]}).json()
    assert [e["episodeId"] for e in inc["episodes"]] == [e["episodeId"] for e in c["episodes"][3:]]    # incremental polling


def test_rollout_viewer_data_has_real_frames_for_captured_episodes(client, done_run):
    eps = client.get(f"/api/rl/runs/{done_run}/episodes").json()
    assert "BOUNDED CAPTURE" in eps["note"] and len(eps["episodes"]) == 3
    e = client.get(f"/api/rl/runs/{done_run}/episodes/{eps['episodes'][0]['sha256']}").json()
    assert len(e["frames"]) == len(e["steps"]) + 1 and e["steps"][0]["tid"] == 0 and len(e["steps"][0]["qValues"]) == 4
    png = client.get(f"/api/rl/runs/{done_run}/frames/{e['frames'][1]}.png")
    assert png.status_code == 200 and png.content[:8] == b"\x89PNG\r\n\x1a\n" and png.headers["content-type"] == "image/png"
    from PIL import Image
    import io
    assert Image.open(io.BytesIO(png.content)).size == (192, 160)
    assert client.get(f"/api/rl/runs/{done_run}/frames/{'0' * 64}.png").status_code == 404
    other = client.get(f"/api/rl/runs/{done_run}/frames/{eps['episodes'][0]['sha256']}.png")   # an episode json is not a frame of this run
    assert other.status_code == 404


def test_replay_buffer_browser_filters_sorts_and_reports_stats(client, done_run):
    b = client.get(f"/api/rl/runs/{done_run}/buffer", params={"limit": 5}).json()
    st = b["buffer"]
    assert st["size"] == min(5000, st["transitionsInserted"]) and b["total"] == st["size"] and len(b["rows"]) == 5 and b["rows"][0]["tid"] == 0
    assert "uniform" in st["samplingWeight"] and st["sampleCounts"]["max"] >= 1
    ends = client.get(f"/api/rl/runs/{done_run}/buffer", params={"flag": "ending", "limit": 200}).json()
    assert ends["total"] == st["terminated"] + st["truncated"] and all(r["terminated"] or r["truncated"] for r in ends["rows"])
    rnd = client.get(f"/api/rl/runs/{done_run}/buffer", params={"source": "random", "sort": "sampleCount", "order": "desc", "limit": 10}).json()
    assert all(r["actionSource"] == "random" for r in rnd["rows"]) and [r["sampleCount"] for r in rnd["rows"]] == sorted((r["sampleCount"] for r in rnd["rows"]), reverse=True)
    assert set(b["rows"][0]["components"]) == {"goal", "step", "bump"} and "reward" in b["rows"][0]
    assert client.get(f"/api/rl/runs/{done_run}/buffer", params={"sort": "nope"}).status_code == 422


def test_a51_transition_trace_over_the_api(client, done_run):
    ov = client.get(f"/api/rl/runs/{done_run}/trace").json()
    assert "BOUNDED CAPTURE" in ov["capturePolicy"]["statement"] and ov["tracked"]
    used = max(ov["tracked"], key=lambda r: r["uses"])
    assert used["uses"] > 0
    t = client.get(f"/api/rl/runs/{done_run}/transitions/{used['tid']}").json()
    assert t["captured"] and [c["stage"] for c in t["chain"]] == ["transition", "buffer insertion", "minibatches", "objective", "policy version"]
    u = t["uses"][0]
    tr = t["transition"]
    assert u["target"] == pytest.approx(tr["reward"] + 0.99 * u["bootstrapMask"] * u["nextValue"], abs=1e-5)
    assert u["bootstrapMask"] == (0.0 if tr["terminated"] else 1.0) and u["policyVersionAfter"] == u["update"] and (u["minibatch"] is None or tr["tid"] in u["minibatch"]["batchTids"])
    # a transition in the final buffer that was NOT tracked says so instead of inventing a chain
    b = client.get(f"/api/rl/runs/{done_run}/buffer", params={"limit": 1, "offset": 700}).json()["rows"][0]
    nt = client.get(f"/api/rl/runs/{done_run}/transitions/{b['tid']}").json()
    assert nt["captured"] is False and "bounded capture" in nt["reason"].lower() and nt["finalBuffer"]["sampleCount"] == b["sampleCount"]
    assert client.get(f"/api/rl/runs/{done_run}/transitions/99999999").json()["captured"] is False


def test_eval_report_over_the_api(client, done_run):
    e = client.get(f"/api/rl/runs/{done_run}/eval").json()
    f = e["final"]
    assert e["evaluationSeeds"] == list(range(1000, 1008)) and f["taskReturn"]["n"] == 8 and f["terminatedCount"] + f["truncatedCount"] == 8 and f["capturedEpisodes"][0]["frames"]
    assert set(f["componentReturns"]) == {"goal", "step", "bump"} and f["successRule"] and "rewardBasis" in f
    assert e["history"][0]["tick"] == 0 and e["history"][-1]["final"]


def test_rl_run_can_be_cancelled(client):
    g = tiny_cartpole(200000)
    g.node("learner").config["trace"] = {"enabled": False}
    rid = submit(client, g, "rl-cancel")
    t0 = time.time()
    while time.time() - t0 < 60 and client.get(f"/api/runs/{rid}").json()["status"] != "running":
        time.sleep(0.2)
    assert client.post(f"/api/runs/{rid}/cancel").status_code in (200, 202)
    assert wait_run(client, rid)["status"] == "cancelled"


# ------------------------------------------------------------------------------------------------ A54 variants as study trials
def test_a54_reward_variants_with_seeds_become_trials_and_a_comparison(client):
    g = tiny_grid(500)
    g.node("learner").config["trace"] = {"enabled": False}
    body = {"name": "step penalty ablation", "hypothesis": "a stronger step penalty shortens episodes", "graph": g.to_json(), "run_config": {},
            "objective": {"metric": {"name": "eval_task_return_mean"}, "direction": "maximize"},
            "search": {"method": "grid", "variables": [{"target": {"scope": "node", "node": "reward", "field": "components"},
                                                       "values": [[{"name": "step", "weight": 0.01}], [{"name": "step", "weight": 0.0}]], "labels": ["step penalty 0.01 (default)", "no step penalty"]}]},
            "repeats": {"seeds": [0, 1]}, "limits": {"max_trials": 10}, "include_baseline": False}
    r = client.post("/api/studies", json=body)
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    end = time.time() + 240
    while time.time() < end:
        s = client.get(f"/api/studies/{sid}").json()
        if s["state"] not in ("running", "planned"):
            break
        time.sleep(0.5)
    assert s["state"] == "completed", s["state"]
    assert s["objective"]["semantics"].startswith("final greedy evaluation") and len(s["trials"]) == 4 and {t["seed"] for t in s["trials"]} == {0, 1}
    assert all(t["metric"]["available"] and t["metric"]["partition"] == "evaluation" for t in s["trials"])
    assert {g_["label"] for g_ in s["groups"]} == {"step penalty 0.01 (default)", "no step penalty"} and all(g_["n"] == 2 for g_ in s["groups"])
    cmp_ = client.get(f"/api/studies/{sid}/rl-comparison").json()
    by = {v["label"]: v for v in cmp_["variants"]}
    nop = by["no step penalty"]
    assert nop["nCompleted"] == 2 and nop["taskReturn"]["n"] == 2 and nop["taskReturn"]["ci"] is not None and "warning" in nop["taskReturn"]
    assert nop["weights"]["step"] == 0.0 and by["step penalty 0.01 (default)"]["weights"]["step"] == 0.01
    assert set(nop["components"]) == {"goal", "step", "bump"} and nop["terminated"] + nop["truncated"] == nop["episodes"] == 16
    assert nop["components"]["step"]["mean"] == 0.0                                  # the disabled component contributes nothing to the training-basis return
    assert nop["rawComponents"]["step"]["mean"] < 0                                  # ... but the raw component is still recorded
    assert cmp_["evaluationSeedSets"] == [list(range(1000, 1008))]
    # same env + eval seeds, reward differs: the task-basis return (default reward) is the comparable one
    ra = nop["runs"][0]
    assert ra["weights"]["goal"] == 1.0 and ra["envSteps"] >= 500 and ra["wrapperChain"][0] == "RewardComponents"
    # a manual comparison of any runs works too
    rids = [x["runId"] for x in nop["runs"]]
    m = client.post("/api/rl/compare", json={"variants": [{"label": "x", "runIds": rids}]}).json()
    assert m["variants"][0]["taskReturn"] == nop["taskReturn"]


def test_study_objective_for_rl_is_validated(client):
    g = tiny_grid(300)
    bad = {"name": "x", "graph": g.to_json(), "run_config": {}, "objective": {"metric": {"name": "val_acc"}, "direction": "maximize"}, "limits": {"max_trials": 3}}
    r = client.post("/api/studies/plan", json=bad)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "objective_invalid"
    ok = {**bad, "objective": {"metric": {"name": "eval_success_rate"}, "direction": "maximize"},
          "search": {"method": "grid", "variables": [{"target": {"scope": "node", "node": "learner", "field": "gamma"}, "values": [0.9, 0.99]}]}}
    p = client.post("/api/studies/plan", json=ok).json()
    assert p["counts"]["total"] == 3 and p["counts"]["invalid"] == 0
    invalid = {**ok, "graph": tiny_cartpole().to_json(), "search": {"method": "grid", "variables": [{"target": {"scope": "node", "node": "env", "field": "env_id"}, "values": ["Pendulum-v1"]}]}}
    p = client.post("/api/studies/plan", json=invalid).json()
    assert p["counts"]["invalid"] == 1 and any("E_RL_ALGO_ACTION_SPACE" == d["code"] for t in p["trials"] for d in t["diagnostics"])
