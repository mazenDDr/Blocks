"""Agent evaluation runs (ADR 0077): declared checks over real child agent runs, Wilson interval, recorded failures."""
import time

import pytest
from fastapi.testclient import TestClient

from agent import samples as sm
from control.app import create_app
from worker.agent_eval import Check, evaluate_check, wilson


def echo_graph():
    nodes = [sm.N("answer", "agent.set_state", assignments=[{"field": "answer", "kind": "template", "template": "SYNTHETIC answer: {question}"}])]
    return sm.make_graph(nodes, sm.chain("START", "answer", "END"), {"state": [sm.S("question"), sm.S("answer")], "limits": {"maxSteps": 4}})


@pytest.mark.parametrize("check,state,ok", [
    ({"field": "a", "kind": "equals", "value": "Paris"}, {"a": " paris "}, True),
    ({"field": "a", "kind": "equals", "value": "Paris", "case_sensitive": True}, {"a": "paris"}, False),
    ({"field": "a", "kind": "contains", "value": "TEAL"}, {"a": "my colour is teal"}, True),
    ({"field": "a", "kind": "not_contains", "value": "sorry"}, {"a": "Sorry, no"}, False),
    ({"field": "a", "kind": "regex", "value": r"^\d{4}$"}, {"a": "2026"}, True),
    ({"field": "a", "kind": "one_of", "value": ["yes", "y"]}, {"a": "Y"}, True),
    ({"field": "a", "kind": "number_close", "value": 12, "tolerance": 0.5}, {"a": "The total is 12.4 units"}, True),
    ({"field": "a", "kind": "number_close", "value": 12}, {"a": "twelve"}, False),
    ({"field": "o.colour", "kind": "equals", "value": "red"}, {"o": {"colour": "red"}}, True),
    ({"field": "missing", "kind": "contains", "value": "x"}, {"a": "x"}, False),
])
def test_checks(check, state, ok):
    r = evaluate_check(Check.model_validate(check), state)
    assert r["passed"] is ok and r["field"] == check["field"]


def test_wilson_interval():
    assert wilson(8, 10) == (0.4902, 0.9433) and wilson(0, 5)[0] == 0.0 and wilson(5, 5)[1] == 1.0 and wilson(0, 0) is None


def wait(c, rid, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        row = c.get(f"/api/runs/{rid}").json()
        if row["status"] in ("completed", "failed", "cancelled"):
            return row
        time.sleep(0.2)
    raise AssertionError("evaluation did not finish")


def test_evaluation_run_over_real_child_runs(tmp_path):
    cases = [{"id": "c1", "input": {"question": "SYNTHETIC alpha"}, "checks": [{"field": "answer", "kind": "contains", "value": "alpha"}]},
             {"id": "c2", "input": {"question": "SYNTHETIC beta"}, "checks": [{"field": "answer", "kind": "regex", "value": r"answer: SYNTHETIC beta$"},
                                                                               {"field": "answer", "kind": "not_contains", "value": "gamma"}]},
             {"id": "c3", "input": {"question": "SYNTHETIC delta"}, "checks": [{"field": "answer", "kind": "equals", "value": "epsilon"}]}]
    with TestClient(create_app(tmp_path / "wb")) as c:
        r = c.post("/api/runs", json={"graph": echo_graph().to_json(), "config": {"kind": "agent_eval", "name": "SYNTHETIC echo", "cases": cases}},
                   headers={"Idempotency-Key": "eval-1"})
        assert r.status_code == 201, r.text
        rid = r.json()["runId"]
        row = wait(c, rid)
        assert row["status"] == "completed" and row["kind"] == "agent_eval" and row["casesDone"] == 3 and row["summary"]["passed"] == 2
        ev = c.get(f"/api/agent/evals/{rid}").json()
        rep = ev["report"]
        assert (rep["cases"], rep["passed"], rep["passRate"], rep["wilson95"]) == (3, 2, 0.6667, [0.2077, 0.9385])
        assert [x["case"] for x in rep["failedChecks"]] == ["c3"] and rep["failedChecks"][0]["observed"] == "SYNTHETIC answer: SYNTHETIC delta"
        for case in ev["cases"]:
            child = c.get(f"/api/runs/{case['childRunId']}").json()
            assert child["status"] == "completed" and child["kind"] == "agent" and child["config"]["evaluationOf"] == rid
            assert c.get(f"/api/agent/runs/{case['childRunId']}/final-state").json()["values"]["question"].startswith("SYNTHETIC")
        bad = c.post("/api/runs", json={"graph": echo_graph().to_json(), "config": {"kind": "agent_eval", "cases": [cases[0], cases[0]]}}, headers={"Idempotency-Key": "eval-2"})
        assert bad.status_code == 422 and "unique" in bad.text
        empty = c.post("/api/runs", json={"graph": echo_graph().to_json(), "config": {"kind": "agent_eval", "cases": []}}, headers={"Idempotency-Key": "eval-3"})
        assert empty.status_code == 422
        assert c.get("/api/agent/evals/nope").status_code == 404


@pytest.mark.live
def test_live_local_model_evaluation(tmp_path):
    import importlib.util
    from pathlib import Path
    from artifact_store import ArtifactStore
    from worker.agent_eval import AgentEvalConfig, run_agent_eval
    spec = importlib.util.spec_from_file_location("fx", Path(__file__).resolve().parents[1] / "examples" / "make_agent_eval_fixture.py")
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    store = ArtifactStore(tmp_path)
    cfg = AgentEvalConfig(name="SYNTHETIC live", cases=[c for c in fx.cases() if c["id"] in ("arith-00", "upper-0")])
    store.create_run("eval", "g", cfg.model_dump())
    assert run_agent_eval(fx.graph("qwen3.5:2b"), cfg, store, "eval") == "completed"
    rep = __import__("json").loads(store.read_artifact(store.artifacts("eval", "agent_eval_report")[-1]["sha256"]))
    assert rep["cases"] == 2 and all(r["status"] == "completed" and r["modelCalls"] == 1 for r in rep["results"])
    print("LIVE EVAL", rep["passed"], rep["cases"], [r["checks"][0]["observed"] for r in rep["results"]])


def test_seeds_repeat_cases_with_each_seed_on_every_model_node(tmp_path):
    from artifact_store import ArtifactStore
    from worker.agent_eval import AgentEvalConfig, run_agent_eval, seeded
    g = sm.make_graph([sm.N("prompt", "agent.prompt", output_field="messages", items=[{"kind": "template", "role": "user", "template": "{question}"}]),
                       sm.N("reply", "agent.chat_model", messages_field="messages", output_field="answer",
                            model={"provider": "fixture", "model": "fixture", "fixture": {"responses": ["SYNTHETIC yes", "SYNTHETIC no"]}})],
                      sm.chain("START", "prompt", "reply", "END"), {"state": [sm.S("question"), sm.S("messages", "messages"), sm.S("answer")], "limits": {"maxSteps": 6, "maxModelCalls": 1}})
    assert next(n for n in seeded(g, 11).to_json()["nodes"] if n["id"] == "reply")["config"]["model"]["seed"] == 11
    store = ArtifactStore(tmp_path)
    cfg = AgentEvalConfig(cases=[{"id": "a", "input": {"question": "q"}, "checks": [{"field": "answer", "kind": "contains", "value": "yes"}]}], seeds=[1, 2])
    store.create_run("ev", "g", cfg.model_dump())
    assert run_agent_eval(g, cfg, store, "ev") == "completed"
    import json as _json
    rep = _json.loads(store.read_artifact(store.artifacts("ev", "agent_eval_report")[-1]["sha256"]))
    assert rep["cases"] == 2 and [r["seed"] for r in rep["results"]] == [1, 2] and len(rep["perSeed"]) == 2
    assert store.get_run("ev-s1c000")["config"]["seed"] == 2
    with pytest.raises(ValueError):
        AgentEvalConfig(cases=cfg.cases, seeds=[1, 1])


def test_deployed_release_evaluation_isolates_cases_and_real_users(tmp_path):
    from test_production_memory import setup, say, texts
    lab, rt, version, rel = setup(tmp_path)
    assert say(rt, "real-1", "SYNTHETIC a real user's note")["status"] == 200
    with TestClient(create_app(lab.wb)) as c:
        rt2 = c.app.state.services.production
        cases = [{"id": f"n{i}", "input": {"note": f"SYNTHETIC eval note {i}"}, "checks": [{"field": "output", "kind": "contains", "value": "recalled 0 "}]} for i in range(3)]
        cases.append({"id": "designed-fail", "input": {"note": "SYNTHETIC x"}, "checks": [{"field": "output", "kind": "contains", "value": "recalled 5 "}]})
        r = c.post(f"/api/production/releases/{rel['id']}/evaluate", json={"name": "SYNTHETIC release check", "cases": cases})
        assert r.status_code == 201, r.text
        rid = r.json()["runId"]
        end = time.time() + 60
        while c.get(f"/api/agent/evals/{rid}").json()["status"] not in ("completed", "failed") and time.time() < end:
            time.sleep(0.2)
        ev = c.get(f"/api/agent/evals/{rid}").json()
        assert ev["kind"] == "release_eval" and ev["status"] == "completed", ev
        assert (ev["report"]["passed"], ev["report"]["cases"]) == (3, 4) and ev["report"]["failedChecks"][0]["case"] == "designed-fail"
        assert all(x["traceSha256"] and x["user"].startswith("ev") for x in ev["cases"])
        assert texts(rt2, rel, "alice") == ["SYNTHETIC a real user's note"]  # real users untouched; each case had its own user
        assert c.get(f"/api/runs/{rid}").json()["kind"] == "release_eval"
        rt2.ps.query("DELETE FROM routes")
        assert c.post(f"/api/production/releases/{rel['id']}/evaluate", json={"cases": cases}).status_code == 409


def test_compare_reports_counts_fixed_regressed_and_exact_mcnemar():
    from worker.agent_eval import compare_reports
    mk = lambda flags: {"results": [{"case": f"c{i}", "seed": None, "passed": f} for i, f in enumerate(flags)]}
    r = compare_reports(mk([True, True, False, False, True]), mk([True, False, True, True, True] + [True]))
    assert (r["shared"], r["fixed"], r["regressed"], r["passedA"], r["passedB"]) == (5, 2, 1, 3, 4)
    assert r["mcnemarExactP"] == 1.0 and r["onlyB"] == 1  # B has one extra case, not compared
    r = compare_reports(mk([False] * 8), mk([True] * 8))
    assert r["fixed"] == 8 and r["mcnemarExactP"] == 0.0078


def test_compare_api_over_two_evaluations(tmp_path):
    with TestClient(create_app(tmp_path / "wb")) as c:
        ids = []
        for i, val in enumerate(("alpha", "zzz")):
            cases = [{"id": "c1", "input": {"question": "SYNTHETIC alpha"}, "checks": [{"field": "answer", "kind": "contains", "value": val}]}]
            r = c.post("/api/runs", json={"graph": echo_graph().to_json(), "config": {"kind": "agent_eval", "cases": cases}}, headers={"Idempotency-Key": f"cmp-{i}"})
            ids.append(r.json()["runId"])
            wait(c, ids[-1])
        cmp_ = c.get("/api/agent/evals/compare", params={"a": ids[0], "b": ids[1]}).json()
        assert (cmp_["shared"], cmp_["regressed"], cmp_["fixed"]) == (1, 1, 0) and cmp_["rows"][0]["change"] == "regressed"
        assert c.get("/api/agent/evals/compare", params={"a": ids[0], "b": "nope"}).status_code == 404
