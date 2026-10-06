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
