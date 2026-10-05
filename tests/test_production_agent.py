"""Native model-free StateGraphs offline; real Ollama is exercised separately with -m live."""
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from agent import samples as sm
from agent_helpers import Lab
from control.app import create_app
from graph_core.schema import Graph
from production import agent_adapter as aa
from production.models import PredictRequest, RegisterVersion, ReleaseCreate
from production.pipeline import ProductionError
from production.runtime import ProductionRuntime

META = {"name": "native turn", "owner": "tests", "intendedUse": "SYNTHETIC control flow", "limitations": "model-free native workflow, no LLM accuracy claim"}


def pure_graph():
    return sm.make_graph([
        sm.N("left", "agent.set_state", assignments=[{"field": "log", "kind": "append_item", "value": "left"}, {"field": "n", "kind": "increment", "by": 1}]),
        sm.N("right", "agent.set_state", assignments=[{"field": "log", "kind": "append_item", "value": "right"}, {"field": "n", "kind": "increment", "by": 2}]),
        sm.N("join", "agent.set_state", assignments=[{"field": "answer", "kind": "template", "template": "{question}: n={n}"}])],
        [sm.E("START", "left"), sm.E("START", "right"), sm.E("left", "join"), sm.E("right", "join"), sm.E("join", "END")],
        {"state": [sm.S("question"), sm.S("n", "integer", "add", default=0), sm.S("log", "list", "append", default=[]), sm.S("answer")],
         "joins": [{"node": "join", "waitFor": ["left", "right"]}],
         "limits": {"maxSteps": 10, "maxModelCalls": 1, "maxTokens": 1024, "maxSeconds": 10}})


def setup(tmp_path, capture=True):
    lab = Lab(tmp_path)
    st, rid = lab.run(pure_graph(), inp={"question": "SYNTHETIC source"})
    assert st == "completed"
    rt = ProductionRuntime(lab.store)
    v = rt.register_version(RegisterVersion(runId=rid, node=aa.NODE, **META))
    release = rt.create_release(ReleaseCreate(versionId=v["id"], config={"namespace": "agents", "maxBatch": 1, "captureInputs": capture}))
    rt.activate(release["id"], None)
    return lab, rt, v, release


def request(id_, question="SYNTHETIC turn", **kw):
    return PredictRequest(requestId=id_, records=[{"question": question}], **kw)


def test_native_parallel_join_reducers_isolation_idempotency_and_restart(tmp_path):
    lab, rt, v, rel = setup(tmp_path)
    before = lab.final("r1")
    contexts_before = lab.store.artifacts("r1")
    t = rt.predict("local", "agents", request("one"))
    assert t["status"] == 200 and t["result"]["predictions"] == ["SYNTHETIC turn: n=3"]
    ev = t["result"]["agent"]
    assert ev["finalState"]["n"] == 3 and sorted(ev["finalState"]["log"]) == ["left", "right"]
    assert ev["graphHash"] == v["manifest"]["graphHash"] and ev["contexts"] == [] and ev["modelCalls"] == 0
    assert [e["seq"] for e in ev["events"]] == list(range(len(ev["events"])))
    assert sum(e["type"] == "node_finished" for e in ev["events"]) == 3
    assert t["sessionState"] is None and rt.ps.query("SELECT * FROM sessions") == []
    again = rt.predict("local", "agents", request("one"))
    assert again["idempotentReplay"] and again["result"] == t["result"]
    assert lab.final("r1") == before and lab.store.artifacts("r1") == contexts_before
    with ThreadPoolExecutor(max_workers=2) as pool:
        outs = list(pool.map(lambda i: rt.predict("local", "agents", request(f"p{i}", user=f"u{i}")), range(2)))
    assert all(o["result"]["agent"]["finalState"]["n"] == 3 for o in outs)
    assert len({o["result"]["agent"]["threadId"] for o in outs} | {ev["threadId"]}) == 3
    restarted = ProductionRuntime(lab.store)
    assert restarted.predict("local", "agents", request("one"))["idempotentReplay"]
    with pytest.raises(ProductionError, match="different input"):
        rt.predict("local", "agents", request("one", "different"))


@pytest.mark.parametrize("records,code", [([], "E_REQUEST_BOUNDS"), ([{"question": "x"}]*2, "E_REQUEST_BOUNDS"),
    ([{"question": "x"*2001}], "E_REQUEST_SCHEMA"), ([{"question": ""}], "E_REQUEST_SCHEMA"),
    ([{"question": 1}], "E_REQUEST_SCHEMA"), ([{"question": "x", "n": 9}], "E_REQUEST_SCHEMA"), ([{"other": "x"}], "E_REQUEST_SCHEMA")])
def test_input_contract(tmp_path, records, code):
    _, rt, v, _ = setup(tmp_path)
    with pytest.raises(ProductionError) as e:
        rt.pipeline(v["id"]).validate_records(records)
    assert e.value.code == code


@pytest.mark.parametrize("change,code", [({"maxBatch": 2}, "E_RELEASE_CONFIG"), ({"sessionMode": "counter"}, "E_RELEASE_CONFIG")])
def test_release_refusals(tmp_path, change, code):
    _, rt, v, _ = setup(tmp_path)
    with pytest.raises(ProductionError) as e:
        rt.create_release(ReleaseCreate(versionId=v["id"], config={"maxBatch": 1, **change}))
    assert e.value.code == code


@pytest.mark.parametrize("mutation,code", [("unbounded", "E_AGENT_LIMITS"), ("thread", "E_AGENT_SCOPE"), ("tool", "E_AGENT_SCOPE"),
                                           ("fixture", "E_AGENT_PROVIDER"), ("remote", "E_AGENT_PROVIDER")])
def test_source_scope_is_refused_without_provider_calls(mutation, code):
    g = pure_graph().to_json()
    if mutation == "unbounded":
        del g["agent"]["limits"]["maxSeconds"]
    elif mutation == "thread":
        g["agent"]["state"][0]["scope"] = "thread"
    elif mutation == "tool":
        g = sm.approval_tools_graph().to_json()
    else:
        g = serving_graph().to_json()
        model = g["nodes"][1]["config"]["model"]
        model.update({"provider": "fixture", "model": "fixture"} if mutation == "fixture" else {"base_url": "https://example.com"})
    with pytest.raises(ProductionError) as e:
        aa.contract(Graph.model_validate(g), {"question": "x"})
    assert e.value.code == code


def test_capture_off_monitoring_reads_only_recorded_data_and_labels(tmp_path, monkeypatch):
    lab, rt, v, rel = setup(tmp_path, capture=False)
    t = rt.predict("local", "agents", request("private", "private input"))
    assert t["records"] is None and t["result"]["agent"]["events"] is None and t["result"]["agent"]["finalState"] is None
    assert not list(lab.wb.glob("agent-turn*"))
    rt.ps.add_labels("local-user", "private", ["independently supplied unequal reference"])
    from production.monitor import monitoring
    monkeypatch.setattr(rt.pipeline(v["id"]), "predict", lambda *_a, **_k: pytest.fail("Monitoring must not execute an agent"))
    m = monitoring(rt, rel["id"])
    assert m["labelBasedQuality"]["values"]["exactStringAgreement"] == 0
    assert not m["inputDrift"]["question.characters"]["available"]
    assert m["usage"]["inputTokens"] is None and m["usage"]["successfulTurnModelCalls"] == 0


def test_deadline_cancellation_and_integrity_refuse_outputs(tmp_path, monkeypatch):
    lab, rt, v, _ = setup(tmp_path)
    p = rt.pipeline(v["id"])
    with pytest.raises(ProductionError) as e:
        p.predict([{"question": "x"}], deadline=time.perf_counter()-1)
    assert e.value.code == "E_REQUEST_TIMEOUT"
    with pytest.raises(ProductionError) as e:
        p.predict([{"question": "x"}], cancelled=lambda: True)
    assert e.value.code == "E_REQUEST_CANCELLED"
    original = aa.implementation
    monkeypatch.setattr(aa, "implementation", lambda: {**original(), "agent/runtime.py": "changed"})
    with pytest.raises(ProductionError) as e:
        rt.pipeline(v["id"])
    assert e.value.code == "E_SERVING_ENVIRONMENT"


def serving_graph():
    from graph_core.project_io import load_project
    from agent_helpers import REPO
    return load_project(REPO / "examples/serving_agent.project.json").graph


def test_http_reference_serve_replay_labels_and_manifest(tmp_path):
    lab, _, v, rel = setup(tmp_path)
    with TestClient(create_app(lab.wb)) as c:
        overview = c.get("/api/production").json()
        assert any(x["adapter"] == "agent" and x["node"] == aa.NODE for x in overview["candidates"])
        ref = c.get(f"/api/production/versions/{v['id']}/reference-input").json()
        assert ref["observedLabels"] is None and "not ground truth" in ref["provenance"]["partition"]
        body = request("http", "SYNTHETIC request").model_dump()
        t = c.post("/api/serve/local/agents/predict", json=body)
        assert t.status_code == 200, t.text
        trace = t.json()
        replay = c.post("/api/production/requests/http/replay", json={})
        assert replay.status_code == 200 and replay.json()["result"]["predictions"] == trace["result"]["predictions"]
        assert replay.json()["result"]["agent"]["executionId"] != trace["result"]["agent"]["executionId"]
        assert c.post("/api/production/requests/http/labels", json={"labels": [1]}).status_code == 422
        assert c.post("/api/production/requests/http/labels", json={"labels": ["reference"]}).status_code == 200
        assert c.get(f"/api/production/releases/{rel['id']}/monitor").json()["labelBasedQuality"]["values"]["exactStringAgreement"] == 0
        assert c.get("/api/serve/local/agents/health").json()["ready"]


def test_native_loop_exhaustion_is_a_failed_serving_request(tmp_path):
    g = sm.counter_loop_graph(stop_at=2, max_steps=6).to_json()
    g["agent"]["state"].append(sm.S("question"))
    g["agent"]["limits"].update(maxModelCalls=1, maxTokens=1024, maxSeconds=10)
    g["agent"]["routes"][0]["cases"][0]["when"] = sm.cmp("question", "==", "stop")
    lab = Lab(tmp_path)
    status, rid = lab.run(Graph.model_validate(g), inp={"question": "stop"})
    assert status == "completed"
    rt = ProductionRuntime(lab.store)
    v = rt.register_version(RegisterVersion(runId=rid, node=aa.NODE, **META))
    rel = rt.create_release(ReleaseCreate(versionId=v["id"], config={"maxBatch": 1}))
    rt.activate(rel["id"], None)
    t = rt.predict("local", "lab", request("loops", "keep looping"))
    assert t["status"] == 422 and t["error"]["code"] == "E_AGENT_BUDGET" and t["result"] is None
    status, rid2 = lab.run(Graph.model_validate(g), thread="budget-thread", inp={"question": "keep looping"})
    assert status == "completed" and lab.finished(rid2)["stoppedBy"] == "recursion_limit"
    with pytest.raises(ProductionError) as e:
        rt.register_version(RegisterVersion(runId=rid2, node=aa.NODE, **META))
    assert e.value.code == "E_AGENT_SOURCE"


def test_native_state_growth_is_stopped_between_supersteps(tmp_path):
    g = sm.counter_loop_graph(stop_at=8, max_steps=12).to_json()
    g["agent"]["state"].extend([sm.S("question"), sm.S("growth")])
    g["agent"]["limits"].update(maxModelCalls=1, maxTokens=1024, maxSeconds=10)
    g["nodes"][0]["config"]["assignments"].append({"field": "growth", "kind": "template", "template": "{growth}{growth}{question}"})
    g["nodes"][1]["config"]["assignments"] = [{"field": "growth", "kind": "copy", "source": "growth"}]
    lab = Lab(tmp_path)
    status, rid = lab.run(Graph.model_validate(g), inp={"question": "x"})
    assert status == "completed"
    m = aa.build_manifest(lab.store, rid, aa.NODE)
    p = aa.AgentPipeline(lab.store, m)
    with pytest.raises(ProductionError) as e:
        p.predict([{"question": "x"*2000}])
    assert e.value.code == "E_AGENT_STATE_BOUNDS"
