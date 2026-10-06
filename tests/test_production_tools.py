"""Native pure calculator serving: contracts, budgets, capture, replay and isolation."""
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent import samples as sm
from agent_helpers import Lab
from control.app import create_app
from graph_core.schema import Graph
from production import tools_agent_adapter as adapter
from production.models import PredictRequest, RegisterVersion, ReleaseCreate
from production.pipeline import ProductionError
from production.runtime import ProductionRuntime
from test_production_agent import META
from test_production_retrieval import graph as retrieval_graph


def graph(*, budget=2, two=False, parallel=False, docs=None):
    g = retrieval_graph(docs).to_json() if docs else sm.make_graph([], [], {
        "state": [sm.S("question"), sm.S("answer")],
        "limits": {"maxSteps": 8, "maxSeconds": 10, "maxModelCalls": 1, "maxTokens": 1024}}).to_json()
    nodes = [sm.N("calc", "agent.tool_call", tool="calculator", args={"expression": "{question}"}, output_field="calculation")]
    g["agent"]["state"].append(sm.S("calculation", "json"))
    if two:
        nodes.append(sm.N("calc2", "agent.tool_call", tool="calculator", args={"expression": "{question}"}, output_field="second"))
        g["agent"]["state"].append(sm.S("second", "json"))
    if docs:
        nodes.append(g["nodes"][0])
    nodes.append(sm.N("answer", "agent.set_state", assignments=[{"field": "answer", "kind": "template", "template": "Native calculator: {calculation.result.value}"}]))
    g["nodes"] = nodes
    if parallel:
        g["edges"] = sm.chain("START", "calc", "answer") + sm.chain("START", "calc2", "answer", "END")
        g["agent"]["joins"] = [{"node": "answer", "waitFor": ["calc", "calc2"]}]
    else:
        g["edges"] = sm.chain("START", *(n["id"] for n in nodes), "END")
    for i, edge in enumerate(g["edges"]): edge["id"] = f"e{i}"
    g["agent"]["limits"]["maxToolCalls"] = budget
    return Graph.model_validate(g)


def register(tmp_path, g=None, capture=True):
    lab = Lab(tmp_path / "lab")
    status, source = lab.run(g or graph(), inp={"question": "(2 + 3) * 4"})
    assert status == "completed"
    rt = ProductionRuntime(lab.store)
    v = rt.register_version(RegisterVersion(runId=source, node=adapter.NODE, **META))
    r = rt.create_release(ReleaseCreate(versionId=v["id"], config={"namespace": "tools", "maxBatch": 1, "captureInputs": capture}))
    rt.activate(r["id"], None)
    return lab, rt, v, r, source


def test_native_calculator_http_replay_restart_labels_and_source_isolation(tmp_path):
    lab, rt, version, release, source = register(tmp_path)
    original = lab.store.events(source), lab.store.artifacts(source)
    assert "agent/tools.py" in version["manifest"]["implementation"]
    with TestClient(create_app(lab.wb)) as c:
        candidates = c.get("/api/production").json()["candidates"]
        assert any(x["node"] == adapter.NODE and x["runId"] == source for x in candidates)
        r = c.post("/api/serve/local/tools/predict", json={"requestId": "calc", "records": [{"question": "sqrt(81) + 7"}]})
        assert r.status_code == 200, r.text
        trace = r.json(); a = trace["result"]["agent"]
        assert trace["result"]["predictions"] == ["Native calculator: 16.0"]
        assert a["toolCalls"] == 1 and a["modelCalls"] == 0
        assert a["tools"][0]["result"] == {"expression": "sqrt(81) + 7", "value": 16.0}
        assert a["tools"][0]["effects"] == []
        replay = c.post("/api/production/requests/calc/replay", json={"user": "local-user"}).json()
        assert replay["result"]["predictions"] == trace["result"]["predictions"]
        assert replay["result"]["agent"]["executionId"] != a["executionId"]
        assert c.post("/api/production/requests/calc/labels", json={"labels": ["Native calculator: 16.0"]}).status_code == 200
        assert c.post("/api/production/requests/calc/labels", json={"labels": [16]}).status_code == 422
        monitor = c.get(f"/api/production/releases/{release['id']}/monitor").json()
        assert monitor["labelBasedQuality"]["values"]["exactStringAgreement"] == 1
        assert monitor["usage"]["successfulTurnModelCalls"] == 0
    restarted = ProductionRuntime(lab.store)
    assert restarted.predict("local", "tools", PredictRequest(requestId="calc", records=[{"question": "sqrt(81) + 7"}]))["result"] == trace["result"]
    assert (lab.store.events(source), lab.store.artifacts(source)) == original


@pytest.mark.parametrize("parallel", [False, True])
def test_native_tool_budget_refuses_before_extra_calls(tmp_path, parallel):
    lab, rt, v, release, source = register(tmp_path, graph(two=True, parallel=parallel))
    # Source reaches END with budget 2. Re-register a recorded graph with budget 1:
    # source worker itself honestly stops by budget and cannot be registered.
    status, limited = lab.run(graph(two=True, parallel=parallel, budget=1), thread="limited", inp={"question": "2+3"})
    if not parallel:
        assert adapter.is_candidate(lab.store, lab.store.get_run(limited)) is None
    # Serving budget is tested on a completed native source with a branch that
    # only invokes the second tool for a different request input.
    doc = graph(two=True, budget=1).to_json()
    doc["edges"] = sm.chain("START", "calc") + sm.chain("calc2", "answer", "END")
    for i, edge in enumerate(doc["edges"]): edge["id"] = f"e{i}"
    doc["agent"]["routes"] = [{"id": "branch", "from": "calc", "cases": [{"id": "second", "when": {"op": "==", "field": "question", "value": "2+3"}, "to": "calc2"}], "default": "answer"}]
    status, source = lab.run(Graph.model_validate(doc), thread="branched", inp={"question": "4+5"})
    assert status == "completed", lab.events(source, "run_finished")
    v = rt.register_version(RegisterVersion(runId=source, node=adapter.NODE, **META))
    p = rt.pipeline(v["id"])
    with pytest.raises(ProductionError, match="maxToolCalls") as err:
        p.predict([{"question": "2+3"}])
    assert err.value.code == "E_AGENT_BUDGET"
    # Force both native branches through the serving wrapper independently of
    # source eligibility, proving atomic reservations at maxToolCalls=1.
    p.graph = graph(two=True, parallel=parallel, budget=1)
    p.spec = adapter.agent_spec(p.graph)
    with pytest.raises(ProductionError) as err:
        p.predict([{"question": "2+3"}])
    assert err.value.code == "E_AGENT_BUDGET"


def test_pinned_retrieval_combined_with_calculator(tmp_path):
    docs = tmp_path / "docs"; docs.mkdir(); (docs / "numbers.txt").write_text("SYNTHETIC apples on trees")
    lab, rt, v, r, source = register(tmp_path, graph(docs=docs))
    before = rt.pipeline(v["id"]).predict([{"question": "7*6"}], capture=True)[0]
    import shutil
    shutil.rmtree(docs); shutil.rmtree(lab.wb / "agent" / "indexes")
    after = rt.pipeline(v["id"]).predict([{"question": "7*6"}], capture=True)[0]
    assert before["predictions"] == after["predictions"] == ["Native calculator: 42"]
    assert before["agent"]["retrievals"] == after["agent"]["retrievals"]
    assert after["agent"]["tools"][0]["result"]["value"] == 42


def test_capture_off_tool_arguments_not_persisted(tmp_path):
    lab, rt, v, r, source = register(tmp_path, capture=False)
    marker = "984321 + 17"
    trace = rt.predict("local", "tools", PredictRequest(requestId="private", records=[{"question": marker}]))
    a = trace["result"]["agent"]
    assert a["toolCalls"] == 1 and a["tools"] is None and a["events"] is None and a["finalState"] is None
    assert marker not in json.dumps(trace)
    assert all(marker.encode() not in f.read_bytes() for f in (lab.wb / "artifacts").iterdir() if f.is_file())


@pytest.mark.parametrize("expression", ["2 ** 99", "1e200", "1e100 * 1e100", "+"*40+"2", "1+"*260+"1", "max("+",".join(["1"]*130)+")"])
def test_expression_bounds(tmp_path, expression):
    _, rt, v, _, _ = register(tmp_path)
    with pytest.raises(ProductionError) as err:
        rt.pipeline(v["id"]).predict([{"question": expression}])
    assert err.value.code == "E_AGENT_TOOL_BOUNDS"


@pytest.mark.parametrize("change", ["file_read", "file_write", "unbounded", "thread", "dynamic", "directory", "no_tools"])
def test_scope_refusals(change):
    doc = graph().to_json()
    cfg = doc["nodes"][0]["config"]
    if change == "file_read": cfg.update(tool="read_text_file", args={"path": "file.txt"}, allowed_dir="/tmp")
    elif change == "file_write": cfg.update(tool="write_note", args={"path": "file.txt", "text": "hi"})
    elif change == "unbounded": doc["agent"]["limits"].pop("maxToolCalls")
    elif change == "thread": doc["agent"]["state"][0]["scope"] = "thread"
    elif change == "dynamic": cfg["args"]["expression"] = "{calculation.result.value}"
    elif change == "directory": cfg["allowed_dir"] = "/tmp"
    elif change == "no_tools": doc["nodes"][0] = sm.N("calc", "agent.set_state", assignments=[])
    with pytest.raises(ProductionError): adapter.contract(Graph.model_validate(doc), {"question": "2+3"})


def test_adapter_storage_pins_match_explicit_migration_compatibility_decision():
    from production import agent_adapter, conversation_adapter, retrieval_agent_adapter
    expected = json.loads((Path(__file__).parent / "fixtures/serving_sources_6f63e09.json").read_text())["files"]
    changed = json.loads((Path(__file__).parent / "fixtures/serving_sources_adr0066.json").read_text())["files"]
    assert set(changed) == {"production/agent_adapter.py", "production/conversation_adapter.py", "agent/memory.py", "agent/index.py", "artifact_store/store.py", "storage/schema.py"}
    for name in set(changed) & set(expected):
        assert changed[name] != expected[name], "ADR0066 declares an intentional pin change"
    expected.update(changed)
    wal = json.loads((Path(__file__).parent / "fixtures/serving_sources_wal.json").read_text())["files"]
    assert set(wal) == {"agent/memory.py", "agent/runtime.py", "artifact_store/store.py", "storage/schema.py"}
    for name in wal:
        assert wal[name] != expected[name], "HANDOFF 78 declares an intentional pin change"
    expected.update(wal)
    for mod in (agent_adapter, conversation_adapter, retrieval_agent_adapter):
        for name, current in mod.implementation().items():
            assert expected[name] == current
