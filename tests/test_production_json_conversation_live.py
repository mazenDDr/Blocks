"""Real installed Ollama: native JSON conversations (ADR 0059) — schema-valid objects per turn on persistent checkpoints."""
import pytest
from fastapi.testclient import TestClient

from agent import samples as sm
from agent.models import ollama_status
from agent_helpers import Lab
from control.app import create_app
from production import json_conversation_adapter as adapter
from production.models import RegisterVersion, ReleaseCreate
from production.pipeline import ProductionError
from production.runtime import ProductionRuntime
from test_production_agent import META
from test_production_json_agent import fixture

pytestmark = pytest.mark.live


def graph():
    base = fixture.graph().to_json()
    nodes = [sm.N("tick", "agent.set_state", assignments=[{"field": "turns", "kind": "increment", "by": 1},
                                                          {"field": "history", "kind": "append_item", "source": "question"}])]
    nodes += [sm.N(n["id"], n["type"], **n["config"]) for n in base["nodes"]]
    state = base["agent"]["state"] + [sm.S("turns", "integer", "add", default=0, scope="thread"),
                                      sm.S("history", "list", "keep_last_n", default=[], scope="thread", n=3)]
    return sm.make_graph(nodes, sm.chain("START", "tick", "prompt", "extract", "END"), {"state": state, "limits": base["agent"]["limits"]})


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    status = ollama_status()
    if not status["reachable"] or "qwen3.5:2b" not in status["models"]:
        pytest.skip("Real installed local Ollama qwen3.5:2b required; no download/provider substitution")
    lab = Lab(tmp_path_factory.mktemp("json-conversation"))
    state, run_id = lab.run(graph(), inp={"question": "SYNTHETIC teaching input: colour red, count 3."})
    assert state == "completed" and lab.final(run_id)["result"] == {"colour": "red", "count": 3} and lab.final(run_id)["turns"] == 1
    rt = ProductionRuntime(lab.store)
    version = rt.register_version(RegisterVersion(runId=run_id, node=adapter.NODE, **META))
    assert version["adapter"] == "conversation_json" and version["family"] == "agent_json"
    release = rt.create_release(ReleaseCreate(versionId=version["id"], config={"namespace": "json-chat", "maxBatch": 1, "sessionMode": "conversation",
                                                                               "timeoutSeconds": 30, "captureInputs": True}))
    rt.activate(release["id"], None)
    return lab, rt, version, release, run_id


def test_turns_return_validated_objects_on_persistent_native_checkpoints(served):
    lab, rt, version, release, run_id = served
    source_before = lab.final(run_id)
    with TestClient(create_app(lab.wb)) as c:
        assert any(x["adapter"] == "conversation_json" and x["node"] == adapter.NODE for x in c.get("/api/production").json()["candidates"])
        turn = lambda rid, text, session="chat": c.post("/api/serve/local/json-chat/predict", json={
            "requestId": rid, "user": "alice", "session": session, "records": [{"question": text}]})
        first = turn("t1", "SYNTHETIC teaching input: colour blue, count 9.")
        assert first.status_code == 200, first.text
        assert first.json()["result"]["predictions"] == [{"colour": "blue", "count": 9}] and first.json()["result"]["family"] == "agent_json"
        second = turn("t2", "SYNTHETIC teaching input: colour red, count 2.")
        assert second.json()["result"]["predictions"] == [{"colour": "red", "count": 2}]
        assert second.json()["result"]["agent"]["threadId"] == first.json()["result"]["agent"]["threadId"]
        head = c.get("/api/production/releases/{}/conversation?user=alice&session=chat".format(release["id"])).json()
        assert head["head"]["revision"] == 2 and head["state"]["turns"] == 2
        assert head["state"]["history"] == ["SYNTHETIC teaching input: colour blue, count 9.", "SYNTHETIC teaching input: colour red, count 2."]
        other = turn("t3", "SYNTHETIC teaching input: colour blue, count 1.", session="other")
        assert other.status_code == 200 and other.json()["result"]["agent"]["threadId"] != first.json()["result"]["agent"]["threadId"]
        assert c.get("/api/production/releases/{}/conversation?user=alice&session=other".format(release["id"])).json()["state"]["turns"] == 1
        label = c.post("/api/production/requests/t1/labels", json={"user": "alice", "labels": [{"count": 9, "colour": "blue"}]})
        assert label.status_code == 200, label.text
        bad = c.post("/api/production/requests/t2/labels", json={"user": "alice", "labels": [{"colour": "green"}]})
        assert bad.status_code == 422 and bad.json()["detail"]["code"] == "E_LABEL_SCHEMA"
    assert lab.final(run_id) == source_before  # the source research thread is never mutated


def test_contract_refusals(served):
    lab, rt, version, release, run_id = served
    with pytest.raises(ProductionError) as e:
        rt.create_release(ReleaseCreate(versionId=version["id"], config={"namespace": "json-chat-stateless", "maxBatch": 1}))
    assert e.value.code == "E_RELEASE_CONFIG"
    state, stateless_run = lab.run(fixture.graph(), inp={"question": "SYNTHETIC teaching input: colour red, count 3."})
    assert state == "completed"
    with pytest.raises(ProductionError) as e:
        rt.register_version(RegisterVersion(runId=stateless_run, node=adapter.NODE, **META))
    assert e.value.code == "E_AGENT_SOURCE"
    graph_doc, _ = adapter.conv._graph(lab.store, stateless_run)
    with pytest.raises(ProductionError) as e:
        adapter.contract(graph_doc, {"question": "x"})
    assert e.value.code == "E_AGENT_SCOPE"
