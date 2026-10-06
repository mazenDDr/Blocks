"""Actual installed Ollama sees native calculator results; no fixture provider."""
import pytest
from fastapi.testclient import TestClient
from agent import samples as sm
from agent.models import ollama_status
from agent_helpers import Lab
from graph_core.schema import Graph
from control.app import create_app
from production import tools_agent_adapter as adapter
from production.models import RegisterVersion, ReleaseCreate
from production.runtime import ProductionRuntime
from test_production_agent import META
from test_production_tools import graph as pure_graph

pytestmark = pytest.mark.live


def graph():
    doc = pure_graph().to_json()
    doc["nodes"] = doc["nodes"][:1] + [
        sm.N("prompt", "agent.prompt", output_field="messages", items=[
            {"kind": "template", "role": "system", "template": "Reply with only the numeric value from the calculator result."},
            {"kind": "template", "role": "user", "template": "Native calculator value: {calculation.result.value}"}]),
        sm.N("reply", "agent.chat_model", messages_field="messages", output_field="answer", model={
            "provider": "ollama", "model": "qwen3.5:2b", "think": False, "max_tokens": 32, "timeout_s": 20, "temperature": 0, "seed": 0})]
    doc["edges"] = sm.chain("START", "calc", "prompt", "reply", "END")
    doc["agent"]["state"].append(sm.S("messages", "messages"))
    doc["agent"]["limits"]["maxSeconds"] = 30
    return Graph.model_validate(doc)


def test_real_model_context_contains_native_calculator_result_and_source_is_unchanged(tmp_path):
    status = ollama_status()
    if not status["reachable"] or "qwen3.5:2b" not in status["models"]:
        pytest.skip("Actual installed local Ollama qwen3.5:2b required; no downloads")
    lab = Lab(tmp_path)
    status, source = lab.run(graph(), inp={"question": "7*6"})
    assert status == "completed"
    before = lab.store.events(source), lab.store.artifacts(source)
    rt = ProductionRuntime(lab.store)
    v = rt.register_version(RegisterVersion(runId=source, node=adapter.NODE, **META))
    r = rt.create_release(ReleaseCreate(versionId=v["id"], config={"namespace": "tools", "maxBatch": 1, "timeoutSeconds": 30, "captureInputs": True}))
    rt.activate(r["id"], None)
    with TestClient(create_app(lab.wb)) as c:
        response = c.post("/api/serve/local/tools/predict", json={"requestId": "real", "records": [{"question": "9*9"}]})
        assert response.status_code == 200, response.text
        a = response.json()["result"]["agent"]
        assert a["toolCalls"] == a["modelCalls"] == 1
        assert a["tools"][0]["result"]["value"] == 81
        context = a["contexts"][0]["value"]
        assert context["provider"] == "ollama" and not context["fixture"]
        assert context["usage"]["source"] == "provider"
        assert "Native calculator value: 81" in str(context["providerRequest"])
        assert c.post("/api/production/requests/real/labels", json={"labels": ["81"]}).status_code == 200
        mon = c.get(f"/api/production/releases/{r['id']}/monitor").json()
        assert mon["usage"]["successfulTurnModelCalls"] == 1
    assert (lab.store.events(source), lab.store.artifacts(source)) == before
