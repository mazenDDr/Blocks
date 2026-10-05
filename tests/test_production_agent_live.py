"""Real local Ollama → native agent source → immutable release → isolated production turns."""
import copy

import pytest
from fastapi.testclient import TestClient

from agent.models import ollama_status
from agent_helpers import Lab
from control.app import create_app
from production import agent_adapter as aa
from production.models import PredictRequest, RegisterVersion, ReleaseCreate
from production.pipeline import ProductionError, read_verified
from production.runtime import ProductionRuntime
from test_production_agent import META, serving_graph

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    status = ollama_status()
    if not status["reachable"] or "qwen3.5:2b" not in status["models"]:
        pytest.skip("Local Ollama with qwen3.5:2b is required; no downloads")
    lab = Lab(tmp_path_factory.mktemp("native-agent-serving"))
    st, rid = lab.run(serving_graph(), inp={"question": "SYNTHETIC teaching prompt: reply briefly with the word hello."})
    assert st == "completed" and not lab.contexts(rid)[0]["fixture"]
    rt = ProductionRuntime(lab.store)
    v = rt.register_version(RegisterVersion(runId=rid, node=aa.NODE, **META))
    rel = rt.create_release(ReleaseCreate(versionId=v["id"], config={"namespace": "live-agent", "maxBatch": 1, "timeoutSeconds": 30, "captureInputs": True}))
    rt.activate(rel["id"], None)
    return lab, rt, v, rel


def test_real_ollama_context_usage_isolation_replay_and_monitoring(native):
    lab, rt, v, rel = native
    before = lab.final("r1")
    source_context = lab.contexts("r1")
    with TestClient(create_app(lab.wb)) as c:
        req = {"requestId": "native-real", "records": [{"question": "SYNTHETIC test: give one word that names a colour."}]}
        out = c.post("/api/serve/local/live-agent/predict", json=req)
        assert out.status_code == 200, out.text
        trace = out.json()
        evidence = trace["result"]["agent"]
        context = evidence["contexts"][0]["value"]
        assert evidence["modelCalls"] == 1 and trace["result"]["predictions"][0].strip()
        assert context["providerRequest"][-1]["content"] == req["records"][0]["question"]
        assert context["messages"][-1]["segments"] and context["usage"]["source"] == "provider"
        assert context["tokens"]["providerInput"] > 0 and 0 < context["tokens"]["providerOutput"] <= 48
        assert context["threadId"] == evidence["executionId"] and not context["fixture"]
        assert context["resolved"]["timeout_s"] <= 20 and context["latencyMs"] > 0
        assert evidence["provider"]["digest"] == aa.provider_identity("qwen3.5:2b")["digest"]
        raw = read_verified(lab.store, evidence["contexts"][0]["sha256"])
        assert req["records"][0]["question"].encode() in raw
        duplicate = c.post("/api/serve/local/live-agent/predict", json=req).json()
        assert duplicate["idempotentReplay"] and duplicate["result"] == trace["result"]
        replay = c.post("/api/production/requests/native-real/replay", json={}).json()
        assert replay["result"]["agent"]["executionId"] != evidence["executionId"] and "may differ" in replay["replayNote"]
        assert lab.final("r1") == before and lab.contexts("r1") == source_context
        # Independently supplied reference, deliberately not copied from generated output.
        assert c.post("/api/production/requests/native-real/labels", json={"labels": ["red"]}).status_code == 200
        monitored = c.get(f"/api/production/releases/{rel['id']}/monitor").json()
        assert monitored["usage"]["successfulTurnModelCalls"] == 1 and monitored["usage"]["providerReportedCalls"] == 1
        assert monitored["labelBasedQuality"]["values"]["exactStringAgreement"] == float(trace["result"]["predictions"][0] == "red")


def test_real_digest_refusal_and_capture_off(native):
    lab, rt, v, _ = native
    changed = copy.deepcopy(v["manifest"])
    changed["provider"]["digest"] = "incorrect digest"
    with pytest.raises(ProductionError) as e:
        aa.verify(lab.store, changed)
    assert e.value.code == "E_AGENT_MODEL_CHANGED"
    rel = rt.create_release(ReleaseCreate(versionId=v["id"], config={"namespace": "private-agent", "maxBatch": 1, "timeoutSeconds": 30}))
    rt.activate(rel["id"], None)
    t = rt.predict("local", "private-agent", PredictRequest(requestId="hidden", records=[{"question": "SYNTHETIC private prompt: say hello."}]))
    assert t["status"] == 200 and t["records"] is None
    evidence = t["result"]["agent"]
    assert evidence["finalState"] is None and evidence["events"] is None and evidence["contexts"][0]["value"] is None
    with pytest.raises(ProductionError):
        read_verified(lab.store, evidence["contexts"][0]["sha256"])
