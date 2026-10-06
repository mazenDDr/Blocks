"""Served model-chosen calculator turns (ADR 0082): contract refusals offline; register → release → served tool call live."""
import importlib.util
import json
from pathlib import Path

import pytest

from graph_core.schema import Graph
from production import tool_choice_adapter as adapter
from production.pipeline import ProductionError

_spec = importlib.util.spec_from_file_location("tafx", Path(__file__).resolve().parents[1] / "examples" / "make_tool_agent_fixture.py")
fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture)


def doc():
    return fixture.graph().to_json()


def test_example_satisfies_the_contract():
    spec, output, cfg = adapter.contract(Graph.model_validate(doc()), {"question": "x"})
    assert output == "answer" and cfg.tools == ["calculator"] and cfg.max_tool_calls == 3


@pytest.mark.parametrize("change,code", [
    (lambda d: d["nodes"][1]["config"].update(tools=["calculator", "read_text_file"], allowed_dir="x"), "E_AGENT_SCOPE"),
    (lambda d: d["nodes"][1]["config"].update(max_tool_calls=6), "E_AGENT_LIMITS"),
    (lambda d: d["nodes"][1]["config"]["model"].update(think=True), "E_AGENT_PROVIDER"),
    (lambda d: d["nodes"][1]["config"]["model"].update(provider="openai_compatible", base_url="http://127.0.0.1:11434/v1"), "E_AGENT_PROVIDER"),
    (lambda d: d["agent"]["limits"].update(maxModelCalls=3), "E_AGENT_LIMITS"),
    (lambda d: d["agent"]["state"][3].update(scope="thread"), "E_AGENT_SCOPE"),
])
def test_contract_refusals(change, code):
    d = doc()
    change(d)
    with pytest.raises(ProductionError) as caught:
        adapter.contract(Graph.model_validate(d), {"question": "x"})
    assert caught.value.code == code, caught.value.message


@pytest.mark.live
def test_live_registered_release_serves_a_model_chosen_calculator_call(tmp_path):
    from agent_helpers import Lab
    from production.models import PredictRequest, RegisterVersion, ReleaseCreate
    from production.runtime import ProductionRuntime
    from test_production_agent import META
    lab = Lab(tmp_path / "lab")
    status, rid = lab.run(fixture.graph(), inp={"question": "SYNTHETIC: what is 1234 * 5678?"})
    assert status == "completed", lab.store.get_run(rid)["error"]
    rt = ProductionRuntime(lab.store)
    version = rt.register_version(RegisterVersion(runId=rid, node=adapter.NODE, **META))
    assert version["adapter"] == "agent_tool_choice" and version["manifest"]["tools"]["offered"] == ["calculator"]
    rel = rt.create_release(ReleaseCreate(versionId=version["id"], config={"maxBatch": 1, "sessionMode": "stateless", "captureInputs": False, "timeoutSeconds": 30}))
    rt.activate(rel["id"], None)
    t = rt.predict("local", "lab", PredictRequest(requestId="q1", records=[{"question": "SYNTHETIC: what is 987 * 654?"}]))
    assert t["status"] == 200, t["error"]
    calls = t["result"]["agent"]["toolCalls"]
    print("LIVE SERVED TOOLS", [(c["tool"], c["args"], c["status"]) for c in calls], repr(t["result"]["predictions"][0][:60]))
    assert any(c["tool"] == "calculator" and c["status"] == "ok" and c["result"]["value"] == 645498 for c in calls)
    stored = rt.ps.trace("local-user", "q1")
    assert stored["result"]["agent"]["toolCalls"] == calls  # recorded in the trace even with capture off
