"""JSON conversation contract (ADR 0059) checked offline: graph inspection only, no model call."""
import copy

import pytest

from graph_core.schema import Graph
from production import json_conversation_adapter as adapter
from production.pipeline import ProductionError
from test_production_json_agent import fixture
from test_production_json_conversation_live import graph


def changed(fn):
    doc = copy.deepcopy(graph().to_json())
    fn(doc)
    return Graph.model_validate(doc)


def code(g, inputs=None):
    with pytest.raises(ProductionError) as e:
        adapter.contract(g, inputs or {"question": "x"})
    return e.value.code


def field(doc, name):
    return next(f for f in doc["agent"]["state"] if f["name"] == name)


def test_accepts_thread_state_with_one_turn_scoped_object_output():
    spec, cfg = adapter.contract(graph(), {"question": "x"})
    assert cfg.output_field == "result" and any(f.scope == "thread" for f in spec.state)


def test_refusals():
    assert code(fixture.graph()) == "E_AGENT_SCOPE"  # no thread state: use the stateless JSON adapter
    assert code(changed(lambda d: field(d, "result").update(scope="thread"))) == "E_AGENT_OUTPUT"
    assert code(changed(lambda d: field(d, "question").update(scope="thread"))) == "E_AGENT_INPUT"
    assert code(changed(lambda d: d["nodes"][-1]["config"]["model"].update(think=True))) == "E_AGENT_PROVIDER"
    assert code(changed(lambda d: d["nodes"][-1]["config"].update(on_failure="route"))) == "E_AGENT_JSON_FAILURE"
    assert code(changed(lambda d: d["agent"]["limits"].update(maxSeconds=60))) == "E_AGENT_LIMITS"


def test_identity_pins_conversation_files_but_not_the_http_layer():
    files = set(adapter.FILES)
    assert {"production/conversation_adapter.py", "production/json_agent_adapter.py", "production/json_conversation_adapter.py", "agent/models.py"} <= files
    assert not any("production_api" in f or f.endswith("runtime.py") and f.startswith("production/") for f in files)
