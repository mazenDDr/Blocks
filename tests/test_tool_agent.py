"""Model-chosen tools (ADR 0080): protocol against a scripted local server, refusals, budget; live Ollama native and /v1."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent import samples as sm
from agent.validate import validate_agent
from agent_helpers import Lab

SEEN = []
SCRIPT = {"mode": "calc"}


class Scripted(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
        offered = [t["function"]["name"] for t in body.get("tools", [])]
        if SCRIPT["mode"] == "calc" and not tool_msgs:
            msg = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "calculator", "arguments": json.dumps({"expression": "12*7"})}}]}
        elif SCRIPT["mode"] == "unoffered" and not tool_msgs:
            msg = {"role": "assistant", "content": "", "tool_calls": [{"id": "x1", "type": "function", "function": {"name": "write_note", "arguments": "{\"path\": \"a\", \"text\": \"b\"}"}}]}
        elif SCRIPT["mode"] == "loop" and offered:
            msg = {"role": "assistant", "content": "", "tool_calls": [{"id": f"l{len(tool_msgs)}", "type": "function", "function": {"name": "calculator", "arguments": json.dumps({"expression": "1+1"})}}]}
        else:
            msg = {"role": "assistant", "content": f"SYNTHETIC final after {len(tool_msgs)} tool results: " + (tool_msgs[-1]["content"] if tool_msgs else "none")}
        raw = json.dumps({"model": "stub", "choices": [{"message": msg, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}],
                          "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def stub():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Scripted)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    SEEN.clear()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def graph(model, max_calls=3, tools=("calculator",), model_calls=6, tool_calls=None):
    nodes = [sm.N("prompt", "agent.prompt", output_field="messages", items=[{"kind": "template", "role": "user", "template": "{question}"}]),
             sm.N("agent", "agent.tool_agent", model=model, messages_field="messages", output_field="answer", tools=list(tools), max_tool_calls=max_calls, calls_field="calls")]
    limits = {"maxSteps": 6, "maxModelCalls": model_calls, **({"maxToolCalls": tool_calls} if tool_calls else {})}
    return sm.make_graph(nodes, sm.chain("START", "prompt", "agent", "END"),
                         {"state": [sm.S("question"), sm.S("messages", "messages"), sm.S("answer"), sm.S("calls", "list", "replace", default=[])], "limits": limits})


def run(tmp_path, g, q="SYNTHETIC: what is 12 times 7?"):
    lab = Lab(tmp_path / "lab")
    status, rid = lab.run(g, inp={"question": q})
    final = json.loads(lab.store.read_artifact(lab.store.artifacts(rid, "final_state")[-1]["sha256"])) if lab.store.artifacts(rid, "final_state") else {}
    return lab, status, rid, final


def test_model_calls_the_offered_tool_and_answers_from_its_result(stub, tmp_path):
    SCRIPT["mode"] = "calc"
    lab, status, rid, final = run(tmp_path, graph({"provider": "openai_compatible", "model": "stub", "base_url": stub}))
    assert status == "completed", lab.store.get_run(rid)["error"]
    assert final["answer"].startswith("SYNTHETIC final after 1 tool results") and '"value": 84' in final["answer"]
    assert final["calls"][0]["tool"] == "calculator" and final["calls"][0]["result"]["value"] == 84 and final["calls"][0]["chosenBy"] == "model"
    first, second = SEEN
    assert [t["function"]["name"] for t in first["tools"]] == ["calculator"] and first["tools"][0]["function"]["parameters"]["required"] == ["expression"]
    assert second["messages"][-2]["tool_calls"][0]["function"]["name"] == "calculator" and second["messages"][-1] == {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"expression": "12*7", "value": 84})}
    events = lab.store.events(rid)
    assert [e["data"]["toolCalls"][0]["name"] for e in events if e["type"] == "model_call" and e["data"].get("toolCalls")] == ["calculator"]
    assert [e["data"]["tool"] for e in events if e["type"] == "tool_call"] == ["calculator"]


def test_unoffered_tools_are_refused_and_the_budget_forces_an_answer(stub, tmp_path):
    SCRIPT["mode"] = "unoffered"
    lab, status, rid, final = run(tmp_path / "a", graph({"provider": "openai_compatible", "model": "stub", "base_url": stub}))
    assert status == "completed" and final["calls"][0]["status"] == "refused" and final["calls"][0]["result"]["code"] == "E_TOOL_NOT_OFFERED"
    assert not [e for e in lab.store.events(rid) if e["type"] == "tool_call" and e["data"]["status"] == "ok"]  # write_note never ran
    assert not list((tmp_path / "a" / "lab").rglob("outbox/*"))
    SCRIPT["mode"] = "loop"
    SEEN.clear()
    lab, status, rid, final = run(tmp_path / "b", graph({"provider": "openai_compatible", "model": "stub", "base_url": stub}, max_calls=2))
    assert status == "completed" and len(final["calls"]) == 2 and final["answer"].startswith("SYNTHETIC final after 2")
    assert "tools" not in SEEN[-1]  # the last request offered no tools, so the model had to answer


def test_validation_refusals():
    for model, tools, code in (({"provider": "fixture", "model": "f"}, ["calculator"], "E_MODEL_TOOLS"),
                               ({"provider": "ollama", "model": "qwen3.5:2b"}, ["read_text_file"], "E_TOOL_BOUNDS")):
        rep = validate_agent(graph(model, tools=tools))
        assert any(d.code == code for d in rep.diagnostics), [d.code for d in rep.diagnostics]
    from pydantic import ValidationError
    from agent.blocks import ToolAgentConfig
    with pytest.raises(ValidationError):  # tools with external effects are not offered to a model; they stay behind an approval node
        ToolAgentConfig.model_validate({"tools": ["write_note"]})


@pytest.mark.live
# qwen3.5:2b answered 1234*5678 wrongly without calling the tool unless thinking was on (probed directly against Ollama);
# qwen3.5:4b calls it with thinking off, so the live check uses 4b.
@pytest.mark.parametrize("model", [{"provider": "ollama", "model": "qwen3.5:4b", "think": False, "temperature": 0.0, "seed": 7, "max_tokens": 128},
                                   {"provider": "openai_compatible", "model": "qwen3.5:4b", "base_url": "http://127.0.0.1:11434/v1", "temperature": 0.0, "seed": 7,
                                    "max_tokens": 128, "reasoning_effort": "none"}])
def test_live_local_model_chooses_the_calculator(tmp_path, model):
    lab, status, rid, final = run(tmp_path, graph(model), q="Use the calculator tool to compute 1234 * 5678, then reply with the result only.")
    assert status == "completed", lab.store.get_run(rid)["error"]
    print("LIVE TOOLS", model["provider"], [(c["tool"], c["args"], c["status"]) for c in final["calls"]], repr(final["answer"][:80]))
    assert any(c["tool"] == "calculator" and c["status"] == "ok" and c["result"]["value"] == 7006652 for c in final["calls"])
