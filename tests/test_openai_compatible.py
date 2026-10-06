"""OpenAI-compatible provider: protocol against a local stub server (offline) and Ollama's real /v1 endpoint (live)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent import samples as sm
from agent.blocks import ChatModelConfig
from agent.models import TOKEN_SINK, ModelSpec, ModelUnavailable, invoke_chat
from agent.openai_compat import check_base_url
from agent.validate import validate_agent
from agent_helpers import Lab
from graph_core.registry import get_op

SEEN = []


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        last = body["messages"][-1]["content"]
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in ("SYNTHETIC ", "echo: ", last):
                self.wfile.write(f"data: {json.dumps({'model': 'stub', 'choices': [{'delta': {'content': piece}}]})}\n\n".encode())
            self.wfile.write(f"data: {json.dumps({'model': 'stub', 'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode())
            self.wfile.write(f"data: {json.dumps({'model': 'stub', 'choices': [], 'usage': {'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10}})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        out = {"model": "stub", "choices": [{"message": {"role": "assistant", "content": f"SYNTHETIC echo: {last}"}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def stub():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    SEEN.clear()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


MSGS = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hello"}]


def test_invoke_and_stream_report_text_usage_and_send_settings(stub, monkeypatch):
    spec = ModelSpec(provider="openai_compatible", model="stub-model", base_url=stub, temperature=0.0, max_tokens=12, seed=3)
    r = invoke_chat(spec, MSGS)
    assert r["text"] == "SYNTHETIC echo: hello" and r["usage"] == {"inputTokens": 7, "outputTokens": 3, "source": "provider"}
    sent = SEEN[-1]
    assert sent["path"] == "/v1/chat/completions" and sent["auth"] is None
    assert sent["body"]["messages"] == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hello"}]
    assert (sent["body"]["model"], sent["body"]["temperature"], sent["body"]["max_tokens"], sent["body"]["seed"], sent["body"]["stream"]) == ("stub-model", 0.0, 12, 3, False)
    assert "reasoning_effort" not in sent["body"]  # only sent when set
    invoke_chat(spec.model_copy(update={"reasoning_effort": "low"}), MSGS)
    assert SEEN[-1]["body"]["reasoning_effort"] == "low"
    deltas = []
    token = TOKEN_SINK.set(deltas.append)
    try:
        s = invoke_chat(spec, MSGS)
    finally:
        TOKEN_SINK.reset(token)
    assert deltas == ["SYNTHETIC ", "echo: ", "hello"] and s["text"] == "SYNTHETIC echo: hello"
    assert s["usage"] == {"inputTokens": 7, "outputTokens": 3, "source": "provider"} and SEEN[-1]["body"]["stream_options"] == {"include_usage": True}
    monkeypatch.setenv("SYNTHETIC_COMPAT_KEY", "SYNTHETIC-secret-value")
    invoke_chat(spec.model_copy(update={"api_key": {"kind": "env", "name": "SYNTHETIC_COMPAT_KEY"}}), MSGS)
    assert SEEN[-1]["auth"] == "Bearer SYNTHETIC-secret-value"


def test_endpoint_rules_validation_and_effects():
    assert check_base_url("http://127.0.0.1:8000/v1/", True) == "http://127.0.0.1:8000/v1"
    assert check_base_url("https://api.example.invalid/v1", True)
    for url, key in ((None, False), ("ftp://x/v1", False), ("http://api.example.invalid/v1", True)):
        with pytest.raises(ValueError):
            check_base_url(url, key)
    with pytest.raises(ModelUnavailable) as caught:
        invoke_chat(ModelSpec(provider="openai_compatible", model="m", base_url="http://api.example.invalid/v1", api_key={"kind": "env", "name": "PATH"}), MSGS)
    assert caught.value.code == "E_MODEL_ENDPOINT"
    op = get_op("agent.chat_model")
    local = ChatModelConfig.model_validate({"model": {"provider": "openai_compatible", "model": "m", "base_url": "http://127.0.0.1:1/v1"}})
    remote = ChatModelConfig.model_validate({"model": {"provider": "openai_compatible", "model": "m", "base_url": "https://api.example.invalid/v1"}})
    assert op.effects(local) == ["local_runtime"] and op.effects(remote) == ["network"]
    g = _graph({"provider": "openai_compatible", "model": "m"})
    assert any(d.code == "E_MODEL_ENDPOINT" for d in validate_agent(g).diagnostics)


def _graph(model):
    nodes = [sm.N("prompt", "agent.prompt", output_field="messages", items=[{"kind": "template", "role": "user", "template": "{question}"}]),
             sm.N("answer", "agent.chat_model", messages_field="messages", output_field="answer", model=model)]
    return sm.make_graph(nodes, sm.chain("START", "prompt", "answer", "END"),
                         {"state": [sm.S("question"), sm.S("messages", "messages"), sm.S("answer")], "limits": {"maxSteps": 6, "maxModelCalls": 1}})


def test_research_run_records_provider_call_through_the_protocol(stub, tmp_path):
    lab = Lab(tmp_path / "lab")
    status, rid = lab.run(_graph({"provider": "openai_compatible", "model": "stub-model", "base_url": stub, "max_tokens": 16}), inp={"question": "SYNTHETIC ping"})
    assert status == "completed", lab.store.get_run(rid)["error"]
    call = next(e for e in lab.store.events(rid) if e["type"] == "model_call")["data"]
    assert call["provider"] == "openai_compatible" and call["response"] == "SYNTHETIC echo: SYNTHETIC ping"
    assert call["usage"]["source"] == "provider" and call["cost"]["amount"] is None


@pytest.mark.live
def test_live_ollama_v1_endpoint_invoke_stream_and_run(tmp_path):
    # qwen3.5 reasons by default; through /v1 the reasoning would consume max_tokens, so ask for none explicitly.
    spec = ModelSpec(provider="openai_compatible", model="qwen3.5:0.8b", base_url="http://127.0.0.1:11434/v1", temperature=0.0, max_tokens=24, seed=7, reasoning_effort="none")
    msgs = [{"role": "user", "content": "Reply with the single word: ready"}]
    r = invoke_chat(spec, msgs)
    assert r["text"].strip() and r["usage"]["source"] == "provider" and r["usage"]["outputTokens"] > 0
    deltas = []
    token = TOKEN_SINK.set(deltas.append)
    try:
        s = invoke_chat(spec.model_copy(update={"max_tokens": 48}), [{"role": "user", "content": "Count from 1 to 10, separated by spaces."}])
    finally:
        TOKEN_SINK.reset(token)
    assert len(deltas) > 1 and "".join(deltas) == s["text"]
    lab = Lab(tmp_path / "lab")
    status, rid = lab.run(_graph({"provider": "openai_compatible", "model": "qwen3.5:0.8b", "base_url": "http://127.0.0.1:11434/v1", "max_tokens": 24, "temperature": 0.0, "reasoning_effort": "none"}),
                          inp={"question": "Reply with the single word: ready"})
    assert status == "completed", lab.store.get_run(rid)["error"]
    call = next(e for e in lab.store.events(rid) if e["type"] == "model_call")["data"]
    print("LIVE", {"invoke": r["text"][:60], "usage": r["usage"], "streamChunks": len(deltas), "run": call["response"][:60]})


def test_structured_output_sends_response_format(stub):
    schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
    invoke_chat(ModelSpec(provider="openai_compatible", model="m", base_url=stub), MSGS, json_schema=schema)
    assert SEEN[-1]["body"]["response_format"] == {"type": "json_schema", "json_schema": {"name": "result", "strict": True, "schema": schema}}
    invoke_chat(ModelSpec(provider="openai_compatible", model="m", base_url=stub), MSGS)
    assert "response_format" not in SEEN[-1]["body"]


@pytest.mark.live
def test_live_structured_output_over_v1(tmp_path):
    from agent.blocks import SchemaField
    nodes = [sm.N("prompt", "agent.prompt", output_field="messages", items=[{"kind": "template", "role": "user", "template": "{question}"}]),
             sm.N("extract", "agent.structured_output", messages_field="messages", output_field="result", on_failure="fail",
                  schema_fields=[{"name": "colour", "type": "text"}, {"name": "count", "type": "integer"}],
                  model={"provider": "openai_compatible", "model": "qwen3.5:2b", "base_url": "http://127.0.0.1:11434/v1", "temperature": 0.0, "max_tokens": 64, "reasoning_effort": "none"})]
    g = sm.make_graph(nodes, sm.chain("START", "prompt", "extract", "END"),
                      {"state": [sm.S("question"), sm.S("messages", "messages"), sm.S("result", "object", properties={"colour": "text", "count": "integer"})], "limits": {"maxSteps": 6, "maxModelCalls": 3}})
    lab = Lab(tmp_path / "lab")
    status, rid = lab.run(g, inp={"question": "The SYNTHETIC record says: colour teal, count 3. Extract them."})
    assert status == "completed", lab.store.get_run(rid)["error"]
    import json as _json
    final = _json.loads(lab.store.read_artifact(lab.store.artifacts(rid, "final_state")[-1]["sha256"]))
    print("LIVE STRUCTURED", final["result"])
    assert final["result"] == {"colour": "teal", "count": 3}
