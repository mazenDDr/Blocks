"""LIVE tests against the local Ollama server (the one tested provider path). Run explicitly:

    pytest -q -m live

They need Ollama at http://localhost:11434 with qwen3.5:2b and nomic-embed-text; they are skipped (with the reason) otherwise. Assertions are about
what was recorded (real provider usage, the exact request, control flow), never about the wording of a generated answer."""
import copy
import json

import pytest

from agent import samples as sm
from agent.memory import MemoryStore
from agent.models import Embedder, ModelSpec, invoke_chat, ollama_context_length, ollama_status
from agent.spec import EmbeddingsSpec
from agent_helpers import Lab, REPO

pytestmark = pytest.mark.live
MODEL = "qwen3.5:2b"


@pytest.fixture(scope="module", autouse=True)
def need_ollama():
    st = ollama_status()
    if not st["reachable"]:
        pytest.skip("Ollama is not reachable at http://localhost:11434")
    missing = [m for m in (MODEL, "nomic-embed-text:latest") if m not in st["models"]]
    if missing:
        pytest.skip(f"Ollama is missing models: {missing}")


def test_live_chat_reports_real_usage_and_honours_settings():
    out = invoke_chat(ModelSpec(provider="ollama", model=MODEL, temperature=0, seed=3, max_tokens=24), [{"role": "user", "content": "Reply with exactly: pong"}])
    assert out["text"].strip() and out["usage"]["source"] == "provider" and out["usage"]["inputTokens"] > 0 and 0 < out["usage"]["outputTokens"] <= 24
    assert out["latencyMs"] > 0
    again = invoke_chat(ModelSpec(provider="ollama", model=MODEL, temperature=0, seed=3, max_tokens=24), [{"role": "user", "content": "Reply with exactly: pong"}])
    assert again["text"] == out["text"]  # same seed + temperature 0 on the same runtime
    assert ollama_context_length(MODEL) and ollama_context_length(MODEL) > 1000


def test_live_embeddings_are_semantic_and_deterministic():
    e = Embedder(EmbeddingsSpec(provider="ollama", model="nomic-embed-text", normalize=True))
    v = e.embed(["a car drives on the road", "an automobile moves along the highway", "a banana is a yellow fruit"])
    from agent.models import cosine

    assert len(v[0]) == 768 and cosine(v[0], v[1]) > cosine(v[0], v[2])
    assert e.embed(["a car drives on the road"])[0] == pytest.approx(v[0], abs=1e-4)
    assert e.identity == "ollama:nomic-embed-text:norm1"


def test_live_structured_output_validates_real_json(tmp_path):
    g = sm.make_graph([sm.N("p", "agent.prompt", output_field="m", items=[{"kind": "template", "role": "system", "template": "You grade answers."},
                                                                     {"kind": "template", "role": "user", "template": "Context: the bath limit is 40 C. Answer: the limit is 40 C. Is the answer supported by the context?"}]),
                       sm.N("s", "agent.structured_output", model=sm.ollama_model(MODEL, max_tokens=120), messages_field="m", output_field="res", error_field="err",
                            schema_fields=[{"name": "supported", "type": "boolean"}, {"name": "reason", "type": "text"}], retry={"maxRetries": 2, "feedback": True})],
                      [sm.E("START", "p"), sm.E("p", "s"), sm.E("s", "END")], {"state": [sm.S("m", "list"), sm.S("res", "object", properties={"supported": "boolean", "reason": "text"}), sm.S("err")],
                                                                                      "limits": {"maxSteps": 6}})
    lab = Lab(tmp_path)
    st, rid = lab.run(g)
    f = lab.final(rid)
    calls = lab.events(rid, "model_call")
    assert st == "completed" and calls and all(c["data"]["fixture"] is False and c["data"]["usage"]["source"] == "provider" for c in calls)
    assert f["err"] == "" and isinstance(f["res"]["supported"], bool) and isinstance(f["res"]["reason"], str)


def test_live_retrieval_with_bounded_revision_example(tmp_path):
    """The VISION 12.3 workflow exactly as shipped in examples/, with the real model and real embeddings."""
    from graph_core.project_io import load_project

    g = load_project(REPO / "examples" / "agent_retrieval_revision.project.json").graph
    lab = Lab(tmp_path)
    st, rid = lab.run(g, inp={"question": "What is the highest temperature the Kinase-7 water bath may be set to?"})
    f = lab.final(rid)
    calls = lab.events(rid, "model_call")
    assert st == "completed" and lab.finished(rid)["stoppedBy"] == "end"
    assert calls and all(not c["data"]["fixture"] and c["data"]["provider"] == "ollama" and c["data"]["usage"]["source"] == "provider" for c in calls)
    ret = lab.events(rid, "retrieval")[0]["data"]
    assert ret["embedding"] == "ollama:nomic-embed-text:norm1" and ret["included"][0]["chunk_id"].startswith("enzyme_assay_protocol.txt")
    assert f["status"] in ("answered", "unresolved") and f["attempts"] <= 2
    chk = f["citation_check"]
    print("\nLIVE retrieval example:", json.dumps({"status": f["status"], "attempts": f["attempts"], "answer": f["answer"], "citations": chk, "grade": f["grade"],
                                                    "modelCalls": len(calls), "latencyMs": round(sum(c["data"]["latencyMs"] for c in calls)), "tokens": sum((c["data"]["usage"]["inputTokens"] or 0) + (c["data"]["usage"]["outputTokens"] or 0) for c in calls)}, indent=1))
    for c in lab.contexts(rid):  # every real call has provider-reported counts next to the estimate
        assert c["tokens"]["providerReported"] and c["tokens"]["providerInput"] > 0 and c["tokens"]["modelLimit"] and not c["fixture"]


def test_live_memory_debugging_journey(tmp_path):
    """The journey with the real model: the answer ignores a constraint that was never in its context; after the visual policy edit the same
    constraint is in the context. (What the model then says is reported, not asserted.)"""
    from agent.preview import preview_policy_edit

    lab = Lab(tmp_path)
    mem = MemoryStore(lab.wb)
    for r in sm.memory_seed():
        mem.put_record(r)
    model = sm.ollama_model(MODEL, max_tokens=120)
    g = sm.memory_debug_graph(model)
    st, r1 = lab.run(g, thread="live1", inp={"question": sm.CONSTRAINT_QUESTION})
    assert st == "completed"
    ctx1 = lab.contexts(r1)[0]
    assert not ctx1["fixture"] and ctx1["usage"]["source"] == "provider"
    assert "40 degrees" not in json.dumps(ctx1["providerRequest"])
    ex = {e["id"]: e for e in ctx1["excluded"]}
    assert ex["mem_constraint"]["stage"] == "rank"
    edited = copy.deepcopy(g.to_json())
    next(s for s in edited["agent"]["policies"][0]["stages"] if s["id"] == "rank")["config"]["weights"] = {"recency": 0.2, "relevance": 1.0, "importance": 0.6}
    from graph_core.schema import Graph

    eg = Graph.model_validate(edited)
    pv = preview_policy_edit(lab.store, mem, eg, r1, ctx1["callId"])
    assert pv["noModelCall"] and "40 degrees" in json.dumps(pv["after"]["messages"])
    st, r2 = lab.run(eg, thread="live2", inp={"question": sm.CONSTRAINT_QUESTION})
    ctx2 = lab.contexts(r2)[0]
    assert "40 degrees" in json.dumps(ctx2["providerRequest"]) and "mem_constraint" not in {e["id"] for e in ctx2["excluded"]}
    a1, a2 = lab.final(r1)["answer"], lab.final(r2)["answer"]
    print("\nLIVE memory journey\n  before edit:", a1.strip()[:300].replace("\n", " "), "\n  after edit: ", a2.strip()[:300].replace("\n", " "))
    assert a1.strip() and a2.strip()


def test_live_unanswerable_question_exercises_revision_and_budget_with_the_real_model(tmp_path):
    from graph_core.project_io import load_project

    g = load_project(REPO / "examples" / "agent_retrieval_revision.project.json").graph
    lab = Lab(tmp_path)
    st, rid = lab.run(g, inp={"question": "How many moons does Jupiter have?"})  # not in the synthetic corpus
    f = lab.final(rid)
    rets = lab.events(rid, "retrieval")
    assert st == "completed" and len(rets) == f["attempts"] + 1 and f["attempts"] <= 2
    if f["status"] == "unresolved":
        assert f["attempts"] == 2 and lab.events(rid, "route_taken")[-1]["data"]["via"] == "default"  # the bounded revision budget ended the loop
    print("\nLIVE revision run:", json.dumps({"status": f["status"], "attempts": f["attempts"], "queries": [r["data"]["query"] for r in rets]}))
