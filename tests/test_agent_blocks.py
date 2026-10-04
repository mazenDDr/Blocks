"""Agent blocks: prompt segments, model calls (fixture), structured output with retries, embeddings, document index and retriever, citations, tools."""
import json
import os
import time
from pathlib import Path

import pytest

from agent import samples as sm
from agent.blocks import SchemaField, json_schema_of, render_prompt, validate_structured, PromptConfig
from agent.index import IndexStore, load_documents, split_documents
from agent.models import Embedder, ModelSpec, ModelUnavailable, build_chat_model, estimate_tokens, resolve_settings
from agent.spec import IndexSpec
from agent.tools import ToolError, TOOLS, inside, safe_eval
from agent.validate import validate_agent
from agent_helpers import Lab, copy_docs
from graph_core.schema import Graph


def one_node_graph(node, state, extra=None, route_edges=None):
    nodes = [node]
    edges = [sm.E("START", node["id"]), sm.E(node["id"], "END")]
    return sm.make_graph(nodes, edges, {"state": state, "limits": {"maxSteps": 10}, **(extra or {})})


# ------------------------------------------------------------------------------------------------ prompt blocks and segments
def test_prompt_segments_link_every_range_to_its_source():
    cfg = PromptConfig(output_field="m", items=[
        {"kind": "template", "role": "system", "template": "Be brief about {topic}."},
        {"kind": "memory", "role": "system", "field": "mem", "header": "Notes:", "template": "- ({id}) {text}"},
        {"kind": "documents", "role": "system", "field": "docs", "header": "Context:", "template": "[{chunk_id}] {text}"},
        {"kind": "tool_results", "role": "system", "field": "tools", "header": "Tools:"},
        {"kind": "template", "role": "user", "template": "{question}"}])
    state = {"topic": "enzymes", "question": "why?", "mem": [{"id": "m1", "text": "note one", "store": "long_term", "kind": "semantic", "applicationId": "app_1"}],
             "docs": [{"chunk_id": "a.txt#0", "doc_id": "a.txt", "text": "chunk text", "score": 0.5, "retrievalId": "ret_1", "source_path": "/x/a.txt"}],
             "tools": [{"tool": "calculator", "args": {"expression": "1+1"}, "result": {"value": 2}, "callId": "tool_1"}]}
    msgs = render_prompt(cfg, state, "p")
    assert [m["role"] for m in msgs] == ["system", "system", "system", "system", "user"]
    assert msgs[0]["content"] == "Be brief about enzymes."
    for m in msgs:  # segments tile the message exactly and each has a source
        assert m["segments"][0]["start"] == 0 and m["segments"][-1]["end"] == len(m["content"])
        assert all(s["source"]["kind"] for s in m["segments"])
    mem = msgs[1]
    seg = mem["segments"][1]
    assert seg["source"]["kind"] == "memory_record" and seg["source"]["recordId"] == "m1" and seg["source"]["applicationId"] == "app_1"
    assert mem["content"][seg["start"]:seg["end"]] == "- (m1) note one\n"
    d = msgs[2]["segments"][1]["source"]
    assert d["kind"] == "retrieved_chunk" and d["chunkId"] == "a.txt#0" and d["retrievalId"] == "ret_1"
    assert msgs[3]["segments"][1]["source"]["kind"] == "tool_result" and "calculator" in msgs[3]["content"]
    assert msgs[4]["segments"][0]["source"]["variables"] == ["question"]


def test_empty_selections_add_no_message_and_prompt_fields_are_type_checked():
    cfg = PromptConfig(items=[{"kind": "memory", "field": "mem", "header": "Notes:"}, {"kind": "template", "role": "user", "template": "{q}"}])
    assert [m["role"] for m in render_prompt(cfg, {"mem": [], "q": "x"}, "p")] == ["user"]
    g = one_node_graph(sm.N("p", "agent.prompt", items=[{"kind": "memory", "field": "q"}], output_field="out"), [sm.S("q"), sm.S("out", "list")])
    assert "E_STATE_TYPE" in {d.code for d in validate_agent(g).diagnostics}


# ------------------------------------------------------------------------------------------------ model calls, labelled fixture
def test_chat_model_call_records_the_exact_request_and_labels_the_fixture(tmp_path):
    g = sm.make_graph([sm.N("p", "agent.prompt", output_field="m", items=[{"kind": "template", "role": "system", "template": "S {q}"}, {"kind": "template", "role": "user", "template": "U"}]),
                       sm.N("c", "agent.chat_model", model=sm.fixture_model(default="FIXTURE hi"), messages_field="m", output_field="a")],
                      [sm.E("START", "p"), sm.E("p", "c"), sm.E("c", "END")], {"state": [sm.S("q"), sm.S("m", "list"), sm.S("a")], "limits": {"maxSteps": 10}})
    lab = Lab(tmp_path)
    st, rid = lab.run(g, inp={"q": "Z"})
    assert st == "completed" and lab.final(rid)["a"] == "FIXTURE hi"
    call = lab.events(rid, "model_call")[0]["data"]
    assert call["fixture"] is True and call["provider"] == "fixture" and call["usage"]["source"] == "unavailable"
    assert call["usage"]["inputTokens"] is None  # unknown stays unknown
    assert call["cost"]["amount"] is None
    ctx = lab.contexts(rid)[0]
    assert ctx["providerRequest"] == [{"role": "system", "content": "S Z"}, {"role": "user", "content": "U"}]
    assert ctx["tokens"]["providerReported"] is False and ctx["tokens"]["estimateTotal"] == estimate_tokens("S Z") + estimate_tokens("U")
    assert "estimate" in ctx["tokens"]["estimateBasis"] and "observable" in ctx["boundary"]
    assert validate_agent(g).analysis["usesFixtureModel"] is True
    assert any(d.code == "W_FIXTURE_MODEL" for d in validate_agent(g).diagnostics)


def test_only_supported_settings_are_applied_and_ignored_ones_are_explained():
    r, ign = resolve_settings(ModelSpec(provider="anthropic", temperature=0.2, max_tokens=50, seed=3, think=True))
    assert r["temperature"] == 0.2 and r["max_tokens"] == 50 and "seed" not in r
    assert set(ign) == {"seed", "think"} and "no sampling seed" in ign["seed"]
    r, ign = resolve_settings(ModelSpec(provider="ollama", seed=3))
    assert r["seed"] == 3 and not ign


def test_structured_output_schema_and_validation():
    fields = [SchemaField(name="ok", type="boolean"), SchemaField(name="n", type="integer"), SchemaField(name="kind", type="enum", choices=["a", "b"]),
              SchemaField(name="tags", type="list_of_text", required=False)]
    s = json_schema_of(fields)
    assert s["required"] == ["ok", "n", "kind"] and s["properties"]["kind"]["enum"] == ["a", "b"] and s["additionalProperties"] is False
    assert validate_structured('{"ok": true, "n": 2, "kind": "a"}', fields)[0] == {"ok": True, "n": 2, "kind": "a"}
    assert validate_structured('```json\n{"ok": false, "n": 1, "kind": "b", "tags": ["x"]}\n```', fields)[1] == []
    assert "missing required field 'n'" in validate_structured('{"ok": true, "kind": "a"}', fields)[1][0]
    assert any("must be integer" in e for e in validate_structured('{"ok": true, "n": true, "kind": "a"}', fields)[1])  # bool is not an integer
    assert any("must be enum" in e for e in validate_structured('{"ok": true, "n": 1, "kind": "z"}', fields)[1])
    assert any("unexpected" in e for e in validate_structured('{"ok": true, "n": 1, "kind": "a", "x": 1}', fields)[1])
    assert "not valid JSON" in validate_structured("oops", fields)[1][0]


def structured_graph(model, retries=2, on_failure="route", feedback=True):
    return one_node_graph(
        sm.N("s", "agent.structured_output", model=model, messages_field="m", output_field="res", error_field="err", schema_fields=[{"name": "supported", "type": "boolean"}],
             retry={"maxRetries": retries, "feedback": feedback}, on_failure=on_failure),
        [sm.S("m", "list", default=[{"role": "user", "content": "grade it", "segments": []}]), sm.S("res", "object", properties={"supported": "boolean"}), sm.S("err")])


def test_structured_output_retries_with_feedback_then_succeeds(tmp_path):
    lab = Lab(tmp_path)
    st, rid = lab.run(structured_graph(sm.fixture_model(fail_first=2, responses=['{"supported": true}'])))
    assert st == "completed" and lab.final(rid)["res"] == {"supported": True} and lab.final(rid)["err"] == ""
    att = lab.events(rid, "structured_attempt")
    assert [(a["data"]["attempt"], a["data"]["valid"]) for a in att] == [(1, False), (2, False), (3, True)]
    assert "not valid JSON" in att[0]["data"]["errors"][0]
    calls = lab.contexts(rid)
    assert [c["attempt"] for c in calls] == [1, 2, 3]
    # the retry request really contains the rejected reply and the validation feedback (visible in the context record)
    kinds = [s["source"]["kind"] for m in calls[2]["messages"] for s in m["segments"]]
    assert "structured_output_retry_feedback" in kinds and "structured_output_instruction" in kinds


def test_structured_output_exhausts_retries_and_routes_or_fails(tmp_path):
    lab = Lab(tmp_path)
    st, rid = lab.run(structured_graph(sm.fixture_model(fail_first=9), retries=1))
    f = lab.final(rid)
    assert st == "completed" and f["res"] == {} and "validation failed after 2 attempts" in f["err"]
    assert len(lab.events(rid, "structured_attempt")) == 2
    st, rid = lab.run(structured_graph(sm.fixture_model(fail_first=9), retries=1, on_failure="fail"))
    assert st == "failed" and "E_STRUCTURED_OUTPUT" in lab.store.get_run(rid)["error"]


def test_schema_must_agree_with_the_declared_state_object(tmp_path):
    g = structured_graph(sm.fixture_model(default='{"supported": true}'))
    g = Graph.model_validate({**g.to_json(), "agent": {**g.agent, "state": [{**f, "properties": {"supported": "text"}} if f["name"] == "res" else f for f in g.agent["state"]]}})
    assert "E_STATE_TYPE" in {d.code for d in validate_agent(g).diagnostics}


# ------------------------------------------------------------------------------------------------ embeddings
def test_local_hash_embedding_is_deterministic_normalized_and_labelled():
    from agent.spec import EmbeddingsSpec

    e = Embedder(EmbeddingsSpec(provider="local_hash", dimension=64))
    a, b = e.embed(["the enzyme assay"]), e.embed(["the enzyme assay"])
    assert a == b and len(a[0]) == 64 and abs(sum(x * x for x in a[0]) - 1) < 1e-6
    assert "NOT a semantic model" in e.label
    assert e.identity == "local_hash:d64:norm1"
    near, far = e.embed(["enzyme assay temperature"])[0], e.embed(["pipette calibration"])[0]
    from agent.models import cosine
    assert cosine(a[0], near) > cosine(a[0], far)


def test_embed_text_block_records_identity_dimension_norm(tmp_path):
    g = one_node_graph(sm.N("e", "agent.embed_text", from_field="q", output_field="info", dimension=32), [sm.S("q"), sm.S("info", "object")])
    lab = Lab(tmp_path)
    _, rid = lab.run(g, inp={"q": "hello world"})
    info = lab.final(rid)["info"]
    assert info["dimension"] == 32 and info["identity"] == "local_hash:d32:norm1" and abs(info["norm"] - 1) < 1e-4
    assert lab.events(rid, "embedding")


# ------------------------------------------------------------------------------------------------ documents, splitter, vector store, retriever
def spec_for(d, **kw):
    return IndexSpec.model_validate({"id": "ix", "loader": {"directory": d, "glob": "*.txt"}, "splitter": {"chunkSize": 300, "chunkOverlap": 30}, "embeddings": {"provider": "local_hash", "dimension": 256}, **kw})


def test_loader_splitter_metadata_and_overlap(tmp_path):
    d = copy_docs(tmp_path)
    docs = load_documents(spec_for(d))
    assert sorted(x["doc_id"] for x in docs) == sorted(p.name for p in Path(d).glob("*.txt")) and all(len(x["sha256"]) == 64 for x in docs)
    ch = split_documents(docs, spec_for(d))
    assert all(c["chunk_id"] == f"{c['doc_id']}#{c['index']}" and len(c["text"]) <= 300 for c in ch)
    one = [c for c in ch if c["doc_id"] == "enzyme_assay_protocol.txt"]
    assert len(one) >= 2 and one[0]["start"] == 0
    # chunk text is the source text at its recorded offset: citations can be traced back to the file
    src = Path(d, "enzyme_assay_protocol.txt").read_text()
    assert all(src[c["start"]:c["start"] + len(c["text"])] == c["text"] for c in one)
    fixed = split_documents(docs, spec_for(d, splitter={"strategy": "fixed", "chunkSize": 100, "chunkOverlap": 0}))
    assert len(fixed) > len(ch)


def test_index_is_persisted_reused_and_incrementally_rebuilt(tmp_path):
    d = copy_docs(tmp_path)
    st = IndexStore(tmp_path / "wb")
    m1 = st.ensure(spec_for(d))
    assert m1["action"] == "built" and m1["chunks"] >= 5 and (tmp_path / "wb" / "agent" / "indexes" / "ix" / "faiss.index").exists()
    assert m1["embeddingsReused"] == 0 and m1["embeddingsComputed"] > 0
    assert IndexStore(tmp_path / "wb").ensure(spec_for(d))["action"] == "reused"  # a fresh store object reads the persisted index
    Path(d, "centrifuge_safety.txt").write_text(Path(d, "centrifuge_safety.txt").read_text() + "\nA new sentence about spill kits.\n")
    m3 = st.ensure(spec_for(d))
    assert m3["action"] == "built" and 1 <= m3["embeddingsComputed"] < m1["embeddingsComputed"] / 2 and m3["embeddingsReused"] > m3["embeddingsComputed"]  # unchanged chunk embeddings come from the cache
    m4 = st.ensure(spec_for(d, embeddings={"provider": "local_hash", "dimension": 128}))
    assert m4["action"] == "built" and m4["embeddingsReused"] == 0  # a different embedding identity never reuses vectors


def test_retrieval_scores_threshold_and_exclusions(tmp_path):
    d = copy_docs(tmp_path)
    st = IndexStore(tmp_path / "wb")
    spec = spec_for(d)
    st.ensure(spec)
    r = st.search(spec, "what temperature must the Kinase-7 water bath never exceed", 3, None)
    ids = [x["chunk_id"] for x in r["documents"]]
    assert ids[0].startswith("enzyme_assay_protocol.txt") and len(ids) == 3
    scores = [x["score"] for x in r["documents"]]
    assert scores == sorted(scores, reverse=True) and all(0 <= s <= 1.0001 for s in scores)
    assert "cosine" in r["scoreInterpretation"]
    assert all("below the selected limit" in x["reason"] for x in r["excluded"] if x["rank"] > 3)
    cut = (scores[0] + scores[1]) / 2
    r2 = st.search(spec, "what temperature must the Kinase-7 water bath never exceed", 3, cut)
    assert [x["chunk_id"] for x in r2["documents"]] == ids[:1] and any("below the score threshold" in x["reason"] for x in r2["excluded"])
    with pytest.raises(ValueError):
        st.search(spec_for(d, embeddings={"provider": "local_hash", "dimension": 64}), "q", 1, None)  # query embedder differs from the index's


def test_retrieve_block_validates_the_index_reference_and_records_the_retrieval(tmp_path):
    d = copy_docs(tmp_path)
    ix = {"id": "ix", "loader": {"directory": d, "glob": "*.txt"}, "splitter": {"chunkSize": 300, "chunkOverlap": 30}, "embeddings": {"provider": "local_hash", "dimension": 256}}
    g = one_node_graph(sm.N("r", "agent.retrieve", index="ix", query="{q}", k=2, output_field="docs"), [sm.S("q"), sm.S("docs", "documents")], {"indexes": [ix]})
    lab = Lab(tmp_path)
    st, rid = lab.run(g, inp={"q": "pipette calibration relative standard deviation"})
    docs = lab.final(rid)["docs"]
    assert st == "completed" and docs[0]["doc_id"] == "pipette_calibration.txt" and len(docs) == 2 and docs[0]["retrievalId"]
    ev = lab.events(rid, "retrieval")[0]["data"]
    assert ev["index"] == "ix" and ev["k"] == 2 and ev["embedding"] == "local_hash:d256:norm1" and [i["chunk_id"] for i in ev["included"]] == [x["chunk_id"] for x in docs]
    assert lab.events(rid, "index_ready")[0]["data"]["action"] == "built"
    bad = one_node_graph(sm.N("r", "agent.retrieve", index="ghost", query="{q}", output_field="docs"), [sm.S("q"), sm.S("docs", "documents")], {"indexes": [ix]})
    assert "E_INDEX_UNKNOWN" in {x.code for x in validate_agent(bad).diagnostics}


def test_citations_are_checked_against_what_was_retrieved(tmp_path):
    g = one_node_graph(sm.N("c", "agent.citations", text_field="a", docs_field="docs", output_field="chk"),
                       [sm.S("a"), sm.S("docs", "documents"), sm.S("chk", "object")])
    lab = Lab(tmp_path)
    docs = [{"chunk_id": "x.txt#0", "doc_id": "x.txt", "source_path": "/p/x.txt", "score": 0.4, "text": "t"}]
    _, rid = lab.run(g, inp={"a": "A fact [x.txt#0] and an invented one [y.txt#9] and again [x.txt#0].", "docs": docs})
    chk = lab.final(rid)["chk"]
    assert chk["count"] == 2 and chk["invalid"] == 1 and chk["allValid"] is False
    good = [c for c in chk["citations"] if c["valid"]][0]
    assert good["source_path"] == "/p/x.txt" and good["score"] == 0.4  # the origin comes from the retrieval record, not from the generated text
    _, rid = lab.run(g, inp={"a": "no markers at all", "docs": docs})
    assert lab.final(rid)["chk"]["allValid"] is False and lab.final(rid)["chk"]["count"] == 0


# ------------------------------------------------------------------------------------------------ tools with bounded capabilities
def test_calculator_is_arithmetic_only():
    assert safe_eval("12 * (3 + 4) - 2 ** 3") == 76 and safe_eval("sqrt(16) + abs(-2)") == 6
    for bad in ("__import__('os').system('true')", "open('x')", "a + 1", "[1][0]", "(lambda: 1)()", "2 ** 1000", "1/0"):
        with pytest.raises(ToolError):
            safe_eval(bad)


def test_file_tools_cannot_leave_their_allowed_directory(tmp_path):
    base = tmp_path / "ok"
    base.mkdir()
    (base / "a.txt").write_text("hello")
    (tmp_path / "secret.txt").write_text("nope")
    (base / "link").symlink_to(tmp_path / "secret.txt")
    t = TOOLS["read_text_file"]
    assert t.run({"path": "a.txt"}, base)["text"] == "hello"
    for p in ("../secret.txt", "/etc/hosts", "link", "sub/../../secret.txt"):
        with pytest.raises(ToolError) as e:
            t.run({"path": p}, base)
        assert e.value.code == "E_TOOL_PATH"
    with pytest.raises(ToolError):
        inside(base, "../ok2")  # a sibling that merely shares the prefix


def test_tool_declarations_effects_and_validation():
    d = {n: t.describe() for n, t in TOOLS.items()}
    assert d["calculator"]["effects"] == [] and not d["calculator"]["external"]
    assert d["read_text_file"]["effects"] == ["file_read"] and not d["read_text_file"]["requiresApproval"]
    assert d["write_note"]["effects"] == ["file_write"] and d["write_note"]["external"] and d["write_note"]["requiresApproval"]
    g = sm.approval_tools_graph("/tmp/x")
    assert not [x for x in validate_agent(g).diagnostics if x.severity == "error"]
    bad = Graph.model_validate({**g.to_json(), "nodes": [{**n, "config": {**n["config"], "require_approval": False}} if n["id"] == "write" else n for n in g.to_json()["nodes"]]})
    assert "E_TOOL_APPROVAL" in {x.code for x in validate_agent(bad).diagnostics}
    bad2 = Graph.model_validate({**g.to_json(), "nodes": [{**n, "config": {**n["config"], "args": {"x": "1"}}} if n["id"] == "calc" else n for n in g.to_json()["nodes"]]})
    assert "E_TOOL_ARGS" in {x.code for x in validate_agent(bad2).diagnostics}


def test_read_tool_inside_a_graph_records_the_call_and_errors_become_results(tmp_path):
    base = tmp_path / "files"
    base.mkdir()
    (base / "a.txt").write_text("alpha")
    g = one_node_graph(sm.N("t", "agent.tool_call", tool="read_text_file", args={"path": "{p}"}, allowed_dir=str(base), output_field="r"), [sm.S("p"), sm.S("r", "object")])
    lab = Lab(tmp_path)
    _, rid = lab.run(g, inp={"p": "a.txt"})
    assert lab.final(rid)["r"]["result"]["text"] == "alpha" and lab.events(rid, "tool_call")[0]["data"]["effects"] == ["file_read"]
    _, rid = lab.run(g, inp={"p": "../escape.txt"})
    r = lab.final(rid)["r"]
    assert r["status"] == "error" and r["result"]["code"] == "E_TOOL_PATH"


# ------------------------------------------------------------------------------------------------ provider adapters
def test_anthropic_adapter_needs_a_secret_reference_and_never_stores_the_key(monkeypatch, tmp_path):
    with pytest.raises(ModelUnavailable) as e:
        build_chat_model(ModelSpec(provider="anthropic"))
    assert e.value.code == "E_MODEL_NO_KEY"
    spec = ModelSpec(provider="anthropic", api_key={"kind": "env", "name": "VOID_TEST_ANTHROPIC_KEY"}, temperature=0.3, max_tokens=77)
    assert spec.model == "claude-sonnet-5-5"  # default model id
    with pytest.raises(ModelUnavailable):  # reference given but the variable is not set
        monkeypatch.delenv("VOID_TEST_ANTHROPIC_KEY", raising=False)
        build_chat_model(spec)
    monkeypatch.setenv("VOID_TEST_ANTHROPIC_KEY", "sk-test-not-a-real-key")
    m = build_chat_model(spec)
    assert m.model == "claude-sonnet-5-5" and m.max_tokens == 77 and m.temperature == 0.3
    assert "sk-test" not in json.dumps(spec.model_dump()) and "sk-test" not in repr(m)  # the value is neither in the config nor printed
    with pytest.raises(Exception):  # a literal key is not a reference
        ModelSpec(provider="anthropic", api_key="sk-literal")
    from connectors.errors import SourceError
    from connectors import secrets
    with pytest.raises(SourceError):
        secrets.normalize_ref({"kind": "env", "name": "x y"})


@pytest.mark.skipif(not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("VOID_ANTHROPIC_KEY")),
                    reason="no Anthropic API key configured (set ANTHROPIC_API_KEY); the adapter is implemented but its live call is not exercised here")
def test_anthropic_live_call():
    from agent.models import invoke_chat

    name = "ANTHROPIC_API_KEY" if os.environ.get("ANTHROPIC_API_KEY") else "VOID_ANTHROPIC_KEY"
    out = invoke_chat(ModelSpec(provider="anthropic", api_key={"kind": "env", "name": name}, max_tokens=20, temperature=0), [{"role": "user", "content": "Say hi in three words."}])
    assert out["text"] and out["usage"]["source"] == "provider"
