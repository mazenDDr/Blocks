"""Control-service endpoints for agent graphs, in the existing style (registry, validate, projects, runs, SSE events)."""
import json

import pytest

from agent import samples as sm
from agent_helpers import REPO, api_client, copy_docs, start, wait_run


@pytest.fixture
def client(tmp_path):
    return api_client(tmp_path)


def test_registry_lists_agent_blocks_with_schema_ports_and_defaults(client):
    ops = {o["type"]: o for o in client.get("/api/registry").json()["ops"] if o["graphKind"] == "agent"}
    assert {"agent.prompt", "agent.chat_model", "agent.structured_output", "agent.embed_text", "agent.retrieve", "agent.citations", "agent.tool_call", "agent.human_interrupt",
            "agent.memory_select", "agent.memory_write", "agent.set_state"} <= ops.keys()
    m = ops["agent.chat_model"]
    assert m["backend"] == "langgraph" and m["inputs"] == ["in"] and m["outputs"] == ["out"] and m["inputKinds"] == {"in": "control"}
    assert m["defaults"]["model"]["provider"] == "ollama" and "model" in m["configSchema"]["properties"] and m["displayName"] == "Chat model"
    assert ops["agent.tool_call"]["category"] == "Tools"


def test_catalog_states_provider_capabilities_reducers_tools_and_the_fixture_label(client):
    c = client.get("/api/agent/catalog").json()
    assert set(c["providers"]) == {"ollama", "anthropic", "fixture"}
    assert "seed" in c["providers"]["ollama"]["settings"] and "seed" not in c["providers"]["anthropic"]["settings"] and "no sampling seed" in c["providers"]["anthropic"]["unsupported"]["seed"]
    assert c["providers"]["ollama"]["ollama"]["reachable"] in (True, False) and c["defaults"]["anthropicModel"] == "claude-sonnet-5-5"
    assert {r["kind"] for r in c["reducers"]} == {"replace", "append", "append_unique", "add", "max", "min", "merge", "keep_last_n"}
    assert [t["name"] for t in c["tools"]] == ["calculator", "read_text_file", "write_note"] and {t["name"]: t["requiresApproval"] for t in c["tools"]}["write_note"] is True
    assert "never real model output" in c["fixtureNote"] and c["stages"]["order"] == ["retrieve", "filter", "rank", "dedupe", "budget", "summarize"]
    assert any(e["provider"] == "local_hash" and "NOT a semantic model" in e["label"] for e in c["embeddings"])


def test_validate_returns_per_node_reads_writes_effects_and_loop_analysis(client):
    v = client.post("/api/validate", json={"graph": sm.counter_loop_graph().to_json()}).json()
    assert v["ok"] and v["graphKind"] == "agent" and v["nodes"]["tick"]["reads"] and v["nodes"]["tick"]["writes"] == ["n", "log"]
    assert v["agent"]["loops"][0]["hasConditionalExit"] is True and [f["name"] for f in v["agent"]["state"]][:2] == ["n", "log"]
    g = sm.parallel_join_graph(conflict=True).to_json()
    v = client.post("/api/validate", json={"graph": g}).json()
    assert not v["ok"] and any(d["code"] == "E_CONCURRENT_WRITE" for d in v["diagnostics"])
    t = client.post("/api/validate", json={"graph": sm.approval_tools_graph("").to_json()}).json()
    assert t["nodes"]["write"]["effects"] == ["file_write"] and "external" in t["nodes"]["write"]["explain"]["summary"].lower()
    f = client.post("/api/validate", json={"graph": sm.memory_debug_graph(sm.CONSTRAINT_FIXTURE).to_json()}).json()
    assert f["nodes"]["answer"]["fixtureModel"] is True and any(d["code"] == "W_FIXTURE_MODEL" for d in f["diagnostics"])


def test_example_projects_are_listed_open_and_valid(client):
    ex = {e["id"]: e for e in client.get("/api/examples").json()["details"]}
    for name in ("agent_retrieval_revision", "agent_memory_debugging", "agent_approval_tools", "agent_bounded_loop", "agent_parallel_join"):
        assert ex[name]["graphKind"] == "agent"
        g = client.get(f"/api/examples/{name}").json()
        v = client.post("/api/validate", json={"graph": g["graph"]}).json()
        assert v["ok"], (name, v["diagnostics"])
    assert ex["agent_retrieval_revision"]["synthetic"] and ex["agent_memory_debugging"]["synthetic"]
    # the committed example files are exactly what the generator builds (deterministic)
    import subprocess, sys
    files = sorted((REPO / "examples").glob("agent_*.json"))
    before = {f.name: f.read_bytes() for f in files}
    assert len(before) == 10 and subprocess.run([sys.executable, str(REPO / "examples" / "make_m4_examples.py")], cwd=REPO, capture_output=True).returncode == 0
    assert {f.name: f.read_bytes() for f in files} == before


def test_save_and_load_an_agent_project_keeps_the_agent_section(client):
    g = sm.retrieval_revision_graph(sm.fixture_model(default="x"), sm.fixture_model(default="{}"), sm.fixture_model(default="y")).to_json()
    r = client.put("/api/projects/agent_p", json={"graph": g, "ui": {"schemaVersion": "1.0.0", "positions": {"init": {"x": 1, "y": 2}}}})
    assert r.status_code == 200
    back = client.get("/api/projects/agent_p").json()
    assert back["graph"]["agent"] == g["agent"] and back["graphHash"] == r.json()["graphHash"] and back["ui"]["positions"]["init"] == {"x": 1, "y": 2}
    assert [p for p in client.get("/api/projects").json()["details"] if p["id"] == "agent_p"][0]["graphKind"] == "agent"


def test_run_via_api_trace_events_summary_and_idempotency(client):
    g = sm.counter_loop_graph(stop_at=4)
    body = {"graph": g.to_json(), "config": {"thread_id": "api-1"}}
    assert client.post("/api/runs", json=body).status_code == 400  # Idempotency-Key required, like every run
    h = {"Idempotency-Key": "agent-idem-1"}
    r = client.post("/api/runs", json=body, headers=h)
    assert r.status_code == 201 and r.json()["threadId"] == "api-1"
    again = client.post("/api/runs", json=body, headers=h).json()
    assert again["idempotentReplay"] is True and again["runId"] == r.json()["runId"]
    rid = r.json()["runId"]
    s = wait_run(client, rid)
    assert s["status"] == "completed" and s["kind"] == "agent" and s["threadId"] == "api-1" and s["stoppedBy"] == "end"
    tr = client.get(f"/api/agent/runs/{rid}/trace").json()
    assert [st["node"] for st in tr["steps"]] == ["tick"] * 4 + ["done"]
    last = tr["steps"][3]
    assert last["route"]["via"] == "case" and last["route"]["evaluated"][0]["values"] == {"n": 4, "limit": 4} and last["route"]["to"] == "done"
    assert last["changes"][0]["field"] == "n" and last["changes"][0]["before"] == 3 and last["changes"][0]["after"] == 4 and last["changes"][0]["reducer"] == "add"
    assert tr["run"]["threadId"] == "api-1" and tr["run"]["libraries"]["langgraph"] and tr["provenance"]["graphHash"]
    ev = client.get(f"/api/runs/{rid}/events")  # SSE: ends when the run is terminal
    types = [l.split(": ", 1)[1] for l in ev.text.splitlines() if l.startswith("event: ")]
    assert types[0] == "run_queued" and "route_taken" in types and types[-2:] == ["run_finished", "end"]
    assert client.get(f"/api/runs?project=").json()["runs"][0]["kind"] == "agent"


def test_threads_state_history_and_fork_with_edits(client):
    g = sm.memory_debug_graph(sm.CONSTRAINT_FIXTURE)
    client.post("/api/agent/memory/seed", json={"example": "memory_debugging"})
    r = start(client, g, thread="th-a", inp={"question": "First question"})
    wait_run(client, r["runId"])
    th = {t["threadId"]: t for t in client.get("/api/agent/threads").json()["threads"]}["th-a"]
    assert th["hasCheckpoint"] and th["runs"][0]["runId"] == r["runId"] and th["pendingInterrupt"] is False
    st = client.get("/api/agent/threads/th-a/state").json()
    assert [m["role"] for m in st["values"]["history"]] == ["user", "assistant"] and st["provenance"]["threadId"] == "th-a"
    hist = client.get("/api/agent/threads/th-a/history").json()["checkpoints"]
    assert len(hist) >= 5 and hist[0]["checkpointId"] == st["checkpointId"]
    early = client.get("/api/agent/threads/th-a/state", params={"checkpointId": hist[-1]["checkpointId"]}).json()
    assert early["values"].get("history", []) == [] or len(early["values"]["history"]) < 2  # rewinding the viewer reads captured state
    edited = [{**st["values"]["history"][0], "content": "EDITED first question", "text": "EDITED first question"}]
    fk = client.post("/api/agent/threads/th-a/fork", json={"edits": {"history": edited}, "newThreadId": "th-fork"})
    assert fk.status_code == 201 and fk.json()["forkedFrom"]["threadId"] == "th-a"
    assert client.post("/api/agent/threads/th-a/fork", json={"newThreadId": "th-fork"}).status_code == 409
    assert client.get("/api/agent/threads/th-fork/state").json()["values"]["history"][0]["content"] == "EDITED first question"
    assert client.get("/api/agent/threads/th-a/state").json()["values"]["history"][0]["content"] == "First question"  # the original thread is untouched
    r2 = start(client, g, thread="th-fork", inp={"question": "Follow-up"})
    wait_run(client, r2["runId"])
    h2 = client.get("/api/agent/threads/th-fork/state").json()["values"]["history"]
    assert [m["content"] for m in h2 if m["role"] == "user"] == ["EDITED first question", "Follow-up"]


def test_resume_rules_and_rerun(client, tmp_path):
    g = sm.approval_tools_graph(str(tmp_path / "out"))
    r = start(client, g, thread="ap")
    s = wait_run(client, r["runId"])
    assert s["status"] == "paused" and s["pendingInterrupt"]["payload"]["actions"] == ["approve", "reject", "edit"]
    ev = client.get(f"/api/runs/{r['runId']}/events")
    assert ev.text.rstrip().endswith("event: end\ndata: {}") and "event: interrupt_raised" in ev.text  # the stream closes when the run pauses
    assert client.post(f"/api/runs/{r['runId']}/resume", json={"value": {"action": "reject"}}).status_code == 200
    assert wait_run(client, r["runId"])["status"] == "completed"
    assert client.post(f"/api/runs/{r['runId']}/resume", json={"value": {"action": "approve"}}).status_code == 409  # not paused any more
    assert client.post("/api/runs/nope/resume", json={}).status_code == 404
    rr = client.post(f"/api/runs/{r['runId']}/rerun", json={}).json()
    assert rr["rerunOf"] == r["runId"] and rr["threadId"] != "ap" and "not guaranteed" in rr["note"]
    assert wait_run(client, rr["runId"])["status"] == "paused"
    plain = start(client, sm.counter_loop_graph(), thread="plain")
    assert client.get(f"/api/agent/runs/{plain['runId']}/trace").status_code == 200
    tab = client.get("/api/runs").json()["runs"]
    assert client.get("/api/agent/runs/zzz/trace").status_code == 404


def test_memory_record_endpoints_validate_and_audit(client):
    r = client.post("/api/agent/memory/records", json={"text": "A lab fact", "namespace": "lab", "scope": "user:x", "kind": "semantic", "importance": 0.8, "metadata": {"entity": "e"}})
    assert r.status_code == 201
    rid = r.json()["id"]
    assert client.post("/api/agent/memory/records", json={"text": "bad", "kind": "weird"}).status_code == 422
    assert client.post("/api/agent/memory/records", json={"text": "", "kind": "semantic"}).status_code == 422
    assert client.post("/api/agent/memory/records", json={"text": "x", "importance": 3}).status_code == 422
    assert client.put(f"/api/agent/memory/records/{rid}", json={"text": "An edited lab fact", "namespace": "lab", "scope": "user:x"}).json()["version"] == 2
    lst = client.get("/api/agent/memory/records", params={"namespace": "lab"}).json()
    assert lst["records"][0]["text"] == "An edited lab fact" and lst["namespaces"] == ["lab"] and lst["scopes"] == ["user:x"]
    hist = client.get(f"/api/agent/memory/records/{rid}/history").json()["writes"]
    assert [w["op"] for w in hist] == ["insert", "update"] and hist[1]["old"]["text"] == "A lab fact" and hist[0]["evidence"] == "manual edit in the memory browser"
    assert client.get("/api/agent/memory/records", params={"scope": "user:other"}).json()["records"] == []
    assert client.delete(f"/api/agent/memory/records/{rid}").status_code == 200 and client.delete(f"/api/agent/memory/records/{rid}").status_code == 404
    assert client.post("/api/agent/memory/seed", json={"example": "zzz"}).status_code == 422


def test_index_endpoints_build_info_chunks_search(client, tmp_path):
    d = copy_docs(tmp_path)
    g = sm.retrieval_revision_graph(sm.fixture_model(default="a"), sm.fixture_model(default="{}"), sm.fixture_model(default="r"), docs_dir=d).to_json()
    body = {"graph": g, "indexId": "lab_docs"}
    assert client.post("/api/agent/indexes/info", json=body).json()["built"] is False
    m = client.post("/api/agent/indexes/build", json=body).json()
    assert m["action"] == "built" and m["chunks"] >= 5 and m["embedding"]["identity"].startswith("local_hash") and "cosine" in m["scoreInterpretation"]
    assert client.post("/api/agent/indexes/build", json=body).json()["action"] == "reused"
    ch = client.post("/api/agent/indexes/chunks", json=body).json()["chunks"]
    assert ch and ch[0]["chunk_id"].endswith("#0")
    s = client.post("/api/agent/indexes/search", json={**body, "query": "centrifuge rotor maximum speed", "k": 2}).json()
    assert s["documents"][0]["doc_id"] == "centrifuge_safety.txt" and len(s["documents"]) == 2 and s["excluded"]
    assert client.post("/api/agent/indexes/search", json={**body, "query": " "}).status_code == 422
    assert client.post("/api/agent/indexes/build", json={"graph": g, "indexId": "ghost"}).status_code == 422
    g2 = json.loads(json.dumps(g))
    g2["agent"]["indexes"][0]["loader"]["directory"] = str(tmp_path / "missing")
    assert client.post("/api/agent/indexes/build", json={"graph": g2, "indexId": "lab_docs"}).json()["detail"]["code"] == "index_build_failed"


def test_secrets_never_appear_in_agent_api_output(client, monkeypatch):
    monkeypatch.setenv("VOID_API_SECRET_CHECK", "sk-SECRET-VALUE-123")
    g = sm.retrieval_revision_graph({"provider": "anthropic", "model": "claude-sonnet-5-5", "api_key": {"kind": "env", "name": "VOID_API_SECRET_CHECK"}},
                                    sm.fixture_model(default="{}"), sm.fixture_model(default="r")).to_json()
    v = client.post("/api/validate", json={"graph": g})
    assert "sk-SECRET" not in v.text and v.json()["nodes"]["draft"]["resolvedConfig"]["model"]["api_key"] == {"kind": "env", "name": "VOID_API_SECRET_CHECK"}
    assert "sk-SECRET" not in client.get("/api/agent/catalog").text
