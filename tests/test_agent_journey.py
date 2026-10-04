"""The memory debugging journey (VISION 12.7; A30, A31, A32) end to end through the API with the FIXTURE model, plus the context inspector.

The fixture is a scripted model that answers correctly only if the constraint text is in the messages it receives. That makes the test
sensitive to what the context really contained; it is labelled as a fixture everywhere. The same journey against the real local model:
tests/test_agent_live.py (pytest -m live)."""
import copy
import json

import pytest

from agent import samples as sm
from agent.memory import MemoryStore
from agent_helpers import api_client, start, wait_run
from graph_core.schema import Graph

CONSTRAINT = "mem_constraint"


def snapshot(client, tmp_path):
    """Everything an isolated preview must leave untouched."""
    mem = MemoryStore(tmp_path / "wb")
    return {"records": json.dumps(mem.list_records(include_deleted=True), sort_keys=True), "apps": len(mem.applications()), "writes": len(mem.writes()),
            "runs": [(r["id"], client.get(f"/api/runs/{r['id']}").json()["maxSeq"]) for r in client.get("/api/runs").json()["runs"]],
            "artifacts": sorted(a.name for a in (tmp_path / "wb" / "artifacts").iterdir()), "effects": len(mem.effects())}


def edit_policy(graph: Graph, **rank_cfg) -> Graph:
    d = copy.deepcopy(graph.to_json())
    for st in d["agent"]["policies"][0]["stages"]:
        if st["id"] == "rank":
            st["config"].update(rank_cfg)
    return Graph.model_validate(d)


def segment_text(ctx):
    return " ".join(m["content"][s["start"]:s["end"]] for m in ctx["messages"] for s in m["segments"])


def test_memory_debugging_journey_end_to_end(tmp_path, monkeypatch):
    c = api_client(tmp_path)
    graph = sm.memory_debug_graph(sm.CONSTRAINT_FIXTURE)
    assert c.post("/api/agent/memory/seed", json={"example": "memory_debugging"}).json()["seeded"] == 8

    # 1. run: the answer ignores the constraint
    r1 = start(c, graph, thread="j1", inp={"question": sm.CONSTRAINT_QUESTION})
    s1 = wait_run(c, r1["runId"])
    assert s1["status"] == "completed", s1
    ans = c.get(f"/api/agent/runs/{r1['runId']}/final-state").json()["values"]["answer"]
    assert "45 C" in ans and ans.startswith("FIXTURE")  # labelled as a fixture reply
    # 2. select the model call and search its assembled context for the constraint
    calls = c.get(f"/api/agent/runs/{r1['runId']}/model-calls").json()["calls"]
    assert len(calls) == 1 and calls[0]["fixture"] is True
    ctx = c.get(f"/api/agent/runs/{r1['runId']}/model-calls/{calls[0]['callId']}").json()
    assert "40 degrees" not in segment_text(ctx) and "40 degrees" not in json.dumps(ctx["providerRequest"])  # absent from what was sent
    # 3. the record is stored (not "used"): it is listed as EXCLUDED with stage and reason, never as included
    stored = {r["id"] for r in c.get("/api/agent/memory/records").json()["records"]}
    assert CONSTRAINT in stored
    ex = {e["id"]: e for e in ctx["excluded"]}
    assert ex[CONSTRAINT]["stage"] == "rank" and ex[CONSTRAINT]["reason"].startswith("ranked below the selected limit") and ex[CONSTRAINT]["status"] == "excluded"
    included_ids = {s["source"]["recordId"] for m in ctx["messages"] for s in m["segments"] if s["source"].get("recordId")}
    assert CONSTRAINT not in included_ids and not (included_ids & set(ex))  # disjoint: nothing is both
    assert "mem_bob" in ex and ex["mem_bob"]["stage"] == "eligible" and "scope filter" in ex["mem_bob"]["reason"]
    assert "namespace filter" in ex["mem_gym"]["reason"]
    # 4. trace the expected record through storage -> scope -> retrieval -> ranking -> budget
    tr = c.get("/api/agent/memory/trace", params={"recordId": CONSTRAINT, "runId": r1["runId"], "callId": calls[0]["callId"]}).json()
    t = tr["applications"][0]
    assert tr["stored"] and t["present"] and t["status"] == "excluded" and t["firstDisappearedAt"]["stage"] == "rank"
    assert [x["stage"] for x in t["trail"]] == ["eligible", "rank"] and t["trail"][0]["action"] == "kept"  # passed storage + scope + eligibility, lost at ranking
    assert "excluded at stage 'rank'" in t["verdict"] and t["scores"]["recency"] < 0.01 and t["scores"]["final"] < 0.2
    # if the record HAD been included the trace must say so (it rules out one explanation)
    inc = next(i for i in included_ids)
    t2 = c.get("/api/agent/memory/trace", params={"recordId": inc, "runId": r1["runId"]}).json()["applications"][0]
    assert t2["usedInContext"] is True and "rules out omission" in t2["verdict"]

    # 5. visually edit the responsible policy and preview the context in isolation
    edited = edit_policy(graph, weights={"recency": 0.2, "relevance": 1.0, "importance": 0.6})
    before = snapshot(c, tmp_path)
    import agent.models as am

    monkeypatch.setattr(am, "invoke_chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a preview must not call a model")))
    pv = c.post("/api/agent/policy/preview", json={"graph": edited.to_json(), "runId": r1["runId"], "callId": calls[0]["callId"]})
    assert pv.status_code == 200, pv.text
    p = pv.json()
    assert p["noModelCall"] is True and p["noStateMutation"] is True and p["embeddingCalls"] is False
    assert "40 degrees" in " ".join(m["content"] for m in p["after"]["messages"]) and "40 degrees" not in " ".join(m["content"] for m in p["before"]["messages"])
    ch = {x["id"]: x for x in p["recordChanges"]}
    assert ch[CONSTRAINT]["before"] == "excluded" and ch[CONSTRAINT]["after"] == "included" and "ranked below" in ch[CONSTRAINT]["beforeReason"]
    added = [s for s in p["segments"] if s["status"] == "added"]
    assert any(s["source"].get("recordId") == CONSTRAINT for s in added)
    assert p["after"]["tokensEstimate"] > 0 and p["applications"][0]["stages"][1]["op"] == "rank"
    assert snapshot(c, tmp_path) == before  # no event, artifact, record, application, write or run was created
    # the same preview with the unedited policy reproduces the recorded context (the preview engine is faithful)
    same = c.post("/api/agent/policy/preview", json={"graph": graph.to_json(), "runId": r1["runId"], "callId": calls[0]["callId"]}).json()
    assert same["recordChanges"] == [] and [m["content"] for m in same["after"]["messages"]] == [m["content"] for m in same["before"]["messages"]]
    monkeypatch.undo()

    # 6. rerun with the edited policy: a new run (new model call) on a new thread
    r2 = start(c, edited, thread="j2", inp={"question": sm.CONSTRAINT_QUESTION})
    assert wait_run(c, r2["runId"])["status"] == "completed"
    ans2 = c.get(f"/api/agent/runs/{r2['runId']}/final-state").json()["values"]["answer"]
    assert "37 C" in ans2 and "45 C" not in ans2
    call2 = c.get(f"/api/agent/runs/{r2['runId']}/model-calls").json()["calls"][0]
    ctx2 = c.get(f"/api/agent/runs/{r2['runId']}/model-calls/{call2['callId']}").json()
    assert CONSTRAINT in {s["source"].get("recordId") for m in ctx2["messages"] for s in m["segments"]} and CONSTRAINT not in {e["id"] for e in ctx2["excluded"]}
    # the preview predicted the real rerun's context exactly (same messages, apart from the unique memory ids of the two applications)
    assert [m["content"] for m in ctx2["messages"][:3]] == [m["content"] for m in p["after"]["messages"][:3]]
    assert c.get(f"/api/runs/{r1['runId']}").json()["status"] == "completed" and ans.startswith("FIXTURE")  # the original run and its record are unchanged


def test_budget_truncation_is_found_as_the_exclusion_stage_and_fixed_by_the_budget(tmp_path):
    c = api_client(tmp_path)
    c.post("/api/agent/memory/seed", json={"example": "memory_debugging"})
    # ranking already favours the constraint, but a tight token budget removes it: the first stage where it disappears is 'budget'
    g = sm.memory_debug_graph(sm.CONSTRAINT_FIXTURE, rank_weights={"recency": 0.1, "relevance": 1.0, "importance": 0.5}, rank_limit=5, budget_tokens=20)
    r = start(c, g, thread="b1", inp={"question": sm.CONSTRAINT_QUESTION})
    wait_run(c, r["runId"])
    call = c.get(f"/api/agent/runs/{r['runId']}/model-calls").json()["calls"][0]
    t = c.get("/api/agent/memory/trace", params={"recordId": CONSTRAINT, "runId": r["runId"], "callId": call["callId"]}).json()["applications"][0]
    assert t["firstDisappearedAt"]["stage"] == "budget" and "removed to meet the configured token budget" in t["firstDisappearedAt"]["reason"]
    assert [x["stage"] for x in t["trail"]] == ["eligible", "rank", "budget"] and t["trail"][1]["action"] == "kept"
    d = copy.deepcopy(g.to_json())
    next(s for s in d["agent"]["policies"][0]["stages"] if s["id"] == "budget")["config"]["max_tokens"] = 200
    pv = c.post("/api/agent/policy/preview", json={"graph": d, "runId": r["runId"], "callId": call["callId"]}).json()
    assert {x["id"]: x["after"] for x in pv["recordChanges"]}[CONSTRAINT] == "included"
    # an unsaved edited policy can also be previewed by passing it directly
    pol = copy.deepcopy(d["agent"]["policies"][0])
    pol["stages"][2]["config"]["max_tokens"] = 25
    pv2 = c.post("/api/agent/policy/preview", json={"graph": g.to_json(), "runId": r["runId"], "callId": call["callId"], "policy": pol}).json()
    assert pv2["after"]["tokensEstimate"] <= pv["after"]["tokensEstimate"]


def test_deleting_a_record_is_not_removing_it_from_an_assembled_context(tmp_path):
    c = api_client(tmp_path)
    c.post("/api/agent/memory/seed", json={"example": "memory_debugging"})
    g = sm.memory_debug_graph(sm.CONSTRAINT_FIXTURE)
    r = start(c, g, thread="d1", inp={"question": sm.CONSTRAINT_QUESTION})
    wait_run(c, r["runId"])
    call = c.get(f"/api/agent/runs/{r['runId']}/model-calls").json()["calls"][0]
    ctx = c.get(f"/api/agent/runs/{r['runId']}/model-calls/{call['callId']}").json()
    used = next(s["source"]["recordId"] for m in ctx["messages"] for s in m["segments"] if s["source"].get("recordId"))
    resp = c.delete(f"/api/agent/memory/records/{used}").json()
    assert resp["retainedSnapshots"][0]["usedInModelContext"] is True and "Historical runs keep" in resp["note"]
    ctx_after = c.get(f"/api/agent/runs/{r['runId']}/model-calls/{call['callId']}").json()
    assert ctx_after["messages"] == ctx["messages"]  # the recorded context of the old call is untouched
    assert used not in {x["id"] for x in c.get("/api/agent/memory/records").json()["records"]}
    r2 = start(c, g, thread="d2", inp={"question": sm.CONSTRAINT_QUESTION})
    wait_run(c, r2["runId"])
    sel = c.get(f"/api/agent/runs/{r2['runId']}/final-state").json()["values"]["memory"]
    assert used not in [x["id"] for x in sel]  # future calls cannot retrieve it
    h = c.get(f"/api/agent/memory/records/{used}/history").json()
    assert [w["op"] for w in h["writes"]] == ["insert", "delete"]
