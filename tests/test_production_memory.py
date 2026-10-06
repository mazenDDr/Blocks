"""Release long-term memory (ADR0075): per release and user, committed only with successful turns, research memory untouched."""
import json
import sqlite3
import time
from contextlib import closing

import pytest
from fastapi.testclient import TestClient

from agent import samples as sm
from agent.memory import MemoryStore
from agent_helpers import Lab
from control.app import create_app
from graph_core.schema import Graph
from production import memory_agent_adapter as adapter
from production import runtime as runtime_module
from production.models import PredictRequest, RegisterVersion, ReleaseCreate
from production.pipeline import ProductionError
from production.runtime import ProductionRuntime
from production.store import SCHEMA, ProductionStore
from storage.schema import migrate, statements
from test_production_agent import META
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("make_memory_serving_fixture", Path(__file__).resolve().parents[1] / "examples" / "make_memory_serving_fixture.py")
fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture)

MARKER = "SYNTHETIC RESEARCH_PRIVATE_MARKER"


def setup(tmp_path, capture=True):
    lab = Lab(tmp_path / "lab")
    status, rid = lab.run(fixture.graph(), inp={"note": "SYNTHETIC source note"})
    assert status == "completed"
    # Research memory in the SAME namespace and scope the release uses: serving must never see it.
    MemoryStore(lab.wb).put_record({"text": MARKER, "namespace": "notes", "scope": "user", "kind": "episodic"})
    rt = ProductionRuntime(lab.store)
    version = rt.register_version(RegisterVersion(runId=rid, node=adapter.NODE, **META))
    rel = rt.create_release(ReleaseCreate(versionId=version["id"], config={"maxBatch": 1, "sessionMode": "stateless", "captureInputs": capture}))
    rt.activate(rel["id"], None)
    return lab, rt, version, rel


def say(rt, id_, note, user="alice"):
    return rt.predict("local", "lab", PredictRequest(requestId=id_, records=[{"note": note}], user=user))


def texts(rt, rel, user):
    return [r["text"] for r in rt.ps.release_memory(rel["id"], user)]


def recalled(trace):
    return trace["result"]["agent"]["finalState"]["recalled"]


def test_turns_recall_only_their_users_records_and_commit_with_the_trace(tmp_path):
    lab, rt, version, rel = setup(tmp_path)
    research_before = MemoryStore(lab.wb).list_records()
    assert version["adapter"] == "agent_memory" and rel["memoryPolicy"].startswith("Long-term records per release and request user")
    assert rt.ps.release_memory(rel["id"], "alice") == []  # the release warmup wrote nothing

    t1 = say(rt, "a1", "SYNTHETIC alice likes teal")
    assert t1["status"] == 200 and t1["result"]["predictions"] == ["SYNTHETIC: I recalled 0 earlier note(s) of yours before storing this one."]
    assert len(t1["memoryWrites"]) == 1 and texts(rt, rel, "alice") == ["SYNTHETIC alice likes teal"]
    t2 = say(rt, "a2", "SYNTHETIC alice owns a bicycle")
    assert t2["result"]["predictions"][0].startswith("SYNTHETIC: I recalled 1 ")
    assert [r["text"] for r in recalled(t2)] == ["SYNTHETIC alice likes teal"]
    assert t2["result"]["agent"]["memorySelections"][0]["selected"] == [t1["memoryWrites"][0]]

    b1 = say(rt, "b1", "SYNTHETIC bob likes red", user="bob")
    assert b1["result"]["predictions"][0].startswith("SYNTHETIC: I recalled 0 ") and recalled(b1) == []
    assert texts(rt, rel, "bob") == ["SYNTHETIC bob likes red"] and len(texts(rt, rel, "alice")) == 2
    assert all(MARKER not in json.dumps(t["result"]) for t in (t1, t2, b1))

    replay = say(rt, "a2", "SYNTHETIC alice owns a bicycle")
    assert replay["idempotentReplay"] and len(texts(rt, rel, "alice")) == 2
    dup = say(rt, "a3", "SYNTHETIC alice likes teal")  # same text: the graph's skip_duplicates stores nothing new
    assert dup["status"] == 200 and "memoryWrites" not in dup and len(texts(rt, rel, "alice")) == 2

    restarted = ProductionRuntime(lab.store)
    t4 = say(restarted, "a4", "SYNTHETIC alice reads poems")
    assert t4["result"]["predictions"][0].startswith("SYNTHETIC: I recalled 2 ") and len(texts(restarted, rel, "alice")) == 3
    assert MemoryStore(lab.wb).list_records() == research_before  # serving never wrote research memory

    with TestClient(create_app(lab.wb)) as c:
        listed = c.get(f"/api/production/releases/{rel['id']}/memory", params={"user": "alice"}).json()
        assert listed["count"] == 3 and listed["maxRecords"] == 200 and [r["request"] for r in listed["records"]] == ["a1", "a2", "a4"]
        gone = listed["records"][0]["id"]
        assert c.delete(f"/api/production/releases/{rel['id']}/memory/{gone}", params={"user": "alice"}).json() == {"deleted": gone}
        assert c.delete(f"/api/production/releases/{rel['id']}/memory/{gone}", params={"user": "alice"}).status_code == 404
        assert c.delete(f"/api/production/releases/{rel['id']}/memory/{gone}", params={"user": "bob"}).status_code == 404
        assert c.get(f"/api/production/releases/{rel['id']}/memory", params={"user": "bob"}).json()["count"] == 1
    events = rt.ps.query("SELECT data FROM lifecycle WHERE type='release_memory_deleted'")
    assert len(events) == 1 and "teal" not in events[0]["data"]
    t5 = say(restarted, "a5", "SYNTHETIC alice again")
    assert t5["result"]["predictions"][0].startswith("SYNTHETIC: I recalled 2 ")


def test_failed_cancelled_late_and_over_cap_turns_commit_no_memory(tmp_path, monkeypatch):
    lab, rt, version, rel = setup(tmp_path, capture=False)
    monkeypatch.setattr(runtime_module, "MEMORY_MAX_RECORDS", 1)
    assert say(rt, "c1", "SYNTHETIC first")["status"] == 200
    full = say(rt, "c2", "SYNTHETIC second")
    assert full["status"] == 409 and full["error"]["code"] == "E_AGENT_MEMORY_FULL" and full["result"] is None and "memoryWrites" not in full
    assert texts(rt, rel, "alice") == ["SYNTHETIC first"]
    bad = rt.predict("local", "lab", PredictRequest(requestId="c3", records=[{"wrong": "x"}], user="alice"))
    assert bad["status"] != 200 and texts(rt, rel, "alice") == ["SYNTHETIC first"]

    write = {"id": "mem_manual", "namespace": "notes", "kind": "episodic", "text": "SYNTHETIC late", "metadata": {}, "importance": 0.5, "generated": False, "evidence": None}
    ok_trace = lambda: {"status": 200, "error": None, "result": {"predictions": ["x"]}}
    rt.ps.begin_request("alice", "late", "f-late", rel["id"])
    late = rt.ps.finish_request("alice", "late", ok_trace(), deadline=time.perf_counter() - 1, memory=(rel["id"], "alice", [write], 10))
    assert late["status"] == 504 and "memoryWrites" not in late
    rt.ps.begin_request("alice", "cancel", "f-cancel", rel["id"])
    rt.ps.cancel("alice", "cancel")
    cancelled = rt.ps.finish_request("alice", "cancel", ok_trace(), memory=(rel["id"], "alice", [write], 10))
    assert cancelled["status"] == 409 and cancelled["error"]["code"] == "E_REQUEST_CANCELLED"
    rt.ps.begin_request("alice", "failed", "f-failed", rel["id"])
    rt.ps.finish_request("alice", "failed", {"status": 503, "error": {"code": "E_X", "message": "x"}, "result": None}, memory=(rel["id"], "alice", [write], 10))
    assert texts(rt, rel, "alice") == ["SYNTHETIC first"]
    rt.ps.begin_request("alice", "good", "f-good", rel["id"])
    assert rt.ps.finish_request("alice", "good", ok_trace(), memory=(rel["id"], "alice", [write], 10))["memoryWrites"] == ["mem_manual"]
    assert texts(rt, rel, "alice") == ["SYNTHETIC first", "SYNTHETIC late"]


def test_production_database_migrates_from_version_one_preserving_rows(tmp_path):
    path = tmp_path / "production.sqlite"
    with closing(sqlite3.connect(path)) as db:
        migrate(db, "production", (statements(SCHEMA),))
        db.execute("INSERT INTO aliases VALUES('SYNTHETIC-alias','v1')")
        db.commit()
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
    from artifact_store import ArtifactStore
    ps = ProductionStore(ArtifactStore(tmp_path))
    assert ps.query("SELECT * FROM aliases") == [{"name": "SYNTHETIC-alias", "version": "v1"}]
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM release_memory").fetchone()[0] == 0


@pytest.mark.parametrize("change", ["scope_global", "short_term_target", "approve_mode", "all_scopes", "short_term_source", "two_writes", "thread_field", "no_write"])
def test_memory_contract_refusals(change):
    doc = fixture.graph().to_json()
    write = next(n for n in doc["nodes"] if n["id"] == "remember")
    take = doc["agent"]["policies"][0]["stages"][0]["config"]
    if change == "scope_global":
        write["config"]["scope"] = "global"
    elif change == "short_term_target":
        write["config"]["target"] = "short_term"
        doc["agent"]["state"].append(sm.S("history", "messages", "keep_last_n", default=[], n=3))
    elif change == "approve_mode":
        write["config"]["mode"] = "approve"
    elif change == "all_scopes":
        take["scopes"] = []
    elif change == "short_term_source":
        take["sources"] = ["long_term", "short_term"]
    elif change == "two_writes":
        doc["nodes"].append({**write, "id": "remember2"})
        doc["edges"] = [e for e in doc["edges"] if e["to"]["node"] != "END"] + sm.chain("remember", "remember2", "END")
    elif change == "thread_field":
        doc["agent"]["state"][2]["scope"] = "thread"
    elif change == "no_write":
        doc["nodes"] = [n for n in doc["nodes"] if n["id"] != "remember"]
        doc["edges"] = [e for e in doc["edges"] if "remember" not in (e["from"]["node"], e["to"]["node"])] + sm.chain("answer", "END")
    with pytest.raises(ProductionError) as caught:
        adapter.contract(Graph.model_validate(doc), {"note": "x"})
    assert caught.value.code == ("E_AGENT_SCOPE" if change == "thread_field" else "E_AGENT_MEMORY_SCOPE"), caught.value.message


def test_erasing_a_user_removes_their_release_memory_bytes(tmp_path):
    from maintenance.erase import erase_user
    from test_erase_user import bytes_containing
    lab, rt, version, rel = setup(tmp_path)
    secret, other = b"SYNTHETIC-ALICE-MEMORY-7f3a", b"SYNTHETIC-BOB-MEMORY-91c2"
    assert say(rt, "e1", secret.decode())["status"] == 200 and say(rt, "e2", other.decode(), user="bob")["status"] == 200
    assert bytes_containing(lab.wb, secret)
    report = erase_user(lab.wb, "alice", apply=True, offline=True)
    assert report["rows"]["release_memory"] == 1
    assert bytes_containing(lab.wb, secret) == [] and bytes_containing(lab.wb, other)
    fresh = ProductionRuntime(lab.store)
    assert fresh.ps.release_memory(rel["id"], "alice") == [] and texts(fresh, rel, "bob") == [other.decode()]
