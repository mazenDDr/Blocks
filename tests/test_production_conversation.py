"""Real native checkpoints/reducers and SQLite commits; no model/provider substitutes."""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from agent import samples as sm
from agent_helpers import Lab
from control.app import create_app
from graph_core.schema import Graph
from production import conversation_adapter as ca
from production.models import PredictRequest, RegisterVersion, ReleaseCreate
from production.pipeline import ProductionError
from production.runtime import ProductionRuntime
from tabular.core import dumps

META = dict(name="SYNTHETIC conversation", owner="tests", intendedUse="native state evidence", limitations="no model accuracy claim")


def graph():
    return sm.make_graph([
        sm.N("tick", "agent.set_state", assignments=[{"field": "n", "kind": "increment", "by": 1},
                                                     {"field": "history", "kind": "append_item", "source": "question"},
                                                     {"field": "turns", "kind": "increment", "by": 1}]),
        sm.N("answer", "agent.set_state", assignments=[{"field": "answer", "kind": "template", "template": "{question}: {n}/{turns}"}])],
        [sm.E("START", "tick"), sm.E("tick", "answer"), sm.E("answer", "END")],
        {"state": [sm.S("question"), sm.S("n", "integer", "add", default=0, scope="thread"),
                   sm.S("history", "list", "keep_last_n", default=[], scope="thread", n=2),
                   sm.S("turns", "integer", "add", default=0), sm.S("answer")],
         "limits": {"maxSteps": 8, "maxModelCalls": 1, "maxTokens": 1024, "maxSeconds": 10}})


def setup(tmp_path, capture=True):
    lab = Lab(tmp_path)
    assert lab.run(graph(), inp={"question": "SYNTHETIC source"})[0] == "completed"
    rt = ProductionRuntime(lab.store)
    v = rt.register_version(RegisterVersion(runId="r1", node=ca.NODE, **META))
    rel = rt.create_release(ReleaseCreate(versionId=v["id"], config={"maxBatch": 1, "sessionMode": "conversation", "captureInputs": capture}))
    rt.activate(rel["id"], None)
    return lab, rt, v, rel


def req(id_, text="SYNTHETIC input", user="alice", session="chat"):
    return PredictRequest(requestId=id_, records=[{"question": text}], user=user, session=session)


def head(rt, rel, user="alice", session="chat"):
    return rt.ps.conversation(dumps([rel["id"], user, session]))


def test_native_reducers_turn_reset_restart_idempotency_source_isolation(tmp_path):
    lab, rt, v, rel = setup(tmp_path)
    source = lab.final("r1")
    assert head(rt, rel) is None  # warmup doesn't seed any live conversation
    for i, text in enumerate(("first", "second", "third"), 1):
        t = rt.predict("local", "lab", req(f"r{i}", text))
        assert t["status"] == 200, t
        assert t["result"]["predictions"] == [f"{text}: {i}/1"]
        assert t["conversationState"]["revision"] == i and t["conversationParent"]["revision"] == i-1
        assert t["conversationState"]["parentCheckpointSha256"] == t["conversationParent"]["checkpointSha256"]
        assert "_checkpoint" not in t["timings"]
        assert rt.predict("local", "lab", req(f"r{i}", text))["idempotentReplay"]
        rt = ProductionRuntime(lab.store)
    state = rt.pipeline(v["id"]).checkpoint_state(head(rt, rel)["checkpointSha256"])
    assert state == {"question": "third", "n": 3, "history": ["second", "third"], "turns": 1, "answer": "third: 3/1"}
    # Independent native research saver is the reference for the whole served state.
    for text in ("first", "second", "third"):
        status, reference = lab.run(graph(), thread="reference", inp={"question": text})
        assert status == "completed"
    assert state == lab.final(reference)
    assert lab.final("r1") == source
    _, next_run = lab.run(graph(), thread="t1", inp={"question": "research next"})
    assert lab.final(next_run)["n"] == 2  # native research saver independently retains its source count
    assert head(rt, rel)["revision"] == 3


def test_concurrent_same_session_serialized_and_user_session_release_isolated(tmp_path):
    _, rt, v, rel = setup(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        outs = list(pool.map(lambda i: rt.predict("local", "lab", req(f"p{i}")), range(4)))
    assert sorted(t["conversationState"]["revision"] for t in outs) == [1, 2, 3, 4]
    assert len({t["conversationState"]["threadId"] for t in outs}) == 1
    for id_, user, session in [("other-user", "bob", "chat"), ("other-session", "alice", "other")]:
        t = rt.predict("local", "lab", req(id_, user=user, session=session))
        assert t["conversationState"]["revision"] == 1 and t["result"]["predictions"] == ["SYNTHETIC input: 1/1"]
    other = rt.create_release(ReleaseCreate(versionId=v["id"], config={"maxBatch": 1, "sessionMode": "conversation", "concurrency": 1}))
    rt.activate(other["id"], rel["id"])
    assert rt.predict("local", "lab", req("rollout"))["conversationState"]["revision"] == 1
    rt.activate(rel["id"], other["id"], rollback=True)
    assert rt.predict("local", "lab", req("rollback"))["conversationState"]["revision"] == 5


def test_failed_cancelled_and_expired_candidates_never_advance_head(tmp_path, monkeypatch):
    lab, rt, v, rel = setup(tmp_path)
    rt.predict("local", "lab", req("success"))
    before = head(rt, rel)
    p = rt.pipeline(v["id"])
    original = p.predict
    def cancel(*a, **kw):
        out = original(*a, **kw)
        rt.ps.cancel("alice", "cancel")
        return out
    monkeypatch.setattr(p, "predict", cancel)
    t = rt.predict("local", "lab", req("cancel"))
    assert t["status"] == 409 and t["result"] is None and t["conversationState"] is None
    assert head(rt, rel) == before
    monkeypatch.setattr(p, "predict", original)
    t = rt.predict("local", "lab", PredictRequest(requestId="bad", records=[{"wrong": "x"}], user="alice", session="chat"))
    assert t["status"] == 422 and head(rt, rel) == before
    t = rt.predict("local", "lab", req("missing", session=None))
    assert t["error"]["code"] == "E_SESSION_REQUIRED" and head(rt, rel) == before
    # Slow immutable snapshot I/O must not sneak a late checkpoint into the head.
    put = lab.store.put_bytes
    def slow(raw):
        if b'"manifestSha256"' in raw:
            time.sleep(.02)
        return put(raw)
    monkeypatch.setattr(lab.store, "put_bytes", slow)
    rt.ps.begin_request("alice", "expired", "fp", rel["id"])
    result, timing = original([{"question": "expired"}], capture=True, checkpoint=before["checkpointSha256"])
    t = rt.ps.finish_request("alice", "expired", {"status": 200, "error": None, "result": result},
                             conversation=(dumps([rel["id"], "alice", "chat"]), before, timing["_checkpoint"]), deadline=time.perf_counter()+.01)
    assert t["status"] == 504 and t["conversationState"] is None and head(rt, rel) == before


def test_sqlite_failure_rolls_back_trace_and_checkpoint_together(tmp_path, monkeypatch):
    _, rt, v, rel = setup(tmp_path)
    rt.predict("local", "lab", req("ok"))
    before = head(rt, rel)
    native_db = rt.ps.db
    def failing_db():
        db = native_db()
        db.execute("CREATE TEMP TRIGGER fail_finish BEFORE UPDATE OF trace ON requests BEGIN SELECT RAISE(ABORT, 'forced persistence failure'); END")
        assert db.execute("SELECT COUNT(*) FROM sqlite_temp_master WHERE name='fail_finish'").fetchone()[0] == 1
        return db
    monkeypatch.setattr(rt.ps, "db", failing_db)
    with pytest.raises(Exception, match="forced persistence failure"):
        rt.predict("local", "lab", req("failed-write"))
    assert head(rt, rel) == before
    assert rt.ps.trace("alice", "failed-write")["available"] is False
    monkeypatch.setattr(rt.ps, "db", native_db)
    restarted = ProductionRuntime(rt.store)
    assert restarted.ps.trace("alice", "failed-write")["error"]["code"] == "E_SERVING_RESTART"
    assert head(restarted, rel) == before
    assert restarted.predict("local", "lab", req("after-crash"))["conversationState"]["revision"] == 2


def test_http_captured_parent_replay_checkpoint_inspection_and_capture_off(tmp_path):
    lab, _, v, rel = setup(tmp_path)
    with TestClient(create_app(lab.wb)) as c:
        assert any(x["adapter"] == "conversation" for x in c.get("/api/production").json()["candidates"])
        for i in range(3):
            r = c.post("/api/serve/local/lab/predict", json=req(f"t{i}", str(i)).model_dump())
            assert r.status_code == 200, r.text
        endpoint = f"/api/production/releases/{rel['id']}/conversation?user=alice&session=chat"
        before = c.get(endpoint).json()
        replay = c.post("/api/production/requests/t1/replay", json={"user": "alice"})
        assert replay.status_code == 200 and replay.json()["result"]["predictions"] == ["1: 2/1"]
        assert "_checkpoint" not in replay.json()["timings"] and c.get(endpoint).json() == before
        assert c.get(endpoint.replace("user=alice", "user=bob")).json()["head"] is None
        assert c.post("/api/production/requests/t1/labels", json={"user": "alice", "labels": ["independent reference"]}).status_code == 200
        assert c.get(f"/api/production/releases/{rel['id']}/monitor").json()["labelBasedQuality"]["values"]["exactStringAgreement"] == 0
    rt = ProductionRuntime(lab.store)
    private = rt.create_release(ReleaseCreate(versionId=v["id"], config={"maxBatch": 1, "sessionMode": "conversation", "captureInputs": False}))
    rt.activate(private["id"], rel["id"])
    trace = rt.predict("local", "lab", req("private", "persisted by explicit session contract"))
    assert trace["records"] is None and trace["result"]["agent"]["finalState"] is None
    assert rt.pipeline(v["id"]).checkpoint_state(head(rt, private)["checkpointSha256"])["question"] == "persisted by explicit session contract"


def test_cas_parent_conflict_and_cross_version_checkpoint_refused(tmp_path):
    _, rt, v, rel = setup(tmp_path)
    trace = rt.predict("local", "lab", req("one"))
    before = head(rt, rel)
    result, timing = rt.pipeline(v["id"]).predict([{"question": "two"}], checkpoint=before["checkpointSha256"])
    rt.ps.begin_request("alice", "stale", "fp", rel["id"])
    t = rt.ps.finish_request("alice", "stale", {"status": 200, "result": result}, conversation=(dumps([rel["id"], "alice", "chat"]), None, timing["_checkpoint"]))
    assert t["error"]["code"] == "E_SESSION_CONFLICT" and head(rt, rel) == before
    payload = json.loads(rt.store.read_artifact(before["checkpointSha256"]))
    payload["manifestSha256"] = "wrong version"
    wrong = rt.store.put_bytes(dumps(payload).encode())
    with pytest.raises(ProductionError, match="another pinned version"):
        rt.pipeline(v["id"]).predict([{"question": "x"}], checkpoint=wrong)


def test_actual_process_death_discards_candidate_and_restart_preserves_committed_turn(tmp_path):
    lab, rt, _, rel = setup(tmp_path)
    rt.predict("local", "lab", req("committed"))
    before = head(rt, rel)
    ready = tmp_path / 'candidate-ready'
    # Child performs the real native turn then pauses exactly before durable commit.
    code = '''import sys, time
from pathlib import Path
from artifact_store import ArtifactStore
from production.runtime import ProductionRuntime
from production.models import PredictRequest
rt=ProductionRuntime(ArtifactStore(Path(sys.argv[1])))
finish=rt.ps.finish_request
def pending(*args, **kwargs):
    assert kwargs['conversation'][2] is not None
    Path(sys.argv[2]).write_text('native candidate ready; not committed')
    while True: time.sleep(.1)
rt.ps.finish_request=pending
rt.predict('local','lab',PredictRequest(requestId='killed',records=[{'question':'SYNTHETIC killed turn'}],user='alice',session='chat'))
'''
    child = subprocess.Popen([sys.executable, '-c', code, str(lab.wb), str(ready)])
    try:
        until = time.monotonic()+20
        while not ready.exists() and child.poll() is None and time.monotonic()<until:
            time.sleep(.05)
        assert ready.exists(), f'child exit {child.poll()}'
        child.kill()
        assert child.wait(timeout=10) != 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
    restarted = ProductionRuntime(lab.store)
    assert head(restarted, rel) == before
    assert restarted.ps.trace('alice','killed')['error']['code'] == 'E_SERVING_RESTART'
    assert restarted.predict('local','lab',req('killed', 'SYNTHETIC killed turn'))['idempotentReplay']
    assert restarted.predict('local','lab',req('after-process-death'))['conversationState']['revision'] == 2


@pytest.mark.parametrize('mutation,code', [('no-thread','E_AGENT_SCOPE'),('thread-input','E_AGENT_INPUT'),('tool','E_AGENT_SCOPE')])
def test_declared_conversation_scope_contract(mutation, code):
    g = graph().to_json()
    if mutation == 'no-thread':
        for field in g['agent']['state']:
            field['scope'] = 'turn'
    elif mutation == 'thread-input':
        g['agent']['state'][0]['scope'] = 'thread'
    else:
        g = sm.approval_tools_graph().to_json()
    with pytest.raises(ProductionError) as error:
        ca.contract(Graph.model_validate(g), {'question':'SYNTHETIC input'})
    assert error.value.code == code


def test_checkpoint_hash_integrity_failure_cannot_commit(tmp_path):
    lab, rt, _, rel = setup(tmp_path)
    t = rt.predict('local','lab',req('one'))
    sha = t['conversationState']['checkpointSha256']
    lab.store.path_of(sha).write_bytes(b'altered native checkpoint')
    t = rt.predict('local','lab',req('bad-integrity'))
    assert t['status'] != 200 and t['result'] is None and t['conversationState'] is None
    row = rt.ps.query('SELECT revision,checkpoint FROM agent_sessions')[0]
    assert row == {'revision':1, 'checkpoint':sha}
