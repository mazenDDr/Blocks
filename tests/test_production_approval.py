"""Native interrupts, pending writes and transactional reviewed serving commits."""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from agent import samples as sm
from agent_helpers import Lab
from control.app import create_app
from production import approval_adapter as aa
from production.approval_requests import ResumeRequest, inspect
from production.models import PredictRequest, RegisterVersion, ReleaseCreate
from production.pipeline import ProductionError
from production.runtime import ProductionRuntime
from tabular.core import dumps

META = dict(name="SYNTHETIC approval", owner="tests", intendedUse="native paused checkpoint evidence", limitations="no file effects or quality claim")


def graph():
    return sm.make_graph([
        sm.N("tick", "agent.set_state", assignments=[{"field":"n", "kind":"increment", "by":1}]),
        sm.N("draft", "agent.set_state", assignments=[{"field":"answer", "kind":"template", "template":"{question}"}]),
        sm.N("review", "agent.human_interrupt", prompt="Review the SYNTHETIC proposal.", show_field="answer", edit_field="answer", decision_field="decision", actions=["approve","reject","edit"]),
        sm.N("finish", "agent.set_state", assignments=[{"field":"answer", "kind":"template", "template":"{decision}: {answer} / {n}"}])],
        [sm.E("START","tick"),sm.E("tick","draft"),sm.E("draft","review"),sm.E("review","finish"),sm.E("finish","END")],
        {"state":[sm.S("question"),sm.S("n","integer","add",default=0,scope="thread"),sm.S("answer"),sm.S("decision")],
         "limits":{"maxSteps":10,"maxSeconds":10,"maxModelCalls":1,"maxTokens":1024}})


def setup(tmp_path, capture=True, g=None):
    lab=Lab(tmp_path);g=g or graph()
    status,rid=lab.run(g,inp={"question":"source"})
    assert status=="paused"
    assert lab.run(g,run_id=rid,resume={"action":"approve"})[0]=="completed"
    rt=ProductionRuntime(lab.store)
    version=rt.register_version(RegisterVersion(runId=rid,node=aa.NODE,**META))
    release=rt.create_release(ReleaseCreate(versionId=version["id"],config={"maxBatch":1,"sessionMode":"conversation","captureInputs":capture}))
    rt.activate(release["id"],None)
    return lab,rt,version,release


def turn(rt,id_="pause",question="proposal",user="alice",session="chat"):
    return rt.predict("local","lab",PredictRequest(requestId=id_,records=[{"question":question}],user=user,session=session))


def resume_request(rt,rel,id_="resume",action="approve",value=None,user="alice",session="chat"):
    snap=inspect(rt,rel["id"],user,session)
    return ResumeRequest(requestId=id_,records=snap["records"],user=user,session=session,expectedRelease=rel["id"],approval={
        "expectedRevision":snap["head"]["revision"],"expectedCheckpointSha256":snap["head"]["checkpointSha256"],
        "interruptId":snap["pending"]["id"],"action":action,"value":value})


def head(rt,rel,user="alice",session="chat"):
    return rt.ps.conversation(dumps([rel["id"],user,session]))


@pytest.mark.parametrize("action,value,result",[("approve",None,"approve: proposal / 1"),("reject",None,"reject: proposal / 1"),("edit","reviewed","edit: reviewed / 1")])
def test_native_pause_restart_resume_end_and_idempotency(tmp_path,action,value,result):
    lab,rt,version,rel=setup(tmp_path)
    source=lab.final("r1")
    assert head(rt,rel) is None  # warmup is paused privately, not an approved fabricated turn
    paused=turn(rt)
    assert paused["status"]==202 and paused["error"] is None and paused["result"]["predictions"]==[]
    assert paused["conversationState"]["revision"]==1 and "_status" not in paused["timings"]
    assert turn(rt)["idempotentReplay"] and head(rt,rel)["revision"]==1
    snap=inspect(rt,rel["id"],"alice","chat")
    assert snap["pending"]["value"]["proposed"]=="proposal" and snap["state"]["n"]==1
    envelope,native=rt.pipeline(version["id"]).decode(head(rt,rel)["checkpointSha256"])
    assert any(w[1]=="__interrupt__" for w in native["pendingWrites"])
    req=resume_request(rt,rel,action=action,value=value)
    rt=ProductionRuntime(lab.store)
    out=rt.predict("local","lab",req)
    assert out["status"]==200,out
    assert out["result"]["predictions"]==[result]
    assert out["conversationState"]["revision"]==2 and out["result"]["pending"] is None
    assert out["conversationState"]["threadId"]==paused["conversationState"]["threadId"]
    assert rt.predict("local","lab",req)["idempotentReplay"] and head(rt,rel)["revision"]==2
    assert lab.final("r1")==source
    # Exact state agrees with the independently paused/resumed native research graph.
    _,reference=lab.run(graph(),thread="reference",inp={"question":"proposal"})
    lab.run(graph(),run_id=reference,resume={"action":action,**({"value":value} if value is not None else {})})
    assert rt.pipeline(version["id"]).checkpoint_state(head(rt,rel)["checkpointSha256"])==lab.final(reference)
    again=turn(rt,"next","second")
    assert again["status"]==202 and inspect(rt,rel["id"],"alice","chat")["state"]["n"]==2


def test_stale_wrong_interrupt_input_and_concurrent_decisions_refuse(tmp_path):
    _,rt,v,rel=setup(tmp_path);turn(rt)
    req=resume_request(rt,rel);before=head(rt,rel)
    blocked=turn(rt,"new-while-paused")
    assert blocked["status"]==409 and blocked["error"]["code"]=="E_APPROVAL_PENDING" and head(rt,rel)==before
    for id_,changes,code in [("wrong-id",{"interruptId":"wrong"},"E_APPROVAL_DECISION"),("stale",{"expectedRevision":2},"E_SESSION_CONFLICT")]:
        bad=req.model_copy(update={"requestId":id_,"approval":req.approval.model_copy(update=changes)})
        out=rt.predict("local","lab",bad)
        assert out["error"]["code"]==code and head(rt,rel)==before
    bad=req.model_copy(update={"requestId":"new-input","records":[{"question":"changed"}]})
    assert rt.predict("local","lab",bad)["error"]["code"]=="E_APPROVAL_INPUT" and head(rt,rel)==before
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda id_:rt.predict("local","lab",req.model_copy(update={"requestId":id_})),["decision-a","decision-b"]))
    assert sorted(t["status"] for t in results)==[200,409] and head(rt,rel)["revision"]==2


def test_failed_cancelled_expired_and_sqlite_failure_preserve_paused_head(tmp_path,monkeypatch):
    lab,rt,v,rel=setup(tmp_path);turn(rt);before=head(rt,rel);p=rt.pipeline(v["id"]);original=p.predict
    def cancel(*a,**kw):
        result=original(*a,**kw);rt.ps.cancel("alice","cancel");return result
    monkeypatch.setattr(p,"predict",cancel)
    out=rt.predict("local","lab",resume_request(rt,rel,"cancel"))
    assert out["status"]==409 and out["conversationState"] is None and head(rt,rel)==before
    monkeypatch.setattr(p,"predict",original)
    req=resume_request(rt,rel,"expired");result,timing=p.predict(req.records,checkpoint=before["checkpointSha256"],approval=req.approval)
    rt.ps.begin_request("alice","expired","fp",rel["id"])
    from production.approval_store import finish_request
    out=finish_request(rt.ps,"alice","expired",{"status":200,"error":None,"result":result},conversation=(dumps([rel["id"],"alice","chat"]),before,timing["_checkpoint"]),deadline=time.perf_counter()-1)
    assert out["status"]==504 and head(rt,rel)==before
    rt.ps.query("CREATE TRIGGER fail_finish BEFORE UPDATE OF trace ON requests BEGIN SELECT RAISE(ABORT, 'forced approval persistence failure'); END")
    with pytest.raises(Exception,match="forced approval persistence failure"):
        rt.predict("local","lab",resume_request(rt,rel,"write-failed"))
    assert head(rt,rel)==before and rt.ps.trace("alice","write-failed")["available"] is False
    rt.ps.query("DROP TRIGGER fail_finish")
    restarted=ProductionRuntime(lab.store)
    assert head(restarted,rel)==before
    assert restarted.predict("local","lab",resume_request(restarted,rel,"after-crash"))["status"]==200


def test_capture_off_still_has_committed_review_and_scope_isolation(tmp_path):
    _,rt,v,rel=setup(tmp_path,False)
    for user,session in [("alice","chat"),("bob","chat"),("alice","other")]:
        out=turn(rt,f"pause-{user}-{session}",user=user,session=session)
        assert out["records"] is None and out["result"]["agent"]["events"] is None and out["conversationState"]["revision"]==1
    assert inspect(rt,rel["id"],"alice","chat")["pending"]
    assert rt.predict("local","lab",resume_request(rt,rel))["status"]==200
    assert inspect(rt,rel["id"],"bob","chat")["pending"]


def test_http_inspection_resume_replay_monitor_and_actions(tmp_path):
    lab,rt,v,rel=setup(tmp_path)
    with TestClient(create_app(lab.wb)) as client:
        body=PredictRequest(requestId="http-pause",records=[{"question":"proposal"}],user="alice",session="chat").model_dump()
        paused=client.post("/api/serve/local/lab/predict",json=body)
        assert paused.status_code==202,paused.text
        r=client.get(f'/api/production/releases/{rel["id"]}/approval?user=alice&session=chat')
        assert r.status_code==200 and r.json()["pending"]["value"]["actions"]==["approve","reject","edit"]
        before=head(rt,rel);snap=r.json()
        blocked=client.post(f'/api/production/releases/{rel["id"]}/conversation/reset',json={"actionId":"reset-pending","user":"alice","session":"chat","expectedRevision":1,"expectedCheckpointSha256":before["checkpointSha256"],"reason":"SYNTHETIC"})
        assert blocked.status_code==409 and head(rt,rel)==before
        # Replay of a pending initial turn pauses in isolation and never commits.
        replay=client.post('/api/production/requests/http-pause/replay',json={"user":"alice"})
        assert replay.status_code==200 and replay.json()["result"]["pending"] and head(rt,rel)==before
        resume=resume_request(rt,rel,"http-edit","edit","edited")
        out=client.post('/api/serve/local/lab/resume',json=resume.model_dump())
        assert out.status_code==200 and out.json()["result"]["predictions"]==["edit: edited / 1"]
        assert client.post('/api/serve/local/lab/resume',json=resume.model_dump()).json()["idempotentReplay"]
        before=head(rt,rel)
        replay=client.post('/api/production/requests/http-edit/replay',json={"user":"alice"})
        assert replay.status_code==200 and replay.json()["result"]["predictions"]==["edit: edited / 1"] and head(rt,rel)==before
        assert client.post('/api/production/requests/http-edit/labels',json={"user":"alice","labels":["edit: edited / 1"]}).status_code==200
        monitor=client.get(f'/api/production/releases/{rel["id"]}/monitor').json()
        assert monitor["health"]["errors"]==0 and monitor["health"]["pendingSegments"]==1
        assert monitor["labelBasedQuality"]["values"]["exactStringAgreement"]==1 and monitor["usage"]["successfulTurnModelCalls"]==0


def test_actual_service_kill_preserves_pending_native_writes(tmp_path):
    from test_agent_persistence import Service
    lab,rt,v,rel=setup(tmp_path)
    service=Service(lab.wb).start()
    try:
        body=PredictRequest(requestId="kill-pause",records=[{"question":"restart proposal"}],user="alice",session="chat").model_dump()
        assert service.post('/api/serve/local/lab/predict',body).status_code==202
        reviewed=resume_request(rt,rel,"kill-resume","edit","after actual SIGKILL")
        before=head(rt,rel)
        service.kill()
        service.start()
        assert head(rt,rel)==before
        restored=service.get(f'/api/production/releases/{rel["id"]}/approval?user=alice&session=chat')
        assert restored.json()["pending"]["id"]==reviewed.approval.interruptId
        out=service.post('/api/serve/local/lab/resume',reviewed.model_dump())
        assert out.status_code==200 and out.json()["result"]["predictions"]==["edit: after actual SIGKILL / 1"]
    finally:
        if service.proc and service.proc.poll() is None:
            service.kill()

@pytest.mark.parametrize('change', ['two-reviews','file-tool','no-thread','edit-integer','mutable-input','cycle','no-actions','duplicate-actions','fixed-fork'])
def test_approval_contract_refuses_unsupported_graphs(change):
    from graph_core.schema import Graph
    doc=graph().to_json()
    if change=='two-reviews':
        doc['nodes'].append(sm.N('other','agent.human_interrupt',decision_field='decision'))
        doc['edges']=sm.chain('START','tick','draft','review','other','finish','END')
    elif change=='file-tool':
        doc['nodes'][1]=sm.N('draft','agent.tool_call',tool='write_note',args={'path':'note.txt','text':'proposal'},output_field='answer',allowed_dir='/tmp')
    elif change=='no-thread':
        for f in doc['agent']['state']: f['scope']='turn'
    elif change=='edit-integer':
        doc['nodes'][2]['config']['edit_field']='n'
    elif change=='mutable-input':
        doc['nodes'][1]['config']['assignments'][0]['field']='question'
    elif change=='cycle':
        doc['edges'].append(sm.E('finish','tick'))
    elif change=='no-actions':
        doc['nodes'][2]['config']['actions']=[]
    elif change=='duplicate-actions':
        doc['nodes'][2]['config']['actions']=['approve','approve']
    elif change=='fixed-fork':
        doc['edges'].append(sm.E('tick','review'))
    with pytest.raises(ProductionError): aa.contract(Graph.model_validate(doc), {'question':'proposal'})


def test_edit_requires_text_and_non_edit_refuses_values():
    from production.approval_requests import Review
    from pydantic import ValidationError
    base={'expectedRevision':1,'expectedCheckpointSha256':'a'*64,'interruptId':'native'}
    for update in ({'action':'edit'},{'action':'edit','value':3},{'action':'approve','value':'arbitrary'},{'action':'other'},{'action':'edit','value':'x'*2001}):
        with pytest.raises(ValidationError): Review.model_validate({**base,**update})


def test_paused_cancellation_and_active_time_budget_do_not_commit(tmp_path,monkeypatch):
    lab,rt,v,rel=setup(tmp_path);p=rt.pipeline(v['id']);original=p.predict
    def cancel(*a,**kw):
        out=original(*a,**kw);rt.ps.cancel('alice','cancel-pause');return out
    monkeypatch.setattr(p,'predict',cancel)
    assert turn(rt,'cancel-pause')['status']==409 and head(rt,rel) is None
    monkeypatch.setattr(p,'predict',original)
    turn(rt)
    before=head(rt,rel);req=resume_request(rt,rel)
    payload,native=p.decode(before['checkpointSha256'])
    payload['budget']['activeSeconds']=p.spec.limits.maxSeconds
    # Real serialized paused native checkpoint, modified only to exhaust recorded active time.
    sha=lab.store.put_bytes(dumps(payload).encode())
    with pytest.raises(ProductionError,match='active-time budget exhausted'):
        original(req.records,checkpoint=sha,approval=req.approval)
    assert head(rt,rel)==before
