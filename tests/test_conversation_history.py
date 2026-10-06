"""Restore actual native historical state with reviewed identities and atomic receipts."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from control.app import create_app
from control.production_api import ConversationRestore
from production.conversations import action
from production.history import inspect_history
from production.pipeline import ProductionError
from production.runtime import ProductionRuntime
from test_conversation_actions import body
from test_production_conversation import setup, req, head, graph


def seed(tmp_path, capture=True):
    lab, rt, version, rel = setup(tmp_path, capture)
    turns = [rt.predict('local', 'lab', req(word, word)) for word in ('first', 'second', 'third')]
    assert all(t['status'] == 200 for t in turns)
    return lab, rt, version, rel, turns


def reviewed(rt, rel, id_='restore', source='second'):
    current = head(rt, rel)
    history = inspect_history(rt, rel['id'], 'alice', 'chat', source)
    return ConversationRestore(actionId=id_, user='alice', session='chat', reason='SYNTHETIC reviewed historical continuation',
        expectedRevision=current['revision'], expectedCheckpointSha256=current['checkpointSha256'],
        **{k: history[k] for k in ('sourceRequestId', 'sourceTraceSha256', 'sourceCheckpointSha256')})


@pytest.mark.parametrize('capture', [True, False])
def test_restore_whole_native_checkpoint_new_thread_monotonic_restart_and_history(tmp_path, capture):
    lab, rt, v, rel, turns = seed(tmp_path, capture)
    research = lab.final('r1')
    selected = inspect_history(rt, rel['id'], 'alice', 'chat', 'second')
    assert selected['state'] == {'question':'second','n':2,'history':['first','second'],'turns':1,'answer':'second: 2/1'}
    original = head(rt, rel)
    request = reviewed(rt, rel)
    restored = action(rt, rel['id'], 'restore', request)
    assert restored['head']['revision'] == 4 and restored['head']['lastRequestId'] is None
    assert restored['sourceHead'] == original
    assert restored['threadId'] != turns[1]['conversationState']['threadId']
    assert rt.pipeline(v['id']).checkpoint_state(restored['head']['checkpointSha256']) == selected['state']
    continued = rt.predict('local', 'lab', req('continued', 'next'))
    assert continued['result']['predictions'] == ['next: 3/1']
    assert continued['conversationState']['revision'] == 5 and continued['conversationState']['threadId'] == restored['threadId']
    for text in ('first', 'second', 'next'):
        status, reference = lab.run(graph(), thread='independent-reference', inp={'question':text})
        assert status == 'completed'
    assert rt.pipeline(v['id']).checkpoint_state(head(rt,rel)['checkpointSha256']) == lab.final(reference)
    restarted = ProductionRuntime(lab.store)
    duplicate = action(restarted, rel['id'], 'restore', request)
    assert duplicate['idempotentReplay'] and duplicate['actionSha256'] == restored['actionSha256']
    assert head(restarted, rel)['revision'] == 5 and lab.final('r1') == research
    for t in turns:
        assert rt.ps.trace('alice', t['requestId'])['traceSha256'] == t['traceSha256']
    assert inspect_history(rt,rel['id'],'alice','chat','second') == selected
    event = rt.ps.query("SELECT data FROM lifecycle WHERE type='conversation_restore'")[0]
    assert json.loads(event['data'])['sourceTraceSha256'] == selected['sourceTraceSha256']


def test_restore_after_reset_and_conflicting_retry(tmp_path):
    _, rt, _, rel, _ = seed(tmp_path)
    action(rt,rel['id'],'reset',body(rt,rel,'reset'))
    request = reviewed(rt,rel,source='first')
    assert request.expectedCheckpointSha256 is None
    result = action(rt,rel['id'],'restore',request)
    assert result['head']['revision'] == 5
    assert rt.predict('local','lab',req('after'))['result']['predictions'] == ['SYNTHETIC input: 2/1']
    with pytest.raises(ProductionError) as e:
        action(rt,rel['id'],'restore',request.model_copy(update={'reason':'different'}))
    assert e.value.code == 'E_SESSION_ACTION_CONFLICT'


@pytest.mark.parametrize('field', ['sourceTraceSha256','sourceCheckpointSha256','expectedCheckpointSha256','expectedRevision'])
def test_reviewed_source_and_target_must_match(tmp_path,field):
    _,rt,_,rel,_ = seed(tmp_path)
    before=head(rt,rel)
    request=reviewed(rt,rel).model_copy(update={field:99 if field=='expectedRevision' else '0'*64})
    with pytest.raises(ProductionError) as e:
        action(rt,rel['id'],'restore',request)
    assert e.value.code == ('E_SESSION_CONFLICT' if field.startswith('expected') else 'E_SESSION_HISTORY_CONFLICT')
    assert head(rt,rel)==before and rt.ps.query('SELECT * FROM conversation_actions')==[]


@pytest.mark.parametrize('user,session', [('bob','chat'),('alice','other')])
def test_source_scope_is_exact(tmp_path,user,session):
    _,rt,_,rel,_=seed(tmp_path)
    other=rt.predict('local','lab',req('other',user=user,session=session))
    assert other['status']==200
    with pytest.raises(ProductionError) as e:
        inspect_history(rt,rel['id'],'alice','chat','other')
    assert e.value.code in ('E_SESSION_HISTORY_SCOPE','E_REQUEST_NOT_FOUND')


def test_competing_restores_one_winner_and_receipt_failure_rolls_back(tmp_path,monkeypatch):
    _,rt,_,rel,_=seed(tmp_path)
    before=head(rt,rel)
    native_db = rt.ps.db
    def failing_db():
        db = native_db()
        db.execute("CREATE TEMP TRIGGER fail_restore BEFORE INSERT ON conversation_actions BEGIN SELECT RAISE(ABORT,'forced restore rollback'); END")
        assert db.execute("SELECT COUNT(*) FROM sqlite_temp_master WHERE name='fail_restore'").fetchone()[0] == 1
        return db
    monkeypatch.setattr(rt.ps, 'db', failing_db)
    request=reviewed(rt,rel)
    with pytest.raises(sqlite3.IntegrityError,match='forced restore rollback'):
        action(rt,rel['id'],'restore',request)
    assert head(rt,rel)==before and rt.ps.query("SELECT * FROM lifecycle WHERE type='conversation_restore'")==[]
    monkeypatch.setattr(rt.ps, 'db', native_db)
    requests=[reviewed(rt,rel,id_=f'restore-{i}') for i in range(2)]
    def restore(r):
        try:
            return action(rt,rel['id'],'restore',r)
        except ProductionError as e:
            return e.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(restore,requests))
    assert sum(isinstance(r,dict) for r in results)==1 and 'E_SESSION_CONFLICT' in results
    assert head(rt,rel)['revision']==4


def test_source_changed_during_clone_refuses_commit(tmp_path,monkeypatch):
    lab,rt,_,rel,_=seed(tmp_path)
    request=reviewed(rt,rel)
    before=head(rt,rel)
    put=lab.store.put_bytes
    def changing(raw):
        sha=put(raw)
        if json.loads(raw).get('threadId'):
            rt.ps.query("UPDATE requests SET trace=NULL WHERE user='alice' AND id='second'")
        return sha
    monkeypatch.setattr(lab.store,'put_bytes',changing)
    with pytest.raises(ProductionError) as e:
        action(rt,rel['id'],'restore',request)
    assert e.value.code=='E_SESSION_HISTORY_CONFLICT' and head(rt,rel)==before


def test_http_history_preview_restore_and_required_review(tmp_path):
    lab,rt,_,rel,_=seed(tmp_path)
    request=reviewed(rt,rel)
    with TestClient(create_app(lab.wb)) as client:
        endpoint=f"/api/production/releases/{rel['id']}/conversation"
        preview=client.get(endpoint+'/history/second?user=alice&session=chat')
        assert preview.status_code==200 and preview.json()['readOnly'] and preview.json()['state']['n']==2
        args=request.model_dump()
        for field in ('sourceRequestId','sourceTraceSha256','sourceCheckpointSha256','expectedCheckpointSha256'):
            assert client.post(endpoint+'/restore',json={k:v for k,v in args.items() if k!=field}).status_code==422
        result=client.post(endpoint+'/restore',json=args)
        assert result.status_code==200,result.text
        assert result.json()['head']['revision']==4
        assert client.post(endpoint+'/restore',json=args).json()['idempotentReplay']


def test_active_turn_serialization_rejects_stale_review(tmp_path,monkeypatch):
    import threading
    _,rt,v,rel,_=seed(tmp_path)
    request=reviewed(rt,rel)
    ready,release=threading.Event(),threading.Event()
    pipeline=rt.pipeline(v['id'])
    predict=pipeline.predict
    def paused(*a,**kw):
        result=predict(*a,**kw)
        ready.set()
        assert release.wait(5)
        return result
    monkeypatch.setattr(pipeline,'predict',paused)
    def restore():
        with pytest.raises(ProductionError) as e:
            action(rt,rel['id'],'restore',request)
        return e.value.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        turn=pool.submit(rt.predict,'local','lab',req('active'))
        assert ready.wait(5)
        mutation=pool.submit(restore)
        try:
            assert not mutation.done()
        finally:
            release.set()
        assert turn.result(timeout=5)['status']==200
        assert mutation.result(timeout=5)=='E_SESSION_CONFLICT'
    assert head(rt,rel)['revision']==4 and rt.ps.query('SELECT * FROM conversation_actions')==[]


@pytest.mark.parametrize('change', ['running','failed','thread','revision','session','version'])
def test_incomplete_failed_or_mismatched_history_refused(tmp_path,change):
    lab,rt,_,rel,_=seed(tmp_path)
    before=head(rt,rel)
    trace=rt.ps.trace('alice','first')
    trace.pop('traceSha256')
    if change=='running':
        rt.ps.query("UPDATE requests SET trace=NULL,state='running' WHERE user='alice' AND id='first'")
    else:
        if change=='failed': trace['status']=500
        elif change=='thread': trace['conversationState']['threadId']='different'
        elif change=='revision': trace['conversationState']['revision']=False
        elif change=='session': trace['session']='other'
        elif change=='version': trace['versionId']='0'*64
        sha=lab.store.put_bytes(json.dumps(trace).encode())
        rt.ps.query("UPDATE requests SET trace=? WHERE user='alice' AND id='first'",(sha,))
    with pytest.raises(ProductionError) as e:
        inspect_history(rt,rel['id'],'alice','chat','first')
    assert e.value.code in ('E_SESSION_HISTORY_EMPTY','E_SESSION_HISTORY_INTEGRITY','E_SESSION_HISTORY_SCOPE')
    assert head(rt,rel)==before


def test_no_inference_corrupt_source_and_busy_lock_refused(tmp_path,monkeypatch):
    from tabular.core import dumps
    from production.models import ReleaseCreate
    lab,rt,v,rel,_=seed(tmp_path)
    request=reviewed(rt,rel,source='first')
    pipeline=rt.pipeline(v['id'])
    monkeypatch.setattr(pipeline,'predict',lambda *_a,**_kw:pytest.fail('Restore/preview must not execute native inference'))
    assert inspect_history(rt,rel['id'],'alice','chat','first')['state']['n']==1
    action(rt,rel['id'],'restore',request)
    before=head(rt,rel)
    corrupt=reviewed(rt,rel,id_='corrupt',source='second')
    lab.store.path_of(corrupt.sourceCheckpointSha256).write_bytes(b'corrupt historical source')
    with pytest.raises(ProductionError) as e:
        action(rt,rel['id'],'restore',corrupt)
    assert e.value.code=='E_ARTIFACT_INTEGRITY' and head(rt,rel)==before
    monkeypatch.undo()
    short=rt.create_release(ReleaseCreate(versionId=v['id'],config={'maxBatch':1,'sessionMode':'conversation','timeoutSeconds':1.0}))  # bounds the busy-lock wait; 0.2 s was too tight for a slow runner's warmup turn
    rt.activate(short['id'],rel['id'])
    assert rt.predict('local','lab',req('short'))['status']==200
    review=reviewed(rt,short,id_='busy',source='short')
    lock=rt.session_locks[dumps([short['id'],'alice','chat'])]
    lock.acquire()
    try:
        with pytest.raises(ProductionError) as e:
            action(rt,short['id'],'restore',review)
        assert e.value.code=='E_SESSION_BUSY'
    finally:
        lock.release()


def test_real_process_death_in_restore_transaction_then_restart_retry(tmp_path):
    import subprocess
    import sys
    import time
    from agent_helpers import REPO
    lab,rt,_,rel,_=seed(tmp_path)
    before=head(rt,rel)
    request=reviewed(rt,rel)
    args=tmp_path/'restore.json'
    args.write_text(json.dumps(request.model_dump()))
    ready=tmp_path/'uncommitted'
    code='''import json,sys,time
from pathlib import Path
sys.path.insert(0,'services')
from artifact_store import ArtifactStore
from production.runtime import ProductionRuntime
from production.conversations import action
from control.production_api import ConversationRestore
rt=ProductionRuntime(ArtifactStore(Path(sys.argv[1])))
original=rt.ps.event
def uncommitted(db,typ,data):
    original(db,typ,data)
    if typ=='conversation_restore':
        Path(sys.argv[2]).write_text('uncommitted head/receipt/event')
        while True:time.sleep(.1)
rt.ps.event=uncommitted
action(rt,sys.argv[3],'restore',ConversationRestore.model_validate(json.loads(Path(sys.argv[4]).read_text())))
'''
    child=subprocess.Popen([sys.executable,'-c',code,str(lab.wb),str(ready),rel['id'],str(args)],cwd=REPO)
    try:
        until=time.monotonic()+20
        while not ready.exists() and child.poll() is None and time.monotonic()<until:time.sleep(.05)
        assert ready.exists(),child.poll()
        child.kill()
        assert child.wait(timeout=10)!=0
    finally:
        if child.poll() is None:
            child.kill();child.wait(timeout=10)
    restarted=ProductionRuntime(lab.store)
    assert head(restarted,rel)==before
    assert restarted.ps.query('SELECT * FROM conversation_actions')==[]
    assert restarted.ps.query("SELECT * FROM lifecycle WHERE type='conversation_restore'")==[]
    result=action(restarted,rel['id'],'restore',request)
    assert action(restarted,rel['id'],'restore',request)['actionSha256']==result['actionSha256']
    assert restarted.ps.query('SELECT COUNT(*) AS n FROM conversation_actions')[0]['n']==1


def test_shared_token_protects_preview_and_restore(tmp_path,monkeypatch):
    lab,rt,_,rel,_=seed(tmp_path)
    args=reviewed(rt,rel).model_dump()
    monkeypatch.setenv('VOID_API_TOKEN','historical-shared-token')
    with TestClient(create_app(lab.wb)) as c:
        endpoint=f"/api/production/releases/{rel['id']}/conversation"
        assert c.get(endpoint+'/history/second?user=alice&session=chat').status_code==401
        assert c.post(endpoint+'/restore',json=args).status_code==401
        response=c.post(endpoint+'/restore',json=args,headers={'Authorization':'Bearer historical-shared-token'})
        assert response.status_code==200,response.text


def test_deadline_after_clone_or_receipt_cas_never_changes_head(tmp_path,monkeypatch):
    import time
    from production.models import ReleaseCreate
    lab,rt,v,rel,_=seed(tmp_path)
    short=rt.create_release(ReleaseCreate(versionId=v['id'],config={'maxBatch':1,'sessionMode':'conversation','timeoutSeconds':1.0}))  # 1 s leaves room for a slow runner's warmup turn; the write below sleeps past it
    rt.activate(short['id'],rel['id'])
    assert rt.predict('local','lab',req('short'))['status']==200
    before=head(rt,short)
    review=reviewed(rt,short,source='short')
    put=lab.store.put_bytes
    def slow(raw):
        if json.loads(raw).get('operation')=='restore':time.sleep(1.1)
        return put(raw)
    monkeypatch.setattr(lab.store,'put_bytes',slow)
    with pytest.raises(ProductionError) as e:
        action(rt,short['id'],'restore',review)
    assert e.value.code=='E_REQUEST_TIMEOUT' and head(rt,short)==before
    assert rt.ps.query('SELECT * FROM conversation_actions')==[]
    assert rt.ps.query("SELECT * FROM lifecycle WHERE type='conversation_restore'")==[]


def test_other_release_and_stateless_history_refused(tmp_path):
    from production.models import ReleaseCreate
    _,rt,v,rel,_=seed(tmp_path)
    other=rt.create_release(ReleaseCreate(versionId=v['id'],config={'maxBatch':1,'sessionMode':'conversation','concurrency':1}))
    with pytest.raises(ProductionError) as e:
        inspect_history(rt,other['id'],'alice','chat','first')
    assert e.value.code=='E_SESSION_HISTORY_SCOPE'
    from test_production_agent import setup as stateless
    _,native,_,isolated=stateless(tmp_path/'isolated')
    with pytest.raises(ProductionError) as e:
        inspect_history(native,isolated['id'],'alice','chat','unknown')
    assert e.value.code=='E_RELEASE_CONFIG'
