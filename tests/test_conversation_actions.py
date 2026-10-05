"""Real native session lineage and atomic SQLite actions; no fabricated model responses."""
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from control.app import create_app
from control.production_api import ConversationFork, ConversationReset
from production.conversations import action
from production.models import ReleaseCreate
from production.pipeline import ProductionError
from production.runtime import ProductionRuntime
from test_production_conversation import setup, req, head
from tabular.core import dumps


def body(rt, rel, id_, session='chat', destination=None):
    h = head(rt, rel, session=session)
    data = dict(actionId=id_, user='alice', session=session, expectedRevision=h['revision'],
                expectedCheckpointSha256=h['checkpointSha256'], reason='SYNTHETIC native continuation test')
    return ConversationFork(**data, destinationSession=destination) if destination else ConversationReset(**data)


def seed(tmp_path):
    lab, rt, v, rel = setup(tmp_path)
    first = rt.predict('local', 'lab', req('one', 'first'))
    second = rt.predict('local', 'lab', req('two', 'second'))
    assert second['status'] == 200
    return lab, rt, v, rel, first, second


def test_native_fork_then_reset_lineage_independence_restart_and_duplicate_receipts(tmp_path):
    lab, rt, v, rel, first, second = seed(tmp_path)
    original = head(rt, rel)
    source_state = rt.pipeline(v['id']).checkpoint_state(original['checkpointSha256'])
    fork_req = body(rt, rel, 'fork-one', destination='alternative')
    fork = action(rt, rel['id'], 'fork', fork_req)
    assert fork['head']['revision'] == 1 and fork['head']['lastRequestId'] is None
    assert fork['sourceHead'] == original and head(rt, rel) == original
    assert rt.pipeline(v['id']).checkpoint_state(fork['head']['checkpointSha256']) == source_state
    assert fork['threadId'] != second['result']['agent']['threadId']
    branch = rt.predict('local', 'lab', req('branch', 'branch', session='alternative'))
    assert branch['result']['predictions'] == ['branch: 3/1'] and branch['conversationState']['revision'] == 2
    assert branch['result']['agent']['threadId'] == fork['threadId'] and head(rt, rel) == original
    duplicate = action(rt, rel['id'], 'fork', fork_req)
    assert duplicate['idempotentReplay'] and duplicate['actionSha256'] == fork['actionSha256']
    assert duplicate['head']['revision'] == 1 and head(rt, rel, session='alternative')['revision'] == 2
    reset_req = body(rt, rel, 'reset-one')
    reset = action(rt, rel['id'], 'reset', reset_req)
    assert reset['head'] == {'revision':3, 'checkpointSha256':None, 'lastRequestId':None}
    rt = ProductionRuntime(lab.store)
    assert action(rt, rel['id'], 'reset', reset_req)['idempotentReplay']
    fresh = rt.predict('local','lab', req('fresh', 'fresh'))
    assert fresh['result']['predictions'] == ['fresh: 1/1'] and fresh['conversationState']['revision'] == 4
    assert fresh['conversationParent'] == {'revision':3, 'checkpointSha256':None}
    assert fresh['result']['agent']['threadId'] != second['result']['agent']['threadId']
    assert rt.ps.trace('alice','two')['traceSha256'] == second['traceSha256']
    assert rt.pipeline(v['id']).checkpoint_state(original['checkpointSha256']) == source_state
    assert rt.predict('local','lab', req('one','first'))['idempotentReplay']
    assert head(rt, rel)['revision'] == 4
    assert [r['type'] for r in rt.ps.query("SELECT type FROM lifecycle WHERE type IN ('conversation_fork','conversation_reset')")] == ['conversation_fork','conversation_reset']
    raw = lab.store.read_artifact(reset['actionSha256'])
    assert json.loads(raw)['sourceHead'] == original


def test_changed_checkpoint_duplicate_input_and_existing_destinations_refused(tmp_path):
    _, rt, _, rel, _, _ = seed(tmp_path)
    stale = body(rt, rel, 'stale')
    rt.predict('local','lab',req('next'))
    before = head(rt, rel)
    with pytest.raises(ProductionError) as e:
        action(rt, rel['id'], 'reset', stale)
    assert e.value.code == 'E_SESSION_CONFLICT' and head(rt, rel) == before
    fork = body(rt, rel, 'fork', destination='other')
    action(rt, rel['id'],'fork',fork)
    for destination in ('other','chat'):
        with pytest.raises(ProductionError) as e:
            action(rt,rel['id'],'fork',body(rt,rel,'another-'+destination,destination=destination))
        assert e.value.code == 'E_SESSION_DESTINATION'
    with pytest.raises(ProductionError) as e:
        action(rt,rel['id'],'fork',fork.model_copy(update={'reason':'changed'}))
    assert e.value.code == 'E_SESSION_ACTION_CONFLICT'
    action(rt,rel['id'],'reset',body(rt,rel,'reset'))
    with pytest.raises(ProductionError) as e:
        action(rt,rel['id'],'reset',body_placeholder(stale,'empty'))
    assert e.value.code == 'E_SESSION_EMPTY'


def body_placeholder(req_, id_):
    return req_.model_copy(update={'actionId':id_})


def test_reset_waits_for_active_native_turn_and_refuses_old_review(tmp_path, monkeypatch):
    _, rt, v, rel, _, _ = seed(tmp_path)
    reviewed = body(rt,rel,'racing-reset')
    candidate_ready, release_candidate = threading.Event(), threading.Event()
    p = rt.pipeline(v['id'])
    native = p.predict
    def paused(*a, **kw):
        result = native(*a, **kw)
        candidate_ready.set()
        assert release_candidate.wait(5)
        return result
    monkeypatch.setattr(p,'predict',paused)
    def reset():
        with pytest.raises(ProductionError) as e:
            action(rt,rel['id'],'reset',reviewed)
        return e.value.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        turn = pool.submit(rt.predict,'local','lab',req('active-native'))
        assert candidate_ready.wait(5)
        mutation = pool.submit(reset)
        try:
            assert not mutation.done()
        finally:
            release_candidate.set()
        assert turn.result(timeout=5)['status'] == 200
        assert mutation.result(timeout=5) == 'E_SESSION_CONFLICT'
    assert head(rt,rel)['revision'] == 3
    assert rt.ps.query('SELECT * FROM conversation_actions') == []


@pytest.mark.parametrize('operation',['reset','fork'])
def test_sqlite_receipt_failure_rolls_back_head_event_and_receipt(tmp_path,operation):
    _, rt, _, rel, _, _ = seed(tmp_path)
    before = head(rt,rel)
    req_ = body(rt,rel,'rollback',destination='other' if operation=='fork' else None)
    rt.ps.query("CREATE TRIGGER fail_action BEFORE INSERT ON conversation_actions BEGIN SELECT RAISE(ABORT,'forced action transaction failure'); END")
    with pytest.raises(sqlite3.IntegrityError,match='forced action transaction failure'):
        action(rt,rel['id'],operation,req_)
    assert head(rt,rel) == before and head(rt,rel,session='other') is None
    assert rt.ps.query('SELECT * FROM conversation_actions') == []
    assert rt.ps.query("SELECT * FROM lifecycle WHERE type IN ('conversation_reset','conversation_fork')") == []
    rt.ps.query('DROP TRIGGER fail_action')
    assert action(rt,rel['id'],operation,req_)['idempotentReplay'] is False


def test_competing_forks_never_overwrite_destination_and_opposite_locks_do_not_deadlock(tmp_path):
    _, rt, _, rel, _, _ = seed(tmp_path)
    requests = [body(rt,rel,f'f{i}',destination='other') for i in range(2)]
    def fork(req_):
        try:
            return action(rt,rel['id'],'fork',req_)['head']
        except ProductionError as e:
            return e.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(fork,requests))
    assert sum(isinstance(x,dict) for x in results) == 1 and 'E_SESSION_DESTINATION' in results
    reverse = body(rt,rel,'reverse',session='other',destination='chat')
    forward = body(rt,rel,'forward',destination='other')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fork,q) for q in (reverse,forward)]
        assert [f.result(timeout=5) for f in futures] == ['E_SESSION_DESTINATION']*2


def test_action_timeout_at_lock_and_after_cas_keeps_native_head(tmp_path,monkeypatch):
    lab, rt, v, rel, _, _ = seed(tmp_path)
    short = rt.create_release(ReleaseCreate(versionId=v['id'],config={'maxBatch':1,'sessionMode':'conversation','timeoutSeconds':.2}))
    rt.activate(short['id'],rel['id'])
    rt.predict('local','lab',req('short'))
    reviewed = body(rt,short,'busy')
    scope = dumps([short['id'],'alice','chat'])
    lock = rt.session_locks[scope]
    lock.acquire()
    try:
        with pytest.raises(ProductionError) as e:
            action(rt,short['id'],'reset',reviewed)
        assert e.value.code == 'E_SESSION_BUSY'
    finally:
        lock.release()
    before = head(rt,short)
    put = lab.store.put_bytes
    def slow(raw):
        if json.loads(raw).get('operation') == 'reset':
            time.sleep(.25)
        return put(raw)
    monkeypatch.setattr(lab.store,'put_bytes',slow)
    with pytest.raises(ProductionError) as e:
        action(rt,short['id'],'reset',reviewed)
    assert e.value.code == 'E_REQUEST_TIMEOUT' and head(rt,short) == before


def test_http_reset_empty_inspection_and_historical_parent_replay(tmp_path):
    lab, _, _, rel, _, second = seed(tmp_path)
    with TestClient(create_app(lab.wb)) as c:
        endpoint=f"/api/production/releases/{rel['id']}/conversation"
        old=c.get(endpoint+'?user=alice&session=chat').json()
        args={'user':'alice','session':'chat','actionId':'http','expectedRevision':old['head']['revision'],
              'expectedCheckpointSha256':old['head']['checkpointSha256'],'reason':'SYNTHETIC reset test'}
        r=c.post(endpoint+'/reset',json=args)
        assert r.status_code == 200, r.text
        assert c.post(endpoint+'/reset',json=args).json()['idempotentReplay']
        empty=c.get(endpoint+'?user=alice&session=chat').json()
        assert empty['state'] is None and empty['head']['revision'] == 3
        replay=c.post('/api/production/requests/two/replay',json={'user':'alice'})
        assert replay.status_code == 200 and replay.json()['result']['predictions'] == ['second: 2/1']
        assert c.get(endpoint+'?user=alice&session=chat').json() == empty
        fresh=c.post('/api/serve/local/lab/predict',json=req('fresh').model_dump()).json()
        assert fresh['conversationState']['revision'] == 4 and fresh['result']['predictions'] == ['SYNTHETIC input: 1/1']
        fork=c.post(endpoint+'/fork',json={**args,'actionId':'http-fork','expectedRevision':4,
                   'expectedCheckpointSha256':fresh['conversationState']['checkpointSha256'],'destinationSession':'fork'} )
        assert fork.status_code == 201
        assert c.get(endpoint+'?user=alice&session=fork').json()['state']['n'] == 1
        assert c.get('/api/production/requests/two?user=alice').json()['traceSha256'] == second['traceSha256']


@pytest.mark.parametrize('changes',[{'reason':''},{'actionId':'../x'},{'expectedRevision':0},{'expectedCheckpointSha256':'wrong'},{'session':'../x'}])
def test_http_actions_require_bounded_reviewed_values(tmp_path,changes):
    lab, rt, _, rel, _, _ = seed(tmp_path)
    args=body(rt,rel,'schema').model_dump()
    with TestClient(create_app(lab.wb)) as c:
        r=c.post(f"/api/production/releases/{rel['id']}/conversation/reset",json={**args,**changes})
        assert r.status_code == 422
    assert rt.ps.query('SELECT * FROM conversation_actions') == []


def test_stateless_release_and_other_user_source_refused(tmp_path):
    from test_production_agent import setup as stateless_setup
    _, rt, _, rel = stateless_setup(tmp_path)
    fake_review=ConversationReset(actionId='not-conversation',user='alice',session='chat',expectedRevision=1,expectedCheckpointSha256='0'*64,reason='SYNTHETIC refusal')
    with pytest.raises(ProductionError) as e:
        action(rt,rel['id'],'reset',fake_review)
    assert e.value.code == 'E_RELEASE_CONFIG'
    _, rt, _, rel, _, _ = seed(tmp_path/'conversation')
    review=body(rt,rel,'other-user').model_copy(update={'user':'bob'})
    with pytest.raises(ProductionError) as e:
        action(rt,rel['id'],'reset',review)
    assert e.value.code == 'E_SESSION_EMPTY'


@pytest.mark.parametrize('operation',['reset','fork'])
def test_real_process_death_before_transaction_commit_then_retry_once(tmp_path,operation):
    import subprocess
    import sys
    from agent_helpers import REPO
    lab, rt, _, rel, _, _ = seed(tmp_path)
    before = head(rt,rel)
    reviewed = body(rt,rel,'killed-action',destination='branch' if operation=='fork' else None)
    args = tmp_path/'args.json'
    args.write_text(json.dumps(reviewed.model_dump()))
    ready = tmp_path/'uncommitted'
    code = '''import json,sys,time
from pathlib import Path
sys.path.insert(0,'services')
from artifact_store import ArtifactStore
from production.runtime import ProductionRuntime
from production.conversations import action
from control.production_api import ConversationReset,ConversationFork
rt=ProductionRuntime(ArtifactStore(Path(sys.argv[1])))
original=rt.ps.event
def before_commit(db,typ,data):
    original(db,typ,data)
    if typ in ('conversation_reset','conversation_fork'):
        Path(sys.argv[2]).write_text('head,receipt,event written inside uncommitted transaction')
        while True: time.sleep(.1)
rt.ps.event=before_commit
model=ConversationFork if sys.argv[4]=='fork' else ConversationReset
req=model.model_validate(json.loads(Path(sys.argv[5]).read_text()))
action(rt,sys.argv[3],sys.argv[4],req)
'''
    child = subprocess.Popen([sys.executable,'-c',code,str(lab.wb),str(ready),rel['id'],operation,str(args)],cwd=REPO)
    try:
        until = time.monotonic()+20
        while not ready.exists() and child.poll() is None and time.monotonic()<until:
            time.sleep(.05)
        assert ready.exists(), child.poll()
        child.kill()
        assert child.wait(timeout=10) != 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
    restarted = ProductionRuntime(lab.store)
    assert head(restarted,rel) == before and head(restarted,rel,session='branch') is None
    assert restarted.ps.query('SELECT * FROM conversation_actions') == []
    assert restarted.ps.query("SELECT * FROM lifecycle WHERE type IN ('conversation_reset','conversation_fork')") == []
    result=action(restarted,rel['id'],operation,reviewed)
    assert action(restarted,rel['id'],operation,reviewed)['actionSha256'] == result['actionSha256']
    assert restarted.ps.query('SELECT COUNT(*) AS n FROM conversation_actions')[0]['n'] == 1


def test_actions_do_not_execute_inference_and_refuse_corrupt_checkpoint(tmp_path,monkeypatch):
    lab, rt, v, rel, _, _ = seed(tmp_path)
    p=rt.pipeline(v['id'])
    monkeypatch.setattr(p,'predict',lambda *_a,**_kw: pytest.fail('Session actions must never invoke native inference'))
    source=head(rt,rel)
    action(rt,rel['id'],'fork',body(rt,rel,'no-infer',destination='other'))
    reviewed=body(rt,rel,'corrupt')
    lab.store.path_of(source['checkpointSha256']).write_bytes(b'corrupt native snapshot')
    with pytest.raises(ProductionError) as e:
        action(rt,rel['id'],'reset',reviewed)
    assert e.value.code == 'E_ARTIFACT_INTEGRITY'
    assert rt.ps.query('SELECT COUNT(*) AS n FROM conversation_actions')[0]['n'] == 1


def test_shared_token_protects_action_routes(tmp_path,monkeypatch):
    lab, rt, _, rel, _, _ = seed(tmp_path)
    monkeypatch.setenv('VOID_API_TOKEN','session-action-shared-token')
    args=body(rt,rel,'protected').model_dump()
    with TestClient(create_app(lab.wb)) as c:
        path=f"/api/production/releases/{rel['id']}/conversation"
        for operation in ('reset','fork'):
            assert c.post(path+'/'+operation,json={**args,'destinationSession':'other'}).status_code == 401
        r=c.post(path+'/reset',json=args,headers={'Authorization':'Bearer session-action-shared-token'})
        assert r.status_code == 200,r.text
