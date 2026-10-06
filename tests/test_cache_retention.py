"""Real native cache pruning, owner lifecycle, revision guards and offline recovery."""
import json
import shutil
import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from artifact_store import ArtifactStore
from control.app import create_app
from graph_core.project_io import Project, save_project
from maintenance.cache_retention import CacheRetention, RetentionPolicy
from tabular_helpers import example, set_cfg, FIX
from test_node_cache import run, outputs
from workbench_backup.core import create, restore


@pytest.fixture
def native(tmp_path):
    store=ArtifactStore(tmp_path/'workbench')
    graph=set_cfg(example('tabular_regression'),'housing',path=str(FIX/'synthetic_housing.csv'))
    save_project(Project(graph),store.root/'projects/p1.project.json')
    run(store,graph,'original')
    run(store,set_cfg(graph.model_copy(deep=True),'ols',fit_intercept=False),'variant')
    assert len(store.node_cache_entries())==19
    return store,graph


def wait_receipt(retention,timeout=10):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        state=retention.inspect('p1')
        if state['receipts']:return state
        time.sleep(.05)
    raise AssertionError('Actual configured periodic retention did not record a native receipt')


def test_disabled_default_no_metadata_or_pruning_and_strict_bounds(tmp_path):
    store=ArtifactStore(tmp_path);retention=CacheRetention(store)
    assert retention.inspect('p1')['policy'] is None
    retention.start()
    try:assert retention.inspect('p1')['running'] and retention.tick()==[] and not retention.path.exists()
    finally:retention.stop()
    assert not retention.inspect('p1')['running'] and not retention.path.exists()
    for invalid in [{'keepLatestPerNode':0},{'enabled':'yes'},{'intervalSeconds':4},{'olderThanHours':float('nan')},{'olderThanHours':float('inf')},{'unknown':True}]:
        with pytest.raises(ValidationError):RetentionPolicy.model_validate(invalid)


def test_actual_periodic_prune_preserves_recorded_runs_protected_bytes_and_restart(native,tmp_path):
    store,graph=native;retention=CacheRetention(store)
    protected=next(e for e in store.node_cache_entries() if e['run_id']=='original' and e['node_id']=='ols')
    store.add_artifact('original','probe',store.read_artifact(protected['sha256']),'complete',None,{})
    before={rid:{'row':store.get_run(rid),'events':store.events(rid),'artifacts':store.artifacts(rid),'outputs':outputs(store,rid)} for rid in ['original','variant']}
    policy=RetentionPolicy(enabled=True,olderThanHours=0,intervalSeconds=5)
    preview=retention.preview('p1',policy);assert preview['removed']==3 and preview['bytesFreed']==0 and len(store.node_cache_entries())==19
    retention.configure('p1',policy,0)
    restarted=CacheRetention(store);assert restarted.inspect('p1')['policy']['revision']==1
    restarted.start()
    try:
        state=wait_receipt(restarted);receipt=state['receipts'][0]
        assert receipt['revision']==1 and receipt['error'] is None
        result=receipt['result'];assert result['removed']==3 and result['bytesFreed']>0 and len(result['entriesSha256'])==64 and result['entriesTruncated'] is False
        assert {e['node_id'] for e in result['entries']}=={'ols','metrics','predictions'} and len(store.node_cache_entries())==16
        assert store.verify(protected['sha256'])
        assert {rid:{'row':store.get_run(rid),'events':store.events(rid),'artifacts':store.artifacts(rid),'outputs':outputs(store,rid)} for rid in before}==before
        for rid in before:assert all(store.verify(a['sha256']) for a in store.artifacts(rid))
        with pytest.raises(ValueError,match='E_CACHE_POLICY_STALE'):restarted.configure('p1',policy,0)
        disabled=restarted.configure('p1',RetentionPolicy(enabled=False,olderThanHours=0,intervalSeconds=5),1)
        assert disabled['policy']['revision']==2 and disabled['receipts']==state['receipts']
    finally:restarted.stop()
    # Actual offline copy preserves policy/receipt and run evidence after source deletion.
    snapshot=restarted.inspect('p1');snapshot.pop('running');snapshot.pop('lastError')
    backup=create(store.root,tmp_path/'backup',offline=True)
    shutil.rmtree(store.root)
    restore(tmp_path/'backup',tmp_path/'restored',trusted=True,manifest_sha256=backup['manifestSha256'])
    recovered=CacheRetention(ArtifactStore(tmp_path/'restored'));observed=recovered.inspect('p1');observed.pop('running');observed.pop('lastError');assert observed==snapshot
    assert recovered.tick()==[]
    run(recovered.store,graph,'recomputed');assert outputs(recovered.store,'recomputed')==before['original']['outputs']
    assert next(e['data']['cache']['status'] for e in recovered.store.events('recomputed',types=('node_finished',)) if e['node_id']=='ols')=='miss'


def test_policy_api_native_preview_stale_scope_auth_and_owner_shutdown(native):
    store,_=native;app=create_app(store.root,api_token='SYNTHETIC-local-retention-test-token')
    headers={'Authorization':'Bearer SYNTHETIC-local-retention-test-token'}
    with TestClient(app) as client:
        path='/api/cache/nodes/policies/p1'
        assert client.get(path).status_code==401
        assert client.get(path,headers=headers).json()['policy'] is None
        payload={'enabled':False,'keepLatestPerNode':1,'olderThanHours':0,'intervalSeconds':5}
        preview=client.post(path+'/preview',json=payload,headers=headers);assert preview.status_code==200 and preview.json()['removed']==3 and preview.json()['bytesFreed']==0
        response=client.put(path,json={'expectedRevision':0,'policy':payload},headers=headers);assert response.status_code==200 and response.json()['running']
        assert len(store.node_cache_entries())==19 and app.state.services.cache_retention.tick()==[]
        stale=client.put(path,json={'expectedRevision':0,'policy':payload},headers=headers);assert stale.status_code==409 and stale.json()['detail']['code']=='E_CACHE_POLICY_STALE'
        missing=client.put('/api/cache/nodes/policies/absent',json={'policy':payload},headers=headers);assert missing.status_code==404 and missing.json()['detail']['code']=='E_CACHE_POLICY_PROJECT'
        assert client.put(path,json={'expectedRevision':True,'policy':payload},headers=headers).status_code==422
    assert not app.state.services.cache_retention.inspect('p1')['running']


def test_actual_paused_native_run_defers_pruning_until_resumed(tmp_path):
    from agent import samples as sm
    from agent_helpers import Lab
    lab=Lab(tmp_path)
    graph=sm.make_graph([sm.N('review','agent.human_interrupt',show_field='question',decision_field='decision')],sm.chain('START','review','END'),
                        {'state':[sm.S('question'),sm.S('decision','object')]})
    status,rid=lab.run(graph,inp={'question':'SYNTHETIC actual paused native review'})
    assert status=='paused'
    before=(lab.store.get_run(rid),lab.store.events(rid),lab.store.artifacts(rid))
    retention=CacheRetention(lab.store)
    retention.configure('p1',RetentionPolicy(enabled=True,olderThanHours=0,intervalSeconds=5),0)
    retention.start()
    try:
        receipt=wait_receipt(retention)['receipts'][0]
        assert receipt['result']['deferred'] and receipt['result']['runIds']==[rid] and receipt['error'] is None
        assert (lab.store.get_run(rid),lab.store.events(rid),lab.store.artifacts(rid))==before
        status,_=lab.run(graph,run_id=rid,resume={'action':'approve'})
        assert status=='completed'
        end=time.monotonic()+10
        while time.monotonic()<end:
            receipts=retention.inspect('p1')['receipts']
            if receipts and receipts[0]['seq']!=receipt['seq']:break
            time.sleep(.05)
        assert receipts[0]['result']['removed']==0 and 'deferred' not in receipts[0]['result']
    finally:retention.stop()
