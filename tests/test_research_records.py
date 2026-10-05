"""Actual native runs plus separately authored metadata; no synthesized execution results."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from agent import samples as sm
from agent_helpers import Lab
from research.records import Records,RecordError
from control.research_api import Annotation


def setup(tmp_path):
    lab=Lab(tmp_path)
    for i in range(3):assert lab.run(sm.counter_loop_graph(stop_at=i+1),thread=f't{i}')[0]=='completed'
    return lab,Records(lab.store)


def request(records,id_='r1',mutation='note-one',note='SYNTHETIC research observation',tags=None,revision=0):
    return Annotation(mutationId=mutation,expectedRevision=revision,expectedGraphHash=records.get(id_)['graphHash'],author='SYNTHETIC researcher',note=note,tags=['teaching'] if tags is None else tags).model_dump()


def test_notes_are_authored_revisioned_atomic_idempotent_and_leave_native_evidence_unchanged(tmp_path):
    lab,records=setup(tmp_path)
    final=lab.final('r1');events=lab.events('r1');artifacts=lab.store.artifacts('r1');run=lab.store.get_run('r1')
    assert records.get('r1')['annotation']=={'revision':0,'note':'','tags':[],'author':None,'recordedAt':None}
    args=request(records)
    saved=records.update('r1',args)
    assert saved['annotation']['revision']==1 and saved['annotation']['recordedAt']>0
    assert not saved['idempotentReplay']
    assert records.update('r1',args)['idempotentReplay']
    cleared=records.update('r1',request(records,mutation='clear',note='',tags=[],revision=1))
    assert cleared['annotation']['revision']==2 and cleared['annotation']['note']==''
    assert Records(lab.store).get('r1')['annotation']==cleared['annotation']
    revisions=records.history('r1',limit=1)
    assert revisions['revisions']==[saved['annotation']] and revisions['nextAfter']==1
    assert records.history('r1',after=1)['revisions']==[cleared['annotation']]
    assert records.update('r1',args)['annotation']==saved['annotation']
    assert lab.final('r1')==final and lab.events('r1')==events and lab.store.artifacts('r1')==artifacts and lab.store.get_run('r1')==run


def test_conflicting_reviews_concurrent_writers_and_mutation_reuse_refused(tmp_path):
    _,records=setup(tmp_path)
    bodies=[request(records,mutation=f'write{i}',note=str(i)) for i in range(2)]
    def update(body):
        try:return records.update('r1',body)
        except RecordError as e:return e.code
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(update,bodies))
    assert sum(isinstance(x,dict) for x in results)==1 and 'E_RECORD_CONFLICT' in results
    winner=next(r for r in results if isinstance(r,dict))
    original=next(b for b in bodies if b['mutationId']==winner['mutationId'])
    with pytest.raises(RecordError) as e:records.update('r1',{**original,'note':'different'})
    assert e.value.code=='E_RECORD_MUTATION'
    assert len(records.history('r1')['revisions'])==1


def test_unicode_literal_catalogue_filters_and_bounded_keyset_pages_without_artifact_reads(tmp_path,monkeypatch):
    lab,records=setup(tmp_path)
    records.update('r1',request(records,note='SYNTHETIC Straße colour_100%',tags=['vision','Καλημέρα']))
    records.update('r2',request(records,id_='r2',mutation='note-two',note='SYNTHETIC unrelated',tags=['vision']))
    monkeypatch.setattr(lab.store,'read_artifact',lambda *_:pytest.fail('Catalogue cannot load artifacts'))
    page=records.catalogue(limit=2)
    assert [r['runId'] for r in page['runs']]==['r1','r2'] and page['nextAfter']=='r2'
    assert [r['runId'] for r in records.catalogue(after='r2',limit=2)['runs']]==['r3']
    for query in ('STRASSE','colour_100%','ΚΑΛΗΜΈΡΑ'):
        assert [r['runId'] for r in records.catalogue(query=query)['runs']]==['r1']
    assert len(records.catalogue(tag='vision')['runs'])==2
    assert records.catalogue(query='missing',tag='vision')['runs']==[]
    assert records.catalogue(kind='model')['runs']==[]
    assert len(records.catalogue(kind='agent',status='completed')['runs'])==3
    assert records.catalogue(project_id='missing')['runs']==[]
    assert records.catalogue(tag='vis%')['runs']==[]


def test_reviewed_graph_and_stored_graph_identity_refused(tmp_path):
    lab,records=setup(tmp_path)
    args=request(records)
    with pytest.raises(RecordError) as e:records.update('r1',{**args,'expectedGraphHash':'0'*64})
    assert e.value.code=='E_RECORD_IDENTITY'
    records.update('r1',args)
    lab.store._exec("UPDATE runs SET graph_hash=? WHERE id='r1'",('0'*64,))
    for op in (lambda:records.get('r1'),lambda:records.catalogue(),lambda:records.update('r1',args)):
        with pytest.raises(RecordError) as e:op()
        assert e.value.code=='E_RECORD_IDENTITY'


def test_transaction_failure_rolls_back_current_revision_receipt_and_history(tmp_path):
    _,records=setup(tmp_path)
    with records.db() as db:db.execute("CREATE TRIGGER fail_revision BEFORE INSERT ON mutations BEGIN SELECT RAISE(ABORT,'forced note rollback'); END")
    args=request(records)
    with pytest.raises(sqlite3.IntegrityError,match='forced note rollback'):records.update('r1',args)
    assert records.get('r1')['annotation']['revision']==0 and records.history('r1')['revisions']==[]
    with records.db() as db:db.execute('DROP TRIGGER fail_revision')
    assert records.update('r1',args)['annotation']['revision']==1


@pytest.mark.parametrize('changes',[{'tags':['']},{'tags':['space ']},{'tags':['a','a']},{'tags':['x'*61]},{'tags':['x']*21},{'author':' '},{'note':'x'*5001},{'expectedRevision':-1},{'expectedGraphHash':'bad'},{'mutationId':'../'}])
def test_metadata_contract_refuses_unbounded_or_ambiguous_fields(tmp_path,changes):
    _,records=setup(tmp_path)
    with pytest.raises(ValueError):Annotation.model_validate({**request(records),**changes})
    assert records.get('r1')['annotation']['revision']==0


def test_http_scope_search_schema_token_and_stale_editor(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from control.app import create_app
    lab,records=setup(tmp_path)
    args=request(records)
    monkeypatch.setenv('VOID_API_TOKEN','research-shared-token')
    with TestClient(create_app(lab.wb)) as c:
        base='/api/research/runs'
        assert c.get(base).status_code==401
        assert c.put(base+'/r1/annotation',json=args).status_code==401
        headers={'Authorization':'Bearer research-shared-token'}
        assert c.get(base,headers=headers).status_code==200
        assert c.get(base+'?limit=0',headers=headers).status_code==422
        assert c.get(base+'?kind=invalid',headers=headers).status_code==422
        assert c.get(base+'/missing',headers=headers).status_code==404
        result=c.put(base+'/r1/annotation',json=args,headers=headers)
        assert result.status_code==200,result.text
        assert c.put(base+'/r1/annotation',json=args,headers=headers).json()['idempotentReplay']
        stale=c.put(base+'/r1/annotation',json={**args,'mutationId':'stale','note':'SYNTHETIC different review'},headers=headers)
        assert stale.status_code==409 and stale.json()['detail']['code']=='E_RECORD_CONFLICT'
        assert c.get(base+'?query=observation&tag=teaching',headers=headers).json()['runs'][0]['annotation']['note']==args['note']
        assert c.get(base+'/r1/annotation/history',headers=headers).json()['revisions'][0]['revision']==1
        assert lab.final('r1')['n']==1


def test_actual_offline_annotation_recovery_after_source_deletion(tmp_path):
    import shutil
    from artifact_store import ArtifactStore
    from control.app import create_app
    from workbench_backup.core import create,restore,verify
    lab,records=setup(tmp_path)
    create_app(lab.wb) # Actual control-owned databases, no running services/worker.
    saved=records.update('r1',request(records))
    records.update('r1',request(records,mutation='note-two',note='SYNTHETIC revised',revision=1))
    history=records.history('r1')
    current=records.get('r1')
    backup=tmp_path/'backup';destination=tmp_path/'restored'
    created=create(lab.wb,backup,offline=True)
    assert 'research.sqlite' in created['databases']
    shutil.rmtree(lab.wb)
    verify(backup)
    restore(backup,destination,trusted=True)
    recovered=Records(ArtifactStore(destination))
    assert recovered.get('r1')==current and recovered.history('r1')==history
    assert recovered.update('r1',request(recovered))['annotation']==saved['annotation']
    assert recovered.catalogue(tag='teaching')['runs'][0]['annotation']['note']=='SYNTHETIC revised'


@pytest.mark.parametrize('corruption',["UPDATE revisions SET tags='invalid'", "UPDATE revisions SET tags='[\"a\",\"a\"]'", 'DELETE FROM revisions'])
def test_malformed_current_annotation_never_appears_unannotated(tmp_path,corruption):
    _,records=setup(tmp_path)
    records.update('r1',request(records))
    with records.db() as db:db.execute(corruption)
    for inspect in (lambda:records.get('r1'),lambda:records.catalogue()):
        with pytest.raises(RecordError) as e:inspect()
        assert e.value.code=='E_RECORD_INTEGRITY'
