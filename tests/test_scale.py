"""Native scale integrations, recovery, inert packaging and extension conformance."""
from __future__ import annotations
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from artifact_store import ArtifactStore
from control.app import create_app
from graph_core.hashing import semantic_hash
from graph_core.project_io import load_project
from graph_core.schema import Graph
from graph_core.validate import validate,require_executable
from scale.common import IntegrationError,digest
from scale.remote import RemoteClient,RemoteRequest,snapshot,materialize
from scale.worker_server import create_worker
from tracking.bridge import Bridge,ExportSelection
from extensions.packages import build_package,inspect_package,import_package
from extensions.sdk import read_manifest
from worker.tabular_run import TabularRunConfig,run_tabular
from conftest import EXAMPLES,ROOT

@pytest.fixture
def recorded(tmp_path):
    store=ArtifactStore(tmp_path/'wb')
    g=load_project(EXAMPLES/'production_sensors.project.json').graph
    cfg=TabularRunConfig()
    store.create_run('native',semantic_hash(g),cfg.model_dump())
    store.add_artifact('native','graph',g.model_dump_json(by_alias=True).encode(),'complete',None,{})
    assert run_tabular(g,cfg,store,'native')=='completed'
    return store,g

def native_python(code,args=()):
    p=subprocess.run([str(ROOT/'.venv-trackers/bin/python'),'-c',code,*map(str,args)],capture_output=True,text=True,timeout=60,
                     env={**os.environ,'MLFLOW_ENABLE_TELEMETRY':'false','WANDB_MODE':'offline'})
    assert p.returncode==0,p.stderr
    return json.loads(p.stdout.splitlines()[-1])

@pytest.mark.parametrize('adapter',['mlflow','wandb'])
def test_native_tracker_reconnect_reuses_confirmed_run_and_artifact_mapping(recorded,adapter):
    store,g=recorded
    bridge=Bridge(store.root)
    sha=next(a['sha256'] for a in store.artifacts('native','node_summary') if a['meta']['node']=='metrics')
    selection=ExportSelection(runId='native',adapter=adapter,shareMetrics=True,artifactSha256=[sha])
    r=bridge.queue(selection);assert r['status']=='pending'
    payload=json.loads(store.read_artifact(r['payloadSha256']))
    assert payload['metrics'] and all('values' not in m for m in payload['metrics'])
    assert not any(k in payload['metadata'] for k in ('config','source','token','data'))
    r=bridge.sync(r['id']);assert r['status']=='confirmed',r['error']
    ext=r['external'];assert len(ext['artifacts'])==1 and ext['artifacts'][0]['sha256']==sha
    # Simulate process death after native confirmation before the local acknowledgement.
    bridge.state.put('tracker',r['id'],{**r,'status':'pending','external':None})
    recovered=Bridge(store.root).sync(r['id']);assert recovered['external']==ext
    assert Bridge(store.root).queue(selection)==recovered
    if adapter=='mlflow':
        measured=native_python('''import json,sys
from mlflow import MlflowClient
c=MlflowClient(tracking_uri='sqlite:///'+sys.argv[1]+'/tracking.sqlite')
e=c.get_experiment_by_name('void-local')
r=c.search_runs([e.experiment_id])
x=r[0]
print(json.dumps({'runs':len(r),'metrics':{k:[{'value':v.value,'step':v.step,'timestamp':v.timestamp} for v in c.get_metric_history(x.info.run_id,k)] for k in x.data.metrics},'artifacts':[a.path for a in c.list_artifacts(x.info.run_id,'cas')]}))''',[store.root/'trackers/mlflow'])
        assert measured['runs']==1 and measured['artifacts']==['cas/'+sha]
        for m in payload['metrics']:
            hist=measured['metrics'][m['key']]
            assert hist==[{k:m[k] for k in ('value','step','timestamp')}]
    else:
        root=store.root/'trackers/wandb'
        assert len(list(root.glob('*/confirmed.json')))==1
        data_files=list(root.glob('*/wandb/offline-run-*/run-*.wandb'))
        assert len(data_files)==1 and data_files[0].stat().st_size>0
        # Decode actual SDK binary records; offline existence alone isn't sufficient evidence.
        observed=native_python('''import sys,json,struct,zlib
from wandb.proto.wandb_internal_pb2 import Record
# Read the native W&B LevelDB record format with checksum verification.
b=open(sys.argv[1],'rb').read();assert b[:7]==b':W&B\\xe1\\xbe\\x00'
pos=7;hist=[];art=[];parts=b''
while pos<len(b):
 remaining=32768-pos%32768
 if remaining<7:pos+=remaining;continue
 crc,n,typ=struct.unpack('<IHB',b[pos:pos+7]);pos+=7
 raw=b[pos:pos+n];pos+=n
 assert zlib.crc32(bytes([typ])+raw)&0xffffffff==crc
 if typ==1:whole=raw
 elif typ==2:parts=raw;continue
 elif typ==3:parts+=raw;continue
 elif typ==4:whole=parts+raw;parts=b''
 else:raise AssertionError('unknown native record type')
 r=Record();r.ParseFromString(whole)
 if r.HasField('history'):hist.append({i.key or '.'.join(i.nested_key):json.loads(i.value_json) for i in r.history.item})
 if r.HasField('artifact'):art.append(r.artifact.name)
print(json.dumps({'history':hist,'artifacts':art}))''',[data_files[0]])
        for m in payload['metrics']:assert any(h.get(m['key'])==m['value'] and h.get('void/source_step')==m['step'] for h in observed['history'])
        assert 'void-'+sha in observed['artifacts']

def test_tracker_sharing_is_explicit_and_failed_sync_preserves_intention(recorded):
    store,g=recorded;b=Bridge(store.root)
    r=b.queue(ExportSelection(runId='native',adapter='mlflow'))
    p=json.loads(store.read_artifact(r['payloadSha256']))
    assert p['metrics']==[] and p['artifacts']==[]
    b.python=Path('/unavailable')
    with pytest.raises(IntegrationError,match='Install'):b.sync(r['id'])
    assert b.state.get('tracker',r['id'])['status']=='pending'
    graph_sha=store.artifacts('native','graph')[0]['sha256']
    with pytest.raises(IntegrationError):b.queue(ExportSelection(runId='native',adapter='wandb',artifactSha256=[graph_sha]))
    with pytest.raises(IntegrationError):b.queue(ExportSelection(runId='not-recorded',adapter='mlflow'))

def test_tracker_artifact_integrity_refuses_mutated_evidence(recorded):
    store,_=recorded;b=Bridge(store.root)
    sha=next(a['sha256'] for a in store.artifacts('native','node_summary') if a['meta']['node']=='metrics')
    r=b.queue(ExportSelection(runId='native',adapter='mlflow',artifactSha256=[sha]))
    store.path_of(sha).write_bytes(b'changed')
    with pytest.raises(IntegrationError,match='changed'):b.sync(r['id'])

@contextmanager
def real_worker(tmp_path,env):
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    endpoint=f'http://127.0.0.1:{port}'
    log=(tmp_path/'worker.log').open('w')
    proc=subprocess.Popen([sys.executable,'-m','scale.worker_server','--workbench',str(tmp_path/'worker'),'--port',str(port)],env=env,stdout=log,stderr=log)
    try:
        end=time.time()+30
        while time.time()<end:
            try:
                r=httpx.get(endpoint+'/v1/health',headers={'Authorization':'Bearer '+env['VOID_WORKER_TOKEN']},timeout=1,trust_env=False)
                if r.status_code==200:break
            except httpx.HTTPError:pass
            if proc.poll() is not None:raise AssertionError((tmp_path/'worker.log').read_text())
            time.sleep(.1)
        else:raise AssertionError('Worker startup timed out')
        yield endpoint,proc
    finally:
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
        log.close()

def test_separate_http_worker_runs_snapshot_and_recovers_without_duplicates(recorded,tmp_path,monkeypatch):
    store,g=recorded
    token='void-local-test-token-123456789';monkeypatch.setenv('VOID_WORKER_TOKEN',token)
    with real_worker(tmp_path,dict(os.environ)) as (url,proc):
        assert httpx.get(url+'/v1/health',trust_env=False).status_code==401
        client=RemoteClient(store.root)
        req=RemoteRequest(graph=g,endpoint=url,tokenEnv='VOID_WORKER_TOKEN',projectId='production_sensors')
        r=client.submit(req);assert r['workerRunId'] and 'separate local' in r['location']
        end=time.time()+30
        while time.time()<end and not r['runId']:
            time.sleep(.1);r=RemoteClient(store.root).refresh(r['id'])
        assert r['status']=='completed',r
        assert proc.pid!=os.getpid()
        native=next(a for a in store.artifacts('native','node_summary') if a['meta']['node']=='metrics')
        remote=next(a for a in store.artifacts(r['runId'],'node_summary') if a['meta']['node']=='metrics')
        x=json.loads(store.read_artifact(native['sha256']));y=json.loads(store.read_artifact(remote['sha256']))
        assert x['values']==y['values'] and x['evaluatedRowIdsSha256']==y['evaluatedRowIdsSha256']
        before=len(store.artifacts(r['runId']));events=len(store.events(r['runId']))
        assert client.submit(req)['runId']==r['runId']
        assert len(store.artifacts(r['runId']))==before and len(store.events(r['runId']))==events
        row=store.get_run(r['runId']);assert row['config']['remote']['originalGraphHash']==semantic_hash(g)
        assert row['graph_hash']!=semantic_hash(g) # materialized source path has distinct execution identity
        for a in store.artifacts(r['runId']):assert store.verify(a['sha256']) and not a['meta'].get('trustedWorkerArtifact')
        worker_state=ArtifactStore(tmp_path/'worker')
        assert len(worker_state.list_runs())==1
        sources=[a for a in worker_state.artifacts(r['workerRunId'],'node_summary') if a['meta']['node']=='sensors']
        ss=json.loads(worker_state.read_artifact(sources[0]['sha256']))
        assert ss['sha256']==snapshot(g)['files']['sensors']['sha256']
        assert ss['path'].startswith(str(tmp_path/'worker'))

@pytest.mark.parametrize('mode',['remote-host','missing-token','unsupported-op','tampered-source','oversize-source'])
def test_worker_contract_refusals(recorded,tmp_path,monkeypatch,mode):
    store,g=recorded
    if mode=='remote-host':
        with pytest.raises(IntegrationError):RemoteClient(store.root).submit(RemoteRequest(graph=g,endpoint='http://example.com:8888',tokenEnv='VOID_WORKER_TOKEN'))
    elif mode=='missing-token':
        monkeypatch.delenv('VOID_WORKER_TOKEN',raising=False)
        with pytest.raises(IntegrationError):RemoteClient(store.root).headers({'tokenEnv':'VOID_WORKER_TOKEN'})
    elif mode=='unsupported-op':
        g.nodes[0].type='core.code_block'
        with pytest.raises(IntegrationError):snapshot(g)
    else:
        bundle=snapshot(g)
        if mode=='tampered-source':bundle['files']['sensors']['base64']=base64.b64encode(b'bad').decode()
        else:
            f=b'x'*(8*1024*1024+1);bundle['files']['sensors']={'base64':base64.b64encode(f).decode(),'sha256':hashlib.sha256(f).hexdigest()}
        with pytest.raises(IntegrationError):materialize(bundle,tmp_path/'isolated')

def test_worker_job_reuse_auth_and_hash_conflicts(recorded,tmp_path):
    _,g=recorded;bundle=snapshot(g);id=digest(bundle)
    app=create_worker(tmp_path/'worker','test-token-123456789')
    with TestClient(app) as c:
        assert c.post('/v1/jobs',json={'id':id,'bundle':bundle}).status_code==401
        headers={'Authorization':'Bearer test-token-123456789'}
        bad=copy.deepcopy(bundle);bad['files']['sensors']['sha256']='0'*64
        assert c.post('/v1/jobs',json={'id':id,'bundle':bad},headers=headers).status_code==422
        assert c.get('/v1/jobs/unknown',headers=headers).status_code==404

def test_portable_package_matches_source_and_does_not_run(recorded,tmp_path):
    store,g=recorded
    ui={'schemaVersion':'1.0.0','positions':{},'synthetic':True}
    p=build_package(g,ui,True);info=inspect_package(p)
    assert info['resources']==1 and not info['executionBlocked'] and not info['installsOrExecutesCode']
    before=len(store.list_runs())
    out=import_package(p,tmp_path/'projects','portable')
    assert len(store.list_runs())==before
    imported=load_project(tmp_path/'projects/portable.project.json').graph
    path=Path(imported.node('sensors').config['path']);assert path.is_file()
    assert path.read_bytes()==Path(g.node('sensors').config['path']).read_bytes()
    assert validate(imported).ok
    assert out['identity']==p['identity']

@pytest.mark.parametrize('mode',['tamper','path','secret','missing-dependency'])
def test_inert_package_refusals_and_unknown_operation_preservation(recorded,tmp_path,mode):
    _,g=recorded;ui={'schemaVersion':'1.0.0','positions':{}}
    if mode=='secret':
        g.nodes[0].config['password']='never-export-this'
        with pytest.raises(IntegrationError):build_package(g,ui)
        return
    p=build_package(g,ui,True)
    if mode=='tamper':
        p['graph']['nodes'][0]['config']['path']='modified'
        with pytest.raises(IntegrationError):inspect_package(p)
    elif mode=='path':
        key=next(iter(p['resources']));p['resources']['../escape']=p['resources'].pop(key)
        p['identity']=digest({k:v for k,v in p.items() if k!='identity'})
        with pytest.raises(IntegrationError):import_package(p,tmp_path/'projects','blocked')
        assert not (tmp_path/'escape').exists()
    else:
        p['graph']['nodes'][0]['type']='community.unavailable'
        p['dependencies'][0]={'operation':'community.unavailable','version':'1.0.0','implementation':{'name':'missing','version':'1.0.0','sha256':'0'*64}}
        p['identity']=digest({k:v for k,v in p.items() if k!='identity'})
        report=import_package(p,tmp_path/'projects','unknown')
        assert report['executionBlocked']
        reread=load_project(tmp_path/'projects/unknown.project.json').graph
        assert reread.nodes[0].type=='community.unavailable' and not validate(reread).ok
        assert any(d.code=='E_PACKAGE_DEPENDENCY' for d in validate(reread).errors)

def test_sdk_real_conformance_and_integrity(tmp_path):
    manifest=EXAMPLES/'plugins/offset/manifest.json'
    p=subprocess.run([sys.executable,'-m','extensions.sdk',str(manifest)],capture_output=True,text=True,timeout=15)
    assert p.returncode==0,p.stderr
    r=json.loads(p.stdout);assert r['passed'] and len(r['cases'])==2
    body=json.loads(manifest.read_text());body['sha256']='0'*64
    (tmp_path/'manifest.json').write_text(json.dumps(body));(tmp_path/'operation.py').write_bytes((manifest.parent/'operation.py').read_bytes())
    with pytest.raises(IntegrationError):read_manifest(tmp_path/'manifest.json')
    # Explicit startup activation is inherited by a real worker and records exact package identities.
    env={**os.environ,'VOID_PLUGIN_MANIFESTS':str(manifest)}
    p=subprocess.run([sys.executable,'-c',"from graph_core.registry import get_op; from extensions.sdk import LOADED; import json; print(json.dumps({'present':get_op('community.table_offset') is not None,'loaded':LOADED}))"],env=env,capture_output=True,text=True,timeout=15)
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)['loaded']['community.table_offset']['sha256']==r['sha256']

def test_integration_ui_endpoints_and_evidence_are_recorded(recorded):
    store,g=recorded
    with TestClient(create_app(store.root)) as c:
        info=c.get('/api/integrations').json();assert info['trackerRuntimeAvailable'] and info['sdkExamples']==['offset']
        assert c.post('/api/extensions/conformance',json={'example':'offset'}).json()['passed']
        compare=c.post('/api/evidence/compare',json={'runIds':['native']}).json()
        assert compare['runs'][0]['summaries'] and compare['runs'][0]['graphHash']==semantic_hash(g)
        record=c.post('/api/evidence',json={'runIds':['native'],'conclusion':'SYNTHETIC evidence only; evaluation stays linked.'}).json()
        assert store.verify(record['sha256'])
        assert c.post('/api/evidence',json={'runIds':['unknown'],'conclusion':'No evidence'}).status_code==422
        assert c.post('/api/packages/export',json={'graph':g.to_json(),'ui':{'schemaVersion':'1.0.0','positions':{}},'includeCsv':True}).status_code==200

def test_acceptance_checklist_covers_all_scenarios_and_has_real_references():
    import re
    data=json.loads((ROOT/'docs/acceptance.json').read_text())
    assert [r['id'] for r in data['rows']]==[f'A{i:02d}' for i in range(1,65)]
    spec=(ROOT/'docs/VISION.md').read_text()
    for r in data['rows']:
        assert re.search(r'\| '+r['id']+r' \| '+re.escape(r['scenario'])+r' \|',spec)
        assert r['status'] in ('bounded evidence','not implemented') and r['scope'] and r['evidence']
        for path in r['evidence']:assert (ROOT/path).exists(),path
    assert all(r['status']=='not implemented' for r in data['rows'] if r['id'] in ('A09','A44'))

def test_a64_connected_postgres_comparison_inspection_conclusion_and_real_serving(tmp_path):
    import importlib.util
    import threading
    import uvicorn
    spec=importlib.util.spec_from_file_location('connected_production_journey',EXAMPLES/'connected_production_journey.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    app=create_app(tmp_path/'control')
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='error'))
    thread=threading.Thread(target=server.run,daemon=True);thread.start()
    try:
        end=time.time()+15
        while not server.started and time.time()<end:time.sleep(.05)
        assert server.started
        result=mod.journey(f'http://127.0.0.1:{port}',namespace='m8-test')
        assert len(result['runIds'])==2 and len(result['evaluation'])==2
        assert result['source']['type']=='postgres.query'
        assert result['traffic']['successfulRequests']>0 and result['traffic']['errors']==0
        assert app.state.services.store.verify(result['conclusion']['sha256'])
        second=app.state.services.store.get_run(result['runIds'][1])
        assert second['config']['source_pins']['sensors']
    finally:
        server.should_exit=True;thread.join(timeout=15)
        assert not thread.is_alive()

def test_worker_retrieval_replays_an_import_after_lost_acknowledgement(recorded,tmp_path,monkeypatch):
    store,g=recorded;monkeypatch.setenv('VOID_WORKER_TOKEN','recovery-test-token-123456')
    with real_worker(tmp_path,dict(os.environ)) as (url,_):
        client=RemoteClient(store.root)
        r=client.submit(RemoteRequest(graph=g,endpoint=url,tokenEnv='VOID_WORKER_TOKEN'))
        end=time.time()+30
        while r['runId'] is None and time.time()<end:time.sleep(.1);r=client.refresh(r['id'])
        assert r['status']=='completed' and r['runId']
        counts=(len(store.list_runs()),len(store.events(r['runId'])),len(store.artifacts(r['runId'])))
        client.state.put('remote',r['id'],{**r,'status':'running','runId':None})
        again=RemoteClient(store.root).refresh(r['id'])
        assert again['runId']==r['runId']
        assert counts==(len(store.list_runs()),len(store.events(r['runId'])),len(store.artifacts(r['runId'])))

def test_worker_disconnection_keeps_exact_pending_bundle(recorded,monkeypatch):
    store,g=recorded;monkeypatch.setenv('VOID_WORKER_TOKEN','pending-test-token-123456')
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    client=RemoteClient(store.root)
    r=client.submit(RemoteRequest(graph=g,endpoint=f'http://127.0.0.1:{port}',tokenEnv='VOID_WORKER_TOKEN'))
    assert r['status']=='pending' and r['error'] and r['runId'] is None
    assert store.verify(r['bundleSha256']) and RemoteClient(store.root).state.get('remote',r['id'])==r

def test_package_dependency_mismatch_blocks_even_available_builtin(recorded,tmp_path):
    _,g=recorded;p=build_package(g,{'schemaVersion':'1.0.0','positions':{}})
    p['dependencies'][0]['version']='999.0.0';p['identity']=digest({k:v for k,v in p.items() if k!='identity'})
    report=import_package(p,tmp_path/'projects','wrong_version');assert report['executionBlocked']
    reread=load_project(tmp_path/'projects/wrong_version.project.json').graph
    assert any(d.code=='E_PACKAGE_DEPENDENCY' for d in validate(reread).errors)

def test_inline_url_credentials_are_refused_in_exports(recorded):
    _,g=recorded;g.nodes[0].config['path']='https://name:password@example.com/data.csv'
    with pytest.raises(IntegrationError,match='credentials'):build_package(g,{'schemaVersion':'1.0.0','positions':{}})

def test_real_worker_two_job_admission_and_between_node_cancel(recorded,tmp_path,monkeypatch):
    _,g=recorded;token='bounded-worker-test-token-123456';monkeypatch.setenv('VOID_WORKER_TOKEN',token)
    with real_worker(tmp_path,dict(os.environ)) as (url,_):
        headers={'Authorization':'Bearer '+token}
        jobs=[]
        with httpx.Client(base_url=url,headers=headers,timeout=20,trust_env=False) as c:
            for seed in (901,902):
                bundle=snapshot(g);bundle['seed']=seed;id=digest(bundle)
                r=c.post('/v1/jobs',json={'id':id,'bundle':bundle});assert r.status_code==200,r.text
                jobs.append(id)
            bundle=snapshot(g);bundle['seed']=903
            assert c.post('/v1/jobs',json={'id':digest(bundle),'bundle':bundle}).status_code==429
            assert c.post('/v1/jobs/'+jobs[0]+'/cancel').status_code==200
            end=time.time()+30;results={}
            while time.time()<end and len(results)<2:
                for id in jobs:
                    r=c.get('/v1/jobs/'+id).json()
                    if r['run']['status'] in ('completed','failed','cancelled'):results[id]=r
                time.sleep(.1)
            assert len(results)==2 and results[jobs[0]]['run']['status']=='cancelled'
            assert any(e['type']=='cancel_acknowledged' for e in results[jobs[0]]['events'])
            assert results[jobs[1]]['run']['status']=='completed'
            # Exact job fingerprint may be reconnected, but cannot be reused for different bytes.
            original=snapshot(g);original['seed']=902
            assert c.post('/v1/jobs',json={'id':jobs[1],'bundle':original}).json()['replayed']
            original['seed']=904
            assert c.post('/v1/jobs',json={'id':jobs[1],'bundle':original}).status_code==409
