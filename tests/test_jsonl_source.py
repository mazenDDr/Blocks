"""Flat JSONL native tables, exact source provenance/cache and snapshot worker transport."""
import base64
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from artifact_store import ArtifactStore
from control.app import create_app
from extensions.packages import build_package, import_package, inspect_package
from graph_core.project_io import load_project
from graph_core.schema import Graph
from graph_core.validate import require_executable, validate
from operations import jsonl_source as source
from production.models import PredictRequest, RegisterVersion, ReleaseCreate
from production.runtime import ProductionRuntime
from scale.common import IntegrationError
from scale.remote import RemoteClient, RemoteRequest, materialize, snapshot
from tabular.core import ExecCtx, ExecutionError, clean
from tabular.engine import run_graph
from tabular_helpers import FIX, example, table, summary
from test_node_cache import run, outputs
from test_scale import real_worker
from workbench_backup.core import create, restore


def graph(path, regression=True):
    g=example('tabular_regression');node=g.node('housing');node.type='tabular.jsonl_source';node.config={'path':str(path)}
    if not regression:
        g.nodes=[g.node('housing'),g.node('profile')]
        g.edges=[e for e in g.edges if e.from_.node=='housing' and e.to.node=='profile']
    return g


def housing(path):
    frame=pd.read_csv(FIX/'synthetic_housing.csv')
    path.write_text(''.join(json.dumps(clean(record),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n' for record in frame.to_dict('records')))
    return frame


def test_actual_native_dataframe_row_ids_nulls_unicode_and_original_bytes(tmp_path):
    path=tmp_path/'record.jsonl'
    records=[{'id':1,'value':2.5,'text':'SYNTHETIC Unicode\u2028inside JSON','flag':True}, {'id':2,'value':None,'flag':False}]
    raw=('\n'+json.dumps(records[0],ensure_ascii=False)+'\n \t\r\n'+json.dumps(records[1])+'\n').encode()
    path.write_bytes(raw);cfg=source.JsonlSourceConfig(path=str(path))
    frame,provenance=source.read_source(cfg)
    pd.testing.assert_frame_equal(frame,pd.DataFrame.from_records(records).rename_axis('row_id'))
    assert frame.index.tolist()==[0,1] and provenance['rows']==2 and provenance['columns']==4
    assert provenance['sha256']==hashlib.sha256(raw).hexdigest() and provenance['bytes']==len(raw) and path.read_bytes()==raw
    op=source.JsonlSource();out=op.infer(cfg,{},'source')['table']
    assert out.info['rows']==2 and out.info['rowsExact'] and out.info['sourceSha256']==provenance['sha256']
    actual,recorded=op.execute(cfg,{},ExecCtx('source'))
    pd.testing.assert_frame_equal(actual['table'].df,frame)
    assert recorded['preview']['rows'][1]==[2,None,None,False]
    assert actual['table'].lineage['source']=={'node':'source',**provenance}


@pytest.mark.parametrize('line',[b'{broken',b'[1,2]',b'{}',b'{"x":{}}',b'{"x":[]}',b'{"x":1,"x":2}',
                               b'{"x":NaN}',b'{"x":Infinity}',b'{"x":1e309}',b'{"x":9007199254740992}',b'{"x":"\\ud800"}',b'{"x":"\xff"}',b'{"x":'+b'['*2000+b'0'+b']'*2000+b'}'])
def test_invalid_records_have_physical_line_diagnostics_without_silent_skips(tmp_path,line):
    path=tmp_path/'bad.jsonl';path.write_bytes(b'\n{"x":1}\n'+line+b'\n')
    with pytest.raises(ExecutionError) as caught:source.read_source(source.JsonlSourceConfig(path=str(path)))
    assert caught.value.code=='E_JSONL_RECORD' and 'line 3:' in caught.value.message
    report=validate(graph(path,False))
    assert not report.ok and any(d.code=='E_JSONL_RECORD' and d.nodeId=='housing' for d in report.diagnostics)


@pytest.mark.parametrize('bound,value,raw',[('MAX_BYTES',8,b'{"x":1234567}'),('MAX_RECORD_BYTES',8,b'{"x":1234567}'),
                                         ('MAX_ROWS',1,b'{"x":1}\n{"x":2}'),('MAX_COLUMNS',1,b'{"x":1,"y":2}'),
                                         ('MAX_CELLS',2,b'{"x":1}\n{"y":2}')])
def test_whole_source_bounds_refuse_in_static_and_runtime_reads(tmp_path,monkeypatch,bound,value,raw):
    monkeypatch.setattr(source,bound,value);path=tmp_path/'bounded.jsonl';path.write_bytes(raw)
    with pytest.raises(ExecutionError):source.read_source(source.JsonlSourceConfig(path=str(path)))
    assert not validate(graph(path,False)).ok


def test_empty_sources_and_absent_files_refuse(tmp_path):
    path=tmp_path/'empty.jsonl';path.write_bytes(b' \r\n\t\n')
    with pytest.raises(ExecutionError,match='no nonblank'):source.read_source(source.JsonlSourceConfig(path=str(path)))
    path.unlink()
    with pytest.raises(ExecutionError) as caught:source.read_source(source.JsonlSourceConfig(path=str(path)))
    assert caught.value.code=='E_SOURCE_NOT_FOUND'


def test_source_is_reread_cache_content_changes_and_fitted_pipeline_survives_source_deletion(tmp_path):
    path=tmp_path/'housing.jsonl';expected=housing(path);g=graph(path);store=ArtifactStore(tmp_path/'wb')
    first=run(store,g,'first');second=run(store,g,'cached')
    assert first['housing']['cache']['status']==second['housing']['cache']['status']=='bypass'
    assert 're-read' in second['housing']['cache']['reason']
    assert all(d['cache']['status']=='hit' for n,d in second.items() if n!='housing')
    assert outputs(store,'first')==outputs(store,'cached')
    actual=table(store,'first','housing');pd.testing.assert_frame_equal(actual,expected.rename_axis('row_id'),check_dtype=False)
    assert store.last_event('first','source_recorded')['data']['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    rt=ProductionRuntime(store)
    v=rt.register_version(RegisterVersion(runId='first',node='ols',name='SYNTHETIC JSONL housing',owner='test',intendedUse='Native provenance verification',limitations='No real-world housing benchmark'))
    assert v['manifest']['source']['type']=='tabular.jsonl_source'
    pipeline=rt.pipeline(v['id']);rows=clean(expected[[c['name'] for c in pipeline.manifest['inputSchema']]].iloc[:5].to_dict('records'))
    predicted=pipeline.predict(rows)[0]['predictions'];native=run_graph(g,require_executable(g))
    from graph_core.registry import get_op
    from tabular.core import Table
    t=Table(pd.DataFrame(rows))
    for apply,fit in [('imp_val','imp_fit'),('oh_val','oh_fit'),('sc_val','sc_fit')]:
        op=get_op('tabular.apply_transform');t=op.execute(op.Config(),{'table':t,'fit':native[fit].outs['fit']},ExecCtx(apply))[0]['table']
    model=native['ols'].outs['model']
    np.testing.assert_allclose(predicted,model.estimator.predict(t.df[model.features].to_numpy()),atol=1e-10)
    records=[json.loads(line) for line in path.read_text().splitlines()];records[0]['price_k']+=1
    path.write_text(''.join(json.dumps(record,allow_nan=False)+'\n' for record in records))
    changed=run(store,g,'changed')
    assert all(d['cache']['status']=='miss' for n,d in changed.items() if n!='housing')
    assert changed['profile']['cache']['changed']==['input table (from housing)']
    assert summary(store,'changed','housing')['sha256']!=summary(store,'first','housing')['sha256']
    path.unlink();release=rt.create_release(ReleaseCreate(versionId=v['id'],config={'captureInputs':True}));rt.activate(release['id'],None)
    backup=create(store.root,tmp_path/'backup',offline=True);shutil.rmtree(store.root)
    restore(tmp_path/'backup',tmp_path/'restored',trusted=True,manifest_sha256=backup['manifestSha256'])
    recovered=ProductionRuntime(ArtifactStore(tmp_path/'restored'))
    trace=recovered.predict('local','lab',PredictRequest(requestId='after-source-deletion',records=rows))
    assert trace['status']==200 and not path.exists()
    np.testing.assert_allclose(trace['result']['predictions'],predicted,atol=1e-10)
    assert trace['lineage']['source']['summary']['sha256']==v['manifest']['source']['summary']['sha256']


def test_registry_bundle_explicit_package_and_verified_snapshot_transport(tmp_path):
    path=tmp_path/'housing.jsonl';housing(path);g=graph(path)
    with TestClient(create_app(tmp_path/'wb')) as client:
        op=next(o for o in client.get('/api/registry').json()['ops'] if o['type']=='tabular.jsonl_source')
        assert op['displayName']=='JSONL table source' and op['summaryKind']=='source'
        assert client.put('/api/projects/jsonl',json={'graph':g.to_json(),'ui':{'schemaVersion':'1.0.0','positions':{},'synthetic':True}}).status_code==200
        files=client.get('/api/projects/jsonl/export/bundle').json()['requirements']['localFiles']
        assert files==[{'node':'housing','path':str(path),'status':'local file; not packaged - supply it or its identity'}]
    ui={'schemaVersion':'1.0.0','positions':{},'synthetic':True}
    assert build_package(g,ui,False)['resources']=={}
    package=build_package(g,ui,True);assert inspect_package(package)['resources']==1
    key=next(iter(package['resources']));assert key.endswith('.jsonl')
    bundle=snapshot(graph(path,False));raw=path.read_bytes();path.unlink()
    imported=import_package(package,tmp_path/'projects','portable')
    restored=load_project(tmp_path/'projects/portable.project.json').graph
    assert Path(restored.node('housing').config['path']).read_bytes()==raw and validate(restored).ok
    remote=materialize(bundle,tmp_path/'worker')
    assert remote.node('housing').config['path'].endswith('.jsonl')
    assert Path(remote.node('housing').config['path']).read_bytes()==raw and validate(remote).ok
    bundle['files']['housing']['base64']=base64.b64encode(b'{"tampered":true}').decode()
    with pytest.raises(IntegrationError):materialize(bundle,tmp_path/'tampered')


def test_real_separate_loopback_worker_uses_jsonl_snapshot_after_sender_file_deletion(tmp_path,monkeypatch):
    path=tmp_path/'housing.jsonl';housing(path);g=graph(path,False)
    token='SYNTHETIC-jsonl-worker-test-token';monkeypatch.setenv('VOID_WORKER_TOKEN',token)
    client=RemoteClient(tmp_path/'sender')
    with real_worker(tmp_path,dict(os.environ)) as (url,proc):
        request=RemoteRequest(graph=g,endpoint=url,tokenEnv='VOID_WORKER_TOKEN',projectId='jsonl')
        sent=client.submit(request);path.unlink()
        end=time.monotonic()+30
        while not sent['runId'] and time.monotonic()<end:
            time.sleep(.1);sent=RemoteClient(tmp_path/'sender').refresh(sent['id'])
        assert sent['status']=='completed',sent
        assert proc.pid!=os.getpid()
        store=ArtifactStore(tmp_path/'sender')
        assert summary(store,sent['runId'],'profile')['rows']==412
        raw=json.loads(store.read_artifact(sent['bundleSha256']))
        assert summary(store,sent['runId'],'housing')['sha256']==raw['files']['housing']['sha256']
        assert summary(store,sent['runId'],'housing')['path'].endswith('.jsonl')
        assert not path.exists() and len(ArtifactStore(tmp_path/'worker').list_runs())==1
        assert RemoteClient(tmp_path/'sender').refresh(sent['id'])['runId']==sent['runId']
