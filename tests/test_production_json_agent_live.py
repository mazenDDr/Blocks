"""Real installed Ollama structured output through native source and production API."""
import copy
import json

import pytest
from fastapi.testclient import TestClient

from agent.models import ollama_status
from agent_helpers import Lab
from control.app import create_app
from production import agent_adapter, json_agent_adapter as adapter
from production.models import PredictRequest, RegisterVersion, ReleaseCreate
from production.pipeline import ProductionError, read_verified
from production.runtime import ProductionRuntime
from test_production_agent import META, pure_graph
from test_production_json_agent import fixture

pytestmark=pytest.mark.live


@pytest.fixture(scope='module')
def native_json(tmp_path_factory):
    status=ollama_status()
    if not status['reachable'] or 'qwen3.5:2b' not in status['models']:
        pytest.skip('Real installed local Ollama qwen3.5:2b required; no download/provider substitution')
    lab=Lab(tmp_path_factory.mktemp('native-json-serving'))
    # Register an original text version BEFORE registering the new adapter.
    state,text_run=lab.run(pure_graph(),inp={'question':'SYNTHETIC legacy source'})
    assert state=='completed'
    runtime=ProductionRuntime(lab.store)
    text_version=runtime.register_version(RegisterVersion(runId=text_run,node=agent_adapter.NODE,**META))
    state,run_id=lab.run(fixture.graph(),inp={'question':'SYNTHETIC teaching input: colour red, count 3.'})
    assert state=='completed' and lab.final(run_id)['result']=={'colour':'red','count':3}
    assert all(not c['fixture'] and c['provider']=='ollama' for c in lab.contexts(run_id))
    version=runtime.register_version(RegisterVersion(runId=run_id,node=adapter.NODE,**META))
    release=runtime.create_release(ReleaseCreate(versionId=version['id'],config={'namespace':'json-agent','maxBatch':1,'timeoutSeconds':30,'captureInputs':True}))
    runtime.activate(release['id'],None)
    return lab,runtime,version,release,text_version


def test_real_json_source_release_request_context_replay_labels_monitor_and_legacy_version(native_json):
    lab,runtime,version,release,text_version=native_json
    before=copy.deepcopy(lab.final(version['runId']))
    contexts_before=lab.contexts(version['runId'])
    with TestClient(create_app(lab.wb)) as client:
        candidates=client.get('/api/production').json()['candidates']
        assert any(c['node']==adapter.NODE and c['adapter']=='agent_json' for c in candidates)
        reference=client.get(f"/api/production/versions/{version['id']}/reference-input").json()
        assert reference['observedLabels'] is None and reference['family']=='agent_json'
        payload={'requestId':'real-json','records':[{'question':'SYNTHETIC teaching input: colour blue, count 9.'}]}
        response=client.post('/api/serve/local/json-agent/predict',json=payload)
        assert response.status_code==200,response.text
        trace=response.json();evidence=trace['result']['agent'];output=trace['result']['predictions'][0]
        assert output=={'colour':'blue','count':9}
        assert trace['result']['family']=='agent_json' and evidence['modelCalls']==1
        context=evidence['contexts'][0]['value']
        assert not context['fixture'] and context['provider']=='ollama' and context['purpose']=='structured_output'
        assert context['usage']['source']=='provider' and context['usage']['outputTokens']>0
        assert payload['records'][0]['question'] in json.dumps(context['providerRequest'])
        assert evidence['threadId']==evidence['executionId']
        assert any(e['type']=='structured_attempt' and e['data']['valid'] for e in evidence['events'])
        assert client.post('/api/serve/local/json-agent/predict',json=payload).json()['idempotentReplay']
        replay=client.post('/api/production/requests/real-json/replay',json={}).json()
        assert replay['result']['agent']['executionId']!=evidence['executionId']
        assert replay['sourceTraceSha256']==trace['traceSha256'] and replay['versionId']==version['id']
        assert replay['result']['agent']['contexts'][0]['value']['purpose']=='structured_output'
        assert replay['result']['agent']['contexts'][0]['value']['fixture'] is False
        assert replay['result']['agent']['finalState']['result']=={'colour':'blue','count':9}
        assert any(e['type']=='structured_attempt' and e['data']['valid'] for e in replay['result']['agent']['events'])
        assert replay['replayNote'].startswith('New native Ollama call;')
        # Reference is independently supplied teaching truth, not copied from prediction.
        assert client.post('/api/production/requests/real-json/labels',json={'labels':[{'count':9,'colour':'blue'}]}).status_code==200
        assert client.post('/api/production/requests/real-json/labels',json={'labels':['text is not the pinned object schema']}).status_code==422
        events_before=len(runtime.ps.traces(release['id']))
        monitor=client.get(f"/api/production/releases/{release['id']}/monitor").json()
        assert monitor['family']=='agent_json' and monitor['labelBasedQuality']['values']['exactJsonAgreement']==1.0
        assert monitor['predictionDrift']['unit'].startswith('canonical JSON UTF8bytes')
        assert monitor['usage']['successfulTurnModelCalls']==1
        assert len(runtime.ps.traces(release['id']))==events_before
        assert lab.final(version['runId'])==before and lab.contexts(version['runId'])==contexts_before
    legacy_release=runtime.create_release(ReleaseCreate(versionId=text_version['id'],config={'namespace':'legacy-json-check','maxBatch':1}))
    runtime.activate(legacy_release['id'],None)
    legacy=runtime.predict('local','legacy-json-check',PredictRequest(requestId='legacy-still-works',records=[{'question':'SYNTHETIC old version'}]))
    assert legacy['status']==200 and legacy['result']['predictions']==['SYNTHETIC old version: n=3']
    assert runtime.pipeline(text_version['id']).manifest['implementation']==text_version['manifest']['implementation']


def test_json_capture_off_digest_schema_modes_bounds_and_cancel_refusal(native_json):
    lab,runtime,version,_,_=native_json
    for change in [{'maxBatch':2},{'sessionMode':'counter'},{'sessionMode':'conversation'}]:
        with pytest.raises(ProductionError) as error:
            runtime.create_release(ReleaseCreate(versionId=version['id'],config={'namespace':'refused-json','maxBatch':1,**change}))
        assert error.value.code=='E_RELEASE_CONFIG'
    pipeline=runtime.pipeline(version['id'])
    with pytest.raises(ProductionError):pipeline.validate_records([{'question':'x','other':'undeclared'}])
    changed=copy.deepcopy(version['manifest']);changed['provider']['digest']='incorrect'
    with pytest.raises(ProductionError) as error:adapter.verify(lab.store,changed)
    assert error.value.code=='E_AGENT_MODEL_CHANGED'
    for key,value in [('limits',{}),('inputFields',['messages']),('outputSchema',{})]:
        changed=copy.deepcopy(version['manifest']);changed[key]=value
        with pytest.raises(ProductionError):adapter.verify(lab.store,changed)
    with pytest.raises(ProductionError) as error:pipeline.predict([{'question':'SYNTHETIC cancelled'}],cancelled=lambda:True)
    assert error.value.code=='E_REQUEST_CANCELLED'
    release=runtime.create_release(ReleaseCreate(versionId=version['id'],config={'namespace':'private-json','maxBatch':1,'timeoutSeconds':30}))
    runtime.activate(release['id'],None)
    trace=runtime.predict('local','private-json',PredictRequest(requestId='hidden-json',records=[{'question':'SYNTHETIC teaching input: colour red, count 4.'}]))
    assert trace['status']==200 and trace['records'] is None
    evidence=trace['result']['agent'];assert evidence['finalState'] is None and evidence['events'] is None
    assert evidence['contexts'][0]['value'] is None
    with pytest.raises(ProductionError):read_verified(lab.store,evidence['contexts'][0]['sha256'])
    restarted=ProductionRuntime(lab.store)
    duplicate=restarted.predict('local','private-json',PredictRequest(requestId='hidden-json',records=[{'question':'SYNTHETIC teaching input: colour red, count 4.'}]))
    assert duplicate['idempotentReplay'] and duplicate['result']==trace['result']
