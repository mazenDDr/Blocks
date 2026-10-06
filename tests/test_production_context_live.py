"""Actual local Ollama context serving: pinned retrieval, JSON, native short-term policy."""
import shutil
import pytest
from fastapi.testclient import TestClient
from agent import samples as sm
from agent.models import ollama_status
from agent_helpers import Lab
from control.app import create_app
from graph_core.schema import Graph
from production import context_agent_adapter as adapter
from production.models import RegisterVersion, ReleaseCreate, PredictRequest
from production.runtime import ProductionRuntime
from test_production_agent import META
from test_production_context import graph as native_graph, head
from test_production_json_agent import fixture

pytestmark=pytest.mark.live


def graph(docs,conversation,json_output):
    doc=native_graph(docs,conversation=conversation,memory=True).to_json()
    doc['nodes']=doc['nodes'][:-1]+[sm.N('prompt','agent.prompt',output_field='messages',items=[
        {'kind':'template','role':'system','template':'Extract the colour and count from the retrieved SYNTHETIC record. Reply with only the requested output.'},
        {'kind':'documents','role':'system','field':'docs','header':'Retrieved record:', 'template':'{text}'},
        {'kind':'memory','role':'system','field':'selected','header':'Recent questions:', 'template':'{text}'},
        {'kind':'template','role':'user','template':'{question}'}])]
    if json_output:
        cfg=fixture.graph().to_json()['nodes'][1]['config']
        doc['nodes'].append(sm.N('model','agent.structured_output',**cfg))
        doc['agent']['state']=[f for f in doc['agent']['state'] if f['name']!='answer']+[sm.S('result','object',default={})]
    else:
        doc['nodes'][ -1]['config']['items'][0]['template']+=' Use the format "colour=count".'
        doc['nodes'].append(sm.N('model','agent.chat_model',messages_field='messages',output_field='answer',model={'provider':'ollama','model':'qwen3.5:2b','think':False,'max_tokens':32,'timeout_s':20,'temperature':0,'seed':0}))
    doc['agent']['state'].append(sm.S('messages','messages'))
    doc['agent']['limits'].update(maxSeconds=30,maxTokens=4096)
    doc['edges']=sm.chain('START',*[n['id'] for n in doc['nodes']],'END')
    return Graph.model_validate(doc)


@pytest.mark.parametrize('conversation,json_output',[(False,False),(True,False),(False,True),(True,True)])
def test_actual_context_modes_receive_pinned_chunks_and_bounded_policy_history(tmp_path,conversation,json_output):
    status=ollama_status()
    if not status['reachable'] or 'qwen3.5:2b' not in status['models']:
        pytest.skip('Actual installed local Ollama qwen3.5:2b required; no downloads')
    docs=tmp_path/'docs';docs.mkdir();(docs/'record.txt').write_text('SYNTHETIC retrieved record: colour blue; count 9.')
    g=graph(docs,conversation,json_output);lab=Lab(tmp_path/'lab')
    state,rid=lab.run(g,inp={'question':'Read the SYNTHETIC colour and count.'});assert state=='completed'
    source=lab.store.events(rid),lab.store.artifacts(rid),lab.final(rid)
    rt=ProductionRuntime(lab.store)
    name=('conversation_context' if conversation else 'agent_context')+('_json' if json_output else '')
    v=rt.register_version(RegisterVersion(runId=rid,node=adapter.NODES[name],**META))
    rel=rt.create_release(ReleaseCreate(versionId=v['id'],config={'maxBatch':1,'sessionMode':'conversation' if conversation else 'stateless','captureInputs':True,'timeoutSeconds':30}))
    rt.activate(rel['id'],None)
    shutil.rmtree(docs);shutil.rmtree(lab.wb/'agent/indexes/notes')
    with TestClient(create_app(lab.wb)) as client:
        for i,question in enumerate(['SYNTHETIC first question: read the record.','SYNTHETIC second question: read it again.'],1):
            response=client.post('/api/serve/local/lab/predict',json={'requestId':f'turn-{i}','records':[{'question':question}],'user':'alice','session':'chat' if conversation else None})
            assert response.status_code==200,response.text
            trace=response.json();a=trace['result']['agent'];ctx=a['contexts'][0]['value']
            assert a['modelCalls']==1 and ctx['provider']=='ollama' and not ctx['fixture'] and ctx['usage']['source']=='provider'
            sent=str(ctx['providerRequest']);assert 'SYNTHETIC retrieved record: colour blue; count 9.' in sent
            assert question in sent
            if i==2:
                assert ('SYNTHETIC first question' in sent)==conversation
            assert len(a['memorySelections'])==1 and a['memorySelections'][0]['value']['final']
            assert any(s['source']['kind']=='retrieved_chunk' for m in ctx['messages'] for s in m['segments'])
            assert ctx['applications'] and ctx['retrievals']
            if json_output:
                assert trace['result']['predictions']==[{'colour':'blue','count':9}]
                assert client.post(f'/api/production/requests/turn-{i}/labels',json={'user':'alice','labels':[{'count':9,'colour':'blue'}]}).status_code==200
            if conversation:
                assert trace['conversationState']['revision']==i
                before=head(rt,rel)
                replay=client.post(f'/api/production/requests/turn-{i}/replay',json={'user':'alice'})
                assert replay.status_code==200 and head(rt,rel)==before
                assert replay.json()['result']['agent']['contexts'][0]['value']['providerRequest']==ctx['providerRequest']
        monitor=client.get(f"/api/production/releases/{rel['id']}/monitor").json()
        assert monitor['usage']['successfulTurnModelCalls']==2
        if json_output: assert monitor['labelBasedQuality']['values']['exactJsonAgreement']==1
    assert (lab.store.events(rid),lab.store.artifacts(rid),lab.final(rid))==source
    assert not (lab.wb/'agent/indexes/notes').exists()
