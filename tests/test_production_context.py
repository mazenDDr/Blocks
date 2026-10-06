"""Native pinned retrieval + bounded short-term policies on END conversation checkpoints."""
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
import pytest
from fastapi.testclient import TestClient
from agent import samples as sm
from agent_helpers import Lab
from agent.memory import MemoryStore
from control.app import create_app
from graph_core.schema import Graph
from production import context_agent_adapter as adapter
from production.models import PredictRequest, RegisterVersion, ReleaseCreate
from production.runtime import ProductionRuntime
from production.pipeline import ProductionError
from tabular.core import dumps
from test_production_agent import META
from test_production_retrieval import graph as rag_graph, DOCS


def graph(docs=None, conversation=True, memory=True):
    doc=rag_graph(docs).to_json() if docs else sm.make_graph([],[],{'state':[]}).to_json()
    state=[sm.S('question'),sm.S('answer'),sm.S('turns','integer','add',default=0,scope='thread' if conversation else 'turn')]
    nodes=[sm.N('tick','agent.set_state',assignments=[{'field':'turns','kind':'increment','by':1}])]
    if docs:
        nodes.append(sm.N('find','agent.retrieve',index='notes',query='{question}',k=1,output_field='docs'))
        state.append(sm.S('docs','documents'))
    if memory:
        state += [sm.S('history','messages','keep_last_n',default=[],scope='thread' if conversation else 'turn',n=3),sm.S('selected','records')]
        nodes += [sm.N('remember','agent.memory_write',target='short_term',mode='direct',field='history',text='{question}',role='user'),
                  sm.N('select','agent.memory_select',policy='recent',short_term_field='history',output_field='selected')]
    nodes.append(sm.N('answer','agent.set_state',assignments=[{'field':'answer','kind':'template','template':'SYNTHETIC {question}: {turns}'}]))
    doc['nodes']=nodes;doc['edges']=sm.chain('START',*[n['id'] for n in nodes],'END')
    doc['agent']['state']=state;doc['agent']['limits']={'maxSteps':12,'maxSeconds':10,'maxModelCalls':1,'maxTokens':1024}
    doc['agent']['policies']=[{'id':'recent','query':'{question}','embeddings':{'provider':'local_hash','dimension':64},'stages':[
        {'id':'take','op':'retrieve','config':{'sources':['short_term'],'method':'recency','k':2}},
        {'id':'dedupe','op':'dedupe','config':{'method':'exact_text'}},
        {'id':'budget','op':'budget','config':{'max_tokens':128,'order':'chronological'}}]}] if memory else []
    doc['agent']['indexes']=doc['agent'].get('indexes',[]) if docs else []
    return Graph.model_validate(doc)


def setup(tmp_path,conversation=True,memory=True,retrieval=True,capture=True):
    docs=tmp_path/'docs';docs.mkdir()
    for name,text in DOCS.items(): (docs/name).write_text(text)
    g=graph(docs if retrieval else None,conversation,memory)
    lab=Lab(tmp_path/'lab');status,rid=lab.run(g,inp={'question':'source yellow bananas'})
    assert status=='completed'
    MemoryStore(lab.wb).put_record({'text':'SYNTHETIC LONG_TERM_PRIVATE_MARKER','namespace':'other-user','scope':'global'})
    rt=ProductionRuntime(lab.store)
    node=adapter.NODES['conversation_context' if conversation else 'agent_context']
    version=rt.register_version(RegisterVersion(runId=rid,node=node,**META))
    rel=rt.create_release(ReleaseCreate(versionId=version['id'],config={'maxBatch':1,'sessionMode':'conversation' if conversation else 'stateless','captureInputs':capture}))
    rt.activate(rel['id'],None)
    return lab,rt,version,rel,docs,g


def ask(rt,id_,question='yellow curved bananas',session='chat',user='alice'):
    return rt.predict('local','lab',PredictRequest(requestId=id_,records=[{'question':question}],user=user,session=session))


def head(rt,rel,session='chat',user='alice'):
    return rt.ps.conversation(dumps([rel['id'],user,session]))


@pytest.mark.parametrize('conversation,memory,retrieval',[(True,True,True),(True,False,True),(True,True,False),(False,True,True),(False,True,False),(False,False,True)])
def test_native_context_modes_retrieval_policy_restart_isolation(tmp_path,conversation,memory,retrieval):
    lab,rt,v,rel,docs,g=setup(tmp_path,conversation,memory,retrieval)
    source=lab.store.events('r1'),lab.store.artifacts('r1'),lab.final('r1')
    outputs=[]
    for i,text in enumerate(['one yellow banana','two apples red','three purple grapes','four yellow bananas'],1):
        out=ask(rt,f'turn-{i}',text)
        assert out['status']==200,out
        assert out['result']['predictions']==[f'SYNTHETIC {text}: {i if conversation else 1}']
        assert out['result']['agent']['modelCalls']==0
        if memory:
            app=out['result']['agent']['memorySelections'][0]['value']
            assert all(r['store']=='short_term' for r in app['records'].values())
            assert 'LONG_TERM_PRIVATE_MARKER' not in json.dumps(out)
            assert len(out['result']['agent']['finalState']['history'])==min(i,3) if conversation else len(out['result']['agent']['finalState']['history'])==1
            assert out['result']['agent']['finalState']['selected'][-1]['text']==text
            assert app['stages'][0]['op']=='retrieve' and app['stages'][-1]['op']=='budget'
        if retrieval:
            assert len(out['result']['agent']['retrievals'][0]['included'])==1
            assert any(e['type']=='index_ready' and e['data']['action']=='pinned' for e in out['result']['agent']['events'])
        if conversation:
            assert out['conversationState']['revision']==i
            assert rt.predict('local','lab',PredictRequest(requestId=f'turn-{i}',records=[{'question':text}],session='chat',user='alice'))['idempotentReplay']
        else: assert out['conversationState'] is None
        outputs.append(out);rt=ProductionRuntime(lab.store)
    if conversation:
        other=ask(rt,'new-session',session='other')
        assert other['result']['predictions']==['SYNTHETIC yellow curved bananas: 1']
        if memory: assert len(other['result']['agent']['finalState']['history'])==1
        # Independent research execution owns the reference; message IDs/timestamps are native runtime identities.
        for text in ['one yellow banana','two apples red','three purple grapes','four yellow bananas']:
            status,reference=lab.run(g,thread='reference',inp={'question':text});assert status=='completed'
        actual=rt.pipeline(v['id']).checkpoint_state(head(rt,rel)['checkpointSha256']);expected=lab.final(reference)
        assert actual['turns']==expected['turns']==4 and actual['answer']==expected['answer']
        if memory:
            assert [m['content'] for m in actual['history']]==[m['content'] for m in expected['history']]
            assert [m['text'] for m in actual['selected']]==[m['text'] for m in expected['selected']]
    assert (lab.store.events('r1'),lab.store.artifacts('r1'),lab.final('r1'))==source


def test_pinned_retrieval_survives_deleted_documents_and_live_index(tmp_path):
    lab,rt,v,rel,docs,g=setup(tmp_path,memory=False)
    before=ask(rt,'before')['result']['agent']['retrievals']
    shutil.rmtree(docs);shutil.rmtree(lab.wb/'agent/indexes/notes')
    assert ask(rt,'after')['result']['agent']['retrievals']==before
    assert not (lab.wb/'agent/indexes/notes').exists()


def test_capture_off_hashes_but_persistent_memory_and_scope_separation(tmp_path):
    lab,rt,v,rel,docs,g=setup(tmp_path,capture=False,retrieval=False)
    out=ask(rt,'hidden','SYNTHETIC-private-short-term')
    assert out['records'] is None and out['result']['agent']['finalState'] is None
    assert out['result']['agent']['memorySelections'][0]['value'] is None
    assert 'SYNTHETIC-private-short-term' not in json.dumps(out['result']['agent'])
    state=rt.pipeline(v['id']).checkpoint_state(head(rt,rel)['checkpointSha256'])
    assert state['history'][0]['content']=='SYNTHETIC-private-short-term'
    other=ask(rt,'other-user',user='bob')
    assert other['conversationState']['revision']==1 and other['result']['agent']['threadId']!=out['result']['agent']['threadId']


def test_concurrent_native_memory_turns_and_http_parent_replay_monitor(tmp_path):
    lab,rt,v,rel,docs,g=setup(tmp_path)
    with ThreadPoolExecutor(max_workers=3) as pool:
        outs=list(pool.map(lambda i:ask(rt,f'parallel-{i}',f'SYNTHETIC turn {i}'),range(3)))
    assert sorted(t['conversationState']['revision'] for t in outs)==[1,2,3]
    with TestClient(create_app(lab.wb)) as c:
        body={'requestId':'http','records':[{'question':'yellow banana'}],'user':'alice','session':'chat'}
        out=c.post('/api/serve/local/lab/predict',json=body)
        assert out.status_code==200,out.text
        assert c.get('/api/production').json()['candidates'][0]['adapter']=='conversation_context'
        before=head(rt,rel)
        replay=c.post('/api/production/requests/http/replay',json={'user':'alice'})
        assert replay.status_code==200 and replay.json()['result']['predictions']==out.json()['result']['predictions'] and head(rt,rel)==before
        assert c.post('/api/production/requests/http/labels',json={'user':'alice','labels':out.json()['result']['predictions']}).status_code==200
        monitor=c.get(f"/api/production/releases/{rel['id']}/monitor").json()
        assert monitor['usage']['successfulTurnModelCalls']==0 and monitor['labelBasedQuality']['values']['exactStringAgreement']==1


@pytest.mark.parametrize('change',['long-term-select','long-term-write','model-summary','embedding-provider','unbounded-history','missing-k','too-many-stages','unused-index','tool','interrupt'])
def test_context_contract_refusals(tmp_path,change):
    doc=graph().to_json()
    if change=='long-term-select':doc['agent']['policies'][0]['stages'][0]['config']['sources']=['long_term']
    if change=='long-term-write':doc['nodes'][1]['config']['target']='long_term'
    if change=='model-summary':doc['agent']['policies'][0]['stages'].append({'id':'summary','op':'summarize','config':{'method':'model'}})
    if change=='embedding-provider':doc['agent']['policies'][0]['embeddings']={'provider':'ollama','model':'nomic-embed-text'}
    if change=='unbounded-history':doc['agent']['state'][3]['reducer']={'kind':'append'}
    if change=='missing-k':doc['agent']['policies'][0]['stages'][0]['config'].pop('k')
    if change=='too-many-stages':doc['agent']['policies'][0]['stages'] += [{'id':f'extra-{i}','op':'dedupe','config':{}} for i in range(5)]
    if change=='unused-index':doc['agent']['indexes']=[{'id':'unused'}]
    if change in ('tool','interrupt'):
        doc['nodes'].append(sm.N('unsupported','agent.tool_call',tool='calculator',args={'expression':'1+2'},output_field='answer') if change=='tool' else sm.N('unsupported','agent.human_interrupt',decision_field='answer'))
        doc['edges']=sm.chain('START',*[n['id'] for n in doc['nodes']],'END')
    with pytest.raises(ProductionError):adapter.contract(Graph.model_validate(doc),{'question':'SYNTHETIC'})
