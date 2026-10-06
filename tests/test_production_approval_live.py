"""Actual Ollama call commits before interrupt; native resume never repeats it."""
import pytest
from agent import samples as sm
from agent.models import ollama_status
from graph_core.schema import Graph
from production.runtime import ProductionRuntime
from production.monitor import monitoring
from test_production_approval import graph as pure_graph, setup, turn, resume_request

pytestmark=pytest.mark.live

def graph():
    doc=pure_graph().to_json()
    doc['nodes']=doc['nodes'][:1]+[
        sm.N('prompt','agent.prompt',output_field='messages',items=[{'kind':'template','role':'user','template':'Repeat this SYNTHETIC identifier exactly: {question}'}]),
        sm.N('reply','agent.chat_model',messages_field='messages',output_field='answer',model={'provider':'ollama','model':'qwen3.5:2b','think':False,'max_tokens':32,'timeout_s':20,'temperature':0,'seed':0})]+doc['nodes'][2:]
    doc['edges']=sm.chain('START','tick','prompt','reply','review','finish','END')
    doc['agent']['state'].append(sm.S('messages','messages'))
    doc['agent']['limits']['maxSeconds']=30
    return Graph.model_validate(doc)

def test_actual_provider_paused_context_survives_resume_without_second_call(tmp_path):
    status=ollama_status()
    if not status['reachable'] or 'qwen3.5:2b' not in status['models']:
        pytest.skip('Actual installed local Ollama qwen3.5:2b required; no download')
    lab,rt,v,rel=setup(tmp_path,g=graph())
    source=lab.store.events('r1'),lab.store.artifacts('r1')
    paused=turn(rt,question='native-approval-marker')
    assert paused['status']==202,paused
    evidence=paused['result']['agent'];assert evidence['modelCalls']==1
    context=evidence['contexts'][0]['value']
    assert context['provider']=='ollama' and not context['fixture'] and context['usage']['source']=='provider'
    assert 'native-approval-marker' in str(context['providerRequest'])
    req=resume_request(rt,rel,action='edit',value='reviewed exact text')
    rt=ProductionRuntime(lab.store)
    out=rt.predict('local','lab',req)
    assert out['status']==200 and out['result']['predictions']==['edit: reviewed exact text / 1']
    assert out['result']['agent']['modelCalls']==0 and out['result']['agent']['logicalTurnModelCalls']==1
    monitor=monitoring(rt,rel['id'])
    assert monitor['usage']['successfulTurnModelCalls']==1 and monitor['usage']['providerReportedCalls']==1
    assert (lab.store.events('r1'),lab.store.artifacts('r1'))==source
