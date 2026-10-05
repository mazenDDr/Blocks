"""Real local Ollama histories across native checkpoint fork and fresh reset."""
import pytest

from agent_helpers import Lab, REPO
from control.production_api import ConversationFork, ConversationReset
from graph_core.project_io import load_project
from production.conversations import action
from production.models import RegisterVersion, ReleaseCreate
from production.runtime import ProductionRuntime
from test_production_conversation import META, req, head

pytestmark = pytest.mark.live


def test_real_ollama_fork_preserves_sent_history_reset_starts_fresh_and_source_isolated(tmp_path):
    lab = Lab(tmp_path)
    graph = load_project(REPO/'examples/serving_conversation.project.json').graph
    assert lab.run(graph,inp={'question':'SYNTHETIC source: say hello briefly.'})[0] == 'completed'
    original_research = lab.final('r1')
    rt = ProductionRuntime(lab.store)
    version = rt.register_version(RegisterVersion(runId='r1',node='__agent_conversation__',**META))
    rel = rt.create_release(ReleaseCreate(versionId=version['id'],config={'maxBatch':1,'sessionMode':'conversation','captureInputs':True,'timeoutSeconds':30}))
    rt.activate(rel['id'],None)
    first = rt.predict('local','lab',req('first','SYNTHETIC teaching prompt: remember the colour blue. Reply briefly.'))
    second = rt.predict('local','lab',req('second','SYNTHETIC teaching prompt: which colour did I mention?'))
    assert first['status'] == second['status'] == 200
    source = head(rt,rel)
    source_history = second['result']['agent']['finalState']['history']
    args={'user':'alice','session':'chat','expectedRevision':source['revision'],'expectedCheckpointSha256':source['checkpointSha256'],'reason':'SYNTHETIC alternative continuation'}
    fork = action(rt,rel['id'],'fork',ConversationFork(**args,actionId='fork',destinationSession='branch'))
    assert len(rt.ps.traces()) == 2 and head(rt,rel) == source
    branch = rt.predict('local','lab',req('branch','SYNTHETIC branch teaching prompt: give a brief greeting.',session='branch'))
    assert branch['status'] == 200 and branch['result']['agent']['threadId'] == fork['threadId']
    context = branch['result']['agent']['contexts'][0]['value']
    assert context['providerRequest'][1:-1] == [{'role':m['role'],'content':m['content']} for m in source_history]
    assert context['usage']['source'] == 'provider' and not context['fixture']
    assert head(rt,rel) == source and lab.final('r1') == original_research
    reset = action(rt,rel['id'],'reset',ConversationReset(**args,actionId='reset'))
    fresh = rt.predict('local','lab',req('fresh','SYNTHETIC fresh-session teaching prompt: give a brief greeting.'))
    assert fresh['status'] == 200 and fresh['conversationState']['revision'] == 4
    assert fresh['conversationParent'] == {'revision':3,'checkpointSha256':None}
    assert len(fresh['result']['agent']['contexts'][0]['value']['providerRequest']) == 2
    assert fresh['result']['agent']['threadId'] != second['result']['agent']['threadId']
    assert len(fresh['result']['agent']['finalState']['history']) == 2
    assert head(rt,rel,session='branch')['revision'] == 2
    assert action(rt,rel['id'],'reset',ConversationReset(**args,actionId='reset'))['actionSha256'] == reset['actionSha256']


def test_real_ollama_historical_restore_resends_original_history_without_restore_model_call(tmp_path):
    from control.production_api import ConversationRestore
    from production.history import inspect_history
    lab = Lab(tmp_path)
    graph = load_project(REPO/'examples/serving_conversation.project.json').graph
    assert lab.run(graph,inp={'question':'SYNTHETIC source: a brief greeting.'})[0] == 'completed'
    rt = ProductionRuntime(lab.store)
    version = rt.register_version(RegisterVersion(runId='r1',node='__agent_conversation__',**META))
    rel = rt.create_release(ReleaseCreate(versionId=version['id'],config={'maxBatch':1,'sessionMode':'conversation','captureInputs':True,'timeoutSeconds':30}))
    rt.activate(rel['id'],None)
    first = rt.predict('local','lab',req('first','SYNTHETIC teaching prompt: remember blue. Reply briefly.'))
    second = rt.predict('local','lab',req('second','SYNTHETIC teaching prompt: remember green too. Reply briefly.'))
    assert first['status'] == second['status'] == 200
    historical = inspect_history(rt,rel['id'],'alice','chat','first')
    current = head(rt,rel)
    restored = action(rt,rel['id'],'restore',ConversationRestore(actionId='restore',user='alice',session='chat',
        expectedRevision=current['revision'],expectedCheckpointSha256=current['checkpointSha256'],reason='SYNTHETIC original context continuation',
        **{k:historical[k] for k in ('sourceRequestId','sourceTraceSha256','sourceCheckpointSha256')}))
    assert len(rt.ps.traces()) == 2 and restored['head']['revision'] == 3
    assert restored['threadId'] != first['conversationState']['threadId']
    assert rt.pipeline(version['id']).checkpoint_state(restored['head']['checkpointSha256']) == historical['state']
    continued = rt.predict('local','lab',req('next','SYNTHETIC continuation: which colour was first?'))
    assert continued['status'] == 200 and continued['conversationState']['revision'] == 4
    context = continued['result']['agent']['contexts'][0]['value']
    assert context['providerRequest'][1:-1] == [{'role':m['role'],'content':m['content']} for m in first['result']['agent']['finalState']['history']]
    assert context['usage']['source'] == 'provider' and not context['fixture']
    assert continued['result']['agent']['threadId'] == restored['threadId']
    assert rt.ps.trace('alice','first')['traceSha256'] == first['traceSha256']
