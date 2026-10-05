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
