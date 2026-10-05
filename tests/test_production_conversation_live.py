"""Actual local Ollama conversation history; no model substitution/downloads."""
import json

import pytest

from agent_helpers import Lab, REPO
from graph_core.project_io import load_project
from production.models import RegisterVersion, ReleaseCreate
from production.runtime import ProductionRuntime
from test_production_conversation import META, req, head

pytestmark = pytest.mark.live


def test_real_ollama_history_native_context_and_replay_parent_isolation(tmp_path):
    lab = Lab(tmp_path)
    graph = load_project(REPO / 'examples/serving_conversation.project.json').graph
    assert lab.run(graph, inp={'question': 'SYNTHETIC source: reply with a greeting.'})[0] == 'completed'
    source = lab.final('r1')
    rt = ProductionRuntime(lab.store)
    v = rt.register_version(RegisterVersion(runId='r1', node='__agent_conversation__', **META))
    rel = rt.create_release(ReleaseCreate(versionId=v['id'], config={'maxBatch':1, 'sessionMode':'conversation', 'captureInputs':True, 'timeoutSeconds':30}))
    rt.activate(rel['id'], None)
    first = rt.predict('local', 'lab', req('one', 'SYNTHETIC teaching prompt: remember my favourite colour is blue. Reply briefly.'))
    assert first['status'] == 200, first
    second = rt.predict('local', 'lab', req('two', 'SYNTHETIC teaching prompt: which colour did I mention?'))
    assert second['status'] == 200, second
    a, b = first['result']['agent'], second['result']['agent']
    ctx = b['contexts'][0]['value']
    assert ctx['usage']['source'] == 'provider' and not ctx['fixture']
    assert ctx['providerRequest'][1:] == [
        {'role':'user','content':first['records'][0]['question']},
        {'role':'assistant','content':first['result']['predictions'][0]},
        {'role':'user','content':second['records'][0]['question']}]
    assert a['threadId'] == b['threadId'] and a['executionId'] != b['executionId']
    assert any(s['source']['kind'] == 'conversation_message' for m in ctx['messages'] for s in m['segments'])
    assert json.loads(lab.store.read_artifact(b['contexts'][0]['sha256'])) == ctx
    before = head(rt, rel)
    replay, timing = rt.pipeline(v['id']).predict(second['records'], capture=True, checkpoint=second['conversationParent']['checkpointSha256'])
    assert replay['agent']['contexts'][0]['value']['providerRequest'] == ctx['providerRequest']
    assert head(rt, rel) == before and lab.final('r1') == source
    other = rt.predict('local', 'lab', req('other', 'SYNTHETIC separate session: greet me.', session='other'))
    assert len(other['result']['agent']['contexts'][0]['value']['providerRequest']) == 2
    assert other['conversationState']['revision'] == 1
    restarted = ProductionRuntime(lab.store)
    third = restarted.predict('local', 'lab', req('three', 'SYNTHETIC third turn: give a short greeting.'))
    assert third['status'] == 200 and third['conversationState']['revision'] == 3
    assert len(third['result']['agent']['finalState']['history']) == 6
