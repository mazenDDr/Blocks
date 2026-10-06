"""SYNTHETIC release long-term memory (ADR0075): each served turn recalls the requesting user's earlier notes and stores the new one."""
import json
from pathlib import Path
from agent import samples as sm


def graph():
    nodes = [sm.N('recall', 'agent.memory_select', policy='mine', output_field='recalled'),
             sm.N('tally', 'agent.set_state', assignments=[{'field': 'count', 'kind': 'length', 'source': 'recalled'}]),
             sm.N('answer', 'agent.set_state', assignments=[{'field': 'answer', 'kind': 'template',
                                                              'template': 'SYNTHETIC: I recalled {count} earlier note(s) of yours before storing this one.'}]),
             sm.N('remember', 'agent.memory_write', target='long_term', mode='direct', scope='user', namespace='notes', kind='episodic',
                  text='{note}', generated=False, evidence='request input', max_chars=500)]
    state = [sm.S('note'), sm.S('recalled', 'records'), sm.S('count', 'integer', default=0), sm.S('answer')]
    return sm.make_graph(nodes, sm.chain('START', *[n['id'] for n in nodes], 'END'), {
        'state': state,
        'policies': [{'id': 'mine', 'query': '{note}', 'embeddings': {'provider': 'local_hash', 'dimension': 64}, 'stages': [
            {'id': 'take', 'op': 'retrieve', 'config': {'sources': ['long_term'], 'scopes': ['user'], 'namespaces': ['notes'], 'method': 'recency', 'k': 5}},
            {'id': 'budget', 'op': 'budget', 'config': {'max_tokens': 256, 'order': 'chronological'}}]}],
        'limits': {'maxSteps': 12, 'maxSeconds': 30, 'maxModelCalls': 1, 'maxTokens': 4096}})


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    (root / 'serving_memory.project.json').write_text(json.dumps(graph().to_json(), indent=2) + '\n')
    ui = {'schemaVersion': '1.0.0', 'positions': {}, 'synthetic': True,
          'description': 'SYNTHETIC long-term memory for a served agent: a release keeps each user\'s notes separately and recalls only that user\'s notes. '
                         'Zero model calls; local hash is lexical hashing, not semantic search.',
          'defaultInput': {'note': 'SYNTHETIC note: my favourite colour is teal.'}}
    (root / 'serving_memory.ui.json').write_text(json.dumps(ui, indent=2) + '\n')
