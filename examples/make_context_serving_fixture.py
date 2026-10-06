"""SYNTHETIC pinned-retrieval / short-term conversation; optional real Ollama JSON."""
import json
from pathlib import Path
from agent import samples as sm
from make_json_serving_fixture import graph as json_graph


def graph(json_output=False):
    nodes=[sm.N('tick','agent.set_state',assignments=[{'field':'turns','kind':'increment','by':1}]),
           sm.N('find','agent.retrieve',index='notes',query='{question}',k=1,output_field='docs'),
           sm.N('remember','agent.memory_write',target='short_term',mode='direct',field='history',text='{question}',role='user'),
           sm.N('select','agent.memory_select',policy='recent',short_term_field='history',output_field='selected')]
    state=[sm.S('question'),sm.S('docs','documents'),sm.S('turns','integer','add',default=0,scope='thread'),
           sm.S('history','messages','keep_last_n',default=[],scope='thread',n=3),sm.S('selected','records')]
    if json_output:
        nodes += [sm.N('prompt','agent.prompt',output_field='messages',items=[
            {'kind':'template','role':'system','template':'Extract colour and count from the retrieved SYNTHETIC record. Use only the declared JSON schema.'},
            {'kind':'documents','role':'system','field':'docs','header':'Retrieved record:','template':'{text}'},
            {'kind':'memory','role':'system','field':'selected','header':'Recent questions:','template':'{text}'},
            {'kind':'template','role':'user','template':'{question}'}]),sm.N('model','agent.structured_output',**json_graph().to_json()['nodes'][1]['config'])]
        state += [sm.S('messages','messages'),sm.S('result','object',properties={'colour':'text','count':'integer'})]
    else:
        nodes.append(sm.N('answer','agent.set_state',assignments=[{'field':'answer','kind':'template','template':'SYNTHETIC {question}: {turns}'}]))
        state.append(sm.S('answer'))
    return sm.make_graph(nodes,sm.chain('START',*[n['id'] for n in nodes],'END'),{
        'state':state,'indexes':[{'id':'notes','loader':{'directory':'examples/fixtures/context_notes','glob':'*.txt'},
                                'splitter':{'strategy':'paragraph','chunkSize':200,'chunkOverlap':0},
                                'embeddings':{'provider':'local_hash','dimension':256,'normalize':True}}],
        'policies':[{'id':'recent','query':'{question}','embeddings':{'provider':'local_hash','dimension':64},'stages':[
            {'id':'take','op':'retrieve','config':{'sources':['short_term'],'method':'recency','k':2}},
            {'id':'dedupe','op':'dedupe','config':{'method':'exact_text'}},
            {'id':'budget','op':'budget','config':{'max_tokens':128,'order':'chronological'}}]}],
        'limits':{'maxSteps':12,'maxSeconds':30,'maxModelCalls':1,'maxTokens':4096}})


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    for structured in (False,True):
        name='serving_context_json' if structured else 'serving_context'
        (root/(name+'.project.json')).write_text(json.dumps(graph(structured).to_json(),indent=2)+'\n')
        ui={'schemaVersion':'1.0.0','positions':{},'synthetic':True,
            'description':'SYNTHETIC pinned retrieval and bounded short-term conversation. Local hash is lexical hashing. '+('Actual installed local Ollama JSON; no quality benchmark.' if structured else 'Zero model calls; deterministic teaching output.'),
            'defaultInput':{'question':'SYNTHETIC source: read the record.'}}
        (root/(name+'.ui.json')).write_text(json.dumps(ui,indent=2)+'\n')
