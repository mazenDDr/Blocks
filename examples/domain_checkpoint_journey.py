"""Real local HTTP checkpoint, native inference and exact epoch-continuation journey. SYNTHETIC fixtures only."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
import httpx
from domain import samples


def journey(base):
    out=[]
    with httpx.Client(base_url=base,timeout=90,trust_env=False) as c:
        def post(path, body):
            r=c.post(path,json=body,headers={'Idempotency-Key':uuid.uuid4().hex});r.raise_for_status();return r.json()
        def get(path):
            r=c.get(path);r.raise_for_status();return r.json()
        def run(graph, project):
            row=post('/api/runs',{'graph':graph,'config':{'project_id':project}})
            rid=row['runId'];deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                summary=get('/api/runs/'+rid)
                if summary['status'] in ('completed','failed','cancelled'):
                    assert summary['status']=='completed',summary
                    return get('/api/domain/models?runId='+rid)['models'][0]
                time.sleep(.1)
            raise TimeoutError(rid)
        for builder,family in [(samples.vision_graph,'vision'),(samples.nlp_graph,'nlp'),(samples.speech_graph,'speech')]:
            graph=builder(epochs=2);graph['nodes'][0]['config']['n']=12
            if family=='vision':graph['nodes'][-1]['config']['width']=4
            if family=='nlp':graph['nodes'][-1]['config'].update(hidden=8,embedding=8)
            if family=='speech':graph['nodes'][-1]['config']['hidden']=8
            project='checkpoint_'+family
            ui={'schemaVersion':'1.0.0','positions':{}}
            r=c.put('/api/projects/'+project,json={'graph':graph,'ui':ui});r.raise_for_status()
            model=run(graph,project);identity=model['modelId']
            request=get('/api/domain/models/'+identity+'/example')
            first=post('/api/domain/models/'+identity+'/predict',{'records':request['records']})
            raw=c.get('/api/domain/models/'+identity+'/checkpoint');raw.raise_for_status()
            assert hashlib.sha256(raw.content).hexdigest()==model['checkpointSha256']
            graph['nodes'][-1]['config'].update(epochs=4,resume_model_id=identity)
            if family=='nlp':graph['nodes'][1]['config']['fitted_model_id']=identity
            continued=run(graph,project)
            assert continued['parentModelId']==identity and continued['epochs']==4
            again=post('/api/domain/models/'+identity+'/predict',{'records':request['records']});assert first==again
            out.append({'family':family,'synthetic':True,'modelId':identity,'runId':model['runId'],'checkpointSha256':model['checkpointSha256'],
                        'continuedModelId':continued['modelId'],'continuedRunId':continued['runId'],'epochs':[2,4],
                        'predictionProvenance':first['provenance'],'parentPredictionsUnchanged':True,'exportBytes':len(raw.content)})
    return {'location':'local CPU native inference; labelled SYNTHETIC fixtures','models':out}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8769');a=p.parse_args()
    print(json.dumps(journey(a.base),indent=2))
