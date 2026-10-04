"""A64 bounded journey: real local PostgreSQL, labelled SYNTHETIC data, native serving.

No manual extraction: seed a test database, then use only the same HTTP interfaces as the editor.
Database lifetime ends in finally. Serving retains pinned artifacts and does not reread the database.
"""
from __future__ import annotations
import argparse
import copy
import json
import time
import uuid
import os
import httpx
import pandas as pd
from connectors.localtest import LocalPostgres
from scale.common import loopback


def journey(base,namespace="m8-connected"):
    pg=LocalPostgres()
    try:
        df=pd.read_csv('examples/fixtures/synthetic_serving_sensors.csv')
        with pg.admin() as db:
            db.execute('CREATE TABLE synthetic_sensors (id integer, signal double precision, background double precision, label integer)')
            with db.cursor() as cur:
                cur.executemany('INSERT INTO synthetic_sensors VALUES(%s,%s,%s,%s)',[(i,float(r.signal),float(r.background),int(r.label)) for i,r in df.iterrows()])
        with httpx.Client(headers=({'Authorization': 'Bearer ' + os.environ['VOID_API_TOKEN']} if os.environ.get('VOID_API_TOKEN') else {}), base_url=loopback(base),timeout=40,trust_env=False) as c:
            def call(method,path,**kw):
                r=c.request(method,path,**kw);r.raise_for_status();return r.json()
            connection='m8-'+uuid.uuid4().hex[:8]
            call('POST','/api/connections',json={'id':connection,'name':'LOCAL PostgreSQL SYNTHETIC sensors','type':'postgres','settings':pg.settings(),'secrets':{}})
            call('POST',f'/api/connections/{connection}/test',json={})
            g=call('GET','/api/examples/production_sensors')['graph']
            src=next(n for n in g['nodes'] if n['id']=='sensors');src['type']='postgres.query'
            src['config']={'connection':connection,'mode':'visual','query':{'base':{'name':'synthetic_sensors'},'base_alias':'s',
                           'columns':[{'table':'s','column':k} for k in ['signal','background','label']], 'order_by':[{'by':'signal'}], 'limit':160}}
            ui={'schemaVersion':'1.0.0','positions':{},'synthetic':True,'description':'LOCAL PostgreSQL labelled SYNTHETIC sensors; pin snapshots, compare C variants, inspect, record a conclusion and test native serving.'}
            project='connected_sensors'
            call('PUT',f'/api/projects/{project}',json={'graph':g,'ui':ui})
            runs=[];pin=None
            for reg in [1.0,0.1]:
                variant=copy.deepcopy(g);next(n for n in variant['nodes'] if n['id']=='classifier')['config']['C']=reg
                submitted=call('POST','/api/runs',json={'graph':variant,'config':{'project_id':project,'source_pins':{'sensors':pin} if pin else {}}},headers={'Idempotency-Key':uuid.uuid4().hex})
                rid=submitted['runId'];end=time.monotonic()+60
                while True:
                    r=call('GET','/api/runs/'+rid)
                    if r['status'] in ('completed','failed','cancelled'):break
                    if time.monotonic()>end:raise RuntimeError('Journey training deadline')
                    time.sleep(.1)
                if r['status']!='completed':raise RuntimeError(r)
                if pin is None:pin=r['snapshots'][0]['snapshotId']
                runs.append(rid)
                # Real intermediate/fitted values; selected-node inspection uses this same route.
                for node in ('sensors','scale','classifier','metrics'):
                    inspected=call('POST',f'/api/runs/{rid}/inspect',json={'kind':'summary','node':node})
                    assert inspected['available']
            comparison=call('POST','/api/evidence/compare',json={'runIds':runs})
            conclusion=call('POST','/api/evidence',json={'runIds':runs,'conclusion':'Controlled regularization comparison on labelled SYNTHETIC sensors. Compare recorded validation metrics; no real-world generalization claim.'})
            v=call('POST','/api/production/versions',json={'runId':runs[0],'node':'classifier','name':'Connected SYNTHETIC sensors','owner':'local researcher','intendedUse':'Bounded connected-workbench evidence','limitations':'Synthetic local test database; not real sensor data'})
            rel=call('POST','/api/production/releases',json={'versionId':v['id'],'config':{'namespace':namespace,'captureInputs':True}})
            ov=call('GET','/api/production');prior=next((r['release'] for r in ov['routes'] if r['namespace']==namespace and r['target']=='local'),None)
            call('POST',f"/api/production/releases/{rel['id']}/deploy",json={'expectedCurrent':prior})
            refs=call('GET',f"/api/production/versions/{v['id']}/reference-input")
            trace=call('POST',f'/api/serve/local/{namespace}/predict',json={'requestId':uuid.uuid4().hex,'records':refs['records']})
            job=call('POST','/api/production/traffic',json={'releaseId':rel['id'],'payloads':[refs['records'],refs['records'][:1]],'rate':5,'durationSeconds':1,'maxRequests':10})
            end=time.monotonic()+40
            while job['state']=='running':
                if time.monotonic()>end:raise RuntimeError('Traffic drain deadline')
                time.sleep(.1);job=call('GET','/api/production/traffic/'+job['id'])
            assert job['state']=='completed' and job['result']['observed']['successfulRequests']>0
            out={'synthetic':True,'location':'real local PostgreSQL and CPU HTTP serving, no external cloud','runIds':runs,'conclusion':conclusion,
                 'versionId':v['id'],'releaseId':rel['id'],'traceSha256':trace['traceSha256'],'source':v['manifest']['source'],
                 'evaluation':[s for r in comparison['runs'] for s in r['summaries'] if s['type']=='sklearn.metrics'],'traffic':job['result']['observed']}
            print(json.dumps(out,indent=2))
            return out
    finally:pg.stop()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',default='http://127.0.0.1:8768');ap.add_argument('--namespace',default='m8-connected');a=ap.parse_args();journey(a.base,a.namespace)
