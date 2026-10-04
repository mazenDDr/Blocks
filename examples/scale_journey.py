"""Measured local CPU protocol journey. Starts no servers; all services are explicit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import uuid
import httpx
from scale.common import loopback


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',default='http://127.0.0.1:8768');ap.add_argument('--worker',default='http://127.0.0.1:8778');ap.add_argument('--token-env',default='VOID_WORKER_TOKEN');ap.add_argument('--output');a=ap.parse_args()
    records=[]
    with httpx.Client(base_url=loopback(a.base),timeout=100,trust_env=False) as c:
        def call(method,path,**kw):
            r=c.request(method,path,**kw);r.raise_for_status();return r.json()
        graph=call('GET','/api/examples/production_sensors')['graph']
        for seed in (71,72,73):
            started=time.perf_counter()
            local=call('POST','/api/runs',json={'graph':graph,'config':{'seed':seed}},headers={'Idempotency-Key':uuid.uuid4().hex})
            rid=local['runId'];end=time.monotonic()+60
            while True:
                status=call('GET','/api/runs/'+rid)['status']
                if status in ('completed','failed','cancelled'):break
                if time.monotonic()>end:raise RuntimeError('Local deadline')
                time.sleep(.05)
            assert status=='completed'
            local_seconds=time.perf_counter()-started
            started=time.perf_counter()
            job=call('POST','/api/integrations/remote',json={'graph':graph,'endpoint':loopback(a.worker),'tokenEnv':a.token_env,'seed':seed})
            end=time.monotonic()+60
            while not job['runId']:
                if time.monotonic()>end:raise RuntimeError(job)
                time.sleep(.05);job=call('POST',f"/api/integrations/remote/{job['id']}/refresh",json={})
            assert job['status']=='completed',job
            worker_seconds=time.perf_counter()-started
            comparison=call('POST','/api/evidence/compare',json={'runIds':[rid,job['runId']]})
            ms=[next(s for s in r['summaries'] if s['node']=='metrics')['summary'] for r in comparison['runs']]
            assert ms[0]['values']==ms[1]['values'] and ms[0]['evaluatedRowIdsSha256']==ms[1]['evaluatedRowIdsSha256']
            recovered=call('POST',f"/api/integrations/remote/{job['id']}/refresh",json={});assert recovered['runId']==job['runId']
            records.append({'seed':seed,'localRunId':rid,'workerJobId':job['id'],'retrievedRunId':job['runId'],'localSubmitToCompleteSeconds':local_seconds,
                            'httpWorkerSubmitToImportedSeconds':worker_seconds,'metrics':ms[0]['values'],'evaluationRowsSha256':ms[0]['evaluatedRowIdsSha256'],
                            'nativeAgreement':'exact recorded evaluation values','workerLocation':job['location']})
        info=call('GET','/api/integrations');rid=records[-1]['localRunId'];r=next(x for x in info['runs'] if x['id']==rid)
        exports=[]
        for adapter in ['mlflow','wandb']:
            q=call('POST','/api/integrations/exports',json={'runId':rid,'adapter':adapter,'shareMetrics':True,'artifactSha256':[x['sha256'] for x in r['shareableMetricsArtifacts']]})
            confirmed=call('POST',f"/api/integrations/exports/{q['id']}/sync",json={});assert confirmed['status']=='confirmed',confirmed
            again=call('POST',f"/api/integrations/exports/{q['id']}/sync",json={});assert again['external']==confirmed['external']
            exports.append(confirmed)
        result={'scope':'labelled SYNTHETIC sensors, 160 rows; same macOS CPU; separate authenticated loopback process, not another machine or GPU',
                'timingScope':'observed wall time including HTTP, polling, worker startup and transfer; cache/host contention uncontrolled, no speed claim',
                'records':records,'exports':exports,'nativeConformance':call('POST','/api/extensions/conformance',json={'example':'offset'})}
        if a.output:Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))

if __name__=='__main__':main()
