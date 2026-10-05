"""Real HTTP native fork/reset teaching journey; SYNTHETIC prompts, actual local Ollama.

Only this journey's chosen local namespace and created sessions are changed.
No downloads, physical deletion or semantic-answer benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from urllib.parse import urlparse

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8780')
    parser.add_argument('--namespace', default='session-actions-cli')
    args = parser.parse_args()
    if urlparse(args.base).hostname not in ('localhost', '127.0.0.1'):
        parser.error('this journey targets a local workbench only')
    headers = {'Authorization': 'Bearer '+os.environ['VOID_API_TOKEN']} if os.environ.get('VOID_API_TOKEN') else {}
    with httpx.Client(base_url=args.base, headers=headers, timeout=40, trust_env=False) as c:
        def call(method, path, **kw):
            response = c.request(method, path, **kw)
            response.raise_for_status()
            return response.json()
        graph = call('GET','/api/examples/serving_conversation')['graph']
        run = call('POST','/api/runs',headers={'Idempotency-Key':uuid.uuid4().hex},json={'graph':graph,
                   'config':{'input':{'question':'SYNTHETIC source teaching prompt: give a brief greeting.'}}})
        rid = run['runId']
        until = time.monotonic()+60
        while True:
            state = call('GET',f'/api/runs/{rid}')
            if state['status'] in ('completed','failed','cancelled','paused'):
                break
            if time.monotonic()>until:
                raise RuntimeError('Source worker exceeded 60 seconds')
            time.sleep(.1)
        assert state['status']=='completed', state
        version = call('POST','/api/production/versions',json={'runId':rid,'node':'__agent_conversation__','name':'SYNTHETIC session management journey',
                       'owner':'local scientist','intendedUse':'native conversation fork/reset evidence','limitations':'bounded teaching prompts; no answer accuracy claim'})
        rel = call('POST','/api/production/releases',json={'versionId':version['id'],'config':{'namespace':args.namespace,'maxBatch':1,'sessionMode':'conversation','captureInputs':True,'timeoutSeconds':30}})
        overview = call('GET','/api/production')
        prior = next((r['release'] for r in overview['routes'] if r['target']=='local' and r['namespace']==args.namespace),None)
        call('POST',f"/api/production/releases/{rel['id']}/deploy",json={'expectedCurrent':prior})
        def turn(session, question):
            return call('POST',f'/api/serve/local/{args.namespace}/predict',json={'requestId':uuid.uuid4().hex,'records':[{'question':question}],'session':session,'expectedRelease':rel['id']})
        first = turn('main','SYNTHETIC teaching prompt: remember my favourite colour is blue. Reply briefly.')
        second = turn('main','SYNTHETIC teaching prompt: which colour did I mention?')
        endpoint = f"/api/production/releases/{rel['id']}/conversation"
        inspect = lambda session: call('GET',endpoint,params={'session':session})
        original = inspect('main')
        reviewed = {'session':'main','expectedRevision':original['head']['revision'],'expectedCheckpointSha256':original['head']['checkpointSha256'],'reason':'SYNTHETIC checkpoint management teaching journey'}
        fork_req = {**reviewed,'actionId':uuid.uuid4().hex,'destinationSession':'alternative'}
        fork = call('POST',endpoint+'/fork',json=fork_req)
        assert inspect('alternative')['state']==original['state'] and inspect('main')==original
        branch = turn('alternative','SYNTHETIC alternative teaching prompt: give a brief greeting.')
        branch_context = branch['result']['agent']['contexts'][0]['value']
        assert branch_context['providerRequest'][1:-1]==[{'role':m['role'],'content':m['content']} for m in original['state']['history']]
        assert branch['result']['agent']['threadId']==fork['threadId'] and inspect('main')==original
        reset_req = {**reviewed,'actionId':uuid.uuid4().hex}
        reset = call('POST',endpoint+'/reset',json=reset_req)
        empty = inspect('main')
        assert empty['state'] is None and empty['head']['revision']==3
        fresh = turn('main','SYNTHETIC fresh teaching prompt: give a brief greeting.')
        assert fresh['conversationState']['revision']==4 and len(fresh['result']['agent']['contexts'][0]['value']['providerRequest'])==2
        assert fresh['result']['agent']['threadId']!=second['result']['agent']['threadId']
        duplicate = call('POST',endpoint+'/reset',json=reset_req)
        assert duplicate['idempotentReplay'] and duplicate['actionSha256']==reset['actionSha256'] and inspect('main')['head']['revision']==4
        before = inspect('main')
        replay = call('POST',f"/api/production/requests/{second['requestId']}/replay",json={})
        assert replay['result']['agent']['contexts'][0]['value']['providerRequest']==second['result']['agent']['contexts'][0]['value']['providerRequest']
        assert inspect('main')==before and inspect('alternative')['head']['revision']==2
        assert call('GET',f"/api/production/requests/{second['requestId']}")['traceSha256']==second['traceSha256']
        print(json.dumps({'teachingPrompts':'SYNTHETIC','runId':rid,'versionId':version['id'],'releaseId':rel['id'],
                          'fork':fork,'reset':reset,'branchPredictions':branch['result']['predictions'],'branchUsage':branch_context['usage'],
                          'exactBranchRequest':branch_context['providerRequest'],'freshPredictions':fresh['result']['predictions'],
                          'freshUsage':fresh['result']['agent']['contexts'][0]['usage'],'freshHead':fresh['conversationState'],
                          'originalTraceSha256':second['traceSha256'],'historicalReplay':'exact prior sent request; live heads unchanged',
                          'duplicateAction':'original immutable receipt; no second reset'},indent=2))


if __name__=='__main__':
    main()
