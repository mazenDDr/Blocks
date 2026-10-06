"""Exact native Gymnasium/PyTorch contracts consumed by the real TS RL outline."""
import json
from pathlib import Path
import subprocess

import pytest
from fastapi.testclient import TestClient
from control.app import create_app

ROOT=Path(__file__).resolve().parents[1]
SCRIPT="""
import fs from 'node:fs';import {pathToFileURL} from 'node:url';
const {rlOutline}=await import(pathToFileURL(process.argv[1]).href);
const p=JSON.parse(fs.readFileSync(0,'utf8'));
console.log(JSON.stringify(rlOutline(p.graph,{},p.native,p.pending,p.query,p.errorsOnly)));
"""


def outline(graph,native,**kw):
    result=subprocess.run(['node','--input-type=module','-e',SCRIPT,str(ROOT/'apps/editor/src/rlOutline.ts')],input=json.dumps({'graph':graph,'native':native,**kw}),capture_output=True,text=True,check=True,timeout=20)
    return json.loads(result.stdout)


@pytest.mark.parametrize('name',['rl_cartpole_dqn','rl_gridworld_builder'])
def test_actual_native_rl_contracts_exact_ports_spaces_rewards_equation_and_pending_identity(name,tmp_path):
    graph=json.loads((ROOT/f'examples/{name}.project.json').read_text())
    graph['nodes'][1]['id']='constructor'
    for edge in graph['edges']:
        for end in ('from','to'):
            if edge[end]['node']=='env':edge[end]['node']='constructor'
    before=json.dumps(graph)
    with TestClient(create_app(tmp_path/'workbench')) as client:
        response=client.post('/api/validate',json={'graph':graph});assert response.status_code==200
        native=response.json();assert native['ok']
        result=outline(graph,native);assert result['graphHash']==native['graphHash']
        for row in result['rows']:
            assert row['node']==next(n for n in graph['nodes'] if n['id']==row['id'])
            assert row['nativeView']==native['nodes'][row['id']]
        env=result['rows'][1]['nativeView']['outputShapes']['env']
        assert env['observationSpace']==native['rl']['environment']['observationSpace']
        assert env['actionSpace']==native['rl']['environment']['actionSpace']
        assert env['weights']==native['rl']['environment']['weights']
        q=next(r for r in result['rows'] if r['id']=='qnet')['nativeView']
        assert q['params']==native['rl']['network']['params']
        assert q['outputShapes']['network']['shapes']==native['rl']['network']['shapes']
        learner=next(r for r in result['rows'] if r['id']=='learner')['nativeView']['outputShapes']['learner']
        assert learner['equation']==native['rl']['learner']['equation']
        assert [r['id'] for r in outline(graph,native,query='observationSpace')['rows']]
        pending=outline(graph,native,pending=True)
        assert pending['graphHash'] is None and all(r['nativeView'] is None and not r['diagnostics'] for r in pending['rows'])
        invalid={**graph,'edges':[e for e in graph['edges'] if e['id']!='e4']}
        refused=client.post('/api/validate',json={'graph':invalid}).json();assert not refused['ok']
        view=outline(invalid,refused,errorsOnly=True)
        assert any(d['code']=='E_MISSING_INPUT' for r in view['rows'] for d in r['diagnostics'])
        assert all(r['diagnostics']==[d for d in refused['diagnostics'] if d['nodeId']==r['id']] for r in view['rows'])
    assert json.dumps(graph)==before
