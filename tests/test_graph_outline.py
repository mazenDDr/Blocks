"""Actual native API validation provenance/diagnostics consumed by the TS outline."""
import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from control.app import create_app

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = """
import fs from 'node:fs';import {pathToFileURL} from 'node:url';
const {graphOutline}=await import(pathToFileURL(process.argv[1]).href);
const p=JSON.parse(fs.readFileSync(0,'utf8'));console.log(JSON.stringify(graphOutline(p.graph,p.ops,p.validation,p.pending,p.query,p.errorsOnly)));
"""


def outline(graph, ops, validation, pending=False, query='', errors_only=False):
    r = subprocess.run(['node','--input-type=module','-e',SCRIPT,str(ROOT/'apps/editor/src/graphOutline.ts')],
        input=json.dumps(dict(graph=graph,ops=ops,validation=validation,pending=pending,query=query,errorsOnly=errors_only)),
        text=True,capture_output=True,check=True,timeout=20)
    return json.loads(r.stdout)


@pytest.mark.parametrize('name',['reference_cnn','residual_cnn','tabular_regression','nlp_token_classification'])
def test_outline_consumes_exact_native_contracts_hashes_wires_and_preserves_input(name,tmp_path):
    graph=json.loads((ROOT/f'examples/{name}.project.json').read_text())
    before=json.dumps(graph)
    with TestClient(create_app(tmp_path/'workbench')) as client:
        response=client.post('/api/validate',json={'graph':graph});assert response.status_code==200
        native=response.json();assert native['ok']
        view=outline(graph,{},native)
        assert view['graphHash']==native['graphHash']
        assert [r['id'] for r in view['rows']]==[n['id'] for n in graph['nodes']]
        for row in view['rows']:
            assert row['nativeView']==native['nodes'].get(row['id'])
            assert len(row['incoming'])==sum(e['to']['node']==row['id'] for e in graph['edges'])
            assert len(row['outgoing'])==sum(e['from']['node']==row['id'] for e in graph['edges'])
        pending=outline(graph,{},native,pending=True)
        assert pending['graphHash'] is None and all(r['nativeView'] is None and not r['diagnostics'] for r in pending['rows'])
    assert json.dumps(graph)==before


def test_actual_nested_native_error_routes_to_instance_and_global_diagnostics_are_retained(tmp_path):
    graph=json.loads((ROOT/'examples/residual_cnn.project.json').read_text())
    module=graph['modules'][0];module['nodes'][0]['config']['out_channels']=0
    with TestClient(create_app(tmp_path/'workbench')) as client:
        native=client.post('/api/validate',json={'graph':graph}).json()
    assert not native['ok']
    nested=[d for d in native['diagnostics'] if d['nodeId'] and d['nodeId'].startswith('res1/')]
    assert nested
    view=outline(graph,{},native,errors_only=True)
    row=next(r for r in view['rows'] if r['id']=='res1')
    assert all(d in row['diagnostics'] for d in nested)
    routed=[d for r in view['rows'] for d in r['diagnostics']]+view['globalDiagnostics']
    assert all(d in routed for d in native['diagnostics'] if d['severity']=='error')
