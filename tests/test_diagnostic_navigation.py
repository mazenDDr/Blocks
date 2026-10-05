"""Actual native diagnostic paths resolve to exact stored definitions with no graph edit."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
from graph_core.hashing import semantic_hash
from graph_core.validate import validate

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('diagnostic_fixture', ROOT/'examples/make_diagnostic_fixture.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
SCRIPT = """
import fs from 'node:fs';import {pathToFileURL} from 'node:url';
const {diagnosticTarget}=await import(pathToFileURL(process.argv[1]).href);
const p=JSON.parse(fs.readFileSync(0,'utf8')), before=JSON.stringify(p.graph);
console.log(JSON.stringify({targets:p.ids.map(id=>({id,target:diagnosticTarget(p.graph,id)})),unchanged:JSON.stringify(p.graph)===before}));
"""


@pytest.mark.parametrize('kind,paths', [
    ('nested', ['call/loop/it0/constructor','call/loop/it1/constructor']),
    ('repeat', ['loop/it0/constructor','loop/it1/constructor']),
    ('select', ['pick/then/constructor','pick/otherwise/constructor']),
])
def test_actual_native_nested_repeat_and_both_branch_errors_open_exact_stored_definition(kind, paths):
    graph = fixture.diagnostic_graph(kind)
    before = graph.to_json()
    report = validate(graph)
    ids = [d.nodeId for d in report.errors if d.code == 'E_MISSING_INPUT']
    assert ids == paths and not report.ok
    assert semantic_hash(graph) == semantic_hash(type(graph).model_validate(before))
    result = subprocess.run(['node','--input-type=module','-e',SCRIPT,str(ROOT/'apps/editor/src/diagnosticNavigation.ts')],
                            input=json.dumps({'graph':before,'ids':ids+['missing/constructor']}),text=True,capture_output=True,check=True,timeout=20)
    data = json.loads(result.stdout)
    assert data['unchanged'] and data['targets'][-1]['target'] is None
    for row in data['targets'][:-1]:
        target = row['target']
        assert target['node'] == 'constructor'
        expected = 'alternate' if '/otherwise/' in row['id'] else 'leaf'
        assert target['scopes'][-1]['module'] == expected
        assert graph.module(expected,'1.0.0').nodes[0].id == target['node']
        assert target['scopes'][-1]['via'] == ('pick/'+row['id'].split('/')[1] if kind=='select' else 'loop/'+row['id'].split('/')[-2])
        if kind == 'nested':assert target['scopes'][0] == {'module':'middle','version':'1.0.0','via':'call'}
    assert graph.to_json() == before
