"""Actual Node module transfer against native Torch; labelled SYNTHETIC teaching tensors.

Independent equal-weight reference setup below is not a clipboard weight-transfer claim.
"""
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import torch

from graph_core.composite import module_harness
from graph_core.hashing import semantic_hash
from graph_core.lower import lower_graph
from graph_core.project_io import load_project
from graph_core.schema import Graph
from graph_core.validate import validate

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = """
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const {copyModuleNodes,pasteModuleNodes}=await import(pathToFileURL(process.argv[1]).href);
const p=JSON.parse(fs.readFileSync(0,'utf8'));
const clip=copyModuleNodes(p.graph,p.ui,p.graph.modules[0],p.selected,'SYNTHETIC native module',p.hash);
console.log(JSON.stringify({clip,...pasteModuleNodes(p.target,p.ui,p.target.modules.at(-1),clip)}));
"""


def transfer(selected, target=None):
    source = load_project(ROOT / 'examples/residual_cnn.project.json').graph
    graph = source.to_json()
    payload = dict(graph=graph, ui={'schemaVersion': '1.0.0', 'positions': {}, 'unknown': {'label': 'SYNTHETIC metadata'}},
                   selected=selected, hash=semantic_hash(module_harness(source, graph['modules'][0]['id'], graph['modules'][0]['version'])[0]), target=target or graph)
    result = subprocess.run(['node', '--input-type=module', '-e', SCRIPT, str(ROOT / 'apps/editor/src/moduleClipboard.ts')],
                            input=json.dumps(payload), text=True, capture_output=True, check=True, timeout=20)
    return source, json.loads(result.stdout)


def test_module_transfer_keeps_original_interfaces_edges_root_and_reports_actual_missing_boundary_inputs():
    source, result = transfer(['conv_a', 'relu_a'])
    original = source.to_json()
    after = result['graph']
    assert after['nodes'] == original['nodes'] and after['edges'] == original['edges']
    old, new = original['modules'][0], after['modules'][0]
    assert new['inputs'] == old['inputs'] and new['outputs'] == old['outputs'] and new['params'] == old['params']
    assert new['nodes'][:len(old['nodes'])] == old['nodes'] and new['edges'][:len(old['edges'])] == old['edges']
    assert result['clip']['omittedBoundaryEdges'] == 2
    assert result['clip']['sourceGraphHash'] == semantic_hash(module_harness(source, old['id'], old['version'])[0])
    assert result['ui']['unknown'] == {'label': 'SYNTHETIC metadata'}
    assert semantic_hash(Graph.model_validate(after)) != semantic_hash(source)
    report = validate(Graph.model_validate(after))
    assert not report.ok
    assert any(d.code == 'E_MISSING_INPUT' and 'conv_a_copy' in (d.nodeId or '') for d in report.errors)


def test_explicit_reconnection_of_transferred_module_matches_native_outputs_loss_gradients_sgd_and_independent_parameters():
    source = load_project(ROOT / 'examples/residual_cnn.project.json').graph
    graph = source.to_json()
    original = graph['modules'][0]
    target = deepcopy(graph)
    empty = {**deepcopy(original), 'id': 'copied_block', 'nodes': [], 'edges': [],
             'outputs': [{**o, 'from': {'node': '', 'port': ''}} for o in original['outputs']]}
    target['modules'].append(empty)
    for node in target['nodes']:
        if node['type'] == 'core.composite':
            node['config']['module'] = 'copied_block'
    _, result = transfer([n['id'] for n in original['nodes']], target)
    copied = result['graph']['modules'][-1]
    assert copied['inputs'] == empty['inputs'] and copied['outputs'] == empty['outputs']
    assert not any(e['from']['node'] == '$in' for e in copied['edges'])
    mapping = result['mapping']
    # Explicit reference fixture reconnects the omitted declared boundary wires; the product never invents them.
    for index, edge in enumerate(e for e in original['edges'] if e['from']['node'] == '$in'):
        copied['edges'].append({**deepcopy(edge), 'id': f'transferred_input_{index}',
                               'to': {**edge['to'], 'node': mapping[edge['to']['node']]}})
    copied['outputs'] = [{**deepcopy(o), 'from': {**o['from'], 'node': mapping[o['from']['node']]}} for o in original['outputs']]
    pasted_graph = Graph.model_validate(result['graph'])
    before, after = validate(source), validate(pasted_graph)
    assert before.ok and after.ok and before.total_params == after.total_params
    torch.manual_seed(37)
    native, pasted = lower_graph(source), lower_graph(pasted_graph)
    def remap(name):
        parts = name.split('/')
        return '/'.join([*parts[:-1], mapping.get(parts[-1], parts[-1])])
    for name, module in native._modules.items():
        other = pasted._modules[remap(name)]
        other.load_state_dict(module.state_dict())
        for p, q in zip(module.parameters(), other.parameters(), strict=True):
            assert p.data_ptr() != q.data_ptr()
    x, labels = torch.randn(2, 3, 64, 64), torch.tensor([1, 2])
    outputs = [m(x) for m in (native, pasted)]
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    losses = [torch.nn.functional.cross_entropy(y, labels) for y in outputs]
    torch.testing.assert_close(*losses, rtol=0, atol=0)
    optimizers = [torch.optim.SGD(m.parameters(), lr=.01) for m in (native, pasted)]
    for loss in losses:
        loss.backward()
    for name, module in native._modules.items():
        for p, q in zip(module.parameters(), pasted._modules[remap(name)].parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    for optimizer in optimizers:
        optimizer.step()
    torch.testing.assert_close(native(x), pasted(x), rtol=0, atol=0)
