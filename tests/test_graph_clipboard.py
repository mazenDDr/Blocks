"""Real Node clipboard transform verified against native graph/Torch behavior.

Inputs are labelled SYNTHETIC bundled teaching graphs/tensors. Copying never supplies
trained weights; equal native weights below are an independent reference comparison.
"""
import json
import subprocess
from pathlib import Path

import pytest
import torch

from graph_core.hashing import semantic_hash
from graph_core.lower import lower_graph
from graph_core.project_io import load_project
from graph_core.schema import Graph
from graph_core.validate import validate

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = """
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const {copyGraphNodes,pasteGraphNodes}=await import(pathToFileURL(process.argv[1]).href);
const p=JSON.parse(fs.readFileSync(0,'utf8'));
const clip=copyGraphNodes(p.graph,p.ui,p.selected,'SYNTHETIC native teaching source',p.hash);
console.log(JSON.stringify({clip,...pasteGraphNodes(p.target,p.targetUi,clip)}));
"""


def copied(project, selected=None, target=None):
    graph = project.graph.to_json()
    ui = project.ui.to_json() if project.ui else {"schemaVersion": "1.0.0", "positions": {}}
    target = target or {**graph, "nodes": [], "edges": [], "modules": [], "codeBlocks": []}
    payload = dict(graph=graph, ui=ui, selected=selected or [n.id for n in project.graph.nodes],
                   hash=semantic_hash(project.graph), target=target,
                   targetUi={"schemaVersion": "1.0.0", "positions": {}})
    result = subprocess.run(["node", "--input-type=module", "-e", SCRIPT,
                             str(ROOT / "apps/editor/src/graphClipboard.ts")],
                            input=json.dumps(payload), text=True, capture_output=True, check=True, timeout=20)
    return json.loads(result.stdout)


@pytest.mark.parametrize("name", ["reference_cnn", "residual_cnn"])
def test_copied_model_native_shapes_outputs_loss_gradients_and_sgd(name):
    project = load_project(ROOT / f"examples/{name}.project.json")
    result = copied(project)
    graph = Graph.model_validate(result["graph"])
    before, after = validate(project.graph), validate(graph)
    assert before.ok and after.ok
    assert before.total_params == after.total_params
    assert result["clip"]["sourceGraphHash"] == semantic_hash(project.graph)
    assert semantic_hash(graph) != semantic_hash(project.graph)  # fresh node identities
    mapping = result["mapping"]
    remap = lambda name: mapping.get(name.split('/')[0], name.split('/')[0]) + (('/' + name.split('/', 1)[1]) if '/' in name else '')
    for node, ports in before.output_types.items():
        assert {p: t.to_json() for p, t in ports.items()} == {p: t.to_json() for p, t in after.output_types[remap(node)].items()}
    torch.manual_seed(37)
    original, pasted = lower_graph(project.graph), lower_graph(graph)
    for node, module in original._modules.items():
        pasted._modules[remap(node)].load_state_dict(module.state_dict())
    for node, module in original._modules.items():
        for p, q in zip(module.parameters(), pasted._modules[remap(node)].parameters(), strict=True):
            assert p.data_ptr() != q.data_ptr()
    x = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([1, 2])
    optimizers = [torch.optim.SGD(m.parameters(), lr=.01) for m in (original, pasted)]
    outputs = [m(x) for m in (original, pasted)]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    losses = [torch.nn.functional.cross_entropy(y, labels) for y in outputs]
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    for loss in losses:
        loss.backward()
    for node, module in original._modules.items():
        for p, q in zip(module.parameters(), pasted._modules[remap(node)].parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    for optimizer in optimizers:
        optimizer.step()
    torch.testing.assert_close(original(x), pasted(x), rtol=0, atol=0)
    for old, new in mapping.items():
        if old in result["clip"]["positions"]:
            point = result["clip"]["positions"][old]
            assert result["ui"]["positions"][new] == dict(x=point["x"]+80, y=point["y"]+80)


@pytest.mark.parametrize("name", ["tabular_regression", "vision_segmentation_synthetic", "nlp_token_classification", "speech_ctc_tones"])
def test_copied_native_typed_workflow_retains_contracts_and_configs(name):
    project = load_project(ROOT / f"examples/{name}.project.json")
    result = copied(project)
    graph = Graph.model_validate(result["graph"])
    assert validate(project.graph).ok and validate(graph).ok
    assert [n.config for n in graph.nodes] == [n.config for n in project.graph.nodes]
    assert [e.kind for e in graph.edges] == [e.kind for e in project.graph.edges]
    assert result["clip"]["omittedBoundaryEdges"] == 0
    for old, new in zip(project.graph.edges, graph.edges, strict=True):
        assert new.from_.node == result["mapping"][old.from_.node]
        assert new.to.node == result["mapping"][old.to.node]
        assert new.from_.port == old.from_.port and new.to.port == old.to.port


def test_partial_copy_does_not_invent_inputs_or_hide_native_validation(cnn_project):
    result = copied(cnn_project, ["conv_1"])
    assert result["clip"]["omittedBoundaryEdges"] == 2
    graph = Graph.model_validate(result["graph"])
    report = validate(graph)
    assert not report.ok
    assert any(d.code == "E_MISSING_INPUT" for d in report.errors)
    graph.nodes[0].type = "future.SYNTHETIC_unknown"
    report = validate(graph)
    assert any(d.code == "E_UNKNOWN_OP" for d in report.errors)


def test_internal_shared_encoder_is_rebound_inside_independent_copied_group():
    project = load_project(ROOT / "examples/shared_encoder.project.json")
    result = copied(project)
    pasted_graph = Graph.model_validate(result["graph"])
    original, pasted = lower_graph(project.graph), lower_graph(pasted_graph)
    mapping = result["mapping"]
    assert original.shared and pasted.shared
    def remap(name):
        root, _, suffix = name.partition('/')
        return mapping[root] + (('/' + suffix) if suffix else '')
    for call, owner in original.shared.items():
        assert pasted.shared[remap(call)] == remap(owner)
        assert pasted._modules[remap(call)].inner is pasted._modules[remap(owner)]
        assert pasted._modules[remap(owner)] is not original._modules[owner]
    assert validate(project.graph).total_params == validate(pasted_graph).total_params
