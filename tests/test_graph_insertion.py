"""Real editor wire insertion compared to native Torch and native validation.

All teaching projects/tensors are labelled SYNTHETIC; no trained weights transferred.
"""
import json
import subprocess
from pathlib import Path

import pytest
import torch

from graph_core.hashing import semantic_hash
from graph_core.lower import lower_graph
from graph_core.project_io import load_project
from graph_core.registry import get_op
from graph_core.schema import Graph
from graph_core.validate import validate

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = """
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const {insertOnWire}=await import(pathToFileURL(process.argv[1]).href);
const {defToView}=await import(pathToFileURL(process.argv[2]).href);
const p=JSON.parse(fs.readFileSync(0,'utf8'));
let graph=p.graph,view=graph,prefix='';
if(p.module){view=defToView(graph.modules.find(d=>d.id===p.module));prefix=p.module+'@1.0.0/';}
try {
 const result=insertOnWire(view,p.ui,p.edge,p.op,p.input,p.output,prefix);
 if(p.module){const d=graph.modules.find(d=>d.id===p.module);const replacement=result.replacement.map(e=>({...e,from:e.from.node.startsWith('__in:')?{node:'$in',port:e.from.node.slice(5)}:e.from}));graph={...graph,modules:graph.modules.map(m=>m===d?{...d,nodes:[...d.nodes,result.graph.nodes.at(-1)],edges:d.edges.flatMap(e=>e.id===p.edge?replacement:[e])}:m)};}
 else graph=result.graph;
 console.log(JSON.stringify({...result,graph}));
}catch(e){console.log(JSON.stringify({error:e.code,message:e.message}));}
"""


def inserted(project, edge, operation="diag.probe", input="input", output="output", module=None):
    op = get_op(operation)
    payload = dict(graph=project.graph.to_json(), ui=project.ui.to_json(), edge=edge,
                   module=module, input=input, output=output,
                   op=dict(type=op.type, version=op.version, backend=op.backend, graphKind=op.graph_kind,
                           inputs=list(op.inputs), outputs=list(op.outputs),
                           inputKinds=getattr(op, "in_kinds", None) or {p: "tensor" for p in op.inputs},
                           outputKinds=getattr(op, "out_kinds", None) or {p: "tensor" for p in op.outputs},
                           defaults=op.Config().model_dump(mode="json")))
    proc = subprocess.run(["node", "--input-type=module", "-e", SCRIPT,
                           str(ROOT/"apps/editor/src/graphInsertion.ts"), str(ROOT/"apps/editor/src/modules.ts")],
                          input=json.dumps(payload), capture_output=True, text=True, check=True, timeout=20)
    return json.loads(proc.stdout)


@pytest.mark.parametrize("name,module,edge_index", [
    ("reference_cnn", None, 0),
    ("residual_cnn", "residual_block", 0),
    ("residual_cnn", "residual_block", 1),
])
def test_inserted_probe_keeps_native_outputs_loss_gradient_sgd_and_other_graph_data(name, module, edge_index):
    project = load_project(ROOT/f"examples/{name}.project.json")
    original_json = project.graph.to_json()
    scope = original_json if module is None else next(d for d in original_json["modules"] if d["id"] == module)
    edge = scope["edges"][edge_index]
    result = inserted(project, edge["id"], module=module)
    graph = Graph.model_validate(result["graph"])
    before, after = validate(project.graph), validate(graph)
    assert before.ok and after.ok
    assert before.total_params == after.total_params
    assert semantic_hash(project.graph) != semantic_hash(graph)
    changed_scope = result["graph"] if module is None else next(d for d in result["graph"]["modules"] if d["id"] == module)
    assert changed_scope["nodes"][:-1] == scope["nodes"]
    assert [e for e in changed_scope["edges"] if e["id"] not in {x["id"] for x in result["replacement"]}] == [e for e in scope["edges"] if e["id"] != edge["id"]]
    if module:
        assert result["graph"]["nodes"] == original_json["nodes"]
        assert result["graph"]["edges"] == original_json["edges"]
        assert changed_scope["inputs"] == scope["inputs"] and changed_scope["outputs"] == scope["outputs"]
    torch.manual_seed(83)
    original, changed = lower_graph(project.graph), lower_graph(graph)
    for node, layer in original._modules.items():
        changed._modules[node].load_state_dict(layer.state_dict())
    inputs, labels = torch.randn(2, 3, 64, 64), torch.tensor([1, 2])
    optimizers = [torch.optim.SGD(m.parameters(), lr=.01) for m in (original, changed)]
    outputs = [m(inputs) for m in (original, changed)]
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    losses = [torch.nn.functional.cross_entropy(y, labels) for y in outputs]
    torch.testing.assert_close(*losses, rtol=0, atol=0)
    for loss in losses:
        loss.backward()
    for node, layer in original._modules.items():
        for a, b in zip(layer.parameters(), changed._modules[node].parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    for optimizer in optimizers:
        optimizer.step()
    torch.testing.assert_close(original(inputs), changed(inputs), rtol=0, atol=0)
    assert project.graph.to_json() == original_json


def test_additional_inputs_remain_disconnected_and_native_error_is_visible():
    project = load_project(ROOT/"examples/reference_cnn.project.json")
    result = inserted(project, project.graph.edges[0].id, "tensor.add", "a", "output")
    report = validate(Graph.model_validate(result["graph"]))
    assert not report.ok
    assert any(d.code == "E_MISSING_INPUT" and d.nodeId == result["id"] and d.port == "b" for d in report.errors)
    assert not any(e["to"] == dict(node=result["id"], port="b") for e in result["graph"]["edges"])


def test_real_tabular_fitted_state_wire_refuses_table_ports_before_mutation():
    project = load_project(ROOT/"examples/tabular_regression.project.json")
    edge = next(e for e in project.graph.edges if e.kind == "fit_state")
    result = inserted(project, edge.id, "tabular.select_columns", "table", "table")
    assert result["error"] == "E_INSERT_KIND"
