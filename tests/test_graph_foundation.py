"""A01, A02, A03, A07, A08 plus validator, hash, codegen and registry behaviour."""
import ast
import json

import pytest

from conftest import EXAMPLES, set_config
from graph_core import registry
from graph_core.codegen import generate_pytorch, source_map
from graph_core.hashing import semantic_hash
from graph_core.lower import lower_graph
from graph_core.project_io import Project, load_project, save_project
from graph_core.schema import Graph, Node, UiDoc
from graph_core.validate import ExecutionBlocked, validate

TABLE = {  # VISION 8.1
    "images": [None, 3, 64, 64],
    "conv_1": [None, 32, 64, 64],
    "relu_1": [None, 32, 64, 64],
    "pool_1": [None, 32, 32, 32],
    "conv_2": [None, 64, 32, 32],
    "relu_2": [None, 64, 32, 32],
    "pool_2": [None, 64, 16, 16],
    "gap": [None, 64, 1, 1],
    "flatten": [None, 64],
    "fc": [None, 10],
}


def out_shape(report, node):
    ports = report.output_types[node]
    return list(next(iter(ports.values())).shape)


def test_a01_reference_cnn_shapes_and_params(cnn_graph):
    r = validate(cnn_graph)
    assert r.ok, r.diagnostics
    for node, expected in TABLE.items():
        assert out_shape(r, node) == ["N"] + expected[1:], node
    assert r.params["conv_1"] == 896
    assert r.params["conv_2"] == 18_496
    assert r.params["fc"] == 650
    assert r.total_params == 20_042
    assert sum(v for k, v in r.params.items() if k not in ("conv_1", "conv_2", "fc")) == 0
    # the formula agrees with the real lowered module
    model = lower_graph(cnn_graph)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 20_042


def test_a02_change_filters_propagates_and_locked_channel_errors(cnn_graph):
    set_config(cnn_graph, "conv_1", out_channels=64)
    r = validate(cnn_graph)
    assert r.ok
    assert r.resolved["conv_2"].in_channels == 64  # inferred
    assert out_shape(r, "conv_1") == ["N", 64, 64, 64]
    assert r.params["conv_1"] == 64 * 3 * 9 + 64 == 1792
    assert r.params["conv_2"] == 64 * 64 * 9 + 64 == 36_928
    assert r.params["fc"] == 650
    assert r.total_params == 1792 + 36_928 + 650

    # conv_2 locked at 32 -> mismatch on conv_2/input
    set_config(cnn_graph, "conv_2", in_channels=32)
    r = validate(cnn_graph)
    assert not r.ok
    [d] = [d for d in r.errors if d.code == "E_CHANNEL_MISMATCH"]
    assert (d.nodeId, d.port) == ("conv_2", "input")
    assert d.path == "/nodes/conv_2/ports/input"
    assert "32" in d.message and "64" in d.message
    assert {f.value for f in d.fixes} == {"infer", 64}
    assert all(f.node == "conv_2" and f.key == "in_channels" for f in d.fixes)
    # nothing downstream is reported as a second, spurious error
    assert [x.code for x in r.errors] == ["E_CHANNEL_MISMATCH"]


def test_a02_locked_matching_value_is_fine(cnn_graph):
    set_config(cnn_graph, "conv_2", in_channels=32)
    assert validate(cnn_graph).ok


def test_a03_incompatible_channels_reported_before_training(cnn_graph):
    set_config(cnn_graph, "conv_1", in_channels=1)  # images have 3 channels
    r = validate(cnn_graph)
    [d] = r.errors
    assert (d.code, d.nodeId, d.port) == ("E_CHANNEL_MISMATCH", "conv_1", "input")
    with pytest.raises(ExecutionBlocked) as e:
        lower_graph(cnn_graph)
    assert e.value.diagnostics[0].nodeId == "conv_1"


def test_linear_locked_features_mismatch(cnn_graph):
    set_config(cnn_graph, "fc", in_features=128)
    [d] = validate(cnn_graph).errors
    assert (d.code, d.nodeId, d.port) == ("E_FEATURE_MISMATCH", "fc", "input")


def test_a07_save_load_round_trip_preserves_hash_and_unknown_ops(cnn_project, tmp_path):
    g = cnn_project.graph
    g.nodes.append(Node.model_validate(
        {"id": "mystery", "type": "plugin.acme.fancy", "version": "9.1.0", "config": {"a": [1, 2], "b": {"c": None}}, "x_future": 7}))
    h = semantic_hash(g)
    save_project(Project(g, cnn_project.ui), tmp_path / "p.project.json")
    loaded = load_project(tmp_path / "p.project.json")
    assert semantic_hash(loaded.graph) == h
    assert loaded.graph.to_json() == g.to_json()
    m = loaded.graph.node("mystery")
    assert m.type == "plugin.acme.fancy" and m.config == {"a": [1, 2], "b": {"c": None}} and m.model_extra == {"x_future": 7}
    assert loaded.ui.positions == cnn_project.ui.positions
    # readable, but execution is blocked with an actionable message
    [d] = [d for d in validate(loaded.graph).errors if d.code == "E_UNKNOWN_OP"]
    assert d.nodeId == "mystery" and d.fixes
    with pytest.raises(ExecutionBlocked):
        lower_graph(loaded.graph)
    # second save is byte-identical
    save_project(loaded, tmp_path / "q.project.json")
    assert (tmp_path / "p.project.json").read_bytes() == (tmp_path / "q.project.json").read_bytes()


def test_a07_example_files_round_trip(tmp_path):
    for name in ("reference_cnn", "mse_teaching"):
        p = load_project(EXAMPLES / f"{name}.project.json")
        save_project(p, tmp_path / f"{name}.project.json")
        assert json.loads((tmp_path / f"{name}.project.json").read_text()) == json.loads((EXAMPLES / f"{name}.project.json").read_text())


def test_a08_moving_nodes_does_not_change_hash(cnn_project, tmp_path):
    h = semantic_hash(cnn_project.graph)
    moved = UiDoc(positions={k: {"x": v["x"] + 500, "y": v["y"] - 33} for k, v in cnn_project.ui.positions.items()})
    save_project(Project(cnn_project.graph, moved), tmp_path / "m.project.json")
    assert semantic_hash(load_project(tmp_path / "m.project.json").graph) == h
    assert moved.to_json() != cnn_project.ui.to_json()


def test_hash_tracks_semantics_not_declaration_order(cnn_graph):
    h = semantic_hash(cnn_graph)
    shuffled = Graph.model_validate(cnn_graph.to_json())
    shuffled.nodes.reverse()
    shuffled.edges.reverse()
    assert semantic_hash(shuffled) == h
    set_config(shuffled, "conv_1", out_channels=33)
    assert semantic_hash(shuffled) != h


def g_from(nodes, edges):
    return Graph.model_validate({"nodes": nodes, "edges": edges})


def n(id, type, **config):
    return {"id": id, "type": type, "config": config}


def e(id, a, b, to_port="input", from_port="output"):
    return {"id": id, "from": {"node": a, "port": from_port}, "to": {"node": b, "port": to_port}}


IN = n("x", "core.tensor_input", shape=["N", 4], dtype="float32", layout="NC")


def codes(g):
    return {(d.code, d.nodeId, d.port) for d in validate(g).diagnostics}


def test_validator_missing_input_and_cycle_and_dangling():
    assert ("E_MISSING_INPUT", "r", "input") in codes(g_from([n("r", "pytorch.nn.relu")], []))
    cyc = g_from([n("a", "pytorch.nn.relu"), n("b", "pytorch.nn.relu")], [e("1", "a", "b"), e("2", "b", "a")])
    assert any(c[0] == "E_CYCLE" for c in codes(cyc))
    dang = g_from([IN], [e("1", "x", "ghost", from_port="value")])
    assert any(c[0] == "E_DANGLING_EDGE" for c in codes(dang))


def test_validator_ports_types_config_ids():
    base = [IN, n("l", "pytorch.nn.linear", out_features=3)]
    assert ("E_UNKNOWN_PORT", "l", "bogus") in codes(g_from(base, [e("1", "x", "l", "bogus", "value")]))
    assert ("E_UNKNOWN_PORT", "x", "output") in codes(g_from(base, [e("1", "x", "l")]))
    two = base + [n("y", "core.tensor_input", shape=["N", 4])]
    assert any(c[0] == "E_MULTIPLE_INPUTS" for c in codes(g_from(two, [e("1", "x", "l", from_port="value"), e("2", "y", "l", from_port="value")])))
    assert any(c[0] == "E_CONFIG" and c[1] == "l" for c in codes(g_from([IN, n("l", "pytorch.nn.linear", out_features=-1)], [e("1", "x", "l", from_port="value")])))
    assert any(c[0] == "E_CONFIG" for c in codes(g_from([n("l", "pytorch.nn.linear", typo=1)], [])))
    assert any(c[0] == "E_BAD_ID" for c in codes(g_from([n("forward", "pytorch.nn.relu"), n("1bad", "pytorch.nn.relu")], [])))
    assert any(c[0] == "E_DUPLICATE_ID" for c in codes(g_from([IN, IN], [])))
    bad_ver = {"id": "r", "type": "pytorch.nn.relu", "version": "2.0.0"}
    assert any(c[0] == "E_UNSUPPORTED_VERSION" for c in codes(g_from([bad_ver], [])))


def test_validator_port_type_dtype_and_shapes():
    ce = n("ce", "pytorch.loss.cross_entropy")
    lg = n("z", "core.tensor_input", shape=["N", 5], dtype="float32")
    bad_t = n("t", "core.tensor_input", shape=["N"], dtype="float32")  # CE needs int64 labels
    c = codes(g_from([ce, lg, bad_t], [e("1", "z", "ce", "logits", "value"), e("2", "t", "ce", "target", "value")]))
    assert ("E_PORT_TYPE", "ce", "target") in c
    sub = g_from([n("a", "core.tensor_input", shape=[3]), n("b", "core.tensor_input", shape=[4]), n("s", "core.sub")],
                 [e("1", "a", "s", "a", "value"), e("2", "b", "s", "b", "value")])
    assert ("E_SHAPE_MISMATCH", "s", "b") in codes(sub)
    pool_on_vec = g_from([n("a", "core.tensor_input", shape=["N", 3]), n("p", "pytorch.nn.max_pool2d")], [e("1", "a", "p", from_port="value")])
    assert ("E_RANK_MISMATCH", "p", "input") in codes(pool_on_vec)


def test_downstream_of_error_not_reported_twice(cnn_graph):
    set_config(cnn_graph, "conv_1", kernel_size=[99, 99])  # does not fit 64x64
    r = validate(cnn_graph)
    assert [d.code for d in r.errors] == ["E_SHAPE_INVALID"]
    assert "conv_2" in r.unresolved_nodes and not r.ok


def test_every_op_declares_explain_and_param_formula(cnn_graph):
    r = validate(cnn_graph)
    types = {n.id: n.type for n in cnn_graph.nodes}
    for nid, t in types.items():
        op = registry.get_op(t)
        ex = op.explain(r.resolved[nid], r.input_types[nid], r.output_types[nid])
        assert ex["equation"] and {"formula", "terms", "total"} <= ex["parameters"].keys()
        assert ex["parameters"]["total"] == r.params[nid]
    conv = registry.get_op("pytorch.nn.conv2d").explain(r.resolved["conv_1"], {}, {})
    assert [t["count"] for t in conv["parameters"]["terms"]] == [864, 32]


def test_registry_has_planned_ops():
    assert {o.type for o in registry.all_ops() if o.graph_kind == "model"} == {  # tabular ops are listed separately (test_tabular_*)
        "core.tensor_input", "pytorch.nn.conv2d", "pytorch.nn.relu", "pytorch.nn.max_pool2d", "pytorch.nn.adaptive_avg_pool2d",
        "pytorch.nn.flatten", "pytorch.nn.linear", "pytorch.loss.cross_entropy", "core.sub", "core.square", "core.mean", "core.sum",
        "core.scalar_mul", "core.add"}


def test_conv_full_config_shape_and_params_match_torch():
    import torch
    g = g_from([n("x", "core.tensor_input", shape=["N", 4, 17, 15]),
                n("c", "pytorch.nn.conv2d", out_channels=6, kernel_size=[3, 5], stride=[2, 1], padding=[1, 2], dilation=[2, 1],
                  groups=2, bias=False, padding_mode="reflect")],
               [e("1", "x", "c", from_port="value")])
    r = validate(g)
    assert r.ok, r.diagnostics
    m = lower_graph(g)
    y = m(torch.zeros(2, 4, 17, 15))
    assert list(y.shape) == [2] + list(r.output_types["c"]["output"].shape[1:])
    assert r.params["c"] == sum(p.numel() for p in m.parameters())


def test_lowered_submodules_named_by_node_id(cnn_graph):
    m = lower_graph(cnn_graph)
    assert set(dict(m.named_children())) == set(TABLE)
    assert {k.split(".")[0] for k in m.state_dict()} == {"conv_1", "conv_2", "fc"}


def test_codegen_deterministic_readable_with_source_map(cnn_graph):
    code = generate_pytorch(cnn_graph)
    assert code == generate_pytorch(Graph.model_validate(cnn_graph.to_json()))
    ast.parse(code)  # syntactically valid Python (parsed, never executed)
    assert "self.conv_1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)" in code
    assert "self.conv_2 = nn.Conv2d(32, 64," in code  # inferred in_channels resolved in the export
    assert "self.fc = nn.Linear(64, 10, bias=True)  # node: fc" in code
    sm = source_map(code)
    assert set(sm) == set(TABLE)
    assert len(sm["conv_1"]) == 2  # constructor line + forward line
    line = code.splitlines()[sm["pool_1"][-1] - 1]
    assert "pool_1 = self.pool_1(relu_1)" in line
    # reversed declaration order generates identical text
    rev = Graph.model_validate(cnn_graph.to_json())
    rev.nodes.reverse()
    assert generate_pytorch(rev) == code


def test_codegen_refuses_non_executable_graph(cnn_graph):
    set_config(cnn_graph, "conv_1", in_channels=1)
    with pytest.raises(ExecutionBlocked):
        generate_pytorch(cnn_graph)
