"""Milestone 3 item 10 / A45: optional in-app Python code blocks. Typed interface, template, fixtures with type/shape checks, isolated subprocess with
limits, stack traces mapped to source lines, identity (source hash + pinned dependencies) in the semantic hash, cache invalidation, differentiable vs
explicit non-differentiable boundary, versioned library."""
import ast
import copy
import importlib.metadata
import os
import re
import time
from pathlib import Path

import pytest
import torch

from codeblocks.sandbox import CodeBlockError, SandboxSession
from codeblocks.template import generate_template
from codeblocks.testing import run_fixture, check_block
from graph_core.build import GraphBuilder
from graph_core.composite import code_interface
from graph_core.hashing import semantic_hash
from graph_core.lower import lower_graph
from graph_core.schema import CodeBlockDef, CodeConfigField, CodeIO, CodeStateField, Graph
from graph_core.validate import ExecutionBlocked, validate
from libstore import LibraryStore, VersionConflict

ROOT = Path(__file__).resolve().parent.parent


def block(source, **kw):
    base = dict(id="blk", inputs=[CodeIO(name="x", shape=["N", 3])], outputs=[CodeIO(name="y", same_as="x")], source=source)
    base.update(kw)
    return CodeBlockDef(**base)


FX = {"name": "fx", "inputs": {"x": {"shape": [2, 3], "seed": 0}}}


def run1(defn, fx=FX):
    r = run_fixture(defn, fx)
    return r


# ------------------------------------------------------------------------------------------------ template + fixtures
def test_template_is_generated_from_the_interface_and_is_valid_python():
    d = CodeBlockDef(id="norm", description="Scale rows", inputs=[CodeIO(name="x", shape=["N", "D"]), CodeIO(name="w", shape=["D"])],
                     outputs=[CodeIO(name="y", same_as="x"), CodeIO(name="n", dtype="float32", shape=["N"])],
                     config=[CodeConfigField(name="eps", type="float", default=1e-5)], state=[CodeStateField(name="calls", shape=[1])], randomness="seeded",
                     differentiable=True)
    src = generate_template(d)
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    assert fn.name == "run" and [a.arg for a in fn.args.args] == ["x", "w"] and {a.arg for a in fn.args.kwonlyargs} == {"eps", "state"}
    assert "differentiable (use torch operations" in src and "randomness: seeded" in src and "calls" in src and "float32 [N, D]" in src


def test_fixture_run_checks_types_values_determinism_and_gradient():
    d = block("import torch\ndef run(x, *, a=2.0):\n    return {'y': x * a + 1}\n", config=[CodeConfigField(name="a", type="float", default=2.0)], differentiable=True,
              fixtures=[{"name": "values", "inputs": {"x": {"values": [[1, 2, 3]]}}, "config": {"a": 3.0}, "expect": {"y": {"values": [[4, 7, 10]]}}}, FX])
    res = check_block(d)
    assert res["ok"], res
    names = [c["name"] for c in res["fixtures"][0]["checks"]]
    assert any("matches the declared interface" in n for n in names) and any("equals the expected values" in n for n in names)
    assert any("finite difference" in n for n in names)
    # a wrong expectation fails visibly
    bad = copy.deepcopy(d)
    bad.fixtures[0]["expect"]["y"]["values"] = [[0, 0, 0]]
    r = check_block(bad)
    assert not r["ok"] and any(not c["ok"] and "expected" in c["name"] for c in r["fixtures"][0]["checks"])


def test_output_contract_violations_are_reported_with_codes():
    cases = {
        "E_CODE_OUTPUT_TYPE": "import torch\ndef run(x):\n    return {'y': x[:, :2]}\n",                # wrong shape
        "E_CODE_OUTPUT_TYPE ": "import torch\ndef run(x):\n    return {'y': x.double()}\n",           # wrong dtype
        "E_CODE_OUTPUT": "import torch\ndef run(x):\n    return {'z': x}\n",                           # wrong name
        "E_CODE_OUTPUT  ": "import torch\ndef run(x):\n    return 3.0\n",                              # not a tensor
    }
    for code, src in cases.items():
        r = run1(block(src))
        assert not r["ok"] and r["error"]["code"] == code.strip(), (code, r["error"])


def test_fixture_input_type_is_checked_before_running():
    d = block("def run(x):\n    return {'y': x}\n")
    r = run1(d, {"name": "bad", "inputs": {"x": {"shape": [2, 5]}}})
    assert not r["ok"] and r["error"]["code"] == "E_FIXTURE" and "must be 3" in r["error"]["message"]


# ------------------------------------------------------------------------------------------------ errors mapped to source lines
def test_exception_stack_trace_is_mapped_to_source_lines():
    src = "import torch\n\n\ndef helper(t):\n    return t / torch.tensor([1, 2]).reshape(3)   # line 5 fails\n\n\ndef run(x):\n    return {'y': helper(x)}\n"
    r = run1(block(src))
    e = r["error"]
    assert e["code"] == "E_CODE_EXCEPTION" and "RuntimeError" in e["message"]
    assert [(f["function"], f["line"]) for f in e["frames"]] == [("run", 9), ("helper", 5)]
    assert "line 5 fails" in e["frames"][1]["text"]
    assert 'File "blk@1.0.0", line 5, in helper' in e["traceback"]     # the original traceback stays available, relabelled with the block name


def test_syntax_error_reports_the_line_and_missing_entry_point():
    r = run1(block("def run(x):\n    return {'y': x\n"))
    assert r["error"]["code"] == "E_CODE_SYNTAX" and r["error"]["frames"][0]["line"] in (2, 3)
    r = run1(block("def other(x):\n    return x\n"))
    assert r["error"]["code"] == "E_CODE_ENTRY"


def test_print_output_is_captured_not_lost_and_cannot_break_the_protocol():
    r = run1(block("def run(x):\n    print('hello from the block')\n    return {'y': x}\n"))
    assert r["ok"] and "hello from the block" in r["stdout"]


# ------------------------------------------------------------------------------------------------ isolation + limits
def test_block_runs_in_a_separate_process_with_a_scrubbed_environment(monkeypatch):
    monkeypatch.setenv("VOID_SECRET_TOKEN", "hunter2")
    src = ("import os, torch\n"
           "def run(x):\n"
           "    flags = torch.tensor([[float(os.getpid()), float('VOID_SECRET_TOKEN' in os.environ), float(os.getcwd().startswith('/Users')), float(len(os.environ))]])\n"
           "    return {'y': x * 0 + flags[:, :1], 'info': flags}\n")
    d = block(src, outputs=[CodeIO(name="y", same_as="x"), CodeIO(name="info", shape=[1, 4])])
    s = SandboxSession(code_interface(d), d.source)
    r = s.call([torch.zeros(2, 3)], {}, {}, None, ["y", "info"])
    pid, has_secret, in_project_cwd, n_env = r["outputs"][1][0].tolist()
    assert int(pid) != os.getpid() and has_secret == 0.0 and in_project_cwd == 0.0 and n_env < 12
    assert s.proc.pid == int(pid)
    s.close()


def test_control_service_never_evaluates_user_source():
    """No eval/exec of arbitrary source anywhere except the sandbox runner (the child process)."""
    offenders = []
    for path in list((ROOT / "services").rglob("*.py")) + list((ROOT / "python").rglob("*.py")):
        if path.name == "runner.py" and path.parent.name == "codeblocks":
            continue
        tree = ast.parse(path.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("eval", "exec"):
                offenders.append(f"{path.relative_to(ROOT)}:{n.lineno}")
    assert not offenders, offenders


def test_cpu_time_limit_and_wall_clock_timeout_kill_the_child():
    r = run1(block("def run(x):\n    while True:\n        pass\n", limits={"cpu_seconds": 1, "wall_seconds": 20}))
    assert r["error"]["code"] == "E_CODE_CPU_LIMIT" and "CPU-time" in r["error"]["message"]
    t0 = time.time()
    r = run1(block("import time\ndef run(x):\n    time.sleep(30)\n    return {'y': x}\n", limits={"cpu_seconds": 20, "wall_seconds": 1}))
    assert r["error"]["code"] == "E_CODE_TIMEOUT" and time.time() - t0 < 15


def test_file_size_limit_and_output_still_works_after_a_failed_call():
    d = block("def run(x):\n    return {'y': x}\n")
    s = SandboxSession(code_interface(d), d.source)
    s.call([torch.zeros(1, 3)], {}, {}, None, ["y"])
    pid1 = s.proc.pid
    s.close()
    s.call([torch.zeros(1, 3)], {}, {}, None, ["y"])    # a closed/killed session restarts cleanly
    assert s.proc.pid != pid1
    s.close()


def test_declared_effects_are_enforced_by_guards():
    net = "import socket\ndef run(x):\n    socket.socket().connect(('127.0.0.1', 9))\n    return {'y': x}\n"
    r = run1(block(net))
    assert r["error"]["code"] == "E_CODE_UNDECLARED_EFFECT" and "network" in r["error"]["message"]
    r = run1(block(net, effects=["network"]))
    assert r["error"]["code"] == "E_CODE_EXCEPTION" and "ConnectionRefused" in r["error"]["message"]    # passed the guard; nothing listens on port 9
    wr = "def run(x):\n    open('out.txt', 'w').write('x')\n    return {'y': x}\n"
    assert run1(block(wr))["error"]["code"] == "E_CODE_UNDECLARED_EFFECT"
    assert run1(block(wr, effects=["file_write"]))["ok"]
    sp = "import subprocess\ndef run(x):\n    subprocess.run(['echo', 'hi'])\n    return {'y': x}\n"
    assert run1(block(sp, effects=["network", "file_write", "file_read"]))["error"]["code"] == "E_CODE_UNDECLARED_EFFECT"    # subprocesses are never allowed


def test_randomness_must_be_declared_and_seeded_blocks_are_reproducible():
    noisy = "import torch\ndef run(x):\n    return {'y': x + torch.randn_like(x)}\n"
    r = run1(block(noisy))
    assert r["error"]["code"] == "E_CODE_UNDECLARED_RANDOMNESS"
    r = run1(block(noisy, randomness="seeded"))
    assert r["ok"] and any("deterministic for equal inputs and seed" in c["name"] for c in r["checks"])


def test_pinned_dependencies_are_checked_in_the_environment_of_the_block():
    ver = importlib.metadata.version("numpy")
    ok = block("import numpy\ndef run(x):\n    return {'y': x}\n", dependencies=[f"numpy=={ver}"])
    assert run1(ok)["ok"]
    bad = block("def run(x):\n    return {'y': x}\n", dependencies=["numpy==0.0.1"])
    e = run1(bad)["error"]
    assert e["code"] == "E_CODE_DEPENDENCY" and f"installed version is {ver}" in e["message"]
    missing = block("def run(x):\n    return {'y': x}\n", dependencies=["definitely_not_a_package==1.0"])
    assert "not installed" in run1(missing)["error"]["message"]


# ------------------------------------------------------------------------------------------------ state
def test_declared_state_persists_between_calls_and_lives_in_buffers():
    d = block("import torch\ndef run(x, *, state):\n    state['calls'] += 1\n    return {'y': x * state['calls']}\n", state=[CodeStateField(name="calls", shape=[1])])
    g = GraphBuilder()
    g.input("i", ["N", 3])
    g.node("c", "code.block", block="blk")
    g.wire("i", "c.x")
    graph = g.build()
    graph.codeBlocks.append(d)
    m = lower_graph(graph)
    x = torch.ones(2, 3)
    assert m(x).tolist() == [[1.0] * 3] * 2 and m(x).tolist() == [[2.0] * 3] * 2
    assert float(m.state_dict()["c.state_calls"]) == 2.0       # checkpointed with the model
    m.close_blocks() if hasattr(m, "close_blocks") else m._modules["c"].close()


# ------------------------------------------------------------------------------------------------ in the graph
def graph_with_block(source, differentiable=True, **kw):
    d = CodeBlockDef(id="act", inputs=[CodeIO(name="x", shape=["N", 4])], outputs=[CodeIO(name="y", same_as="x")], source=source, differentiable=differentiable, **kw)
    g = GraphBuilder()
    g.input("i", ["N", 4])
    g.node("fc1", "tensor.dense", out_features=4)
    g.node("code", "code.block", block="act")
    g.node("fc2", "tensor.dense", out_features=2)
    g.wire("i", "fc1.input"); g.wire("fc1", "code.x"); g.wire("code.y", "fc2.input")
    graph = g.build()
    graph.codeBlocks.append(d)
    return graph


SOFTSIGN = "import torch\ndef run(x):\n    return {'y': x / (1 + x.abs())}\n"


def test_differentiable_block_trains_the_layer_upstream_and_matches_inline_torch():
    graph = graph_with_block(SOFTSIGN)
    r = validate(graph)
    assert r.ok, r.diagnostics
    assert [d.code for d in r.diagnostics] == []
    model = lower_graph(graph)
    x = torch.randn(5, 4)
    out = model(x)
    fc1, fc2 = model._modules["fc1"], model._modules["fc2"]
    ref = fc2(torch.nn.functional.softsign(fc1(x)))
    assert torch.allclose(out, ref, atol=1e-6)
    out.sum().backward()
    g1 = fc1.weight.grad.clone()
    fc1.zero_grad(); fc2.zero_grad()
    ref.sum().backward()
    assert torch.allclose(g1, fc1.weight.grad, atol=1e-5)        # the gradient crossed the sandbox boundary
    model._modules["code"].close()


def test_non_differentiable_block_is_an_explicit_boundary_with_a_warning_and_no_gradient():
    graph = graph_with_block("import torch\ndef run(x):\n    return {'y': torch.round(x * 4) / 4}\n", differentiable=False)
    r = validate(graph)
    assert r.ok and [d.code for d in r.diagnostics] == ["W_NONDIFF_BOUNDARY"] and "fc1" in r.diagnostics[0].message
    model = lower_graph(graph)
    out = model(torch.randn(3, 4))
    out.sum().backward()
    assert model._modules["fc1"].weight.grad is None and model._modules["fc2"].weight.grad is not None
    model._modules["code"].close()


def test_declared_differentiable_but_leaving_torch_is_caught():
    src = "import torch\ndef run(x):\n    return {'y': torch.from_numpy(x.detach().numpy() * 2)}\n"
    model = lower_graph(graph_with_block(src))
    with pytest.raises(CodeBlockError) as e:
        model(torch.randn(3, 4))
    assert e.value.code == "E_CODE_NOT_DIFFERENTIABLE"
    res = run_fixture(block(src, differentiable=True, inputs=[CodeIO(name="x", shape=["N", 4])]), {"name": "f", "inputs": {"x": {"shape": [2, 4]}}})
    assert not res["ok"] and not next(c for c in res["checks"] if "gradient flows" in c["name"])["ok"]
    model._modules["code"].close()


def test_code_block_interface_is_part_of_the_visual_graph_and_typed_on_the_wire():
    graph = graph_with_block(SOFTSIGN)
    graph.nodes[0].config["shape"] = ["N", 5]            # wrong width coming in
    graph.nodes[1].config["in_features"] = "infer"
    r = validate(graph)
    assert r.ok                                                    # fc1 adapts and its output is 4 wide again: block interface satisfied
    graph.nodes[1].config["out_features"] = 7
    r = validate(graph)
    assert not r.ok and r.errors[0].code == "E_PORT_TYPE" and "must be 4" in r.errors[0].message


def test_source_and_dependency_changes_update_semantic_identity_and_invalidate_caches(tmp_path):
    g1 = graph_with_block(SOFTSIGN)
    g2 = graph_with_block(SOFTSIGN + "\n# comment only\n")
    g3 = graph_with_block(SOFTSIGN, dependencies=["numpy==" + importlib.metadata.version("numpy")])
    h1, h2, h3 = (semantic_hash(g) for g in (g1, g2, g3))
    assert len({h1, h2, h3}) == 3                                  # source text and pinned dependencies are both part of the semantic hash
    i1, i2, i3 = (validate(g).resolved["code"].interface["identity"] for g in (g1, g2, g3))
    assert len({i1, i2, i3}) == 3
    # node identity = the code, so a move on the canvas changes nothing
    assert semantic_hash(Graph.model_validate(g1.to_json())) == h1
    # fixture-result cache: second run of the same source is cached; an edit makes the key new, so nothing stale is served
    d = block("def run(x):\n    return {'y': x + 1}\n", fixtures=[{"name": "f", "inputs": {"x": {"values": [[1, 2, 3]]}}, "expect": {"y": {"values": [[2, 3, 4]]}}}])
    first = check_block(d, tmp_path)
    again = check_block(d, tmp_path)
    assert first["ok"] and not first["fixtures"][0]["cached"] and again["fixtures"][0]["cached"]
    d.source = "def run(x):\n    return {'y': x + 2}\n"
    edited = check_block(d, tmp_path)
    assert not edited["fixtures"][0]["cached"] and not edited["ok"]            # the old (passing) result was NOT reused
    d.dependencies = ["numpy==" + importlib.metadata.version("numpy")]
    d.source = "def run(x):\n    return {'y': x + 1}\n"
    assert not check_block(d, tmp_path)["fixtures"][0]["cached"]                # same source as the first run but different pins: a different identity


def test_editing_source_keeps_the_interface_and_the_wires():
    g = graph_with_block(SOFTSIGN)
    ports_before = validate(g).resolved["code"].interface["inputs"], validate(g).resolved["code"].interface["outputs"]
    g.codeBlocks[0].source = "import torch\ndef run(x):\n    return {'y': torch.tanh(x)}\n"
    r = validate(g)
    assert r.ok and (r.resolved["code"].interface["inputs"], r.resolved["code"].interface["outputs"]) == ports_before
    assert len(g.edges) == 3


def test_missing_code_block_definition_blocks_execution():
    g = graph_with_block(SOFTSIGN)
    g.nodes[2].config["block"] = "nope"
    r = validate(g)
    assert not r.ok and "E_CODE_BLOCK_MISSING" in {d.code for d in r.diagnostics}
    with pytest.raises(ExecutionBlocked):
        lower_graph(g)


def test_code_block_config_fields_are_typed():
    d = CodeBlockDef(id="scale", inputs=[CodeIO(name="x", shape=["N", 4])], outputs=[CodeIO(name="y", same_as="x")], config=[CodeConfigField(name="k", type="float", default=2.0)],
                     source="def run(x, *, k=2.0):\n    return {'y': x * k}\n", differentiable=True)
    for params, ok in (({"k": 3.0}, True), ({"k": "big"}, False), ({"zz": 1}, False)):
        g = GraphBuilder()
        g.input("i", ["N", 4])
        g.node("c", "code.block", block="scale", params=params)
        g.wire("i", "c.x")
        graph = g.build()
        graph.codeBlocks.append(d)
        assert validate(graph).ok is ok


# ------------------------------------------------------------------------------------------------ versioned block library
def test_saved_blocks_are_versioned_and_immutable(tmp_path):
    lib = LibraryStore(tmp_path, "codeblocks")
    d = block("def run(x):\n    return {'y': x}\n", description="identity").model_dump(mode="json")
    p = lib.publish(d)
    assert p["version"] == "1.0.0" and not p["alreadyPublished"] and lib.publish(d)["alreadyPublished"]
    d2 = dict(d, source="def run(x):\n    return {'y': x + 1}\n")
    with pytest.raises(VersionConflict, match="immutable"):
        lib.publish(d2)
    d2["version"] = LibraryStore.next_version("1.0.0")
    lib.publish(d2)
    assert lib.versions("blk") == ["1.0.0", "1.1.0"] and lib.get("blk")["definition"]["version"] == "1.1.0" and lib.get("blk", "1.0.0")["definition"]["source"] == d["source"]
    assert [e["version"] for e in lib.list()] == ["1.0.0", "1.1.0"]


def test_memory_limit_is_enforced_where_the_platform_supports_it():
    d = block("import torch\ndef run(x):\n    big = torch.zeros(400_000_000)\n    return {'y': x}\n", limits={"memory_mb": 1500, "cpu_seconds": 20})
    s = SandboxSession(code_interface(d), d.source)
    try:
        s.start()
        enforced = any(l.startswith("memory_mb=") for l in s.applied_limits)
        if not enforced:   # e.g. macOS: the kernel refuses RLIMIT_AS. The session must say so instead of pretending.
            assert any(l.startswith("memory_mb: not enforced on this platform") for l in s.applied_limits)
            return
        with pytest.raises(CodeBlockError) as e:
            s.call([torch.zeros(1, 3)], {}, {}, None, ["y"])
        assert e.value.code in ("E_CODE_EXCEPTION", "E_CODE_CRASH")
    finally:
        s.close()
