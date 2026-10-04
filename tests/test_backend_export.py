"""Native code export for the new backends: deterministic, readable, `# node:` source map; the exported code is executed in a SEPARATE Python process
(no project imports) and must reproduce the outputs of the graph executed in-process."""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import backends
from backend_helpers import EXAMPLES, cmp, cnn_with_loss, skip_unless_available
from backends import BackendError
from backends.tolerances import tol
from graph_core.build import GraphBuilder
from graph_core.codegen import source_map
from graph_core.project_io import load_project
from backends.workloads import shared_graph


@pytest.mark.parametrize("backend", ["pytorch", "keras", "jax"])
def test_export_is_deterministic_and_maps_every_node(backend):
    g = cnn_with_loss()
    a, b = backends.export_code(g, backend), backends.export_code(g, backend)
    assert a == b and a.startswith("# Generated from graph ")
    sm = source_map(a)
    assert {n.id for n in g.nodes} <= set(sm), set(n.id for n in g.nodes) - set(sm)
    compile(a, f"<{backend}-export>", "exec")  # valid Python
    assert "eval(" not in a and "exec(" not in a


def test_exports_do_not_need_the_framework_installed_to_be_generated(monkeypatch):
    """Code generation is pure Python: it works even when the framework cannot be imported."""
    import builtins

    real = builtins.__import__

    def deny(name, *a, **k):
        if name.split(".")[0] in ("tensorflow", "keras", "jax", "jaxlib"):
            raise ImportError(f"{name} blocked for this test")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", deny)
    assert "tf.transpose" in backends.export_code(cnn_with_loss(), "keras")
    assert "lax.conv_general_dilated" in backends.export_code(cnn_with_loss(), "jax")


def test_keras_export_marks_every_layout_and_padding_conversion():
    code = backends.export_code(cnn_with_loss(), "keras")
    assert "(layout: NCHW -> NHWC)" in code and "(padding: explicit zeros)" in code and "weight_layout: OIHW -> HWIO" in code
    assert "images__nhwc = tf.transpose(images, [0, 2, 3, 1])  # node: conv_1" in code
    assert "Initialization:" in code and "differ" in code  # the init difference is stated in the file itself


def test_name_collisions_with_generated_code_are_refused_not_renamed():
    b = GraphBuilder()
    b.input("x", ["N", 4])
    b.node("tf", "pytorch.nn.linear", out_features=2)
    b.chain("x", "tf")
    with pytest.raises(BackendError) as ei:
        backends.export_code(b.build(), "keras")
    assert ei.value.code == "E_BACKEND_NAME_COLLISION"


HARNESS = {
    "keras": textwrap.dedent('''
        import sys
        import numpy as np
        sys.path.insert(0, ".")
        import exported
        d = np.load("data.npz")
        params = {}
        for k in d.files:
            if k.startswith("p/"):
                _, nid, name = k.split("/", 2)
                params.setdefault(nid, {})[name] = d[k]
        m = exported.Model(seed=0)
        m.set_graph_weights(params)
        out = m(*[d["x/" + i] for i in INPUTS])
        out = out if isinstance(out, tuple) else (out,)
        np.savez("out.npz", *[o.numpy() for o in out])
        assert not any(mod.startswith(("graph_core", "backends", "operations")) for mod in sys.modules), "exported code must not import the project"
    '''),
    "jax": textwrap.dedent('''
        import sys
        import numpy as np
        sys.path.insert(0, ".")
        import exported
        d = np.load("data.npz")
        init = exported.init_params(0)
        np.savez("init.npz", **{f"{n}/{k}": np.asarray(v) for n, ps in init.items() for k, v in ps.items()})
        params = {}
        for k in d.files:
            if k.startswith("p/"):
                _, nid, name = k.split("/", 2)
                params.setdefault(nid, {})[name] = d[k]
        out = exported.apply(params, *[d["x/" + i] for i in INPUTS])
        out = out if isinstance(out, tuple) else (out,)
        np.savez("out.npz", *[np.asarray(o) for o in out])
        assert not any(mod.startswith(("graph_core", "backends", "operations")) for mod in sys.modules), "exported code must not import the project"
    '''),
}


def run_exported(tmp_path, graph, backend, inputs, params):
    (tmp_path / "exported.py").write_text(backends.export_code(graph, backend))
    ids = sorted(inputs)
    (tmp_path / "harness.py").write_text(f"INPUTS = {ids!r}\n" + HARNESS[backend])
    np.savez(tmp_path / "data.npz", **{f"x/{k}": v for k, v in inputs.items()}, **{f"p/{n}/{k}": v for n, ps in params.items() for k, v in ps.items()})
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    p = subprocess.run([sys.executable, "harness.py"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-3000:]
    out = np.load(tmp_path / "out.npz")
    return [out[f] for f in sorted(out.files, key=lambda s: int(s.split("_")[1]))], tmp_path


@pytest.mark.parametrize("backend", ["keras", "jax"])
def test_exported_code_runs_in_a_subprocess_and_matches_the_graph(backend, tmp_path):
    skip_unless_available(backend)
    g = cnn_with_loss()
    ref = backends.compile_graph(g, "pytorch", seed=4)
    rng = np.random.default_rng(4)
    inputs = {"images": rng.standard_normal((3, 3, 64, 64)).astype(np.float32), "labels": rng.integers(0, 10, (3,)).astype(np.int64)}
    params = ref.get_params()
    (loss,), d = run_exported(tmp_path, g, backend, inputs, params)
    cmp(loss, ref.forward(inputs).outputs["ce"], tol(backend, "float32", "loss"), "exported loss vs the PyTorch graph")
    ex = backends.compile_graph(g, backend, compiled=False)  # and vs the same backend executed in-process
    ex.set_params(params)
    cmp(loss, ex.forward(inputs).outputs["ce"], tol(backend, "float32", "loss"), "exported loss vs in-process")
    if backend == "jax":  # same key splitting as the runtime: the exported init_params(0) IS the runtime's seed-0 initialization
        init = np.load(d / "init.npz")
        for n, ps in backends.compile_graph(g, "jax", seed=0, compiled=False).get_params().items():
            for k, v in ps.items():
                assert np.array_equal(init[f"{n}/{k}"], v), (n, k)


@pytest.mark.parametrize("backend", ["keras", "jax"])
def test_exported_code_without_a_loss_returns_the_logits(backend, tmp_path):
    skip_unless_available(backend)
    g = load_project(EXAMPLES / "reference_cnn.project.json").graph
    ref = backends.compile_graph(g, "pytorch", seed=1)
    x = np.random.default_rng(1).standard_normal((2, 3, 64, 64)).astype(np.float32)
    (out,), _ = run_exported(tmp_path, g, backend, {"images": x}, ref.get_params())
    cmp(out, ref.forward({"images": x}).outputs["fc"], tol(backend, "float32", "forward"), "exported logits")


@pytest.mark.parametrize("backend", ["keras", "jax"])
def test_exported_shared_parameters_run_and_match(backend, tmp_path):
    skip_unless_available(backend)
    g = shared_graph()
    ref = backends.compile_graph(g, "pytorch", seed=2)
    rng = np.random.default_rng(2)
    inputs = {"u": rng.standard_normal((2, 2, 6, 6)).astype(np.float32), "v": rng.standard_normal((2, 2, 6, 6)).astype(np.float32)}
    (loss,), _ = run_exported(tmp_path, g, backend, inputs, ref.get_params())
    cmp(loss, ref.forward(inputs).outputs["loss"], tol(backend, "float32", "loss"), "shared export")
