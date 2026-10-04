"""Backends for model graphs: PyTorch (reference), TensorFlow/Keras 3 and JAX (portable subset), compatibility reports, native code export.

    compat_report(graph, backend)        what runs, what is converted, what is unsupported (before anything executes)
    compile_graph(graph, backend)        an Executable (forward / loss / gradients / one SGD step) in graph layout; refuses incompatible graphs
    export_code(graph, backend)          deterministic readable native source with `# node:` comments
"""
from __future__ import annotations

from .base import BackendError, Executable, ForwardResult, Params
from .compat import CompatReport, compat_report, require_compatible
from .registry import BACKEND_IDS, INFO, availability, describe

__all__ = ["BackendError", "Executable", "ForwardResult", "Params", "CompatReport", "compat_report", "require_compatible", "BACKEND_IDS", "INFO",
           "availability", "describe", "compile_graph", "export_code"]


def _validated(graph, backend):
    from graph_core.validate import validate

    g = graph.model_copy(deep=True)
    g.backend = backend
    return g, validate(g)


def compile_graph(graph, backend: str, seed: int = 0, compiled: bool = True) -> Executable:
    """Refuses (BackendError with the compatibility report attached) unless every node is supported on `backend` and the library runs here.
    `compiled`: jax.jit / tf.function (the benchmark reports compile time separately); eager when False."""
    from .plan import make_plan
    from graph_core.hashing import semantic_hash

    require_compatible(graph, backend)
    g, report = _validated(graph, backend)
    if backend == "pytorch":
        from .torch_runtime import TorchExecutable

        return TorchExecutable(g, report, seed)
    plan = make_plan(report, semantic_hash(graph))
    if backend == "keras":
        from .keras_runtime import KerasExecutable

        return KerasExecutable(plan, seed, compiled)
    from .jax_runtime import JaxExecutable

    return JaxExecutable(plan, seed, compiled)


def export_code(graph, backend: str) -> str:
    """Readable native source for `backend`. Raises BackendError unless the graph is compatible (unsupported nodes are never skipped).
    Does not need the framework installed."""
    from graph_core.hashing import semantic_hash

    from .compat import compat_report
    from .plan import make_plan

    rep = compat_report(graph, backend)
    if not rep.ok:
        require_compatible_static(rep, backend)
    if backend == "pytorch":
        from graph_core.codegen import generate_pytorch

        g = graph.model_copy(deep=True)
        g.backend = "pytorch"
        return generate_pytorch(g)
    g, report = _validated(graph, backend)
    plan = make_plan(report, semantic_hash(graph))
    if backend == "keras":
        from . import keras_spec

        return keras_spec.generate_keras(plan, keras_spec.analyze(plan))
    from . import jax_spec

    return jax_spec.generate_jax(plan, jax_spec.analyze(plan))


def require_compatible_static(rep: CompatReport, backend: str):
    bad = [f"{n} ({v['type']}): {v['code']} {v['reason']}" for n, v in rep.nodes.items() if v["status"] == "unsupported"]
    bad += [f"{d.get('nodeId')}: {d['code']} {d['message']}" for d in rep.structural]
    bad += [f"{n}: could not be typed" for n in rep.unchecked if not any(d.get("nodeId") == n for d in rep.structural)]
    raise BackendError("E_BACKEND_INCOMPATIBLE", f"graph cannot be exported for '{backend}':\n  " + "\n  ".join(bad), rep)
