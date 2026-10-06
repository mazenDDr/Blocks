"""Serve a PyTorch-trained image classifier through Keras 3 or JAX (ADR 0061).

Keras/JAX have no worker training runs; the portable subset (ADR 0028/0029) runs the same lowered graph on those backends with
PyTorch-layout parameters converted by declared rules. A portable version starts from an image-classifier version's pinned
manifest (ADR 0018: checkpoint, preprocessing, classes, frozen reference images), loads the checkpoint into the PyTorch
lowering, copies its parameters into the Keras/JAX executable and, at registration, measures the logits of both on every
frozen reference image. Registration refuses unless every logit agrees within the declared float32 forward tolerance
(tolerances.py) and every predicted class matches. Requests and outputs are exactly those of the PyTorch image adapter; only
the forward pass runs on the other backend. The model adapter's source is part of registered identities, so this lives here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np

from backends import compile_graph
from backends.tolerances import tol
from . import model_adapter
from .model_adapter import ModelGraphPipeline, _graph
from .pipeline import ProductionError, read_verified

BACKENDS = ("keras", "jax")
FILES = ("production/portable_model_adapter.py", "backends/__init__.py", "backends/base.py", "backends/plan.py", "backends/params.py",
         "backends/tolerances.py", "backends/compat.py", "backends/keras_runtime.py", "backends/keras_spec.py",
         "backends/jax_runtime.py", "backends/jax_spec.py", "backends/torch_runtime.py")


def backend_versions():
    """Installed versions of the frameworks that run the portable forward pass (not the whole requirements file)."""
    from importlib.metadata import version
    return {name: version(name) for name in ("keras", "tensorflow", "jax", "jaxlib")}


def implementation():
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES}


def split(node: str):
    backend, _, out = node.partition(":")
    if backend not in BACKENDS or not out:
        raise ProductionError("E_PORTABLE_NODE", "Register a portable version with node 'keras:<output node>' or 'jax:<output node>'.")
    return backend, out


def _executables(store, manifest, backend):
    """The run's PyTorch checkpoint copied into the other backend's executable (graph-layout parameters)."""
    import io

    from worker.train import load_checkpoint

    g, _ = _graph(store, manifest["runId"])
    torch_exec = compile_graph(g, "pytorch", compiled=False)
    state = load_checkpoint(io.BytesIO(read_verified(store, manifest["checkpointSha256"])))
    torch_exec.model.load_state_dict(state["model"], strict=True)
    torch_exec.model.eval()
    other = compile_graph(g, backend, compiled=True)
    other.set_params(torch_exec.get_params())
    return other


def candidates(store, row):
    out = model_adapter.is_candidate(store, row)
    return [f"{b}:{out}" for b in BACKENDS] if out else []


def build_manifest(store, run_id, node):
    backend, out = split(node)
    base = model_adapter.build_manifest(store, run_id, out)
    pipe = PortablePipeline(store, {**base, "portable": {"backend": backend}}, measuring=True)
    reference = pipe.reference_records()
    if not reference:
        raise ProductionError("E_PORTABLE_REFERENCE", "The version has no frozen reference images to measure agreement on.")
    torch_pipe = ModelGraphPipeline(store, base)
    expected = np.array(torch_pipe.predict(reference)[0]["logits"], dtype="float64")
    got = np.array(pipe.predict(reference)[0]["logits"], dtype="float64")
    t = tol(backend, "float32", "forward")
    excess = np.abs(got - expected) - (t.atol + t.rtol * np.abs(expected))
    agree = bool((got.argmax(1) == expected.argmax(1)).all())
    measured = {"images": len(reference), "maxAbsDiff": float(np.abs(got - expected).max()), "maxToleranceExcess": float(excess.max()),
                "declared": {"atol": t.atol, "rtol": t.rtol, "rule": "|backend - pytorch| <= atol + rtol*|pytorch| for every logit"},
                "predictedClassAgreement": agree}
    if excess.max() > 0 or not agree:
        raise ProductionError("E_PORTABLE_TOLERANCE", f"{backend} logits differ from PyTorch beyond the declared tolerance on the frozen reference: {measured}.", 409)
    return {**base, "adapter": f"portable-{backend}-local", "portable": {"backend": backend, "source": "PyTorch checkpoint parameters copied in graph layout",
            "versions": backend_versions(), "implementation": implementation(), "measuredAgainstPyTorch": measured}}


def verify(store, manifest):
    model_adapter.verify(store, manifest)
    p = manifest["portable"]
    if p["versions"] != backend_versions() or p["implementation"] != implementation():
        raise ProductionError("E_SERVING_ENVIRONMENT", "Pinned Keras/JAX versions or portable implementation changed; register a new version.", 409)


class PortablePipeline(ModelGraphPipeline):
    """The PyTorch image adapter's request/response contract with the forward pass on Keras or JAX."""

    def __init__(self, store, manifest, measuring=False):
        super().__init__(store, manifest)
        if not measuring:
            verify(store, manifest)
        g, _ = _graph(store, manifest["runId"])
        self.backend = manifest["portable"]["backend"]
        self.exec = _executables(store, manifest, self.backend)
        self.input_id, self.output_id = self.exec.input_ids[0], self.exec.output_ids[0]
        del self.model  # nothing below may fall back to the PyTorch forward pass

    def predict(self, records):
        import torch

        t0 = time.perf_counter()
        x = torch.stack([self._tensor(r["imagePng"]) for r in records]).numpy()
        t1 = time.perf_counter()
        with self.lock:
            logits = np.asarray(self.exec.forward({self.input_id: x}).outputs[self.output_id], dtype="float64")
        t2 = time.perf_counter()
        z = logits - logits.max(1, keepdims=True)
        probs = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        classes = self.manifest["outputSchema"]["classes"]
        result = {"predictions": [classes[i] for i in probs.argmax(1).tolist()], "probabilities": probs.tolist(), "logits": logits.tolist(),
                  "classes": classes, "backend": self.backend}
        return result, {"preprocessingMs": (t1 - t0) * 1000, "inferenceMs": (t2 - t1) * 1000, "postprocessingMs": (time.perf_counter() - t2) * 1000,
                        "inputShape": list(x.shape)}
