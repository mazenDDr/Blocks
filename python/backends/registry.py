"""The backends this project knows, their roles (VISION 14.1), pinned libraries, and whether each is actually usable in this environment.

Availability is probed by really importing and exercising the library once (a tiny convolution). A backend that cannot run is reported
`unavailable` with the exact reason; nothing is faked and its tests skip with that reason."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .jax_spec import INIT as JAX_INIT
from .keras_spec import INIT as KERAS_INIT

BACKEND_IDS = ("pytorch", "keras", "jax")


@dataclass(frozen=True)
class BackendInfo:
    id: str
    title: str
    role: str
    packages: tuple[str, ...]  # pinned distribution names (versions are read from python/requirements.txt)
    layout: str
    training: str
    init: str
    facets: dict[str, str] = field(default_factory=dict)


def _facets_common(**kw: str) -> dict[str, str]:
    return kw


INFO: dict[str, BackendInfo] = {
    "pytorch": BackendInfo(
        "pytorch", "PyTorch", "First deep-learning execution backend: modules, parameters, autograd, training, checkpoints.", ("torch",),
        "graph layout is PyTorch's own (NCHW, OIHW)", "Full: worker runs, checkpoints, events (Milestones 0-5).",
        "PyTorch module defaults (Kaiming-uniform kernels, uniform biases).",
        _facets_common(
            layout="NCHW activations, OIHW weights: no conversions.", dtype="float32 and float64 for the ops that declare them; int64 labels.",
            padding="zeros / reflect / replicate / circular, explicit, 'same' and 'valid'; dilation, groups, ceil_mode: all native.",
            randomness="Initialization from torch's global generator (seeded by the run).", gradients="Reverse-mode autograd.",
            serialization="state_dict checkpoints content-addressed in the artifact store; deterministic PyTorch code export.",
            hardware="CPU here (this machine has no CUDA/MPS use in the adapter).",
            sharing="Shared call sites reuse one nn.Module (one set of tensors).")),
    "keras": BackendInfo(
        "keras", "TensorFlow / Keras 3", "Later deep-learning backend: variables, layers, differentiation (Keras 3 on the TensorFlow backend).",
        ("keras", "tensorflow"), "native NHWC; the graph is NCHW, every transpose is recorded as a conversion", "Not implemented: forward, loss, gradients and one SGD step only (no worker training run, no checkpoints).",
        KERAS_INIT,
        _facets_common(
            layout="Graph NCHW -> native NHWC at the nodes that need it; weights OIHW <-> HWIO, Linear (out,in) <-> (in,out). All recorded per node.",
            dtype="float32 and float64 are native; int64 labels native. No silent dtype change.",
            padding="Zero padding only: explicit padding becomes tf.pad + 'valid'; 'same' maps to Keras 'same'. reflect/replicate/circular are unsupported (reported). Dilation > 1 requires stride 1. max_pool2d padding/dilation/ceil_mode unsupported.",
            randomness="Keras initializers (Glorot-uniform / zeros) seeded by keras.utils.set_random_seed; values differ from PyTorch. No dropout/augmentation ops in the portable subset.",
            gradients="tf.GradientTape reverse-mode, eager. ReLU gradient at exactly 0 is 0 as in PyTorch; max-pool gradient ties may route differently (not exercised by the continuous random fixtures).",
            serialization="No checkpoint format; weights cross as NumPy arrays in graph layout. Deterministic Keras code export with a source map. tf.function tracing is used only by the benchmark, with the same eager semantics for this subset (no data-dependent control flow).",
            hardware="CPU (macOS arm64 build); no GPU tested.",
            sharing="Shared call sites call the same Keras layer object (one set of variables); the switch preserves sharing.")),
    "jax": BackendInfo(
        "jax", "JAX", "Later functional numerical backend: explicit state and randomness, transformations, compilation.", ("jax", "jaxlib"),
        "graph layout (NCHW/OIHW) via lax.conv_general_dilated: no conversions", "Not implemented: forward, loss, gradients and one SGD step only (no worker training run, no checkpoints).",
        JAX_INIT,
        _facets_common(
            layout="No conversions: NCHW activations and OIHW weights are used as they are.",
            dtype="float32 only (float64 is unsupported: JAX would silently downcast without jax_enable_x64); int64 labels become int32 (recorded; out-of-range values rejected).",
            padding="zeros / reflect / replicate / circular via jnp.pad; explicit, 'same', 'valid'; dilation, groups, ceil_mode via lax. adaptive pooling only when the output size divides the input.",
            randomness="Explicit jax.random keys; deterministic given the seed. No random ops in the portable subset.",
            gradients="jax.value_and_grad over the pure apply function. Max-pool gradient ties may route differently from PyTorch.",
            serialization="Parameters are a plain pytree of arrays (NumPy exchange in graph layout). Deterministic jax.numpy code export with a source map. jax.jit is used by the executable (compile time reported separately by the benchmark).",
            hardware="CPU (CpuDevice); no accelerator tested.",
            sharing="Shared call sites read the owner's entry in the parameter pytree (one set of arrays); the switch preserves sharing.")),
}

_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def pinned_versions() -> dict[str, str]:
    """Distribution -> pinned version from python/requirements.txt."""
    out: dict[str, str] = {}
    req = _ROOT / "python" / "requirements.txt"
    if req.exists():
        for line in req.read_text().splitlines():
            m = re.match(r"^([A-Za-z0-9_.\-\[\]]+)==([^\s#]+)", line.strip())
            if m:
                out[m.group(1).split("[")[0].lower().replace("_", "-")] = m.group(2)
    return out


@dataclass
class Availability:
    available: bool
    versions: dict[str, str]
    reason: str | None = None

    def to_json(self) -> dict:
        return {"available": self.available, "versions": self.versions, "reason": self.reason}


@lru_cache(maxsize=None)
def availability(backend: str) -> Availability:
    if backend == "pytorch":
        try:
            import torch
            return Availability(True, {"torch": torch.__version__})
        except Exception as e:  # noqa: BLE001
            return Availability(False, {}, f"torch import failed: {type(e).__name__}: {e}")
    if backend == "keras":
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        try:
            import keras
            import tensorflow as tf
            if keras.backend.backend() != "tensorflow":
                return Availability(False, {"keras": keras.__version__}, f"Keras is running on the '{keras.backend.backend()}' backend (KERAS_BACKEND), not TensorFlow")
            y = tf.nn.conv2d(tf.ones((1, 4, 4, 1)), tf.ones((3, 3, 1, 1)), 1, "SAME")
            float(y[0, 1, 1, 0])
            return Availability(True, {"keras": keras.__version__, "tensorflow": tf.__version__})
        except Exception as e:  # noqa: BLE001
            return Availability(False, {}, f"TensorFlow/Keras could not be imported or run on this Python/platform: {type(e).__name__}: {e}")
    if backend == "jax":
        try:
            import jax
            import jax.numpy as jnp
            float(jnp.ones((2, 2)).sum())
            return Availability(True, {"jax": jax.__version__, "jaxlib": __import__("jaxlib").__version__})
        except Exception as e:  # noqa: BLE001
            return Availability(False, {}, f"JAX could not be imported or run on this Python/platform: {type(e).__name__}: {e}")
    raise KeyError(backend)


def describe(backend: str) -> dict:
    i = INFO[backend]
    a = availability(backend)
    pins = {p: pinned_versions().get(p) for p in i.packages}
    return {"id": i.id, "title": i.title, "role": i.role, "layout": i.layout, "training": i.training, "init": i.init, "pinned": pins,
            "available": a.available, "versions": a.versions, "reason": a.reason, "facets": i.facets}
