"""Declared cross-backend numerical tolerances (VISION 14.2: "Cross-backend numerical comparison needs documented tolerances").

Every comparison is against PyTorch CPU, with IDENTICAL copied weights (in graph layout), the same inputs, and the same float dtype. They were
chosen after measuring the actual differences on the reference CNN (batch 8, 5 seeds, this machine; docs/CAPABILITIES.md lists them): worst forward
activation difference 1.8e-6 (declared atol 2e-5), loss 2.4e-7 (2e-5), gradient 1.4e-5 (declared atol 5e-5: about 3.5x margin; the relative error
reaches 2.6e-2 on near-zero gradient elements, which is why the absolute term dominates), one SGD update 1.5e-8 (5e-6). They are bounds on floating-point
reassociation (different convolution / reduction kernels), not a claim of bit equality. A comparison passes when |a - b| <= atol + rtol * |b| elementwise.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tol:
    atol: float
    rtol: float


# (backend, dtype) -> quantity -> tolerance. "forward": node outputs/activations/logits; "loss": scalar loss; "grad": parameter and input
# gradients; "update": parameters after one SGD step (lr declared by the test).
TOLERANCES: dict[tuple[str, str], dict[str, Tol]] = {
    ("pytorch", "float32"): {"forward": Tol(1e-6, 1e-5), "loss": Tol(1e-6, 1e-5), "grad": Tol(1e-6, 1e-4), "update": Tol(1e-6, 1e-5)},  # graph-lowered vs handwritten torch: same kernels
    ("keras", "float32"): {"forward": Tol(2e-5, 1e-4), "loss": Tol(2e-5, 1e-4), "grad": Tol(5e-5, 1e-3), "update": Tol(5e-6, 1e-4)},
    ("jax", "float32"): {"forward": Tol(2e-5, 1e-4), "loss": Tol(2e-5, 1e-4), "grad": Tol(5e-5, 1e-3), "update": Tol(5e-6, 1e-4)},
    ("pytorch", "float64"): {"forward": Tol(1e-12, 1e-10), "loss": Tol(1e-12, 1e-10), "grad": Tol(1e-12, 1e-10), "update": Tol(1e-12, 1e-10)},
    ("keras", "float64"): {"forward": Tol(1e-10, 1e-9), "loss": Tol(1e-10, 1e-9), "grad": Tol(1e-10, 1e-9), "update": Tol(1e-10, 1e-9)},
    # no ("jax", "float64"): unsupported (JAX truncates to float32 unless jax_enable_x64 is set; this adapter refuses instead of changing global state)
}


def tol(backend: str, dtype: str, quantity: str) -> Tol:
    return TOLERANCES[(backend, dtype)][quantity]
