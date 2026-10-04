"""Shared helpers for the backend tests (Milestone 6a)."""
import copy

import numpy as np
import pytest

from conftest import EXAMPLES
from graph_core.project_io import load_project
from graph_core.schema import Edge, Node

import backends
from backends.registry import BACKEND_IDS, availability


def requires_backend(name):
    a = availability(name)
    return pytest.mark.skipif(not a.available, reason=f"backend '{name}' is unavailable: {a.reason}")


def skip_unless_available(name):
    a = availability(name)
    if not a.available:
        pytest.skip(f"backend '{name}' is unavailable: {a.reason}")


from backends.workloads import cnn_with_loss  # noqa: E402,F401


def cmp(a, b, tol, what=""):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"{what}: shape {a.shape} != {b.shape}"
    err = np.abs(a - b)
    bound = tol.atol + tol.rtol * np.abs(b)
    assert (err <= bound).all(), f"{what}: max abs diff {err.max():.3e} exceeds atol={tol.atol} rtol={tol.rtol}"
