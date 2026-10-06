"""Keras/JAX training runs (ADR 0070): same seeded start as PyTorch, plain SGD on the backend, PyTorch-format checkpoints."""
import io

import pytest

from artifact_store import ArtifactStore
from graph_core.lower import lower_graph
from worker.train import RunConfig, load_checkpoint, run_training


def train(graph, data, tmp_path, name, **kw):
    store = ArtifactStore(tmp_path / name)
    cfg = RunConfig(data=str(data), batch_size=16, seed=5, lr=0.05, epochs=2, **kw)
    store.create_run("r1", "g", cfg.model_dump())
    return store, run_training(graph, cfg, store, "r1")


def losses(store):
    return [e["data"]["loss"] for e in store.events("r1", types=("train_step",))]


@pytest.mark.parametrize("backend", ["keras", "jax"])
def test_backend_run_matches_pytorch_start_and_checkpoints_load_in_pytorch(cnn_graph, shapes_dir, tmp_path, backend):
    ref, ref_status = train(cnn_graph, shapes_dir, tmp_path, "pytorch")
    store, status = train(cnn_graph, shapes_dir, tmp_path, backend, backend=backend)
    assert ref_status == status == "completed", store.get_run("r1")["error"]
    started = store.last_event("r1", "run_started")["data"]
    assert started["backend"] == backend and started["backendVersions"]
    a, b = losses(ref), losses(store)
    assert len(a) == len(b) and abs(a[0] - b[0]) < 1e-4  # same initialization and first batch: only kernel reassociation differs
    print("BACKEND_LOSSES", backend, {"pytorchFirst": a[0], "backendFirst": b[0], "pytorchLast": a[-1], "backendLast": b[-1]})
    epochs = [e["data"] for e in store.events("r1", types=("epoch_end",))]
    assert len(epochs) == 2 and all(0.0 <= e["val_acc"] <= 1.0 for e in epochs)
    ckpt = [x for x in store.artifacts("r1", "checkpoint") if x["status"] == "complete"][-1]
    state = load_checkpoint(io.BytesIO(store.read_artifact(ckpt["sha256"])))
    assert state["backend"] == backend
    model = lower_graph(cnn_graph)
    model.load_state_dict(state["model"], strict=True)


@pytest.mark.parametrize("kw", [{"optimizer": "adam"}, {"momentum": 0.9}, {"device": "cuda"}])
def test_unsupported_backend_settings_are_refused(cnn_graph, shapes_dir, tmp_path, kw):
    store, status = train(cnn_graph, shapes_dir, tmp_path, "x", backend="jax", **kw)
    assert status == "failed" and "E_BACKEND_OPTIMIZER" in store.get_run("r1")["error"]
    assert not store.artifacts("r1", "checkpoint")
