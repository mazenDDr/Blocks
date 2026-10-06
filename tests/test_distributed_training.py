"""Data-parallel training across worker processes (ADR 0073): same result as one process, real gloo process group."""
import io

import pytest
import torch

from artifact_store import ArtifactStore
from worker.train import RunConfig, load_checkpoint, run_training


def train(graph, data, tmp_path, name, **kw):
    store = ArtifactStore(tmp_path / name)
    cfg = RunConfig(**{"data": str(data), "batch_size": 10, "seed": 11, "lr": 0.05, "epochs": 2, **kw})  # 48 training images: last batch of 8 is uneven
    store.create_run("r1", "g", cfg.model_dump())
    return store, run_training(graph, cfg, store, "r1")


def final_state(store):
    ck = [a for a in store.artifacts("r1", "checkpoint") if a["status"] == "complete"][-1]
    return load_checkpoint(io.BytesIO(store.read_artifact(ck["sha256"])))["model"]


@pytest.mark.parametrize("workers,optimizer", [(2, "sgd"), (3, "adam")])
def test_data_parallel_equals_single_process(cnn_graph, shapes_dir, tmp_path, workers, optimizer):
    one, s1 = train(cnn_graph, shapes_dir, tmp_path, "one", optimizer=optimizer, lr=0.05 if optimizer == "sgd" else 1e-3)
    many, s2 = train(cnn_graph, shapes_dir, tmp_path, "many", optimizer=optimizer, lr=0.05 if optimizer == "sgd" else 1e-3, workers=workers)
    assert s1 == s2 == "completed", many.get_run("r1")["error"]
    started = many.last_event("r1", "run_started")["data"]
    assert started["workers"] == workers and "data-parallel" in started["parallelism"]
    a = [e["data"]["loss"] for e in one.events("r1", types=("train_step",))]
    b = [e["data"]["loss"] for e in many.events("r1", types=("train_step",))]
    assert len(a) == len(b) and max(abs(x - y) for x, y in zip(a, b)) < 1e-5
    pa, pb = final_state(one), final_state(many)
    worst = max(float((pa[k] - pb[k]).abs().max()) for k in pa)
    print("DISTRIBUTED", workers, optimizer, {"maxLossDiff": max(abs(x - y) for x, y in zip(a, b)), "maxParamDiff": worst})
    # SGD updates are linear in the gradient, so summation order changes parameters by ~1e-8. Adam divides by sqrt(v) per parameter,
    # which amplifies the same tiny gradient differences on near-zero gradients; its bound is wider for that declared reason.
    assert worst < (1e-6 if optimizer == "sgd" else 1e-4)


@pytest.mark.parametrize("kw", [{"backend": "jax", "optimizer": "sgd"}, {"device": "cuda"}])
def test_unsupported_combinations_are_refused(cnn_graph, shapes_dir, tmp_path, kw):
    store, status = train(cnn_graph, shapes_dir, tmp_path, "x", workers=2, **kw)
    assert status == "failed" and "E_DISTRIBUTED_CONFIG" in store.get_run("r1")["error"]
