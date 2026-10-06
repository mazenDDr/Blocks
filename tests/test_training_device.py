"""Explicit training device (ADR 0067): CPU default, CUDA refused when absent, CUDA runs verified on a real GPU (pytest -m gpu)."""
import io

import pytest
import torch

from artifact_store import ArtifactStore
from graph_core.lower import lower_graph
from worker.train import RunConfig, device_info, load_checkpoint, run_training


def train(graph, data, tmp_path, name, **kw):
    store = ArtifactStore(tmp_path / name)
    cfg = RunConfig(data=str(data), batch_size=16, seed=7, **kw)
    store.create_run("r1", "g", cfg.model_dump())
    return store, run_training(graph, cfg, store, "r1")


def test_cpu_is_the_default_and_recorded(cnn_graph, shapes_dir, tmp_path):
    store, status = train(cnn_graph, shapes_dir, tmp_path, "cpu", epochs=1)
    assert status == "completed" and RunConfig(data="x").device == "cpu"
    assert store.last_event("r1", "run_started")["data"]["hardware"]["device"] == "cpu"


@pytest.mark.skipif(torch.cuda.is_available(), reason="this machine has CUDA; the refusal applies only without it")
def test_cuda_request_is_refused_without_cuda(cnn_graph, shapes_dir, tmp_path):
    store, status = train(cnn_graph, shapes_dir, tmp_path, "nocuda", epochs=1, device="cuda")
    assert status == "failed" and "E_DEVICE_UNAVAILABLE" in store.get_run("r1")["error"]
    assert not store.artifacts("r1", "checkpoint")
    with pytest.raises(Exception, match="E_DEVICE_UNAVAILABLE"):
        device_info("cuda")


@pytest.mark.gpu
def test_real_cuda_training_matches_cpu_start_and_checkpoints_load_on_cpu(cnn_graph, shapes_dir, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no usable CUDA device")
    cpu_store, cpu_status = train(cnn_graph, shapes_dir, tmp_path, "cpu", epochs=3)
    gpu_store, gpu_status = train(cnn_graph, shapes_dir, tmp_path, "gpu", epochs=3, device="cuda")
    assert cpu_status == gpu_status == "completed"
    hw = gpu_store.last_event("r1", "run_started")["data"]["hardware"]
    assert hw["device"] == "cuda" and hw["name"] and hw["cudaRuntime"]
    cpu_steps = [e["data"]["loss"] for e in cpu_store.events("r1", types=("train_step",))]
    gpu_steps = [e["data"]["loss"] for e in gpu_store.events("r1", types=("train_step",))]
    assert len(cpu_steps) == len(gpu_steps) and abs(cpu_steps[0] - gpu_steps[0]) < 1e-4  # same seed: same init and data order
    ckpt = [a for a in gpu_store.artifacts("r1", "checkpoint") if a["status"] == "complete"][-1]
    state = load_checkpoint(io.BytesIO(gpu_store.read_artifact(ckpt["sha256"])))
    assert all(t.device.type == "cpu" for t in state["model"].values())
    model = lower_graph(cnn_graph)
    model.load_state_dict(state["model"], strict=True)  # the CPU serving path loads a CUDA-trained checkpoint
    epochs = [e["data"] for e in gpu_store.events("r1", types=("epoch_end",))]
    print("GPU_EVIDENCE", {"hardware": hw, "cpuFirstLoss": cpu_steps[0], "gpuFirstLoss": gpu_steps[0], "cpuLastLoss": cpu_steps[-1],
                           "gpuLastLoss": gpu_steps[-1], "gpuValAcc": [e["val_acc"] for e in epochs],
                           "cpuValAcc": [e["data"]["val_acc"] for e in cpu_store.events("r1", types=("epoch_end",))]})


def _api_run(tmp_path, data, device):
    from fastapi.testclient import TestClient
    from control.app import create_app
    from test_api import submit, wait_for
    with TestClient(create_app(tmp_path / "wb")) as c:
        r = submit(c, data, epochs=1, seed=3, device=device)
        assert r.status_code == 201, r.text
        return wait_for(c, r.json()["runId"]), c.app.state.services.store, r.json()["runId"]


@pytest.mark.skipif(torch.cuda.is_available(), reason="this machine has CUDA")
def test_api_worker_process_refuses_cuda_without_it(shapes_dir, tmp_path):
    run, store, rid = _api_run(tmp_path, shapes_dir, "cuda")
    assert run["status"] == "failed" and "E_DEVICE_UNAVAILABLE" in store.get_run(rid)["error"]


@pytest.mark.gpu
def test_api_worker_process_trains_on_cuda(shapes_dir, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no usable CUDA device")
    run, store, rid = _api_run(tmp_path, shapes_dir, "cuda")
    assert run["status"] == "completed" and store.get_run(rid)["config"]["device"] == "cuda"
    assert store.last_event(rid, "run_started")["data"]["hardware"]["device"] == "cuda"
