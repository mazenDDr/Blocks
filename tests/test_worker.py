"""Worker end to end: events, checkpoints, cancellation, blocked execution, store behaviour."""
import hashlib
import json
import subprocess
import sys
import time

import pytest
import torch

from conftest import EXAMPLES, ROOT, _load_generator, set_config
from artifact_store import ArtifactStore, IllegalTransition
from graph_core.hashing import semantic_hash
from graph_core.lower import lower_graph
from graph_core.schema import Node
from graph_core.validate import ExecutionBlocked
from worker.process import submit_run
from worker.train import RunConfig, load_checkpoint, run_training


def run_in_process(graph, data, tmp_path, should_cancel=lambda: False, **kw):
    store = ArtifactStore(tmp_path / "wb")
    cfg = RunConfig(data=str(data), batch_size=16, **kw)
    store.create_run("r1", semantic_hash(graph), cfg.model_dump())
    status = run_training(graph, cfg, store, "r1", should_cancel)
    return store, status


def test_dataset_generator_is_deterministic(tmp_path):
    gen = _load_generator()
    for d in ("a", "b"):
        gen.generate(tmp_path / d, per_class=2, seed=3)

    def digest(root):
        h = hashlib.sha256()
        for f in sorted(root.rglob("*.png")):
            h.update(str(f.relative_to(root)).encode() + f.read_bytes())
        return h.hexdigest()

    assert digest(tmp_path / "a") == digest(tmp_path / "b")
    assert len(list((tmp_path / "a").rglob("*.png"))) == 20 and len(list((tmp_path / "a").iterdir())) == 10


def test_training_run_events_and_checkpoint(cnn_graph, shapes_dir, tmp_path):
    store, status = run_in_process(cnn_graph, shapes_dir, tmp_path, epochs=2, seed=7)
    assert status == "completed"
    assert store.get_run("r1")["status"] == "completed"
    evs = store.events("r1")
    assert [e["seq"] for e in evs] == list(range(len(evs)))  # gapless, ordered
    gh = semantic_hash(cnn_graph)
    assert all(e["graph_hash"] == gh and e["run_id"] == "r1" and e["ts"] > 0 for e in evs)
    types = [e["type"] for e in evs]
    assert types[0] == "run_queued" and types[-1] == "run_finished"
    assert types.count("epoch_end") == 2 and types.count("checkpoint") == 2
    steps = [e for e in evs if e["type"] == "train_step"]
    n_train = next(e for e in evs if e["type"] == "run_started")["data"]["n_train"]
    assert len(steps) == 2 * -(-n_train // 16)
    assert [s["data"]["step"] for s in steps] == list(range(1, len(steps) + 1))
    assert all(isinstance(s["data"]["loss"], float) for s in steps)
    started = next(e for e in evs if e["type"] == "run_started")["data"]
    assert started["total_params"] == 20_042 and len(started["classes"]) == 10
    ws = [e for e in evs if e["type"] == "weight_stats"]
    assert {e["node_id"] for e in ws} == {"conv_1", "conv_2", "fc"}

    # checkpoint artifact exists, is content-addressed, and reloads into a freshly lowered model
    [ck1, ck2] = store.artifacts("r1", "checkpoint")
    assert ck2["status"] == "complete" and ck2["step"] == len(steps)
    assert store.verify(ck2["sha256"])
    ck = load_checkpoint(store.path_of(ck2["sha256"]))
    assert {"model", "optimizer", "rng", "step", "graph_hash"} <= ck.keys()
    assert ck["graph_hash"] == gh and ck["step"] == len(steps) and ck["status"] == "complete"
    fresh = lower_graph(cnn_graph)
    fresh.load_state_dict(ck["model"])
    assert fresh(torch.zeros(1, 3, 64, 64)).shape == (1, 10)
    opt = torch.optim.SGD(fresh.parameters(), lr=0.05)
    opt.load_state_dict(ck["optimizer"])


def test_seeded_runs_are_reproducible(cnn_graph, shapes_dir, tmp_path):
    losses = []
    for i in range(2):
        store, _ = run_in_process(cnn_graph, shapes_dir, tmp_path / str(i), epochs=1, seed=5)
        losses.append([e["data"]["loss"] for e in store.events("r1") if e["type"] == "train_step"])
    assert losses[0] == losses[1]
    store, _ = run_in_process(cnn_graph, shapes_dir, tmp_path / "other", epochs=1, seed=6)
    assert [e["data"]["loss"] for e in store.events("r1") if e["type"] == "train_step"] != losses[0]


def test_adam_runs(cnn_graph, shapes_dir, tmp_path):
    _, status = run_in_process(cnn_graph, shapes_dir, tmp_path, epochs=1, optimizer="adam", lr=0.001)
    assert status == "completed"


def test_cooperative_cancel_in_process_stops_at_batch_boundary(cnn_graph, shapes_dir, tmp_path):
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2  # allow exactly 2 batches (mid first epoch of 3)

    store, status = run_in_process(cnn_graph, shapes_dir, tmp_path, cancel, epochs=5)
    assert status == "cancelled" and store.get_run("r1")["status"] == "cancelled"
    assert len([e for e in store.events("r1") if e["type"] == "train_step"]) == 2
    [ck] = store.artifacts("r1", "checkpoint")
    assert ck["status"] == "partial" and ck["step"] == 2
    assert load_checkpoint(store.path_of(ck["sha256"]))["status"] == "partial"


def test_cancel_running_worker_process_yields_partial_checkpoint(cnn_graph, shapes_dir, tmp_path):
    wb = tmp_path / "wb"
    h = submit_run(cnn_graph, RunConfig(data=str(shapes_dir), epochs=500, batch_size=8), wb)
    deadline = time.time() + 120
    while not any(e["type"] == "train_step" for e in h.store.events(h.run_id)):
        assert time.time() < deadline and h.is_alive(), "worker never started training"
        time.sleep(0.1)
    h.cancel()
    assert h.wait(60) == "cancelled"
    evs = h.store.events(h.run_id)
    assert evs[-1]["type"] == "run_finished" and evs[-1]["data"]["status"] == "cancelled"
    assert "cancel_acknowledged" in [e["type"] for e in evs]
    *complete, ck = h.store.artifacts(h.run_id, "checkpoint")  # per-epoch ones may precede it
    assert all(c["status"] == "complete" for c in complete)
    assert ck["status"] == "partial" and h.store.verify(ck["sha256"])
    assert h.process.exitcode == 0


def test_unknown_op_blocks_execution_with_actionable_error(cnn_graph, shapes_dir, tmp_path):
    cnn_graph.nodes.append(Node.model_validate({"id": "gelu_x", "type": "plugin.acme.gelu", "config": {"approx": "tanh"}}))
    wb = tmp_path / "wb"
    with pytest.raises(ExecutionBlocked) as e:
        submit_run(cnn_graph, RunConfig(data=str(shapes_dir)), wb)
    [d] = e.value.diagnostics
    assert (d.code, d.nodeId) == ("E_UNKNOWN_OP", "gelu_x")
    assert "cannot execute until it is resolved" in d.message and d.fixes
    assert ArtifactStore(wb).list_runs() == []  # nothing was queued

    # the worker itself also refuses, and records why on the run
    store, status = run_in_process(cnn_graph, shapes_dir, tmp_path / "direct", epochs=1)
    assert status == "failed"
    [ve] = [x for x in store.events("r1") if x["type"] == "validation_error"]
    assert ve["node_id"] == "gelu_x" and ve["data"]["code"] == "E_UNKNOWN_OP"
    assert store.artifacts("r1") == []


def test_channel_mismatch_blocks_before_training(cnn_graph, shapes_dir, tmp_path):
    set_config(cnn_graph, "conv_2", in_channels=7)
    with pytest.raises(ExecutionBlocked) as e:
        submit_run(cnn_graph, RunConfig(data=str(shapes_dir)), tmp_path / "wb")
    assert (e.value.diagnostics[0].code, e.value.diagnostics[0].nodeId) == ("E_CHANNEL_MISMATCH", "conv_2")


def test_class_count_mismatch_fails_run_with_reason(cnn_graph, shapes_dir, tmp_path):
    set_config(cnn_graph, "fc", out_features=3)
    store, status = run_in_process(cnn_graph, shapes_dir, tmp_path, epochs=1)
    assert status == "failed"
    assert "3 logits" in store.get_run("r1")["error"]


def test_run_lifecycle_rejects_illegal_transitions(tmp_path):
    s = ArtifactStore(tmp_path)
    s.create_run("a", "h", {})
    with pytest.raises(IllegalTransition):
        s.set_status("a", "running")  # queued -> running skips preparing
    s.set_status("a", "preparing")
    s.set_status("a", "running")
    s.set_status("a", "completed")
    with pytest.raises(IllegalTransition):
        s.set_status("a", "running")


def test_artifact_store_is_content_addressed_and_atomic(tmp_path):
    s = ArtifactStore(tmp_path)
    a = s.put_bytes(b"hello")
    assert a == hashlib.sha256(b"hello").hexdigest() == s.put_bytes(b"hello")
    assert s.path_of(a).read_bytes() == b"hello" and s.verify(a)
    assert not list(s.artifact_dir.glob("*.tmp"))
    s.path_of(a).write_bytes(b"corrupt")
    assert not s.verify(a)


def test_cli_end_to_end_prints_json_events(shapes_dir, tmp_path):
    p = subprocess.run(
        [sys.executable, "-m", "worker.cli", str(EXAMPLES / "reference_cnn.project.json"), "--data", str(shapes_dir),
         "--epochs", "1", "--workbench", str(tmp_path / "wb")],
        capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert p.returncode == 0, p.stderr
    lines = [json.loads(l) for l in p.stdout.splitlines()]
    assert lines[-1]["status"] == "completed"
    events = lines[:-1]
    assert events[-1]["type"] == "run_finished" and {"run_id", "graph_hash", "seq", "ts", "type", "node_id", "data"} <= events[0].keys()
    store = ArtifactStore(tmp_path / "wb")
    [ck] = store.artifacts(lines[-1]["run_id"], "checkpoint")
    assert store.verify(ck["sha256"])


def test_cli_blocks_unknown_op(tmp_path, shapes_dir):
    proj = json.loads((EXAMPLES / "reference_cnn.project.json").read_text())
    proj["nodes"].append({"id": "weird", "type": "plugin.x.y", "version": "1.0.0", "config": {}})
    f = tmp_path / "bad.project.json"
    f.write_text(json.dumps(proj))
    p = subprocess.run([sys.executable, "-m", "worker.cli", str(f), "--data", str(shapes_dir), "--workbench", str(tmp_path / "wb")],
                       capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert p.returncode == 2 and "E_UNKNOWN_OP" in p.stderr and "weird" in p.stderr
