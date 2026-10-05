"""Deterministic real SQLite interleaving for worker/control cancellation (no timing luck)."""
from artifact_store import ArtifactStore
from graph_core.hashing import semantic_hash
from worker.train import RunConfig, load_checkpoint, run_training


def test_control_cancellation_wins_worker_transition_but_keeps_partial_checkpoint(cnn_graph, shapes_dir, tmp_path, monkeypatch):
    wb = tmp_path / "wb"
    worker = ArtifactStore(wb)
    control = ArtifactStore(wb)
    cfg = RunConfig(data=str(shapes_dir), epochs=2, batch_size=8)
    worker.create_run("race", semantic_hash(cnn_graph), cfg.model_dump())
    set_status = worker.set_status
    interleavings = []

    def control_wins(run_id, new, error=None):
        if new == "cancelling":
            assert control.get_run(run_id)["status"] == "running"
            control.set_status(run_id, "cancelling")  # separate real writer, immediately before worker's write
            interleavings.append(new)
        return set_status(run_id, new, error)

    monkeypatch.setattr(worker, "set_status", control_wins)
    checks = []

    def cancel_after_one_batch():
        checks.append(True)
        return len(checks) > 1

    status = run_training(cnn_graph, cfg, worker, "race", cancel_after_one_batch)
    assert interleavings == ["cancelling"] and status == "cancelled"
    assert control.get_run("race")["status"] == "cancelled"
    events = worker.events("race")
    assert not any(e["type"] == "error" for e in events)
    assert sum(e["type"] == "train_step" for e in events) == 1
    assert sum(e["type"] == "cancel_acknowledged" for e in events) == 1
    assert events[-1]["type"] == "run_finished" and events[-1]["data"]["status"] == "cancelled"
    [checkpoint] = worker.artifacts("race", "checkpoint")
    assert checkpoint["status"] == "partial" and checkpoint["step"] == 1 and worker.verify(checkpoint["sha256"])
    assert load_checkpoint(worker.path_of(checkpoint["sha256"]))["status"] == "partial"
