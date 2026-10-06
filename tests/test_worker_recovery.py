"""Crash recovery of run workers: leases, heartbeats and reconciliation of lost workers (ADR 0057)."""
import json
import os
import signal
import time

from fastapi.testclient import TestClient

from artifact_store import ArtifactStore
from control.app import create_app
from worker.process import HEARTBEAT_SECONDS, submit_run
from worker.train import RunConfig


def started(store, run_id, status="running"):
    store.create_run(run_id, "g" * 64, {"kind": "SYNTHETIC"})
    path = {"preparing": ["preparing"], "running": ["preparing", "running"], "cancelling": ["preparing", "running", "cancelling"],
            "paused": ["preparing", "running", "paused"], "queued": []}[status]
    for s in path:
        store.set_status(run_id, s)


def test_silent_active_runs_fail_with_evidence_and_others_are_untouched(tmp_path):
    store = ArtifactStore(tmp_path)
    for rid, status in (("q", "queued"), ("p", "preparing"), ("r", "running"), ("c", "cancelling"), ("z", "paused")):
        started(store, rid, status)
    store.create_run("done", "g" * 64, {})
    store.set_status("done", "failed", "SYNTHETIC earlier failure")
    store.lease("r", 4242, "n1")
    assert store.reconcile_lost_workers() == []  # everything changed just now
    lost = store.reconcile_lost_workers(now=time.time() + 121)
    assert {x["runId"]: x["previousStatus"] for x in lost} == {"q": "queued", "p": "preparing", "r": "running", "c": "cancelling"}
    assert [x["workerPid"] for x in lost if x["runId"] == "r"] == [4242]
    for rid in "qprc":
        row = store.get_run(rid)
        assert row["status"] == "failed" and row["error"].startswith("E_WORKER_LOST")
        last = store.events(rid)[-1]
        assert last["type"] == "run_finished" and last["data"]["recovered"] and last["data"]["status"] == "failed"
    assert store.get_run("z")["status"] == "paused" and store.get_run("done")["error"] == "SYNTHETIC earlier failure"
    assert store.reconcile_lost_workers(now=time.time() + 999) == []  # idempotent


def test_fresh_heartbeat_or_event_keeps_a_run_alive(tmp_path):
    store = ArtifactStore(tmp_path)
    started(store, "beat")
    store.lease("beat", 1, "n")
    started(store, "talk")
    later = time.time() + 121
    store._exec("UPDATE run_leases SET heartbeat=? WHERE run_id='beat'", (later - 5,))
    store.append_event("talk", 0, later - 5, "train_step", "g" * 64, None, {})
    store._exec("UPDATE runs SET updated_at=? WHERE id IN ('beat','talk')", (later - 200,))
    assert store.reconcile_lost_workers(now=later) == []
    store.heartbeat("beat", "wrong-nonce")  # a stale worker cannot refresh another worker's lease
    assert store._exec("SELECT heartbeat FROM run_leases WHERE run_id='beat'")[0]["heartbeat"] == later - 5


def test_killed_real_worker_is_recovered_and_a_live_one_is_not(cnn_graph, shapes_dir, tmp_path):
    wb = tmp_path / "wb"
    live = submit_run(cnn_graph, RunConfig(data=str(shapes_dir), epochs=500, batch_size=8), wb)
    doomed = submit_run(cnn_graph, RunConfig(data=str(shapes_dir), epochs=500, batch_size=8), wb)
    store = live.store
    deadline = time.time() + 120
    for h in (live, doomed):
        while not any(e["type"] == "train_step" for e in store.events(h.run_id)):
            assert time.time() < deadline and h.is_alive(), "worker never started training"
            time.sleep(0.1)
    lease = store._exec("SELECT * FROM run_leases WHERE run_id=?", (doomed.run_id,))[0]
    assert lease["pid"] == doomed.process.pid
    os.kill(doomed.process.pid, signal.SIGKILL)  # an OOM kill / crash: no terminal status is ever written
    doomed.process.join(10)
    assert store.get_run(doomed.run_id)["status"] == "running"
    time.sleep(HEARTBEAT_SECONDS + 1.5)
    beat = store._exec("SELECT heartbeat, started FROM run_leases WHERE run_id=?", (live.run_id,))[0]
    assert beat["heartbeat"] > beat["started"]  # the live worker's heartbeat thread is refreshing its lease
    # Window shorter than the doomed run's silence but longer than one heartbeat: only the killed worker is lost.
    store._exec("UPDATE runs SET updated_at=updated_at-1000 WHERE id IN (?,?)", (live.run_id, doomed.run_id))
    last_doomed = max(e["ts"] for e in store.events(doomed.run_id))
    lost = store.reconcile_lost_workers(lost_after=HEARTBEAT_SECONDS + 1, now=max(time.time(), last_doomed + HEARTBEAT_SECONDS + 2))
    assert [x["runId"] for x in lost] == [doomed.run_id] and lost[0]["workerPid"] == doomed.process.pid
    assert store.get_run(doomed.run_id)["error"].startswith("E_WORKER_LOST")
    assert store.artifacts(doomed.run_id, "graph")  # recorded artifacts are kept
    assert store.get_run(live.run_id)["status"] == "running" and live.is_alive()
    live.cancel()
    assert live.wait(60) == "cancelled"


def test_control_startup_reconciles_a_run_left_by_a_crash(tmp_path):
    wb = tmp_path / "wb"
    store = ArtifactStore(wb)
    started(store, "orphan")
    store._exec("UPDATE runs SET updated_at=updated_at-1000 WHERE id='orphan'")
    with TestClient(create_app(wb)):
        deadline = time.time() + 10
        while store.get_run("orphan")["status"] != "failed":
            assert time.time() < deadline
            time.sleep(0.1)
        assert store.get_run("orphan")["error"].startswith("E_WORKER_LOST")
    assert json.loads(json.dumps(store.events("orphan")[-1]))["data"]["previousStatus"] == "running"
