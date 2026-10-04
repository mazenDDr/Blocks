"""Checkpoints, thread identity, interrupts, resume after a real service restart, and protected effects that must not repeat (A15)."""
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from agent import samples as sm
from agent.memory import MemoryStore
from agent_helpers import Lab, REPO


def notes(out: Path) -> list[str]:
    p = out / "notes.txt"
    return p.read_text().splitlines() if p.exists() else []


def test_interrupt_pauses_with_a_pending_form_and_resumes_from_the_checkpoint(tmp_path):
    out = tmp_path / "out"
    lab = Lab(tmp_path)
    g = sm.approval_tools_graph(str(out))
    st, rid = lab.run(g, thread="th1")
    assert st == "paused" and lab.store.get_run(rid)["status"] == "paused"
    ev = lab.events(rid, "interrupt_raised")[0]
    assert ev["node_id"] == "review" and ev["data"]["payload"]["type"] == "human_input" and ev["data"]["payload"]["proposed"] == "calculation 12 * (3 + 4) = 84"
    assert ev["data"]["payload"]["actions"] == ["approve", "reject", "edit"] and ev["data"]["checkpointId"]
    assert lab.final(rid) is None and notes(out) == []
    # a brand-new saver connection (as a restarted process would open) sees the persisted thread state
    from agent.runtime import open_checkpointer

    tup = open_checkpointer(lab.wb).get_tuple({"configurable": {"thread_id": "th1"}})
    assert tup.checkpoint["channel_values"]["note"] == "calculation 12 * (3 + 4) = 84" and any(w[1] == "__interrupt__" for w in tup.pending_writes)
    st, _ = lab.run(g, run_id=rid, resume={"action": "edit", "value": "reviewed: 84"})
    assert st == "paused"  # the external write has its own approval interrupt
    ev2 = lab.events(rid, "interrupt_raised")[-1]
    assert ev2["node_id"] == "write" and ev2["data"]["payload"]["type"] == "approval" and ev2["data"]["payload"]["effects"] == ["file_write"]
    assert ev2["data"]["payload"]["args"] == {"path": "notes.txt", "text": "reviewed: 84"} and notes(out) == []  # nothing written before approval
    st, _ = lab.run(g, run_id=rid, resume={"action": "approve"})
    assert st == "completed" and notes(out) == ["reviewed: 84"]
    f = lab.final(rid)
    assert f["note"] == "reviewed: 84" and f["decision"] == "edit" and f["write_result"]["status"] == "ok"
    # the interrupted node is marked as a replay in the trace: its code ran again from the top on resume
    starts = [(e["node_id"], e["data"]["replay"]) for e in lab.events(rid, "node_started")]
    assert ("review", False) in starts and ("review", True) in starts and ("write", True) in starts
    seqs = [e["seq"] for e in lab.events(rid)]
    assert seqs == list(range(len(seqs)))  # one continuous ordered event log across the resumes


def test_reject_skips_the_effect_and_edit_changes_state(tmp_path):
    out = tmp_path / "out"
    lab = Lab(tmp_path)
    g = sm.approval_tools_graph(str(out))
    _, rid = lab.run(g, thread="rej")
    st, _ = lab.run(g, run_id=rid, resume={"action": "reject"})
    assert st == "completed" and lab.final(rid)["trail"] == ["drafted", "rejected"] and notes(out) == []
    _, rid2 = lab.run(g, thread="rej2")
    lab.run(g, run_id=rid2, resume={"action": "approve"})
    st, _ = lab.run(g, run_id=rid2, resume={"action": "reject"})  # reject the external effect itself
    assert st == "completed" and lab.final(rid2)["write_result"]["status"] == "rejected" and notes(out) == []


def test_replaying_a_protected_node_does_not_repeat_its_effect(tmp_path):
    """Time travel: re-run the write node from the checkpoint before it. The interrupt asks again; after approval the effect ledger
    recognises the already-performed effect and the file keeps exactly one line."""
    from langgraph.types import Command

    from agent.runtime import Runtime, compile_graph, open_checkpointer
    from agent.spec import agent_spec
    from graph_core.hashing import semantic_hash

    out = tmp_path / "out"
    lab = Lab(tmp_path)
    g = sm.approval_tools_graph(str(out))
    _, rid = lab.run(g, thread="tt")
    lab.run(g, run_id=rid, resume={"action": "approve"})
    lab.run(g, run_id=rid, resume={"action": "approve"})
    assert notes(out) == ["calculation 12 * (3 + 4) = 84"]
    saver = open_checkpointer(lab.wb)
    rt = Runtime(g, agent_spec(g), lab.store, lab.wb, "rt-replay", "tt", semantic_hash(g))
    lab.store.create_run("rt-replay", "h", {"kind": "agent"})
    c = compile_graph(g, rt, saver)
    before_write = next(s for s in c.get_state_history({"configurable": {"thread_id": "tt"}}) if s.next == ("write",))
    cfg = before_write.config
    list(c.stream(None, cfg, stream_mode="updates"))  # re-executes `write`: pauses at the approval again
    assert any(w for w in c.get_state({"configurable": {"thread_id": "tt"}}).tasks if w.interrupts)
    list(c.stream(Command(resume={"action": "approve"}), {"configurable": {"thread_id": "tt"}}, stream_mode="updates"))
    assert notes(out) == ["calculation 12 * (3 + 4) = 84"]  # not duplicated
    calls = [e for e in lab.store.events("rt-replay", -1, ("tool_call",))]
    assert calls and calls[-1]["data"]["replaySkipped"] is True
    eff = [e for e in MemoryStore(lab.wb).effects("tt") if e["tool"] == "write_note"]
    assert len(eff) == 1 and eff[0]["status"] == "done"


def test_claiming_an_effect_is_atomic_and_a_crash_mid_effect_is_not_repeated(tmp_path):
    mem = MemoryStore(tmp_path / "wb")
    a = mem.claim_effect("k1", run_id="r", thread_id="t", node="n", tool="x", args_hash="h")
    assert a == ("claimed", None)
    assert mem.claim_effect("k1", run_id="r2", thread_id="t", node="n", tool="x", args_hash="h") == ("uncertain", None)  # started, never completed: surfaced, not repeated
    mem.complete_effect("k1", {"ok": 1})
    assert mem.claim_effect("k1", run_id="r3", thread_id="t", node="n", tool="x", args_hash="h") == ("done", {"ok": 1})


# ------------------------------------------------------------------------------------------------ real control-service restarts
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Service:
    def __init__(self, wb: Path):
        self.wb, self.port, self.proc = wb, free_port(), None

    def start(self):
        env = {**os.environ, "VOID_WORKBENCH": str(self.wb), "PYTHONPATH": f"{REPO}/python:{REPO}/services"}
        self.proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "control.app:create_app", "--factory", "--app-dir", str(REPO / "services"), "--host", "127.0.0.1",
                                      "--port", str(self.port)], cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                if httpx.get(self.url("/api/agent/catalog"), timeout=1).status_code == 200:
                    return self
            except httpx.HTTPError:
                time.sleep(0.15)
        raise RuntimeError("service did not start")

    def url(self, p):
        return f"http://127.0.0.1:{self.port}{p}"

    def kill(self):
        self.proc.send_signal(signal.SIGKILL)  # no graceful shutdown: an actual crash of the control service
        self.proc.wait(10)

    def get(self, p):
        return httpx.get(self.url(p), timeout=10)

    def post(self, p, body, headers=None):
        return httpx.post(self.url(p), json=body, headers=headers or {}, timeout=30)

    def wait_status(self, rid, want, timeout=40):
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = self.get(f"/api/runs/{rid}").json()
            if s["status"] in want:
                return s
            time.sleep(0.2)
        raise AssertionError(f"run {rid} did not reach {want}: {s['status']}")


def test_a15_paused_agent_survives_service_restarts_and_resumes_without_duplicating_the_effect(tmp_path):
    wb, out = tmp_path / "wb", tmp_path / "out"
    g = sm.approval_tools_graph(str(out)).to_json()
    svc = Service(wb).start()
    try:
        r = svc.post("/api/runs", {"graph": g, "config": {"thread_id": "restart-1"}}, {"Idempotency-Key": "k-restart-1"})
        assert r.status_code == 201, r.text
        rid = r.json()["runId"]
        s = svc.wait_status(rid, ("paused", "failed"))
        assert s["status"] == "paused" and s["threadId"] == "restart-1" and s["pendingInterrupt"]["payload"]["type"] == "human_input"
        svc.kill()  # crash #1 while paused at the review form
        svc = Service(wb).start()
        s = svc.get(f"/api/runs/{rid}").json()
        assert s["status"] == "paused" and s["pendingInterrupt"]["node"] == "review"  # persisted run ledger + checkpoint, nothing in memory
        th = svc.get("/api/agent/threads/restart-1/state").json()
        assert th["values"]["note"] == "calculation 12 * (3 + 4) = 84" and th["pendingInterrupt"] is True
        assert svc.post("/api/runs", {"graph": g, "config": {"thread_id": "restart-1"}}, {"Idempotency-Key": "other"}).status_code == 409  # thread busy: no second turn over a pending one
        assert svc.post(f"/api/runs/{rid}/resume", {"value": {"action": "edit", "value": "approved by a person"}}).status_code == 200
        assert svc.post(f"/api/runs/{rid}/resume", {"value": {"action": "approve"}}).status_code == 409  # double resume is rejected
        s = svc.wait_status(rid, ("paused", "failed", "completed"))
        assert s["status"] == "paused" and s["pendingInterrupt"]["payload"]["type"] == "approval" and notes(out) == []
        svc.kill()  # crash #2 while paused at the effect approval
        svc = Service(wb).start()
        assert svc.get(f"/api/runs/{rid}").json()["status"] == "paused" and notes(out) == []
        assert svc.post(f"/api/runs/{rid}/resume", {"value": {"action": "approve"}}).status_code == 200
        s = svc.wait_status(rid, ("completed", "failed"))
        assert s["status"] == "completed"
        assert notes(out) == ["approved by a person"]  # performed exactly once, with the edited text
        fin = svc.get(f"/api/agent/runs/{rid}/final-state").json()["values"]
        assert fin["note"] == "approved by a person" and fin["write_result"]["result"]["bytesAppended"] == len("approved by a person") + 1
        evs = [json.loads(json.dumps(e)) for e in MemoryStore(wb).effects("restart-1")]
        assert len([e for e in evs if e["tool"] == "write_note"]) == 1
        tr = svc.get(f"/api/agent/runs/{rid}/trace").json()
        assert [i["node"] for i in tr["interrupts"]] == ["review", "write"] and len(tr["run"]["resumes"]) == 2
        hist = svc.get("/api/agent/threads/restart-1/history").json()["checkpoints"]
        assert len(hist) >= 6 and hist[0]["step"] > hist[-1]["step"]
    finally:
        svc.kill()


def test_a_loop_may_repeat_an_effect_only_when_its_arguments_differ(tmp_path):
    out = tmp_path / "out"
    state = [sm.S("i", "integer", "add", default=0)]
    nodes = [sm.N("bump", "agent.set_state", assignments=[{"field": "i", "kind": "increment", "by": 1}]),
             sm.N("w", "agent.tool_call", tool="write_note", args={"path": "notes.txt", "text": "line {i}"}, allowed_dir=str(out), output_field="r")]
    g = sm.make_graph(nodes, [sm.E("START", "bump"), sm.E("bump", "w")],
                      {"state": state + [sm.S("r", "object", default={})], "limits": {"maxSteps": 12},
                       "routes": [{"id": "again", "from": "w", "cases": [{"id": "more", "when": sm.cmp("i", "<", 3), "to": "bump"}], "default": "END"}]})
    lab = Lab(tmp_path)
    st, rid = lab.run(g, thread="loop")
    assert st == "paused"
    for _ in range(3):  # each iteration needs its own approval; each writes a different line exactly once
        st, _ = lab.run(g, run_id=rid, resume={"action": "approve"})
    assert st == "completed" and notes(out) == ["line 1", "line 2", "line 3"]
