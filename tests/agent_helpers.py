"""Helpers for the agent (Milestone 4) tests: run a graph in-process through the same worker entry point the API uses."""
import json
import shutil
import time
from pathlib import Path

from agent import samples as sm
from artifact_store import ArtifactStore
from graph_core.hashing import semantic_hash
from worker.agent_run import AgentRunConfig, run_agent

REPO = Path(__file__).resolve().parent.parent


class Lab:
    def __init__(self, tmp_path: Path):
        self.wb = tmp_path / "wb"
        self.store = ArtifactStore(self.wb)
        self.n = 0

    def run(self, graph, thread="t1", inp=None, run_id=None, resume=None, cancel=lambda: False):
        """Start a run (or resume `run_id` with `resume`) in this process. Returns (status, run_id)."""
        if resume is None:
            self.n += 1
            run_id = run_id or f"r{self.n}"
            cfg = AgentRunConfig(thread_id=thread, input=inp or {})
            self.store.create_run(run_id, semantic_hash(graph), cfg.model_dump())
            self.store.add_artifact(run_id, "graph", json.dumps(graph.to_json(), sort_keys=True).encode(), "complete", None, {})
            return run_agent(graph, cfg, self.store, run_id, cancel), run_id
        row = self.store.get_run(run_id)
        self.store.set_status(run_id, "running")
        return run_agent(graph, AgentRunConfig.model_validate(row["config"]), self.store, run_id, cancel, resume), run_id

    def events(self, run_id, *types):
        return self.store.events(run_id, -1, types or None)

    def final(self, run_id):
        a = self.store.artifacts(run_id, "final_state")
        return json.loads(self.store.read_artifact(a[-1]["sha256"])) if a else None

    def finished(self, run_id):
        return self.events(run_id, "run_finished")[-1]["data"]

    def contexts(self, run_id):
        return [json.loads(self.store.read_artifact(a["sha256"])) for a in self.store.artifacts(run_id, "model_context")]


def copy_docs(tmp_path: Path) -> str:
    """The synthetic example documents, copied so tests may edit them."""
    dst = tmp_path / "docs"
    shutil.copytree(REPO / "examples" / "agent_docs", dst, dirs_exist_ok=True)
    return str(dst)


def api_client(tmp_path):
    from fastapi.testclient import TestClient

    from control.app import create_app

    return TestClient(create_app(tmp_path / "wb"))


def wait_run(client, rid, want=("completed", "failed", "cancelled", "paused"), timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get(f"/api/runs/{rid}").json()
        if s["status"] in want:
            return s
        time.sleep(0.15)
    raise AssertionError(f"run {rid} still {s['status']}")


_KEY = iter(range(10**6))


def start(client, graph, thread=None, inp=None, project=None):
    body = {"graph": graph.to_json() if hasattr(graph, "to_json") else graph, "config": {"input": inp or {}, **({"thread_id": thread} if thread else {})}}
    r = client.post("/api/runs", json=body, headers={"Idempotency-Key": f"agent-test-{next(_KEY)}-{time.time()}"})
    assert r.status_code == 201, r.text
    return r.json()
