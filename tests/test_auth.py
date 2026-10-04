"""Optional bearer-token boundary (VOID_API_TOKEN): without a token nothing changes; with one, every request needs the exact token, the
server refuses weak tokens, and the server's own loopback traffic generator still authenticates."""
import socket
import threading
import time

import pytest
from fastapi.testclient import TestClient

from control.app import create_app

TOKEN = "t0k3n-for-tests-0123456789"


def test_no_token_keeps_the_service_open(tmp_path, monkeypatch):
    monkeypatch.delenv("VOID_API_TOKEN", raising=False)
    with TestClient(create_app(tmp_path / "wb")) as c:
        assert c.get("/api/examples").status_code == 200


def test_token_is_required_everywhere(tmp_path):
    with TestClient(create_app(tmp_path / "wb", api_token=TOKEN)) as c:
        for path in ("/api/examples", "/api/production", "/openapi.json", "/docs"):
            r = c.get(path)
            assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer", path
        assert c.get("/api/examples", headers={"Authorization": "Bearer wrong-token-0123456789"}).status_code == 401
        assert c.get("/api/examples", headers={"Authorization": TOKEN}).status_code == 401           # scheme required
        ok = c.get("/api/examples", headers={"Authorization": f"Bearer {TOKEN}"})
        assert ok.status_code == 200 and "tabular_regression" in ok.json()["examples"]
        assert TOKEN not in ok.text


def test_environment_token_and_weak_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("VOID_API_TOKEN", TOKEN)
    with TestClient(create_app(tmp_path / "wb")) as c:
        assert c.get("/api/examples").status_code == 401
    for weak in ("short", " " + TOKEN):
        with pytest.raises(ValueError, match="at least 16"):
            create_app(tmp_path / "wb2", api_token=weak)


def test_loopback_traffic_generator_authenticates(tmp_path):
    """The traffic generator calls this server's own HTTP route; with a token configured it must send it."""
    import uvicorn

    from artifact_store import ArtifactStore
    from conftest import EXAMPLES
    from graph_core.hashing import semantic_hash
    from graph_core.project_io import load_project
    from production.models import PredictRequest, RegisterVersion, ReleaseCreate, ServingConfig, TrafficSpec
    from worker.tabular_run import TabularRunConfig, run_tabular

    graph = load_project(EXAMPLES / "production_sensors.project.json").graph
    store = ArtifactStore(tmp_path / "wb")
    store.create_run("trained", semantic_hash(graph), TabularRunConfig().model_dump())
    store.add_artifact("trained", "graph", graph.model_dump_json(by_alias=True).encode(), "complete", None, {})
    assert run_tabular(graph, TabularRunConfig(), store, "trained") == "completed"
    app = create_app(tmp_path / "wb", api_token=TOKEN)
    rt = app.state.services.production
    v = rt.register_version(RegisterVersion(runId="trained", node="classifier", name="s", owner="o", intendedUse="u", limitations="l"))
    rel = rt.create_release(ReleaseCreate(versionId=v["id"], config=ServingConfig()))
    rt.activate(rel["id"], None)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        import httpx
        assert httpx.get(f"http://127.0.0.1:{port}/api/production", trust_env=False).status_code == 401
        runner = app.state.services.traffic
        records = PredictRequest(requestId="x", records=[{"signal": 1.2, "background": -0.5}]).records
        job = runner.start(TrafficSpec(releaseId=rel["id"], payloads=[records], pattern="closed_loop", rate=20, durationSeconds=0.3, concurrency=1,
                                       warmupRequests=1, maxRequests=5), port)
        deadline = time.monotonic() + 10
        while job["state"] == "running" and time.monotonic() < deadline:
            time.sleep(0.02)
            job = runner.view(job["id"])
        out = job["result"]
        assert out["state"] == "completed" and out["observed"]["successfulRequests"] > 0 and out["observed"]["errors"] == 0
    finally:
        server.should_exit = True
        thread.join(5)
        sock.close()
