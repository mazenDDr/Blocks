"""Real local Ollama: streamed serving deltas reproduce the exact recorded model text of the same native request."""
import json

import pytest
from fastapi.testclient import TestClient

from control.app import create_app
from test_production_agent_live import native  # noqa: F401  (module fixture: real source run + active release)

pytestmark = pytest.mark.live


def events(body: str):
    out = []
    for frame in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_streamed_deltas_equal_recorded_response_and_trace_matches_plain_route(native):  # noqa: F811
    lab, rt, v, rel = native
    with TestClient(create_app(lab.wb)) as c:
        req = {"requestId": "stream-real", "records": [{"question": "SYNTHETIC test: list three fruits, comma separated."}]}
        with c.stream("POST", "/api/serve/local/live-agent/predict/stream", json=req) as r:
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            body = "".join(r.iter_text())
        seq = events(body)
        kinds = [k for k, _ in seq]
        assert kinds[-2:] == ["result", "end"] and set(kinds[:-2]) == {"token"} and len(kinds) >= 4
        trace, end = seq[-2][1], seq[-1][1]
        streamed = "".join(d["delta"] for k, d in seq if k == "token")
        recorded = trace["result"]["agent"]["contexts"][0]["value"]["response"]
        assert streamed == recorded and streamed.strip()
        assert end == {"deltas": len(kinds) - 2, "chars": len(streamed), "status": 200}
        # The stored trace is the one the stream ended with, and the plain route replays it idempotently.
        stored = c.get("/api/production/requests/stream-real").json()
        assert stored["result"] == trace["result"] and stored["status"] == 200
        assert c.post("/api/serve/local/live-agent/predict", json=req).json()["idempotentReplay"]
        assert trace["result"]["agent"]["modelCalls"] == 1 and trace["result"]["agent"]["contexts"][0]["value"]["usage"]["source"] == "provider"


def test_stream_route_keeps_existing_refusals(native):  # noqa: F811
    lab, rt, v, rel = native
    with TestClient(create_app(lab.wb)) as c:
        bad = c.post("/api/serve/local/live-agent/predict/stream", content='{"requestId": "nan", "records": [{"question": NaN}]}',
                     headers={"Content-Type": "application/json"})
        assert bad.status_code == 422
        with c.stream("POST", "/api/serve/local/missing/predict/stream", json={"requestId": "x", "records": [{"question": "q"}]}) as r:
            seq = events("".join(r.iter_text()))
        assert seq[-1][0] == "end" and seq[-1][1]["status"] >= 400 and not [k for k, _ in seq if k == "token"]
