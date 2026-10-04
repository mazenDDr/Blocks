"""Control API for backends: availability, compatibility report, per-backend export, coverage ledger, validation with a backend selected,
and the rule that worker training runs stay PyTorch-only."""
import copy

import pytest
from fastapi.testclient import TestClient

from backend_helpers import cnn_with_loss
from control.app import create_app
from graph_core.build import GraphBuilder


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with TestClient(create_app(tmp_path_factory.mktemp("wb"))) as c:
        yield c


def graph_json(backend="pytorch", g=None):
    g = copy.deepcopy(g or cnn_with_loss())
    g.backend = backend
    return g.to_json()


def reflect_json(backend):
    b = GraphBuilder()
    b.input("x", ["N", 3, 8, 8])
    b.node("c", "pytorch.nn.conv2d", out_channels=4, kernel_size=[3, 3], padding=[1, 1], padding_mode="reflect")
    b.chain("x", "c")
    g = b.build()
    g.backend = backend
    return g.to_json()


def test_backends_listing_reports_versions_and_availability(client):
    bs = {b["id"]: b for b in client.get("/api/backends").json()["backends"]}
    assert set(bs) == {"pytorch", "keras", "jax"}
    for b in bs.values():
        assert b["available"] or b["reason"], b  # unavailable backends say why
        assert b["role"] and b["facets"]["layout"] and b["facets"]["padding"] and b["facets"]["randomness"] and b["facets"]["gradients"] and b["facets"]["serialization"]
    assert bs["keras"]["pinned"] == {"keras": "3.15.1", "tensorflow": "2.21.0"} and bs["jax"]["pinned"]["jax"] == "0.11.2"
    if bs["keras"]["available"]:
        assert bs["keras"]["versions"]["tensorflow"] == "2.21.0"
    assert "Not implemented" in bs["keras"]["training"] and "Not implemented" in bs["jax"]["training"]


def test_compat_report_endpoint_lists_nodes_conversions_and_codes(client):
    r = client.post("/api/backends/compat", json={"graph": graph_json(), "backend": "keras"}).json()
    assert r["ok"] and r["counts"]["unsupported"] == 0 and r["nodes"]["conv_1"]["status"] == "converted"
    assert any(c["kind"] == "layout" for c in r["nodes"]["conv_1"]["conversions"])
    assert r["facets"]["layout"] and r["init"]
    bad = client.post("/api/backends/compat", json={"graph": reflect_json("pytorch"), "backend": "keras"}).json()
    assert not bad["ok"] and bad["nodes"]["c"]["code"] == "E_BACKEND_UNSUPPORTED_PADDING_MODE" and "reflect" in bad["nodes"]["c"]["reason"]
    assert client.post("/api/backends/compat", json={"graph": graph_json(), "backend": "mxnet"}).status_code == 404


def test_validate_with_a_selected_backend_marks_unsupported_nodes(client):
    v = client.post("/api/validate", json={"graph": reflect_json("keras")}).json()
    assert not v["ok"] and any(d["code"] == "E_BACKEND_UNSUPPORTED_PADDING_MODE" and d["nodeId"] == "c" for d in v["diagnostics"])
    assert client.post("/api/validate", json={"graph": reflect_json("jax")}).json()["ok"]
    assert client.post("/api/validate", json={"graph": graph_json("keras")}).json()["ok"]
    assert not client.post("/api/validate", json={"graph": graph_json("tensorflow")}).json()["ok"]  # an unknown backend is still E_UNSUPPORTED_BACKEND


@pytest.mark.parametrize("backend,marker", [("keras", "tf.transpose"), ("jax", "lax.conv_general_dilated"), ("pytorch", "class Model(nn.Module)")])
def test_export_endpoint_per_backend_for_drafts_and_saved_projects(client, backend, marker):
    r = client.post("/api/export", json={"graph": graph_json(), "backend": backend})
    assert r.status_code == 200 and marker in r.json()["code"] and r.json()["backend"] == backend
    pid = f"m6a_{backend}"
    assert client.put(f"/api/projects/{pid}", json={"graph": graph_json(), "ui": None}).status_code == 200
    r2 = client.get(f"/api/projects/{pid}/export/{backend}")
    assert r2.status_code == 200 and r2.json()["code"] == r.json()["code"]


def test_export_of_an_incompatible_graph_is_422_with_node_diagnostics(client):
    r = client.post("/api/export", json={"graph": reflect_json("pytorch"), "backend": "keras"})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "e_backend_incompatible"
    assert any(d["code"] == "E_BACKEND_UNSUPPORTED_PADDING_MODE" and d["nodeId"] == "c" for d in r.json()["detail"]["diagnostics"])
    assert client.post("/api/export", json={"graph": reflect_json("pytorch"), "backend": "jax"}).status_code == 200
    assert client.post("/api/export", json={"graph": reflect_json("pytorch"), "backend": "nope"}).status_code == 404


def test_coverage_endpoint_serves_the_generated_ledger(client):
    from conftest import ROOT

    L = client.get("/api/coverage").json()
    assert L["markdown"] == (ROOT / "docs" / "COVERAGE.md").read_text()
    row = next(r for r in L["model"] if r["type"] == "pytorch.nn.conv2d")
    assert row["execution"] == {"pytorch": "tested", "keras": "tested", "jax": "tested"} and row["explain"] and row["architecture"] == "dedicated schematic"
    assert next(r for r in L["model"] if r["type"] == "tensor.dense")["execution"]["keras"] == "unsupported"


def test_worker_training_runs_stay_pytorch_only(client, shapes_dir):
    from backend_helpers import EXAMPLES, load_project

    cnn = load_project(EXAMPLES / "reference_cnn.project.json").graph
    cnn.backend = "keras"
    before = len(client.get("/api/runs").json()["runs"])
    r = client.post("/api/runs", json={"graph": cnn.to_json(), "config": {"data": str(shapes_dir), "epochs": 1}}, headers={"Idempotency-Key": "m6a-keras-run"})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "backend_training_unsupported" and "PyTorch only" in r.json()["detail"]["message"]
    assert len(client.get("/api/runs").json()["runs"]) == before  # nothing was queued
