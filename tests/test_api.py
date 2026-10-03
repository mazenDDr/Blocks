"""Control service over HTTP (FastAPI TestClient): validate, projects, runs, SSE, inspection, inference."""
import base64
import copy
import json
import time
import uuid

import pytest
import torch
from fastapi.testclient import TestClient

from conftest import EXAMPLES, ROOT
from control.app import create_app
from graph_core.export_schema import DEFAULT, schema_text
from graph_core.hashing import semantic_hash
from graph_core.lower import lower_graph
from graph_core.project_io import load_project
from graph_core.schema import Graph
from worker.dataset import load_image_folder
from worker.train import load_checkpoint

TERMINAL = {"completed", "failed", "cancelled"}


def wait_for(client, rid, states=TERMINAL, timeout=240):
    end = time.time() + timeout
    while time.time() < end:
        r = client.get(f"/api/runs/{rid}").json()
        if r["status"] in states:
            return r
        time.sleep(0.2)
    raise AssertionError(f"run {rid} still {r['status']}")


def cnn_json():
    return load_project(EXAMPLES / "reference_cnn.project.json").graph.to_json()


def parse_sse(text):
    out = []
    for block in text.split("\n\n"):
        fields = dict(l.split(": ", 1) for l in block.splitlines() if ": " in l and not l.startswith(":"))
        if "id" in fields:
            out.append((int(fields["id"]), fields["event"], json.loads(fields["data"])))
    return out


def submit(client, data, key=None, **cfg):
    body = {"graph": cnn_json(), "config": {"data": str(data), "batch_size": 16, **cfg}}
    return client.post("/api/runs", json=body, headers={"Idempotency-Key": key or uuid.uuid4().hex})


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with TestClient(create_app(tmp_path_factory.mktemp("wb"))) as c:
        yield c


@pytest.fixture(scope="module")
def trained(client, shapes_dir):
    r = submit(client, shapes_dir, epochs=2, seed=3)
    assert r.status_code == 201, r.text
    rid = r.json()["runId"]
    assert wait_for(client, rid)["status"] == "completed"
    return rid


# ------------------------------------------------------------------------------------------ registry / validate
def test_registry_exposes_config_schema_and_defaults(client):
    ops = {o["type"]: o for o in client.get("/api/registry").json()["ops"] if o["graphKind"] == "model"}  # tabular ops: test_tabular_api.py
    assert len(ops) == 14
    conv = ops["pytorch.nn.conv2d"]
    assert conv["inputs"] == ["input"] and conv["outputs"] == ["output"] and conv["backend"] == "pytorch"
    assert {"in_channels", "out_channels", "kernel_size", "padding_mode"} <= conv["configSchema"]["properties"].keys()
    assert conv["defaults"]["in_channels"] == "infer"


def test_validate_returns_shapes_params_and_explain(client):
    v = client.post("/api/validate", json={"graph": cnn_json()}).json()
    assert v["ok"] and v["totalParams"] == 20_042 and v["graphHash"] == semantic_hash(Graph.model_validate(cnn_json()))
    assert v["paramCounts"]["conv_1"] == 896 and v["paramCounts"]["conv_2"] == 18_496 and v["paramCounts"]["fc"] == 650
    n = v["nodes"]["conv_1"]
    assert n["inputShapes"]["input"]["shape"] == ["N", 3, 64, 64] and n["outputShapes"]["output"]["shape"] == ["N", 32, 64, 64]
    assert n["resolvedConfig"]["in_channels"] == 3 and "equation" in n["explain"] and n["explain"]["parameters"]["total"] == 896


def test_validate_propagates_edit_and_reports_locked_mismatch(client):
    g = cnn_json()
    for n in g["nodes"]:
        if n["id"] == "conv_1":
            n["config"]["out_channels"] = 64
    v = client.post("/api/validate", json={"graph": g}).json()
    assert v["ok"] and v["nodes"]["conv_2"]["resolvedConfig"]["in_channels"] == 64
    assert v["paramCounts"]["conv_1"] == 1792
    for n in g["nodes"]:
        if n["id"] == "conv_2":
            n["config"]["in_channels"] = 32
    v = client.post("/api/validate", json={"graph": g}).json()
    assert not v["ok"]
    d = v["nodes"]["conv_2"]["diagnostics"]
    assert d[0]["code"] == "E_CHANNEL_MISMATCH" and d[0]["port"] == "input" and d[0]["fixes"]
    assert v["nodes"]["conv_2"]["typed"] is False


def test_validate_keeps_unknown_op_as_diagnostic(client):
    g = cnn_json()
    g["nodes"].append({"id": "weird", "type": "plugin.x.y", "version": "1.0.0", "config": {}})
    v = client.post("/api/validate", json={"graph": g}).json()
    assert v["nodes"]["weird"]["known"] is False and v["nodes"]["weird"]["diagnostics"][0]["code"] == "E_UNKNOWN_OP"


# ------------------------------------------------------------------------------------------ projects
def test_project_save_load_round_trip_with_separate_ui(client):
    g, ui = cnn_json(), {"schemaVersion": "1.0.0", "positions": {"conv_1": {"x": 1.5, "y": 2.5}}, "pinnedBaseline": "abc"}
    r = client.put("/api/projects/p1", json={"graph": g, "ui": ui})
    assert r.status_code == 200
    got = client.get("/api/projects/p1").json()
    assert Graph.model_validate(got["graph"]).to_json() == Graph.model_validate(g).to_json()
    assert got["ui"]["positions"] == ui["positions"] and got["ui"]["pinnedBaseline"] == "abc"
    assert got["graphHash"] == r.json()["graphHash"]
    ui["positions"]["conv_1"]["x"] = 999
    assert client.put("/api/projects/p1", json={"graph": g, "ui": ui}).json()["graphHash"] == got["graphHash"]  # layout is not semantic
    assert "p1" in client.get("/api/projects").json()["projects"]
    assert client.get("/api/projects/missing").status_code == 404
    assert client.put("/api/projects/bad id!", json={"graph": g}).status_code == 422


def test_export_pytorch_and_blocked_export(client):
    g = cnn_json()
    client.put("/api/projects/exp", json={"graph": g})
    e = client.get("/api/projects/exp/export/pytorch").json()
    assert "class Model(nn.Module)" in e["code"] and "# node: conv_1" in e["code"] and e["graphHash"]
    for n in g["nodes"]:
        if n["id"] == "conv_2":
            n["config"]["in_channels"] = 5
    client.put("/api/projects/exp_bad", json={"graph": g})
    r = client.get("/api/projects/exp_bad/export/pytorch")
    assert r.status_code == 422 and r.json()["detail"]["diagnostics"][0]["code"] == "E_CHANNEL_MISMATCH"


def test_schema_file_is_current():
    assert DEFAULT.read_text() == schema_text()


# ------------------------------------------------------------------------------------------ runs
def test_submit_rejects_blocked_graph_and_missing_key(client, shapes_dir):
    g = cnn_json()
    for n in g["nodes"]:
        if n["id"] == "conv_2":
            n["config"]["in_channels"] = 5
    before = len(client.get("/api/runs").json()["runs"])
    r = client.post("/api/runs", json={"graph": g, "config": {"data": str(shapes_dir)}}, headers={"Idempotency-Key": "k-bad"})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "execution_blocked"
    assert client.post("/api/runs", json={"graph": cnn_json(), "config": {"data": str(shapes_dir)}}).status_code == 400
    r = client.post("/api/runs", json={"graph": cnn_json(), "config": {"data": "/no/such/dir"}}, headers={"Idempotency-Key": "k-nodir"})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "dataset_missing"
    assert len(client.get("/api/runs").json()["runs"]) == before


def test_idempotent_submit_returns_same_run(client, shapes_dir):
    key = "idem-" + uuid.uuid4().hex
    n0 = len(client.get("/api/runs").json()["runs"])
    a = submit(client, shapes_dir, key, epochs=1, seed=11)
    b = submit(client, shapes_dir, key, epochs=1, seed=11)
    assert a.status_code == 201 and b.status_code == 200
    assert a.json()["runId"] == b.json()["runId"] and b.json()["idempotentReplay"] is True
    assert submit(client, shapes_dir, key, epochs=1, seed=12).status_code == 409  # same key, different request
    assert len(client.get("/api/runs").json()["runs"]) == n0 + 1
    wait_for(client, a.json()["runId"])


def test_cancel_running_run_and_illegal_cancel(client, shapes_dir):
    rid = submit(client, shapes_dir, epochs=500, batch_size=8).json()["runId"]
    end = time.time() + 120
    while client.get(f"/api/runs/{rid}").json()["progress"]["step"] < 1:
        assert time.time() < end
        time.sleep(0.1)
    r = client.post(f"/api/runs/{rid}/cancel")
    assert r.status_code == 202 and r.json()["status"] == "cancelling"
    done = wait_for(client, rid, timeout=60)
    assert done["status"] == "cancelled"
    cks = client.get(f"/api/runs/{rid}/checkpoints").json()["checkpoints"]
    assert cks[-1]["status"] == "partial"
    assert client.post(f"/api/runs/{rid}/cancel").status_code == 409
    assert client.post("/api/runs/nope/cancel").status_code == 404


def test_run_continues_when_the_event_client_disconnects(client, shapes_dir):
    rid = submit(client, shapes_dir, epochs=2).json()["runId"]
    with client.stream("GET", f"/api/runs/{rid}/events") as r:
        for line in r.iter_lines():
            if line.startswith("id: "):
                break  # drop the connection after the first event
    assert wait_for(client, rid)["status"] == "completed"


def test_run_status_split_and_metrics(client, trained):
    s = client.get(f"/api/runs/{trained}").json()
    assert s["status"] == "completed" and s["totalParams"] == 20_042 and len(s["classes"]) == 10
    assert s["config"]["seed"] == 3 and s["config"]["val_fraction"] == 0.2 and s["split"]["seed"] == 3 and s["split"]["nVal"] == 12
    assert s["final"]["epoch"] == 1 and 0 <= s["final"]["val_acc"] <= 1
    m = client.get(f"/api/runs/{trained}/metrics").json()
    assert len(m["epochs"]) == 2 and m["trainLoss"][0][0] == 1 and m["graph"]["nodes"]
    assert [r["id"] for r in client.get("/api/runs").json()["runs"] if r["id"] == trained]


def test_split_is_seeded_and_deterministic(shapes_dir):
    a = load_image_folder(shapes_dir, 64, 64, 3, 0.2)
    b = load_image_folder(shapes_dir, 64, 64, 3, 0.2)
    c = load_image_folder(shapes_dir, 64, 64, 4, 0.2)
    assert a.val_files == b.val_files and a.val_files != c.val_files
    assert not set(a.val_files) & set(a.train_files) and len(a.val_files) + len(a.train_files) == 60


def test_sse_resume_from_last_event_id_has_no_duplicates(client, trained):
    with client.stream("GET", f"/api/runs/{trained}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        full = parse_sse("".join(r.iter_text()))
    seqs = [s for s, _, _ in full]
    assert seqs == list(range(len(seqs))) and full[0][1] == "run_queued" and full[-1][1] == "run_finished"
    cut = len(seqs) // 2
    with client.stream("GET", f"/api/runs/{trained}/events", headers={"Last-Event-ID": str(seqs[cut])}) as r:
        resumed = parse_sse("".join(r.iter_text()))
    assert [s for s, _, _ in resumed] == seqs[cut + 1:]
    assert full[cut + 1:] == resumed
    p = full[1][2]
    assert p["run_id"] == trained and p["graph_hash"] and {"ts", "type", "node_id", "data"} <= p.keys()
    assert client.get("/api/runs/nope/events").status_code == 404


# ------------------------------------------------------------------------------------------ inspect
def test_inspect_weights_are_checkpoint_values_with_provenance(client, trained):
    r = client.post(f"/api/runs/{trained}/inspect", json={"kind": "weights", "node": "conv_1", "offset": 2, "limit": 3}).json()
    assert r["available"]
    w = r["params"]["weight"]
    assert w["shape"] == [32, 3, 3, 3] and w["slice"] == {"offset": 2, "limit": 3, "of": 32, "truncated": True}
    assert len(w["values"]) == 3 and len(w["values"][0]) == 3 and len(w["values"][0][0][0]) == 3
    cks = client.get(f"/api/runs/{trained}/checkpoints").json()["checkpoints"]
    last = cks[-1]
    p = r["provenance"]
    assert p["runId"] == trained and p["nodeId"] == "conv_1" and p["checkpointStep"] == last["step"] and p["checkpointSha256"] == last["sha256"]
    assert p["graphHash"] == client.get(f"/api/runs/{trained}").json()["graphHash"]
    # values are exactly what is stored in the checkpoint file
    sd = load_checkpoint(client.app.state.services.store.path_of(last["sha256"]))["model"]
    full = client.post(f"/api/runs/{trained}/inspect", json={"kind": "weights", "node": "conv_1"}).json()["params"]["weight"]
    assert torch.allclose(torch.tensor(full["values"]), sd["conv_1.weight"], atol=1e-5)
    assert full["stats"]["min"] == pytest.approx(float(sd["conv_1.weight"].min()))
    first = client.post(f"/api/runs/{trained}/inspect", json={"kind": "weights", "node": "conv_1", "checkpointStep": cks[0]["step"]}).json()
    assert first["provenance"]["checkpointStep"] == cks[0]["step"] != last["step"]
    assert first["params"]["weight"]["values"] != full["values"]  # training changed the weights between checkpoints


def test_inspect_weights_are_bounded(client, trained, monkeypatch):
    import control.inspection as insp
    monkeypatch.setattr(insp, "MAX_ELEMS", 2_000)
    r = client.post(f"/api/runs/{trained}/inspect", json={"kind": "weights", "node": "conv_2", "limit": 10_000}).json()
    w = r["params"]["weight"]
    assert w["shape"] == [64, 32, 3, 3] and w["slice"]["limit"] == 2_000 // (32 * 9) == 6 and len(w["values"]) == 6 and w["slice"]["truncated"]
    assert client.post(f"/api/runs/{trained}/inspect", json={"kind": "weights", "node": "relu_1"}).json()["reason"] == "no_parameters"
    assert client.post(f"/api/runs/{trained}/inspect", json={"kind": "weights", "node": "nope"}).status_code == 422
    assert client.post(f"/api/runs/{trained}/inspect", json={"kind": "weights", "node": "conv_1", "checkpointStep": 99999}).status_code == 422


def test_inspect_activations_match_a_forward_pass(client, trained):
    r = client.post(f"/api/runs/{trained}/inspect", json={"kind": "activations", "node": "conv_1", "sample": 4, "limit": 5}).json()
    assert r["available"] and r["layout"] == "feature_maps" and r["shape"] == [1, 32, 64, 64]
    assert r["slice"]["limit"] == 5 and len(r["values"]) == 5 and len(r["values"][0]) == 64 and len(r["channelStats"]) == 5
    p = r["provenance"]
    samples = client.get(f"/api/runs/{trained}/samples").json()["samples"]
    assert p["sampleIndex"] == 4 and p["sampleId"] == samples[4]["id"] and p["nodeId"] == "conv_1" and p["runId"] == trained
    # independent recomputation with the stored graph + checkpoint
    from PIL import Image
    from worker.dataset import preprocess_image
    sv = client.app.state.services
    split = json.loads(sv.store.read_artifact(sv.store.artifacts(trained, "split")[0]["sha256"]))
    with Image.open(f"{split['root']}/{split['val'][4]}") as im:
        x = preprocess_image(im, 64, 64).unsqueeze(0)
    model = lower_graph(load_project(EXAMPLES / "reference_cnn.project.json").graph)
    ck = sv.store.artifacts(trained, "checkpoint")[-1]
    model.load_state_dict(load_checkpoint(sv.store.path_of(ck["sha256"]))["model"])
    expected = model.conv_1(x)[0, :5]
    assert torch.allclose(torch.tensor(r["values"]), expected, atol=1e-4)
    v = client.post(f"/api/runs/{trained}/inspect", json={"kind": "activations", "node": "fc", "sample": 0}).json()
    assert v["layout"] == "vector" and v["shape"] == [1, 10]
    assert client.post(f"/api/runs/{trained}/inspect", json={"kind": "activations", "node": "conv_1", "sample": 9999}).status_code == 422
    assert client.post(f"/api/runs/{trained}/inspect", json={"kind": "activations", "node": "conv_1"}).status_code == 422


def test_inspect_sample_loss_and_confusion_are_recorded_values(client, trained):
    w = client.post(f"/api/runs/{trained}/inspect", json={"kind": "sample_loss", "limit": 4}).json()
    assert w["available"] and len(w["worst"]) == 4 and w["nVal"] == 12
    losses = [x["loss"] for x in w["worst"]]
    assert losses == sorted(losses, reverse=True) and w["worst"][0]["predName"] and w["provenance"]["epoch"] == 1
    c = client.post(f"/api/runs/{trained}/inspect", json={"kind": "confusion", "epoch": 0}).json()
    assert sum(map(sum, c["matrix"])) == 12 and len(c["classes"]) == 10 and c["provenance"]["epoch"] == 0


def test_inspect_before_any_run_says_not_recorded(tmp_path):
    with TestClient(create_app(tmp_path / "wb")) as c:
        for kind, extra in (("weights", {"node": "conv_1"}), ("activations", {"node": "conv_1", "sample": 0}), ("sample_loss", {}), ("confusion", {})):
            r = c.post("/api/inspect", json={"kind": kind, **extra})
            assert r.status_code == 200
            j = r.json()
            assert j["available"] is False and j["reason"] == "no_run" and j["provenance"]["runId"] is None and "values" not in j
        # a run that exists but has no checkpoint yet is also "not recorded"
        sv = c.app.state.services
        g = Graph.model_validate(cnn_json())
        sv.store.create_run("r0", semantic_hash(g), {})
        sv.store.add_artifact("r0", "graph", json.dumps(g.to_json()).encode(), "complete", None, {})
        j = c.post("/api/runs/r0/inspect", json={"kind": "weights", "node": "conv_1"}).json()
        assert j["available"] is False and j["reason"] == "not_recorded" and j["provenance"]["runId"] == "r0"
        j = c.post("/api/runs/r0/inspect", json={"kind": "activations", "node": "conv_1", "sample": 0}).json()
        assert j["available"] is False and j["reason"] == "not_recorded"
        assert c.post("/api/infer", json={"runId": "r0", "sample": 0}).json()["available"] is False
        assert c.get("/api/runs/r0/checkpoints").json()["checkpoints"] == []


def test_inspecting_never_trains_or_adds_events(client, trained):
    n_runs = len(client.get("/api/runs").json()["runs"])
    seq = client.get(f"/api/runs/{trained}").json()["maxSeq"]
    for _ in range(2):
        client.post(f"/api/runs/{trained}/inspect", json={"kind": "weights", "node": "fc"})
        client.post(f"/api/runs/{trained}/inspect", json={"kind": "activations", "node": "fc", "sample": 1})
        client.post("/api/infer", json={"runId": trained, "sample": 1})
    assert len(client.get("/api/runs").json()["runs"]) == n_runs
    assert client.get(f"/api/runs/{trained}").json()["maxSeq"] == seq


# ------------------------------------------------------------------------------------------ infer
def test_infer_from_checkpoint_on_val_sample_and_upload(client, trained):
    r = client.post("/api/infer", json={"runId": trained, "sample": 2}).json()
    assert r["available"] and len(r["logits"]) == 10 and abs(sum(r["probabilities"]) - 1) < 1e-4
    assert r["predictedName"] == r["classes"][r["predicted"]] == r["classes"][max(range(10), key=r["logits"].__getitem__)]
    p = r["provenance"]
    assert p["runId"] == trained and p["sampleIndex"] == 2 and p["checkpointStep"] and p["trueLabel"] in r["classes"]
    img = client.get(f"/api/runs/{trained}/samples/2/image")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    up = client.post("/api/infer", json={"runId": trained, "imageBase64": base64.b64encode(img.content).decode()}).json()
    assert up["logits"] == pytest.approx(r["logits"], abs=1e-5) and up["provenance"]["sampleId"] == "uploaded image"
    assert client.post("/api/infer", json={"runId": trained}).status_code == 422
    assert client.post("/api/infer", json={"runId": trained, "imageBase64": "!!notbase64"}).status_code == 422
    assert client.post("/api/infer", json={"runId": "nope", "sample": 0}).status_code == 404
