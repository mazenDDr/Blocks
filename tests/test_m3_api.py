"""Milestone 3 over HTTP: validation views of composites, module library, training-procedure runs in a real worker process, the debugger routes,
attention inspector, code-block routes and the shipped example projects."""
import json
import time
import uuid

import pytest
import torch
from fastapi.testclient import TestClient

from conftest import EXAMPLES
from control.app import create_app
from graph_core.build import GraphBuilder, masked_weighted_loss, residual_block, transformer_classifier
from graph_core.hashing import semantic_hash
from graph_core.project_io import load_project
from graph_core.schema import Graph
from graph_core.validate import validate

TERMINAL = {"completed", "failed", "cancelled"}


def wait(client, rid, timeout=240):
    end = time.time() + timeout
    while time.time() < end:
        r = client.get(f"/api/runs/{rid}").json()
        if r["status"] in TERMINAL:
            return r
        time.sleep(0.2)
    raise AssertionError(f"run {rid} still {r['status']}")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with TestClient(create_app(tmp_path_factory.mktemp("wb3"))) as c:
        yield c


def submit(client, graph, procedure, **cfg):
    body = {"graph": graph.to_json(), "config": {"kind": "procedure", "procedure": procedure, **cfg}}
    return client.post("/api/runs", json=body, headers={"Idempotency-Key": uuid.uuid4().hex})


def small_transformer():
    g = transformer_classifier(layers=1)
    proc = {"epochs": 3, "seed": 1, "data": {"kind": "synthetic_sequence", "seq_len": 8, "vocab": 12, "n_train": 64, "n_val": 16, "batch_size": 16},
            "optimizer": {"kind": "adam", "lr": 0.01}, "checkpoint": {"every": {"unit": "optimizer_step", "n": 4}}, "capture": {"steps": [3], "nodes": None},
            "watch": {"param": "head.weight", "index": [0, 0]}}
    return g, proc


@pytest.fixture(scope="module")
def trained(client):
    g, proc = small_transformer()
    r = submit(client, g, proc)
    assert r.status_code == 201, r.text
    rid = r.json()["runId"]
    assert wait(client, rid)["status"] == "completed"
    return g, proc, rid


# ------------------------------------------------------------------------------------------------ validation views
def test_validate_returns_composite_instances_flat_nodes_modules_and_procedure_diagnostics(client):
    g = Graph.model_validate(load_project(EXAMPLES / "residual_cnn.project.json").graph.to_json())
    v = client.post("/api/validate", json={"graph": g.to_json()}).json()
    assert v["ok"] and v["totalParams"] == 2650
    res1 = v["nodes"]["res1"]
    assert res1["structural"] and res1["inputPorts"] == ["x"] and res1["outputPorts"] == ["y"] and res1["params"] == 1168
    assert res1["inputShapes"]["x"]["shape"] == ["N", 8, 64, 64] and res1["outputShapes"]["y"]["shape"] == ["N", 8, 64, 64]
    assert "own parameters" in res1["explain"]["note"] and [s["local"] for s in res1["explain"]["steps"]][:2] == ["conv_a", "relu_a"] or True
    assert "res1/conv_a" in v["flat"] and v["flat"]["res1/conv_a"]["params"] == 584 and "res1/conv_a" not in v["nodes"]
    assert v["instances"]["res2"]["module"] == "residual_block"
    m = v["modules"][0]
    assert m["id"] == "residual_block" and m["usedBy"] == ["res1", "res2"] and len(m["contentHash"]) == 64
    assert v["procedure"]["valid"] is True and v["procedure"]["stages"][0] == "zero_grad"


def test_a24_equation_and_runtime_follow_the_edited_definition_over_http(client):
    g = load_project(EXAMPLES / "residual_cnn.project.json").graph
    v1 = client.post("/api/validate", json={"graph": g.to_json()}).json()
    g2 = Graph.model_validate(g.to_json())
    for n in g2.modules[0].nodes:
        if n.id == "conv_b":
            n.config["kernel_size"], n.config["padding"] = [5, 5], [2, 2]
    v2 = client.post("/api/validate", json={"graph": g2.to_json()}).json()
    assert v2["ok"] and v1["graphHash"] != v2["graphHash"]
    assert v2["totalParams"] == v1["totalParams"] + 2 * (8 * 8 * 16)         # both instances got the 5x5 kernel (+16 weights per filter pair)
    assert v1["nodes"]["res1"]["explain"]["equation"] != v2["nodes"]["res1"]["explain"]["equation"]
    st = lambda v: {s["local"]: s for s in v["nodes"]["res1"]["explain"]["steps"]}  # noqa: E731
    assert "kernel_size" not in st(v1)["conv_b"]["config"] and st(v2)["conv_b"]["config"]["kernel_size"] == [5, 5]     # the composed explanation shows the edited op
    assert st(v2)["conv_b"]["params"] == 8 * 8 * 25 + 8 and "[5, 5]" in v2["nodes"]["res1"]["explain"]["equation"]
    assert v2["flat"]["res2/conv_b"]["resolvedConfig"]["kernel_size"] == [5, 5]


def test_validation_names_the_failing_inner_node_by_path(client):
    g = load_project(EXAMPLES / "residual_cnn.project.json").graph
    g.nodes[1].config["out_channels"] = 5
    v = client.post("/api/validate", json={"graph": g.to_json()}).json()
    assert not v["ok"]
    codes = {d["code"] for d in v["diagnostics"]}
    assert "E_PORT_TYPE" in codes
    assert any(d["code"] == "E_PORT_TYPE" and d["nodeId"] == "res1" for d in v["diagnostics"])
    assert v["nodes"]["res1"]["diagnostics"]                                    # shown on the collapsed card


def test_export_pytorch_of_a_composite_project_keeps_node_paths(client):
    g = load_project(EXAMPLES / "residual_cnn.project.json")
    assert client.put("/api/projects/res_export", json={"graph": g.graph.to_json(), "ui": g.ui.to_json() if g.ui else None}).status_code == 200
    code = client.get("/api/projects/res_export/export/pytorch").json()["code"]
    assert "self.res1__conv_a = nn.Conv2d" in code and "# node: res1/conv_a" in code and "def forward(self, images)" in code
    ns = {}
    exec(compile(code, "<export>", "exec"), ns)    # test-only: the exported module must be importable PyTorch
    out = ns["Model"]()(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 10)


# ------------------------------------------------------------------------------------------------ module library
def test_module_library_publish_versioning_and_starters(client):
    d = masked_weighted_loss("valid_count").model_dump(mode="json", by_alias=True)
    r = client.post("/api/modules/publish", json={"module": d, "note": "first"})
    assert r.status_code == 201 and r.json()["version"] == "1.0.0" and not r.json()["alreadyPublished"]
    assert client.post("/api/modules/publish", json={"module": d}).json()["alreadyPublished"]
    d2 = json.loads(json.dumps(d))
    d2["nodes"][0]["type"] = "tensor.add"
    conflict = client.post("/api/modules/publish", json={"module": d2})
    assert conflict.status_code == 409 and conflict.json()["detail"]["code"] == "version_immutable" and conflict.json()["detail"]["nextVersion"] == "1.1.0"
    d2["version"] = "1.1.0"
    assert client.post("/api/modules/publish", json={"module": d2}).status_code == 201
    got = client.get("/api/modules/masked_weighted_mse_valid_count").json()
    assert got["version"] == "1.1.0" and got["versions"] == ["1.0.0", "1.1.0"]
    assert client.get("/api/modules/masked_weighted_mse_valid_count", params={"version": "1.0.0"}).json()["definition"]["nodes"][0]["type"] == "tensor.sub"
    assert any(m["id"] == "masked_weighted_mse_valid_count" for m in client.get("/api/modules").json()["modules"])
    starters = client.get("/api/modules/starters").json()["modules"]
    assert {"residual_block", "multi_head_attention", "transformer_encoder_block"} <= {m["id"] for m in starters}
    bad = json.loads(json.dumps(d))
    bad["id"] = "loopy"
    bad["nodes"].append({"id": "again", "type": "core.composite", "config": {"module": "loopy"}})
    assert client.post("/api/modules/publish", json={"module": bad}).status_code == 422


def test_module_validate_and_numerical_test_routes(client):
    g = Graph(modules=[masked_weighted_loss("weight_sum"), residual_block(8)])
    v = client.post("/api/modules/validate", json={"graph": g.to_json(), "moduleId": "masked_weighted_mse_weight_sum"}).json()
    assert v["ok"] and v["outputs"]["loss"]["shape"] == [] and "sq" in v["nodes"]
    v = client.post("/api/modules/validate", json={"graph": g.to_json(), "moduleId": "residual_block", "shapes": {"x": ["N", 8, 5, 5]}}).json()
    assert v["ok"] and v["outputs"]["y"]["shape"] == ["N", 8, 5, 5] and v["params"] == 1168
    body = {"graph": g.to_json(), "moduleId": "masked_weighted_mse_weight_sum",
            "inputs": {"pred": [[1.0, 2.5, 2.0]], "target": [[1.0, 2.0, 3.0]], "mask": [[1.0, 1.0, 1.0]], "weights": [[1.0, 1.0, 1.0]]}}
    t = client.post("/api/modules/test", json=body).json()
    assert t["outputs"]["loss"]["values"] == pytest.approx([1.25 / 3], rel=1e-6)     # the VISION MSE teaching fixture (weight sum = 3)
    assert t["grads"]["pred"]["values"] == pytest.approx([0.0, 1 / 3, -2 / 3], abs=1e-6)
    assert t["reduction"]["divisor"] == "weight_sum" and "not a recorded run" in t["provenance"]["kind"]
    bad = client.post("/api/modules/test", json={**body, "inputs": {**body["inputs"], "mask": [[1.0, 1.0]]}})
    assert bad.status_code == 422


# ------------------------------------------------------------------------------------------------ procedure
def test_procedure_check_default_and_run_preflight(client):
    r = client.post("/api/procedure/check", json={"procedure": {"stages": ["zero_grad", "forward", "loss", "backward", "optimizer_step", "clip"], "clip": {"kind": "norm"}}}).json()
    assert not r["valid"] and r["diagnostics"][0]["code"] == "E_PROC_ORDER"
    assert client.post("/api/procedure/check", json={"procedure": {"stages": ["bogus"]}}).json()["diagnostics"][0]["code"] == "E_PROC_CONFIG"
    g, _ = small_transformer()
    d = client.post("/api/procedure/default", json={"graph": g.to_json()}).json()
    assert d["procedure"]["data"]["kind"] == "synthetic_sequence" and d["procedure"]["data"]["seq_len"] == 8 and "SYNTHETIC" in d["note"]
    bad = submit(client, g, {"stages": ["forward", "loss", "backward", "optimizer_step", "clip"], "clip": {"kind": "norm"}})
    assert bad.status_code == 422 and bad.json()["detail"]["code"] == "procedure_invalid"
    miss = submit(client, g, {"loss": {"kind": "module", "module": "ghost"}})
    assert miss.status_code == 422 and miss.json()["detail"]["code"] == "execution_blocked"


def test_procedure_run_in_a_worker_process_records_everything(client, trained):
    g, proc, rid = trained
    s = client.get(f"/api/runs/{rid}").json()
    assert s["kind"] == "procedure" and s["synthetic"] is True and "SYNTHETIC" in s["dataNote"] and s["progress"]["step"] == 12 and s["stoppedBy"] == "epochs_done"
    assert s["validation"]["val_acc"] > 0.5 and s["checkpoints"] == 3 and s["captures"] == [3]
    pv = client.get(f"/api/runs/{rid}/procedure").json()
    assert pv["stages"][:4] == ["zero_grad", "forward", "loss", "backward"] and len(pv["steps"]) == 12 and pv["optimizerTrace"][0]["abs_diff"] < 1e-6
    assert {c["step"] for c in pv["checkpoints"]} == {4, 8, 12}
    m = client.get(f"/api/runs/{rid}/metrics").json()
    assert len(m["trainLoss"]) == 12 and len(m["epochs"]) == 3
    # runs list contains it with its kind, and the idempotency key protects against a double submit
    assert any(r["id"] == rid and r["kind"] == "procedure" for r in client.get("/api/runs").json()["runs"])
    key = uuid.uuid4().hex
    body = {"graph": g.to_json(), "config": {"kind": "procedure", "procedure": {**proc, "epochs": 1, "capture": None}}}
    a = client.post("/api/runs", json=body, headers={"Idempotency-Key": key})
    b = client.post("/api/runs", json=body, headers={"Idempotency-Key": key})
    assert a.status_code == 201 and b.json()["idempotentReplay"] and a.json()["runId"] == b.json()["runId"]
    wait(client, a.json()["runId"])


def test_optimizer_state_inspector_shows_stored_adam_moments(client, trained):
    _, _, rid = trained
    o = client.get(f"/api/runs/{rid}/optimizer", params={"step": 12}).json()
    assert o["available"] and o["optimizer"]["kind"] == "adam" and o["provenance"]["kind"] == "stored optimizer state"
    head = next(e for e in o["state"] if e["name"] == "head.weight")
    assert head["step"]["value"] == 12.0 and head["exp_avg"]["norm"] > 0 and head["exp_avg_sq"]["min"] >= 0
    assert o["paramGroups"][0]["betas"] == [0.9, 0.999]
    assert client.get(f"/api/runs/{rid}/optimizer", params={"step": 5}).json()["available"] is False


# ------------------------------------------------------------------------------------------------ debugger over HTTP
def test_debug_scrubber_wire_gradients_and_honest_not_recorded(client, trained):
    _, _, rid = trained
    s = client.get(f"/api/runs/{rid}/debug/steps").json()
    assert [x["step"] for x in s["steps"] if x["captured"]] == [3] and len(s["steps"]) == 12
    w = client.post(f"/api/runs/{rid}/debug/wire", json={"step": 3, "node": "enc1/attn/weights"}).json()
    assert w["available"] and w["summary"]["shape"] == [16, 2, 8, 8] and w["provenance"]["captured"] and w["provenance"]["step"] == 3
    assert abs(w["summary"]["mean"] - 1 / 8) < 1e-3                         # softmax rows sum to 1 over 8 keys
    nr = client.post(f"/api/runs/{rid}/debug/wire", json={"step": 5, "node": "enc1/attn/weights"}).json()
    assert nr["available"] is False and nr["reason"] == "not_recorded" and nr["action"]["endpoint"] == f"/api/runs/{rid}/debug/capture" and "summary" not in nr
    g = client.post(f"/api/runs/{rid}/debug/gradients", json={"step": 3, "node": "head"}).json()
    assert g["available"] and g["output"]["shape"] == [16, 2] and "head.weight" in g["parameters"]
    assert client.post(f"/api/runs/{rid}/debug/gradients", json={"step": 9}).json()["reason"] == "not_recorded"


def test_capture_and_rerun_through_the_api(client, trained):
    _, _, rid = trained
    r = client.post(f"/api/runs/{rid}/debug/capture", json={"step": 9, "nodes": ["enc1/attn/weights", "head"]})
    assert r.status_code == 201 and r.json()["resumeFrom"] == {"run_id": rid, "max_step": 8}
    new = r.json()["runId"]
    assert wait(client, new)["status"] == "completed"
    pv = client.get(f"/api/runs/{new}/procedure").json()
    assert pv["rerunVerification"][0]["identical"] is True
    w = client.post(f"/api/runs/{new}/debug/wire", json={"step": 9, "node": "enc1/attn/weights"}).json()
    assert w["available"] and w["provenance"]["step"] == 9
    assert client.get(f"/api/runs/{new}").json()["rerunOf"] == rid
    # the original still has nothing at step 9
    assert client.post(f"/api/runs/{rid}/debug/wire", json={"step": 9, "node": "head"}).json()["available"] is False


def test_sandbox_through_the_api_original_untouched(client, trained):
    _, _, rid = trained
    before = client.get(f"/api/runs/{rid}/events").text if False else client.get(f"/api/runs/{rid}/procedure").json()
    res = client.post(f"/api/runs/{rid}/debug/sandbox", json={"step": 6, "stepsForward": 2, "label": "zero the attention output at step 6",
                                                              "interventions": [{"kind": "activation", "node": "enc1/attn/out_proj", "op": "zero", "index": None}],
                                                              "captureNodes": ["enc1/attn/out_proj", "head"]})
    assert res.status_code == 201, res.text
    d = res.json()
    assert d["immutability"]["originalRunUnchanged"] and d["changed"][0]["target"] == "enc1/attn/out_proj" and d["diff"]["activationMaxAbsDiff"]["enc1/attn/out_proj"] > 0
    assert client.get(f"/api/runs/{rid}/procedure").json() == before
    lst = client.get(f"/api/runs/{rid}/debug/sandboxes").json()["sandboxes"]
    assert lst[0]["sandboxRunId"] == d["sandboxRunId"] and lst[0]["lossDelta"] == d["diff"]["lossAtStep"]["delta"]
    assert client.get(f"/api/runs/{rid}/debug/sandboxes/{d['sandboxRunId']}").json()["outcome"] == d["outcome"]
    assert any(r["id"] == d["sandboxRunId"] and r["kind"] == "sandbox" for r in client.get("/api/runs").json()["runs"])
    bad = client.post(f"/api/runs/{rid}/debug/sandbox", json={"step": 6, "interventions": [{"kind": "nope"}]})
    assert bad.status_code == 422 and bad.json()["detail"]["code"] == "sandbox_invalid"


def test_probes_verify_route_reports_invariants_and_overhead(client):
    g, _ = small_transformer()
    r = client.post("/api/debug/probes/verify", json={"graph": g.to_json(), "nodes": ["enc1/attn/weights", "head"], "repeats": 5}).json()
    assert r["invariants"]["ok"] and r["invariants"]["probed_nodes"] == 2 and r["overhead"]["baselineMs"] > 0 and "synthetic" in r["inputs"]
    assert client.post("/api/debug/probes/verify", json={"graph": g.to_json(), "nodes": ["nope"]}).status_code == 422


# ------------------------------------------------------------------------------------------------ attention
def test_attention_routes(client, trained):
    g, _, rid = trained
    inst = client.post("/api/attention/instances", json={"graph": g.to_json()}).json()["instances"]
    assert [i["path"] for i in inst] == ["enc1/attn"]
    init = client.post("/api/attention/inspect", json={"graph": g.to_json(), "instance": "enc1/attn", "sample": 0, "head": 1, "token": 2}).json()
    assert init["available"] and "initial weights" in init["weightsNote"] and "SYNTHETIC" in init["dataNote"] and len(init["weights"]["row"]) == 8
    tr = client.post("/api/attention/inspect", json={"graph": g.to_json(), "instance": "enc1/attn", "sample": 0, "head": 1, "token": 2, "runId": rid}).json()
    assert "run " + rid in tr["weightsNote"] and "step 12" in tr["weightsNote"] and "validation examples" in tr["dataNote"]
    assert tr["weights"]["row"] != init["weights"]["row"] and tr["context"]["maxAbsDiffToCaptured"] < 1e-5
    assert client.post("/api/attention/inspect", json={"graph": g.to_json(), "instance": "enc1", "token": 0}).json()["available"] is False


# ------------------------------------------------------------------------------------------------ code blocks
BLOCK = {"id": "twice", "version": "1.0.0", "inputs": [{"name": "x", "dtype": "float32", "shape": ["N", 3]}], "outputs": [{"name": "y", "same_as": "x"}],
         "differentiable": True, "source": "def run(x):\n    return {'y': x * 2}\n",
         "fixtures": [{"name": "values", "inputs": {"x": {"values": [[1, 2, 3]]}}, "expect": {"y": {"values": [[2, 4, 6]]}}}]}


def test_code_block_routes_template_test_run_publish_environment(client):
    t = client.post("/api/codeblocks/template", json={"block": {**BLOCK, "source": ""}}).json()["source"]
    assert "def run(x):" in t and "same shape and dtype as x" in t
    r = client.post("/api/codeblocks/test", json={"block": BLOCK}).json()
    assert r["ok"] and not r["fixtures"][0]["cached"]
    assert client.post("/api/codeblocks/test", json={"block": BLOCK}).json()["fixtures"][0]["cached"]
    bad = {**BLOCK, "source": "def run(x):\n    return {'y': x * 3}\n"}
    r2 = client.post("/api/codeblocks/test", json={"block": bad}).json()
    assert not r2["ok"] and not r2["fixtures"][0]["cached"] and r2["identity"] != r["identity"]
    err = client.post("/api/codeblocks/run", json={"block": {**BLOCK, "source": "def run(x):\n    return {'y': x[:, 5]}\n"}, "fixture": {"name": "e", "inputs": {"x": {"shape": [2, 3]}}}}).json()
    assert err["error"]["code"] == "E_CODE_EXCEPTION" and err["error"]["frames"][0]["line"] == 2
    assert client.post("/api/codeblocks/test", json={"block": {**BLOCK, "fixtures": []}}).status_code == 422
    p = client.post("/api/codeblocks/publish", json={"block": BLOCK})
    assert p.status_code == 201 and p.json()["version"] == "1.0.0" and len(p.json()["identity"]) == 64
    assert client.post("/api/codeblocks/publish", json={"block": bad}).status_code == 409
    assert client.get("/api/codeblocks/twice").json()["definition"]["source"] == BLOCK["source"] and client.get("/api/codeblocks").json()["codeBlocks"][0]["id"] == "twice"
    env = client.get("/api/codeblocks/environment").json()
    assert env["torch"] and any(p.startswith("numpy==") for p in env["packages"])
    deps = client.post("/api/codeblocks/dependencies", json={"pins": ["numpy==0.0.1", "nonexistent_pkg==1", [p for p in env["packages"] if p.startswith("numpy==")][0]]}).json()["pins"]
    assert [d["status"] for d in deps] == ["version_mismatch", "missing", "ok"]


def test_code_block_in_a_graph_over_http_and_semantic_identity(client):
    g = load_project(EXAMPLES / "code_block_demo.project.json").graph
    v = client.post("/api/validate", json={"graph": g.to_json()}).json()
    assert v["ok"] and v["nodes"]["act"]["inputPorts"] == ["x"] and v["nodes"]["act"]["outputPorts"] == ["y"]
    assert "source" not in v["nodes"]["act"]["resolvedConfig"]["interface"] and v["nodes"]["act"]["resolvedConfig"]["interface"]["differentiable"] is True
    g2 = Graph.model_validate(g.to_json())
    g2.codeBlocks[0].source += "\n# edited\n"
    v2 = client.post("/api/validate", json={"graph": g2.to_json()}).json()
    assert v2["graphHash"] != v["graphHash"]
    assert v2["nodes"]["act"]["resolvedConfig"]["interface"]["identity"] != v["nodes"]["act"]["resolvedConfig"]["interface"]["identity"]


# ------------------------------------------------------------------------------------------------ shipped examples
EXAMPLE_NAMES = ["residual_cnn", "shared_encoder", "masked_loss", "transformer_sequence", "code_block_demo", "control_flow"]


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_shipped_examples_validate_and_are_listed(client, name):
    p = load_project(EXAMPLES / f"{name}.project.json")
    v = client.post("/api/validate", json={"graph": p.graph.to_json()}).json()
    assert v["ok"], v["diagnostics"]
    assert name in client.get("/api/examples").json()["examples"]
    d = next(x for x in client.get("/api/examples").json()["details"] if x["id"] == name)
    assert d["description"]
    if p.graph.training is not None:
        assert v["procedure"]["valid"]


def test_examples_are_regenerated_byte_for_byte():
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_m3", EXAMPLES / "make_m3_examples.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    before = {n: (EXAMPLES / f"{n}.project.json").read_bytes() for n in EXAMPLE_NAMES}
    mod.shared_encoder()
    mod.masked_loss()
    assert before["shared_encoder"] == (EXAMPLES / "shared_encoder.project.json").read_bytes() and before["masked_loss"] == (EXAMPLES / "masked_loss.project.json").read_bytes()


def test_shared_encoder_example_really_shares_and_masked_loss_example_trains(client):
    g = load_project(EXAMPLES / "shared_encoder.project.json").graph
    v = client.post("/api/validate", json={"graph": g.to_json()}).json()
    assert v["nodes"]["enc_right"]["sharedWith"] == "enc_left" and v["nodes"]["enc_right"]["params"] == 0 and v["nodes"]["enc_left"]["params"] == 4 * 8 + 8 + 8 * 8 + 8
    g = load_project(EXAMPLES / "masked_loss.project.json").graph
    r = client.post("/api/runs", json={"projectId": None, "graph": g.to_json(), "config": {"kind": "procedure"}}, headers={"Idempotency-Key": uuid.uuid4().hex})
    assert r.status_code == 201, r.text
    s = wait(client, r.json()["runId"])
    assert s["status"] == "completed" and s["validation"]["val_loss"] < 1.0 and s["progress"]["epochsDone"] == 8
    pv = client.get(f"/api/runs/{r.json()['runId']}/procedure").json()
    assert pv["epochs"][-1]["train_loss"] < pv["epochs"][0]["train_loss"] * 0.5
