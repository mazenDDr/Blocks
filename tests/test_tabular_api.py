"""Control service for tabular graphs: registry, validate, runs through the worker process, inspect, downloads."""
import hashlib
import time
import uuid

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from control.app import create_app
from tabular_helpers import FIX, example, rewire, set_cfg

TERMINAL = {"completed", "failed", "cancelled"}


def wait_for(client, rid, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        r = client.get(f"/api/runs/{rid}").json()
        if r["status"] in TERMINAL:
            return r
        time.sleep(0.1)
    raise AssertionError(f"run {rid} still {r['status']}")


def submit(client, g, key=None, project=None):
    body = {"graph": g.to_json(), "config": {}} if project is None else {"projectId": project, "config": {}}
    return client.post("/api/runs", json=body, headers={"Idempotency-Key": key or uuid.uuid4().hex})


def inspect(client, rid, **body):
    return client.post(f"/api/runs/{rid}/inspect", json=body)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with TestClient(create_app(tmp_path_factory.mktemp("wb"))) as c:
        yield c


@pytest.fixture(scope="module")
def reg_run(client):
    r = submit(client, example("tabular_regression"))
    assert r.status_code == 201, r.text
    s = wait_for(client, r.json()["runId"])
    assert s["status"] == "completed", s
    return s


@pytest.fixture(scope="module")
def gamma_run(client):
    r = submit(client, example("gamma_teaching"))
    return wait_for(client, r.json()["runId"])


def test_registry_lists_tabular_ops_with_typed_ports(client):
    ops = {o["type"]: o for o in client.get("/api/registry").json()["ops"]}
    tab = {t: o for t, o in ops.items() if o["graphKind"] == "tabular"}
    assert set(tab) >= {"tabular.csv_source", "tabular.profile", "tabular.duplicates", "tabular.select_columns", "tabular.drop_missing",
                        "tabular.train_validation_split", "tabular.fit_standardize", "tabular.fit_onehot", "tabular.fit_impute", "tabular.apply_transform",
                        "sklearn.linear_regression", "sklearn.logistic_regression", "sklearn.metrics", "tabular.predictions_export",
                        "scipy.gamma_function", "scipy.gamma_distribution", "scipy.tail_probability", "scipy.hypothesis_test", "scipy.two_group_comparison"}
    fit = tab["tabular.fit_standardize"]
    assert fit["inputKinds"] == {"train": "table"} and fit["outputKinds"] == {"fit": "fit_state"} and fit["backend"] == "scikit-learn"
    assert {"columns", "with_mean", "with_std"} <= fit["configSchema"]["properties"].keys()
    assert tab["scipy.gamma_distribution"]["defaults"]["shape"] == 2.0 and tab["sklearn.linear_regression"]["category"] == "Models"
    assert ops["pytorch.nn.conv2d"]["graphKind"] == "model" and ops["pytorch.nn.conv2d"]["inputKinds"] == {"input": "tensor"}
    # no tabular op has a learning rate / epoch control for ordinary least squares
    assert not {"lr", "learning_rate", "epochs"} & tab["sklearn.linear_regression"]["configSchema"]["properties"].keys()


def test_validate_returns_schema_rows_and_explain_per_node(client):
    v = client.post("/api/validate", json={"graph": example("tabular_regression").to_json()}).json()
    assert v["ok"] and v["graphKind"] == "tabular" and v["diagnostics"] == []
    h = v["nodes"]["housing"]["outputShapes"]["table"]
    assert h["kind"] == "table" and h["rows"] == 412 and h["partition"] == "full" and h["columns"][1] == {"name": "area_m2", "dtype": "float"}
    assert v["nodes"]["split"]["outputShapes"]["train"]["rows"] == 309 or v["nodes"]["split"]["outputShapes"]["train"]["rows"] is None
    assert v["nodes"]["split"]["outputShapes"]["validation"]["partition"] == "validation"
    assert v["nodes"]["sc_fit"]["outputShapes"]["fit"]["kind"] == "fit_state"
    assert "equation" in v["nodes"]["ols"]["explain"] and v["nodes"]["ols"]["resolvedConfig"]["target"] == "price_k"


def test_validate_reports_leakage_with_path_and_fix(client):
    g = example("tabular_regression")
    rewire(g, "oh_fit", "train", "split", "validation")
    v = client.post("/api/validate", json={"graph": g.to_json()}).json()
    d = [x for x in v["diagnostics"] if x["code"] == "E_LEAKAGE_FIT_ON_HELDOUT"]
    assert not v["ok"] and len(d) == 1 and d[0]["nodeId"] == "oh_fit" and d[0]["port"] == "train" and d[0]["path"] == "/nodes/oh_fit/ports/train"
    assert d[0]["fixes"] and "split.train" in d[0]["fixes"][0]["label"]
    assert v["nodes"]["oh_fit"]["diagnostics"][0]["code"] == "E_LEAKAGE_FIT_ON_HELDOUT"
    r = submit(client, g)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "execution_blocked"
    assert r.json()["detail"]["diagnostics"][0]["code"] == "E_LEAKAGE_FIT_ON_HELDOUT"


def test_examples_are_listed_with_their_graph_kind(client):
    ex = {d["id"]: d for d in client.get("/api/examples").json()["details"]}
    assert ex["tabular_regression"]["graphKind"] == "tabular" and ex["tabular_regression"]["synthetic"] is True
    assert ex["gamma_teaching"]["description"].startswith("TEACHING FIXTURE") and ex["reference_cnn"]["graphKind"] == "model"
    e = client.get("/api/examples/two_group_comparison").json()
    assert e["graph"]["graphKind"] == "tabular" and "synthetic" in e["ui"]["description"].lower()
    assert client.put("/api/projects/my_gamma", json={"graph": e["graph"], "ui": e["ui"]}).status_code == 200
    assert {"id": "my_gamma", "graphKind": "tabular"} .items() <= next(d for d in client.get("/api/projects").json()["details"] if d["id"] == "my_gamma").items()
    assert client.get("/api/projects/my_gamma/export/pytorch").status_code == 422


def test_run_goes_through_the_worker_with_status_events_and_provenance(client, reg_run):
    rid = reg_run["id"]
    assert reg_run["kind"] == "tabular" and reg_run["progress"] == {"nodesDone": 17, "nodes": 17}
    assert reg_run["sources"][0]["sha256"] == hashlib.sha256((FIX / "synthetic_housing.csv").read_bytes()).hexdigest()
    assert reg_run["splits"][0]["seed"] == 42 and reg_run["splits"][0]["nTrain"] == 300 and reg_run["failure"] is None
    assert all(n["status"] == "finished" for n in reg_run["nodes"])
    sse = client.get(f"/api/runs/{rid}/events").text
    types = [l[len("event: "):] for l in sse.splitlines() if l.startswith("event: ")]
    assert types[0] == "run_queued" and "run_started" in types and types.count("node_finished") == 17 and types[-2:] == ["run_finished", "end"]
    assert client.get("/api/runs", params={"project": "nope"}).json()["runs"] == []


def test_inspect_table_is_bounded_and_carries_provenance(client, reg_run):
    rid = reg_run["id"]
    t = inspect(client, rid, kind="table", node="housing", limit=5000).json()
    assert t["available"] and t["limit"] == 200 and len(t["rows"]) == 200 and t["total"] == 412 and t["truncated"]
    assert t["columns"][0]["name"] == "id" and t["rowIds"][:3] == [0, 1, 2]
    assert t["provenance"]["sourceSha256"] == hashlib.sha256((FIX / "synthetic_housing.csv").read_bytes()).hexdigest()
    page = inspect(client, rid, kind="table", node="housing", offset=400, limit=50).json()
    assert page["offset"] == 400 and len(page["rows"]) == 12 and not page["truncated"] and page["rowIds"][0] == 400
    raw = pd.read_csv(FIX / "synthetic_housing.csv")
    assert page["rows"][0][0] == int(raw.loc[400, "id"])
    tr = inspect(client, rid, kind="table", node="split", port="train", limit=3).json()
    assert tr["partition"] == "train" and tr["total"] == 300 and tr["provenance"]["split"]["seed"] == 42 and tr["provenance"]["graphHash"] == reg_run["graphHash"]
    va = inspect(client, rid, kind="table", node="sc_val", limit=3).json()
    assert va["partition"] == "validation" and va["total"] == 100 and "neighborhood_central" in [c["name"] for c in va["columns"]]
    assert inspect(client, rid, kind="table", node="split", port="nope").status_code == 422
    assert inspect(client, rid, kind="table", node="nope").status_code == 422


def test_inspect_profile_fit_state_coefficients_and_metrics(client, reg_run):
    rid = reg_run["id"]
    p = inspect(client, rid, kind="profile", node="profile").json()
    assert p["available"] and p["data"]["rows"] == 412 and p["data"]["duplicateRows"] == 12 and p["data"]["exact"] is True
    f = inspect(client, rid, kind="fit_state", node="sc_fit").json()
    assert f["data"]["columns"] == ["area_m2", "rooms", "age_years"] and set(f["data"]["mean"]) == {"area_m2", "rooms", "age_years"}
    assert f["provenance"]["fittedOn"]["partition"] == "train" and f["provenance"]["fittedOn"]["node"] == "split"
    assert inspect(client, rid, kind="fit_state", node="oh_fit").json()["data"]["categories"]["neighborhood"] == ["central", "north", "south"]
    c = inspect(client, rid, kind="coefficients", node="ols").json()
    assert [x["feature"] for x in c["data"]["coefficients"]][:3] == ["area_m2", "rooms", "age_years"] and "neighborhood_central" in str(c["data"]["coefficients"])
    m = inspect(client, rid, kind="metrics", node="metrics").json()
    assert set(m["data"]["values"]) == {"mse", "rmse", "mae", "r2"} and m["data"]["evaluatedPartition"] == "validation" and len(m["data"]["points"]) == 100
    wrong = inspect(client, rid, kind="metrics", node="ols")
    assert wrong.status_code == 422 and "coefficients" in wrong.json()["detail"]["message"]
    assert inspect(client, rid, kind="weights", node="ols").status_code == 422  # model-graph inspection does not apply


def test_predictions_download_matches_the_recorded_table(client, reg_run):
    rid = reg_run["id"]
    r = client.get(f"/api/runs/{rid}/tables/predictions/predictions.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    df = pd.read_csv(pd.io.common.StringIO(r.text), index_col="row_id")
    assert list(df.columns) == ["observed", "predicted", "residual"] and len(df) == 100
    assert client.get(f"/api/runs/{rid}/tables/housing/nope.csv").status_code == 404


def test_gamma_run_inspect_distribution_tail_and_test_result(client, gamma_run):
    rid = gamma_run["id"]
    assert gamma_run["status"] == "completed"
    d = inspect(client, rid, kind="distribution", node="null_distribution").json()["data"]
    assert d["shape"] == 2.0 and len(d["curve"]["x"]) == 300
    t = inspect(client, rid, kind="tail", node="upper_tail").json()
    assert t["data"]["pValue"] == pytest.approx(0.0404276820, abs=5e-11) and t["provenance"]["runId"] == rid
    r = inspect(client, rid, kind="test_result", node="test").json()
    assert r["data"]["decision"] == "reject_h0" and r["data"]["teachingFixture"] is True and r["provenance"]["graphHash"] == gamma_run["graphHash"]


def test_alpha_change_via_api_keeps_p_value(client, gamma_run):
    g = example("gamma_teaching")
    set_cfg(g, "test", alpha=0.04)
    s = wait_for(client, submit(client, g).json()["runId"])
    r = inspect(client, s["id"], kind="test_result", node="test").json()["data"]
    assert r["decision"] == "fail_to_reject_h0"
    assert r["pValue"] == inspect(client, gamma_run["id"], kind="test_result", node="test").json()["data"]["pValue"]
    assert s["graphHash"] != gamma_run["graphHash"]


def test_two_group_run_via_api(client):
    s = wait_for(client, submit(client, example("two_group_comparison")).json()["runId"])
    assert s["status"] == "completed"
    r = inspect(client, s["id"], kind="test_result", node="compare").json()["data"]
    assert r["method"] == "welch" and r["design"]["sampleUnit"] and r["meanDifference"]["ciLow"] < r["meanDifference"]["value"] < r["meanDifference"]["ciHigh"]


def test_failed_node_is_reported_with_code_and_later_nodes_stay_absent(client):
    g = example("two_group_comparison")
    set_cfg(g, "compare", group_b="missing_group")
    s = wait_for(client, submit(client, g).json()["runId"])
    assert s["status"] == "failed" and s["failure"]["node"] == "compare" and s["failure"]["code"] == "E_GROUP_NOT_FOUND"
    assert [n["status"] for n in s["nodes"]] == ["finished", "finished", "failed"]
    u = inspect(client, s["id"], kind="test_result", node="compare").json()
    assert u["available"] is False and u["reason"] == "not_recorded"


def test_idempotent_submit_for_tabular_runs(client):
    g = example("gamma_teaching")
    key = uuid.uuid4().hex
    a = submit(client, g, key)
    b = submit(client, g, key)
    assert a.status_code == 201 and b.status_code == 200 and b.json()["idempotentReplay"] and a.json()["runId"] == b.json()["runId"]
    g2 = example("gamma_teaching")
    set_cfg(g2, "test", alpha=0.1)
    assert submit(client, g2, key).status_code == 409
    wait_for(client, a.json()["runId"])


def test_invalid_run_configs_are_rejected(client):
    r = client.post("/api/runs", json={"graph": example("gamma_teaching").to_json(), "config": {"nonsense": 1}}, headers={"Idempotency-Key": "k-bad"})
    assert r.status_code == 422 and "nonsense" in r.json()["detail"]["message"]
    from conftest import EXAMPLES
    from graph_core.project_io import load_project
    cnn = load_project(EXAMPLES / "reference_cnn.project.json").graph.to_json()
    r = client.post("/api/runs", json={"graph": cnn, "config": {}}, headers={"Idempotency-Key": "k-bad2"})
    assert r.status_code == 422 and "data" in r.json()["detail"]["message"]
