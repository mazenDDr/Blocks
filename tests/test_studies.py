"""Studies and sweeps: bounded grid/random plans, repeat identities, failed vs retried vs repeated trials, baselines (A47, A48)."""
import copy
import json
import math
import shutil
import time

import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient

from control.app import create_app
from conftest import EXAMPLES
from connectors import journey
from tabular_helpers import example

OBJ = {"metric": {"name": "rmse", "node": "metrics"}, "direction": "minimize"}


def wait_study(c, sid, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        s = c.get(f"/api/studies/{sid}").json()
        if s["state"] not in ("running", "planned") and all(t["status"] not in ("running",) for t in s["trials"]):
            return s
        time.sleep(0.3)
    raise AssertionError(f"study {sid} still {s['state']}: {[(t['trialId'], t['status']) for t in s['trials']]}")


def body(g, **kw):
    b = {"name": "test study", "graph": g.to_json(), "run_config": {}, "objective": OBJ, "limits": {"max_trials": 20}, **kw}
    return b


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with TestClient(create_app(tmp_path_factory.mktemp("wb"))) as c:
        yield c


# ------------------------------------------------------------------------------------------------ planning
def test_plan_expands_grid_with_explicit_limits_and_never_truncates(client):
    g = example("tabular_regression")
    search = {"method": "grid", "variables": [{"target": {"scope": "node", "node": "ols", "field": "fit_intercept"}, "values": [True, False]},
                                              {"target": {"scope": "node", "node": "imp_fit", "field": "strategy"}, "values": ["median", "mean"]}]}
    r = client.post("/api/studies/plan", json=body(g, search=search, repeats={"seeds": [1, 2, 3]}, limits={"max_trials": 100}))
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["counts"] == {"total": 15, "groups": 5, "seeds": 3, "folds": 1, "invalid": 0}  # (4 grid configurations + baseline) x 3 seeds
    assert {t["groupIdx"] for t in p["trials"]} == {0, 1, 2, 3, 4} and p["trials"][0]["isBaseline"] and p["trials"][0]["label"] == "baseline"
    assert len({(t["groupIdx"], t["seed"]) for t in p["trials"]}) == 15
    over = client.post("/api/studies/plan", json=body(g, search=search, repeats={"seeds": [1, 2, 3]}, limits={"max_trials": 14}))
    assert over.status_code == 422 and over.json()["detail"]["code"] == "E_SWEEP_LIMIT" and over.json()["detail"]["detail"] == {"trials": 15, "maxTrials": 14}
    assert client.post("/api/studies/plan", json=body(g, limits={"max_trials": 5, "max_concurrency": 2})).status_code == 422  # concurrency is 1 for now
    assert client.post("/api/studies/plan", json=body(g, limits={"max_trials": 100000})).status_code == 422  # hard cap
    assert not client.get("/api/studies").json()["studies"]  # planning stored nothing


def test_plan_random_search_is_seeded_and_bounded(client):
    g = example("tabular_regression")
    vs = [{"target": {"scope": "node", "node": "split", "field": "validation_fraction"}, "range": {"low": 0.1, "high": 0.4}}]
    mk = lambda seed: client.post("/api/studies/plan", json=body(g, search={"method": "random", "variables": vs, "random_trials": 4, "sampler_seed": seed}, include_baseline=False)).json()  # noqa: E731
    a, b, other = mk(1), mk(1), mk(2)
    vals = lambda p: [t["assignments"][0]["value"] for t in p["trials"]]  # noqa: E731
    assert vals(a) == vals(b) and vals(a) != vals(other) and len(vals(a)) == 4 and all(0.1 <= v <= 0.4 for v in vals(a))


def test_plan_rejects_bad_targets_objectives_and_flags_invalid_variants(client):
    g = example("tabular_regression")
    t = lambda **k: {"scope": "node", "node": "ols", "field": "fit_intercept", **k}  # noqa: E731
    bad_field = client.post("/api/studies/plan", json=body(g, search={"method": "grid", "variables": [{"target": t(field="learning_rate"), "values": [0.1]}]}))
    assert bad_field.status_code == 422 and bad_field.json()["detail"]["code"] == "E_SWEEP_TARGET"
    assert client.post("/api/studies/plan", json=body(g, search={"method": "grid", "variables": [{"target": {"scope": "run", "field": "lr"}, "values": [0.1]}]})).json()["detail"]["code"] == "E_SWEEP_TARGET"
    assert client.post("/api/studies/plan", json=body(g, objective={"metric": {"name": "accuracy", "node": "metrics"}, "direction": "maximize"})).json()["detail"]["code"] == "objective_invalid"
    assert client.post("/api/studies/plan", json=body(g, repeats={"folds": 3})).json()["detail"]["code"] == "E_SWEEP_FOLD_UNSUPPORTED"
    # a variant that fails validation is kept in the plan as `invalid` with its diagnostics and is never scheduled
    ok = client.post("/api/studies/plan", json=body(g, search={"method": "grid", "variables": [{"target": {"scope": "node", "node": "imp_fit", "field": "strategy"}, "values": ["mean", "bogus"]}]}))
    p = ok.json()
    inv = [x for x in p["trials"] if x["status"] == "invalid"]
    assert p["counts"]["invalid"] == 1 and inv[0]["assignments"][0]["value"] == "bogus" and inv[0]["diagnostics"][0]["code"] == "E_CONFIG"


# ------------------------------------------------------------------------------------------------ A47 / A48 end to end
@pytest.fixture(scope="module")
def grid_study(client):
    g = example("tabular_regression")
    search = {"method": "grid", "variables": [{"target": {"scope": "node", "node": "ols", "field": "fit_intercept"}, "values": [False], "labels": ["no intercept (ablation)"]}]}
    r = client.post("/api/studies", json=body(g, search=search, repeats={"seeds": [3, 4]}, hypothesis="the intercept matters", limits={"max_trials": 10}))
    assert r.status_code == 201, r.text
    return wait_study(client, r.json()["id"])


def test_a48_sweep_retains_validated_variants_objective_limits_and_baseline_comparison(client, grid_study):
    s = grid_study
    assert s["state"] == "completed" and s["counts"] == {"completed": 4} and s["limits"] == {"max_trials": 10, "max_concurrency": 1, "max_attempts": 3}
    assert s["objective"]["direction"] == "minimize" and "single evaluation" in s["objective"]["semantics"] and s["hypothesis"] == "the intercept matters"
    assert [g["label"] for g in s["groups"]] == ["baseline", "no intercept (ablation)"]
    assert s["baseline"]["mode"] == "baseline_group_mean" and s["baseline"]["n"] == 2
    base_vals = [t["value"] for t in s["trials"] if t["isBaseline"]]
    assert s["baseline"]["value"] == pytest.approx(np.mean(base_vals))
    var = [t for t in s["trials"] if not t["isBaseline"]]
    for t in var:
        assert t["delta"] == pytest.approx(t["value"] - s["baseline"]["value"]) and t["changedFields"] == ["ols.fit_intercept"] and t["attributionWarning"] is None
        assert t["better"] == (t["delta"] < 0)  # minimize: lower is better
        assert t["assignments"] == [{"target": {"scope": "node", "node": "ols", "field": "fit_intercept"}, "value": False}]
    g1 = s["groups"][1]
    assert g1["mean"] == pytest.approx(np.mean([t["value"] for t in var])) and g1["std"] == pytest.approx(np.std([t["value"] for t in var], ddof=1)) and g1["n"] == 2
    assert g1["rank"] in (1, 2) and s["best"]["label"] in ("baseline", "no intercept (ablation)")


def test_a47_seeds_and_attempts_are_distinct_identities_and_metrics_carry_semantics(client, grid_study):
    s = grid_study
    assert sorted({t["seed"] for t in s["trials"]}) == [3, 4] and all(t["fold"] is None and t["attemptCount"] == 1 and not t["retried"] for t in s["trials"])
    assert len({t["runId"] for t in s["trials"]}) == 4  # one run per trial attempt
    base = [t for t in s["trials"] if t["isBaseline"]]
    assert base[0]["value"] != base[1]["value"]  # a different seed gives a different split and a different metric: repeats are not one curve
    for t in s["trials"]:
        m = t["metric"]
        assert m["available"] and m["metric"] == "rmse" and m["step"] is None and "single evaluation" in m["aggregation"] and m["partition"] == "validation"
        assert m["n"] > 0 and m["provenance"]["runId"] == t["runId"] and m["provenance"]["node"] == "metrics"
        run = client.get(f"/api/runs/{t['runId']}").json()
        ident = run["config"]["trial"]
        assert ident == {"studyId": s["id"], "trialId": t["trialId"], "attempt": 1, "groupKey": t["groupKey"], "seed": t["seed"], "fold": None, "isBaseline": t["isBaseline"]}
        assert run["runSeed"]["seed"] == t["seed"] and run["runSeed"]["applied"] == [{"node": "split", "was": 42, "now": t["seed"]}]
        insp = client.post(f"/api/runs/{t['runId']}/inspect", json={"kind": "metrics", "node": "metrics"}).json()
        assert insp["data"]["values"]["rmse"] == pytest.approx(t["value"])
    # structural + parameter diff: same seed, one config field changed -> exactly one difference, no attribution warning
    a = next(t for t in s["trials"] if t["isBaseline"] and t["seed"] == 3)
    b = next(t for t in s["trials"] if not t["isBaseline"] and t["seed"] == 3)
    d = client.get(f"/api/runs/{a['runId']}/diff/{b['runId']}").json()
    assert [(c["node"], c["field"], c["from"], c["to"]) for c in d["graph"]["changes"]] == [("ols", "fit_intercept", True, False)] and d["runConfig"] == [] and d["warning"] is None
    # different seed, same graph: only the run-config seed differs
    d2 = client.get(f"/api/runs/{a['runId']}/diff/{next(t for t in s['trials'] if t['isBaseline'] and t['seed'] == 4)['runId']}").json()
    assert d2["graph"]["changes"] == [] and [c["field"] for c in d2["runConfig"]] == ["seed"]
    # the baseline can be pinned to one run; trials are then compared with that single run
    pinned = client.put(f"/api/studies/{s['id']}/baseline", json={"runId": a["runId"]}).json()
    assert pinned["baseline"]["mode"] == "pinned_run" and pinned["baseline"]["runId"] == a["runId"] and pinned["baseline"]["value"] == pytest.approx(a["value"])
    assert pinned["trials"][3]["delta"] == pytest.approx(pinned["trials"][3]["value"] - a["value"])
    assert client.put(f"/api/studies/{s['id']}/baseline", json={"runId": "nope"}).status_code == 422
    assert client.put(f"/api/studies/{s['id']}/baseline", json={"runId": None}).json()["baseline"]["mode"] == "baseline_group_mean"


def test_failed_retried_and_repeated_trials_are_distinguished(client, tmp_path):
    late = tmp_path / "late.csv"
    shutil.copy(EXAMPLES / "fixtures" / "synthetic_housing.csv", late)
    g = example("tabular_regression")
    search = {"method": "grid", "variables": [{"target": {"scope": "node", "node": "housing", "field": "path"}, "values": [str(late)], "labels": ["late-arriving file"]}]}
    r = client.post("/api/studies", json=body(g, search=search, repeats={"seeds": [1, 2]}, start=False))
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    assert r.json()["state"] == "planned" and r.json()["counts"] == {"planned": 4}
    late.unlink()  # the source disappears between planning and execution
    client.post(f"/api/studies/{sid}/start")
    s = wait_study(client, sid)
    assert s["state"] == "completed_with_failures" and s["counts"] == {"completed": 2, "failed": 2}
    failed = [t for t in s["trials"] if t["status"] == "failed"]
    assert {t["seed"] for t in failed} == {1, 2} and all("E_SOURCE_NOT_FOUND" in t["error"] for t in failed)  # failed trials stay in the record, with their error
    assert s["groups"][1]["failed"] == 2 and s["groups"][1]["mean"] is None and s["groups"][1]["n"] == 0
    # retry one failed trial after the file is back: a NEW attempt of the SAME trial; the failed attempt stays recorded
    shutil.copy(EXAMPLES / "fixtures" / "synthetic_housing.csv", late)
    t0 = failed[0]
    assert client.post(f"/api/studies/{sid}/trials/{t0['trialId']}/retry").status_code == 202
    s = wait_study(client, sid)
    t = next(x for x in s["trials"] if x["trialId"] == t0["trialId"])
    assert t["status"] == "completed" and t["retried"] and t["attemptCount"] == 2
    assert [(a["attempt"], a["kind"], a["status"]) for a in t["attempts"]] == [(1, "initial", "failed"), (2, "retry", "completed")]
    assert "E_SOURCE_NOT_FOUND" in t["attempts"][0]["error"] and t["attempts"][0]["runId"] != t["attempts"][1]["runId"]
    assert client.get(f"/api/runs/{t['attempts'][1]['runId']}").json()["config"]["trial"]["attempt"] == 2
    # same seed, different attempt != different seed: the other failed trial is untouched; the group now has one completed repeat
    other = next(x for x in s["trials"] if x["trialId"] == failed[1]["trialId"])
    assert other["status"] == "failed" and other["attemptCount"] == 1 and s["groups"][1]["n"] == 1 and s["groups"][1]["std"] is None
    # illegal retries
    assert client.post(f"/api/studies/{sid}/trials/{t0['trialId']}/retry").status_code == 409  # completed trials are not retried
    assert client.post(f"/api/studies/{sid}/trials/t999/retry").status_code == 404


def test_folds_are_a_repeat_dimension_with_disjoint_validation_partitions(client):
    g = example("tabular_regression")
    r = client.post("/api/studies", json=body(g, repeats={"folds": 3, "fold_node": "split"}, limits={"max_trials": 3}, name="3-fold"))
    assert r.status_code == 201, r.text
    s = wait_study(client, r.json()["id"])
    assert s["counts"] == {"completed": 3} and [t["fold"] for t in s["trials"]] == [0, 1, 2] and all(t["seed"] is None for t in s["trials"])
    splits = []
    for t in s["trials"]:
        run = client.get(f"/api/runs/{t['runId']}").json()
        sp = run["splits"][0]
        assert (sp["nFolds"], sp["fold"]) == (3, t["fold"]) and run["config"]["trial"]["fold"] == t["fold"]
        splits.append(sp)
    n = splits[0]["nTrain"] + splits[0]["nValidation"]
    assert all(sp["nTrain"] + sp["nValidation"] == n for sp in splits) and sum(sp["nValidation"] for sp in splits) == n  # each row validates exactly once
    assert len({sp["validationRowIdsSha256"] for sp in splits}) == 3
    assert s["groups"][0]["n"] == 3 and s["groups"][0]["repeats"][1]["fold"] == 1


def test_cancel_stops_remaining_trials_and_keeps_the_record(client):
    g = example("tabular_regression")
    r = client.post("/api/studies", json=body(g, repeats={"seeds": [1, 2, 3, 4, 5, 6]}, limits={"max_trials": 10}, name="to cancel"))
    sid = r.json()["id"]
    for _ in range(100):
        if any(t["status"] in ("running", "completed") for t in client.get(f"/api/studies/{sid}").json()["trials"]):
            break
        time.sleep(0.1)
    assert client.post(f"/api/studies/{sid}/cancel").status_code == 202
    s = wait_study(client, sid)
    assert s["state"] == "cancelled" and not {t["status"] for t in s["trials"]} & {"planned", "running"} and s["counts"].get("cancelled", 0) >= 1
    assert client.post(f"/api/studies/{sid}/cancel").status_code == 409


# ------------------------------------------------------------------------------------------------ model graphs
def test_model_graph_sweep_over_node_config_and_run_config(client, shapes_dir):
    from test_api import cnn_json

    g = cnn_json()
    from graph_core.schema import Graph

    graph = Graph.model_validate(g)
    b = {"name": "cnn width x lr", "graph": g, "run_config": {"data": str(shapes_dir), "epochs": 1, "batch_size": 16},
         "objective": {"metric": {"name": "val_loss", "select": "last"}, "direction": "minimize"}, "limits": {"max_trials": 5},
         "search": {"method": "grid", "variables": [{"target": {"scope": "node", "node": "conv_1", "field": "out_channels"}, "values": [8]},
                                                    {"target": {"scope": "run", "field": "lr"}, "values": [0.1]}]}}
    plan = client.post("/api/studies/plan", json=b).json()
    assert plan["counts"]["total"] == 2 and plan["trials"][1]["label"] == "conv_1.out_channels=8, lr=0.1"
    # validated before scheduling: a width that breaks the interface is flagged invalid, not run
    bad = copy.deepcopy(b)
    bad["search"]["variables"][0]["values"] = [0]
    assert client.post("/api/studies/plan", json=bad).json()["counts"]["invalid"] == 1
    assert client.post("/api/studies/plan", json={**b, "repeats": {"folds": 2, "fold_node": "x"}}).json()["detail"]["code"] == "E_SWEEP_FOLD_UNSUPPORTED"
    assert client.post("/api/studies/plan", json={**b, "objective": {"metric": {"name": "r2", "node": "m"}, "direction": "maximize"}}).json()["detail"]["code"] == "objective_invalid"
    r = client.post("/api/studies", json=b)
    assert r.status_code == 201, r.text
    s = wait_study(client, r.json()["id"], timeout=400)
    assert s["state"] == "completed" and s["counts"] == {"completed": 2}
    t = s["trials"][1]
    m = t["metric"]
    assert m["metric"] == "val_loss" and m["step"] > 0 and m["epoch"] == 0 and "last over 1 epoch-end" in m["aggregation"] and m["partition"] == "validation"
    assert t["attributionWarning"] and t["changedFields"] == ["conv_1.out_channels", "lr"]  # two things changed: not attributable to either one
    d = client.get(f"/api/runs/{s['trials'][0]['runId']}/diff/{t['runId']}").json()
    assert [(c["node"], c["field"], c["from"], c["to"]) for c in d["graph"]["changes"]] == [("conv_1", "out_channels", 32, 8)]
    assert [(c["field"], c["from"], c["to"]) for c in d["runConfig"]] == [("lr", 0.05, 0.1)] and d["warning"]
    run = client.get(f"/api/runs/{t['runId']}").json()
    assert run["kind"] == "model" and run["config"]["lr"] == 0.1 and run["config"]["trial"]["trialId"] == t["trialId"]


# ------------------------------------------------------------------------------------------------ connected journey
def test_connected_journey_ablation_with_pinned_sources_is_independent_of_database_changes(lab, client_factory):
    c = client_factory(lab)
    g = journey.graph(db="lab", files="files")
    c.put("/api/projects/journey", json={"graph": g})
    v = c.post("/api/validate", json={"graph": g}).json()
    assert v["ok"], v["diagnostics"]
    rid = c.post("/api/runs", json={"projectId": "journey", "config": {}}, headers={"Idempotency-Key": "j1"}).json()["runId"]
    for _ in range(300):
        run = c.get(f"/api/runs/{rid}").json()
        if run["status"] in ("completed", "failed"):
            break
        time.sleep(0.2)
    assert run["status"] == "completed", run["failure"]
    pins = {x["node"]: x["snapshotId"] for x in run["snapshots"]}
    assert set(pins) == {"assays", "spectra", "images"} and {x["mode"] for x in run["snapshots"]} == {"live"}
    first = c.post(f"/api/runs/{rid}/inspect", json={"kind": "metrics", "node": "metrics"}).json()["data"]["values"]["rmse"]
    # the database changes underneath: every response drifts
    with psycopg.connect(lab.pg.admin_uri, autocommit=True) as conn:
        conn.execute("UPDATE assays SET response = response + 5")
    ab = {"method": "grid", "variables": [{"target": {"scope": "node", "node": "ols", "field": "features"},
                                           "values": [["ph", "temp_c", "absorbance_a", "size"], ["ph", "temp_c", "absorbance_a", "absorbance_b"]],
                                           "labels": ["drop image size", "drop absorbance_b"]}]}
    r = c.post("/api/studies", json={"name": "feature ablation on pinned sources", "projectId": "journey", "run_config": {"source_pins": pins}, "objective": OBJ,
                                    "search": ab, "repeats": {"seeds": [7, 8]}, "limits": {"max_trials": 10}})
    assert r.status_code == 201, r.text
    s = wait_study(c, r.json()["id"], timeout=400)
    assert s["state"] == "completed" and s["counts"] == {"completed": 6}
    base7 = next(t for t in s["trials"] if t["isBaseline"] and t["seed"] == 7)
    assert base7["value"] == pytest.approx(first)  # same pinned data and split seed => the first run's result, despite the database change
    for t in s["trials"]:
        run = c.get(f"/api/runs/{t['runId']}").json()
        assert {x["snapshotId"] for x in run["snapshots"]} == set(pins.values()) and {x["mode"] for x in run["snapshots"]} == {"pinned"}
    # without pins the same graph sees the changed database (and says so with a different snapshot)
    rid2 = c.post("/api/runs", json={"projectId": "journey", "config": {}}, headers={"Idempotency-Key": "j2"}).json()["runId"]
    for _ in range(300):
        run2 = c.get(f"/api/runs/{rid2}").json()
        if run2["status"] in ("completed", "failed"):
            break
        time.sleep(0.2)
    assert {x["snapshotId"] for x in run2["snapshots"] if x["node"] == "assays"} != {pins["assays"]}
