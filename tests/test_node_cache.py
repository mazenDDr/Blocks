"""A09: dependency-scoped node result cache for tabular graphs. Editing one node invalidates it and its dependents only; reused results are
byte-identical to a fresh execution; sources are re-read every run; corrupted entries are never trusted."""
import json
import shutil

import pandas as pd
import pytest

from artifact_store import ArtifactStore
from graph_core.hashing import semantic_hash
from graph_core.schema import Graph
from tabular import cache as C
from tabular_helpers import FIX, example, set_cfg
from worker.tabular_run import TabularRunConfig, run_tabular

UPSTREAM = ["profile", "dedupe", "select", "split", "imp_fit", "imp_train", "imp_val", "oh_fit", "oh_train", "oh_val"]
SCALER_AND_BELOW = ["sc_fit", "sc_train", "sc_val", "ols", "metrics", "predictions"]


def run(store: ArtifactStore, g: Graph, rid: str, mode: str = "reuse", **kw) -> dict:
    cfg = TabularRunConfig(cache=mode, project_id="p1", **kw)
    store.create_run(rid, semantic_hash(g), cfg.model_dump())
    assert run_tabular(g, cfg, store, rid) == "completed", store.get_run(rid)
    return {e["node_id"]: e["data"] for e in store.events(rid, types=("node_finished",))}


def outputs(store: ArtifactStore, rid: str) -> dict:
    """Every recorded output and summary of a run by (node, port) -> sha256: the content-addressed bytes a researcher inspects."""
    out = {(a["meta"]["node"], a["meta"]["port"]): a["sha256"] for a in store.artifacts(rid, "node_output")}
    out.update({(a["meta"]["node"], "summary"): a["sha256"] for a in store.artifacts(rid, "node_summary")})
    return out


def status(fin: dict) -> dict:
    return {n: d["cache"]["status"] for n, d in fin.items()}


@pytest.fixture
def graph(tmp_path):
    """The regression example reading a private copy of its fixture, so a test can change the file's content."""
    data = tmp_path / "housing.csv"
    shutil.copy(FIX / "synthetic_housing.csv", data)
    return set_cfg(example("tabular_regression"), "housing", path=str(data))


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "wb")


def test_first_run_misses_and_identical_rerun_reuses_every_cacheable_node(store, graph):
    first = run(store, graph, "r1")
    assert first["housing"]["cache"]["status"] == "bypass" and "re-read" in first["housing"]["cache"]["reason"]
    assert {n for n, s in status(first).items() if s == "miss"} == set(UPSTREAM + SCALER_AND_BELOW)
    assert all(first[n]["cache"]["changed"] == [] for n in UPSTREAM)

    second = run(store, graph, "r2")
    assert status(second) == {"housing": "bypass", **{n: "hit" for n in UPSTREAM + SCALER_AND_BELOW}}
    assert all(second[n]["cache"]["fromRun"] == "r1" and second[n]["cache"]["key"] == first[n]["cache"]["key"] for n in UPSTREAM + SCALER_AND_BELOW)
    # Reused results are recorded in the new run, byte-identical to the original execution and to an uncached execution.
    assert outputs(store, "r2") == outputs(store, "r1")
    run(store, graph, "plain", mode="off")
    assert outputs(store, "plain") == outputs(store, "r1")


def test_editing_one_node_invalidates_only_it_and_its_dependents(store, graph, tmp_path):
    first = run(store, graph, "r1")
    edited = set_cfg(graph.model_copy(deep=True), "sc_fit", columns=["area_m2", "age_years"])
    after = run(store, edited, "r2")
    assert {n for n, s in status(after).items() if s == "hit"} == set(UPSTREAM)
    assert {n for n, s in status(after).items() if s == "miss"} == set(SCALER_AND_BELOW)
    assert after["sc_fit"]["cache"]["changed"] == ["settings"]
    assert after["sc_train"]["cache"]["changed"] == ["input fit (from sc_fit)"]
    assert set(after["metrics"]["cache"]["changed"]) == {"input model (from ols)", "input table (from sc_val)"}
    assert "Invalidated" in after["ols"]["cache"]["reason"] and "r1" in after["ols"]["cache"]["reason"]
    assert all(after[n]["cache"]["key"] != first[n]["cache"]["key"] for n in SCALER_AND_BELOW)

    # The edited graph's results equal a fresh execution without any cache.
    fresh = ArtifactStore(tmp_path / "fresh")
    run(fresh, edited, "f", mode="off")
    assert outputs(store, "r2") == outputs(fresh, "f")

    # A late edit (the estimator) leaves every upstream preprocessing result valid, including the new scaler result.
    late = set_cfg(edited.model_copy(deep=True), "ols", fit_intercept=False)
    last = run(store, late, "r3")
    assert {n for n, s in status(last).items() if s == "miss"} == {"ols", "metrics", "predictions"}
    assert last["sc_fit"]["cache"]["fromRun"] == "r2" and last["profile"]["cache"]["fromRun"] == "r1"

    # Reverting to the original settings reuses the original results again.
    back = run(store, graph, "r4")
    assert status(back) == status(run(store, graph, "r5"))
    assert all(back[n]["cache"]["status"] == "hit" for n in UPSTREAM + SCALER_AND_BELOW)
    assert outputs(store, "r4") == outputs(store, "r1")


def test_changed_source_content_invalidates_everything_downstream(store, graph):
    run(store, graph, "r1")
    path = graph.node("housing").config["path"]
    df = pd.read_csv(path)
    df.loc[0, "price_k"] = float(df.loc[0, "price_k"]) + 1.0
    df.to_csv(path, index=False)
    after = run(store, graph, "r2")  # same configuration, different bytes on disk
    assert after["housing"]["cache"]["status"] == "bypass"
    assert {n for n, s in status(after).items() if s == "miss"} == set(UPSTREAM + SCALER_AND_BELOW)
    assert after["profile"]["cache"]["changed"] == ["input table (from housing)"]


def test_run_seed_override_invalidates_the_split_and_its_dependents_only(store, graph):
    run(store, graph, "r1")
    after = run(store, graph, "r2", seed=7)
    assert {n for n, s in status(after).items() if s == "hit"} == {"profile", "dedupe", "select"}
    assert after["split"]["cache"]["changed"] == ["settings"]


def test_implementation_or_environment_change_invalidates_all_cacheable_nodes(store, graph, monkeypatch):
    run(store, graph, "r1")
    real = C.implementation_identity
    monkeypatch.setattr(C, "implementation_identity", lambda: {**real(), "operations/tabular_ops.py": "0" * 64})
    after = run(store, graph, "r2")
    assert {n for n, s in status(after).items() if s == "miss"} == set(UPSTREAM + SCALER_AND_BELOW)
    assert "implementation" in after["profile"]["cache"]["changed"]
    monkeypatch.setattr(C, "implementation_identity", real)
    real_env = C.environment_identity
    monkeypatch.setattr(C, "environment_identity", lambda libs: {**real_env(libs), "scikit-learn": "0.0"})
    env = run(store, graph, "r3")
    assert env["split"]["cache"]["status"] == "miss" and "environment" in env["split"]["cache"]["changed"]


def test_a_corrupted_entry_is_never_trusted(store, graph):
    first = run(store, graph, "r1")
    sha = first["ols"]["cache"]["key"] and store.get_node_cache(first["ols"]["cache"]["key"])["sha256"]
    store.path_of(sha).write_bytes(b"not the recorded pickle")
    after = run(store, graph, "r2")
    assert after["ols"]["cache"]["status"] == "miss" and "integrity" in after["ols"]["cache"]["reason"]
    assert after["sc_fit"]["cache"]["status"] == "hit"
    assert outputs(store, "r2") == outputs(store, "r1")


def test_cache_off_records_nothing_and_changes_no_events(store, graph):
    fin = run(store, graph, "r1", mode="off")
    assert all("cache" not in d for d in fin.values())
    assert store._exec("SELECT COUNT(*) AS n FROM node_cache")[0]["n"] == 0
    started = store.last_event("r1", "run_started")["data"]
    assert started["cache"] == {"mode": "off"}


def test_non_cacheable_and_domain_nodes_always_run():
    class Op:
        type, version = "tabular.csv_source", "1.0.0"

    nc = C.NodeCache(None, None, {}, "tabular")
    assert nc.decide("s", Op, None, {}).status == "bypass"
    Op.type = "postgres.query"
    assert "not declared cacheable" in nc.decide("q", Op, None, {}).reason
    Op.type = "tabular.profile"
    assert "no content identity" in nc.decide("p", Op, None, {"table": ("q", "table")}).reason
    dom = C.NodeCache(None, None, {}, "domain")
    assert "never cached" in dom.decide("t", Op, None, {}).reason


def test_api_run_with_cache_reports_hits_in_the_run_summary(tmp_path):
    import time
    import uuid

    from fastapi.testclient import TestClient

    from control.app import create_app

    with TestClient(create_app(tmp_path / "wb")) as c:
        def go(g, cache):
            r = c.post("/api/runs", json={"graph": g.to_json(), "config": {"cache": cache}}, headers={"Idempotency-Key": uuid.uuid4().hex})
            assert r.status_code == 201, r.text
            for _ in range(1200):
                s = c.get(f"/api/runs/{r.json()['runId']}").json()
                if s["status"] in ("completed", "failed", "cancelled"):
                    return s
                time.sleep(0.1)
            raise AssertionError(s)

        g = example("tabular_regression")
        a = go(g, "reuse")
        b = go(set_cfg(g.model_copy(deep=True), "ols", fit_intercept=False), "reuse")
        assert a["status"] == b["status"] == "completed"
        assert a["cache"]["mode"] == "reuse" and len(a["cache"]["implementationSha256"]) == 64
        by = {n["node"]: n["cache"] for n in b["nodes"]}
        assert by["housing"]["status"] == "bypass" and by["sc_val"]["status"] == "hit" and by["sc_val"]["fromRun"] == a["id"]
        assert by["ols"]["changed"] == ["settings"] and by["predictions"]["status"] == "miss"
        bad = c.post("/api/runs", json={"graph": g.to_json(), "config": {"cache": "always"}}, headers={"Idempotency-Key": uuid.uuid4().hex})
        assert bad.status_code == 422
        assert json.dumps(go(g, "off")["nodes"]).count('"cache"') == 0


# ------------------------------------------------------------------------------------------------ retention
def entries(store):
    return store._exec("SELECT key, sha256, node_id, run_id FROM node_cache")


def test_prune_dry_run_reports_and_changes_nothing(store, graph):
    from tabular.cache import cache_summary, prune

    run(store, graph, "r1")
    s = cache_summary(store)
    assert s["entries"] == 16 and s["projects"][0]["projectId"] == "p1" and s["projects"][0]["nodes"] == 16 and s["bytes"] > 0
    rep = prune(store, project_id="p1", dry_run=True)
    assert rep["removed"] == 16 and rep["bytesFreed"] == 0 and len(entries(store)) == 16
    assert all(store.verify(e["sha256"]) for e in entries(store))


def test_keep_latest_per_node_drops_older_variants_and_only_cache_bytes(store, graph):
    from tabular.cache import prune

    run(store, graph, "r1")
    run(store, set_cfg(graph.model_copy(deep=True), "ols", fit_intercept=False), "r2")   # ols, metrics, predictions get a second entry
    assert len(entries(store)) == 19
    old = {e["node_id"]: e for e in entries(store) if e["run_id"] == "r1" and e["node_id"] in ("ols", "metrics", "predictions")}
    rep = prune(store, project_id="p1", keep_latest_per_node=1)
    assert rep["removed"] == 3 and {e["node_id"] for e in rep["entries"]} == {"ols", "metrics", "predictions"} and rep["bytesFreed"] > 0
    assert len(entries(store)) == 16 and all(not store.path_of(e["sha256"]).exists() for e in old.values())
    # recorded run artifacts are untouched, and the pruned original result is simply recomputed (a miss), identical to before
    assert all(store.verify(a["sha256"]) for a in store.artifacts("r1"))
    again = run(store, graph, "r3")
    assert {n for n, s in status(again).items() if s == "miss"} == {"ols", "metrics", "predictions"}
    assert again["ols"]["cache"]["changed"] == ["settings"] and "(run r2)" in again["ols"]["cache"]["reason"]  # compared with the kept r2 entry
    assert outputs(store, "r3") == outputs(store, "r1")


def test_older_than_and_project_scope(store, graph):
    import time

    from tabular.cache import prune

    run(store, graph, "r1")
    assert prune(store, project_id="other", dry_run=False)["removed"] == 0            # another project's scope selects nothing
    assert prune(store, project_id="p1", older_than_seconds=3600)["removed"] == 0      # everything is newer than an hour
    rep = prune(store, all_projects=True, older_than_seconds=0, now=time.time() + 1)
    assert rep["removed"] == 16 and entries(store) == []
    assert all(d["cache"]["status"] in ("miss", "bypass") for d in run(store, graph, "r2").values())


def test_shared_bytes_are_kept_while_referenced(store, graph):
    run(store, graph, "r1")
    e = entries(store)[0]
    store.add_artifact("r1", "probe", store.read_artifact(e["sha256"]), "complete", None, {})  # same bytes recorded as a run artifact
    assert store.delete_node_cache([e["key"]]) == 0 and store.path_of(e["sha256"]).exists()


def test_cache_api_summary_and_prune(tmp_path):
    import uuid

    from fastapi.testclient import TestClient

    from control.app import create_app

    with TestClient(create_app(tmp_path / "wb")) as c:
        r = c.post("/api/runs", json={"projectId": None, "graph": example("tabular_regression").to_json(), "config": {"cache": "reuse", "project_id": "api"}},
                   headers={"Idempotency-Key": uuid.uuid4().hex})
        assert r.status_code == 201, r.text
        import time
        for _ in range(1200):
            if c.get(f"/api/runs/{r.json()['runId']}").json()["status"] == "completed":
                break
            time.sleep(0.1)
        s = c.get("/api/cache/nodes").json()
        assert s["entries"] == 16 and s["projects"][0]["projectId"] == "api"
        assert c.post("/api/cache/nodes/prune", json={}).status_code == 422                       # scope must be explicit
        assert c.post("/api/cache/nodes/prune", json={"projectId": "api"}).json()["dryRun"] is True  # default is a report
        assert c.get("/api/cache/nodes").json()["entries"] == 16
        done = c.post("/api/cache/nodes/prune", json={"projectId": "api", "dryRun": False}).json()
        assert done["removed"] == 16 and done["bytesFreed"] > 0 and c.get("/api/cache/nodes").json()["entries"] == 0
