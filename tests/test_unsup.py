"""Unsupervised and representation workflows (VISION 9.6, A55): native scikit-learn equivalence, method-appropriate diagnostics, stability, projection limits."""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn import metrics as skm
from sklearn.cluster import DBSCAN, KMeans, kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from conftest import EXAMPLES
from control.app import create_app
from graph_core.schema import Graph
from graph_core.validate import validate
from tabular_helpers import codes, run_inproc, summary, table
from unsup.samples import cells_graph, graph as mk_graph, moons_graph, _e, _n

CELLS = pd.read_csv(EXAMPLES / "fixtures" / "synthetic_cells.csv")
FEATS = ["area_um2", "intensity", "granularity", "elongation"]


def run(gj, tmp_path):
    g = Graph.model_validate(gj)
    rep = validate(g)
    assert rep.ok, [(d.code, d.message) for d in rep.errors]
    store, status = run_inproc(g, tmp_path)
    assert status == "completed", store.get_run("r1")["error"]
    return store


def set_(gj, node, **kw):
    gj = copy.deepcopy(gj)
    next(n for n in gj["nodes"] if n["id"] == node)["config"].update(kw)
    return gj


# ------------------------------------------------------------------------------------------------ k-means equals native scikit-learn
def test_kmeans_equals_a_handwritten_scikit_learn_pipeline_and_exposes_the_objective(tmp_path):
    store = run(cells_graph(), tmp_path)
    s = summary(store, "r1", "kmeans")
    a = table(store, "r1", "kmeans", "assignments")
    Xs = StandardScaler().fit_transform(CELLS[FEATS].to_numpy())
    c0 = kmeans_plusplus(Xs, 3, random_state=0 + s["selectedInit"])[0]
    ref = KMeans(n_clusters=3, init=c0, n_init=1, max_iter=300, tol=1e-4, algorithm="lloyd").fit(Xs)
    assert s["inertia"] == pytest.approx(ref.inertia_) and a["cluster"].tolist() == ref.labels_.tolist()
    # objective bookkeeping: per-cluster contributions and per-row distances both sum to the inertia
    assert sum(c["inertiaContribution"] for c in s["clusters"]) == pytest.approx(s["inertia"])
    assert (a["distance_to_centroid"] ** 2).sum() == pytest.approx(s["inertia"])
    assert sum(c["size"] for c in s["clusters"]) == len(CELLS) and sum(c["inertiaShare"] for c in s["clusters"]) == pytest.approx(1.0)
    assert (a["distance_to_second_centroid"] >= a["distance_to_centroid"] - 1e-12).all()
    # silhouette per row, assignments keep the label column next to the cluster (external labels stay visible, not used)
    assert a["silhouette"].mean() == pytest.approx(skm.silhouette_score(Xs, ref.labels_)) and "true_group" in a.columns and list(a.columns[:2]) == ["cell_id", "area_um2"]
    # iteration changes: a real replay of Lloyd's algorithm; inertia never increases and the last replayed iteration is the final solution
    it = s["iterations"]
    assert [i["iteration"] for i in it] == list(range(1, len(it) + 1)) and len(it) == min(s["nIter"], 30)
    assert all(b["inertia"] <= a_["inertia"] + 1e-9 for a_, b in zip(it, it[1:])) and it[-1]["inertia"] == pytest.approx(s["inertia"])
    assert it[0]["reassigned"] is None and it[1]["reassigned"] is not None
    assert s["emptyClusters"] == [] and "relocates" in s["emptyClusterNote"] and s["scaler"]["mean"]["area_um2"] == pytest.approx(CELLS["area_um2"].mean())
    # centres reported in the original units
    c = s["clusters"][0]["center"]
    assert c["area_um2"] == pytest.approx(CELLS.loc[a["cluster"].to_numpy() == 0, "area_um2"].mean(), rel=1e-6)


def test_kmeans_is_seeded_and_scaling_changes_the_partition(tmp_path):
    g = cells_graph()
    a = table(run(g, tmp_path / "a"), "r1", "kmeans", "assignments")["cluster"]
    b = table(run(g, tmp_path / "b"), "r1", "kmeans", "assignments")["cluster"]
    assert a.tolist() == b.tolist()
    raw = table(run(set_(g, "kmeans", scale="none"), tmp_path / "c"), "r1", "kmeans", "assignments")["cluster"]
    assert skm.adjusted_rand_score(a, raw) < 0.95      # unscaled: the area column dominates every distance


def test_empty_selection_uses_every_numeric_column_and_exclude_removes_ids(tmp_path):
    g = set_(cells_graph(), "kmeans", features=[], exclude=["cell_id"])
    s = summary(run(g, tmp_path), "r1", "kmeans")
    assert s["features"] == FEATS


# ------------------------------------------------------------------------------------------------ Gaussian mixture
def test_gmm_matches_scikit_learn_and_keeps_membership_and_density_separate(tmp_path):
    store = run(cells_graph(), tmp_path)
    s = summary(store, "r1", "gmm")
    a = table(store, "r1", "gmm", "assignments")
    Xs = StandardScaler().fit_transform(CELLS[FEATS].to_numpy())
    ref = GaussianMixture(3, covariance_type="full", n_init=3, max_iter=200, random_state=0).fit(Xs)
    assert s["bic"] == pytest.approx(ref.bic(Xs)) and s["aic"] == pytest.approx(ref.aic(Xs)) and a["cluster"].tolist() == ref.predict(Xs).tolist()
    assert a["log_density"].to_numpy() == pytest.approx(ref.score_samples(Xs)) and a["max_responsibility"].between(1 / 3, 1).all()
    assert "cluster" in a.columns and "log_density" in a.columns and "SCORE" in s["densityNote"] and sum(c["weight"] for c in s["components"]) == pytest.approx(1)
    assert s["bic"] == pytest.approx(s["nParameters"] * np.log(len(Xs)) - 2 * s["logLikelihood"])


# ------------------------------------------------------------------------------------------------ DBSCAN
def test_dbscan_core_border_noise_status_and_no_predict(tmp_path):
    store = run(moons_graph(), tmp_path)
    s = summary(store, "r1", "dbscan")
    a = table(store, "r1", "dbscan", "assignments")
    X = pd.read_csv(EXAMPLES / "fixtures" / "synthetic_moons.csv")[["x", "y"]].to_numpy()
    ref = DBSCAN(eps=0.2, min_samples=5).fit(X)
    assert a["cluster"].tolist() == ref.labels_.tolist() and s["nClusters"] == 2
    core = set(ref.core_sample_indices_.tolist())
    assert {i for i, v in enumerate(a["point_status"]) if v == "core"} == core
    assert ((a["point_status"] == "noise") == (a["cluster"] == -1)).all() and (a.loc[a["point_status"] == "core", "neighbors_within_eps"] >= 5).all()
    assert s["canPredict"] is False and "no predict" in s["predictNote"] and "Noise (-1) is not a cluster" in s["noiseNote"]
    assert s["nCore"] + s["nBorder"] + s["nNoise"] == 300 and len(s["kDistance"]["sortedDescending"]) <= 400


def test_dbscan_noise_is_excluded_from_internal_metrics_and_says_so(tmp_path):
    g = set_(moons_graph(), "dbscan", eps=0.12)
    store = run(g, tmp_path)
    d = summary(store, "r1", "db_diag")
    assert any("noise points (label -1) excluded" in n for n in d["internal"]["notes"]) and d["internal"]["nUsed"] < 300
    assert d["values"]["noiseFraction"] > 0


def test_diagnostics_that_need_predict_refuse_a_different_table_for_dbscan(tmp_path):
    g = moons_graph()
    g["nodes"].append(_n("head", "tabular.train_validation_split", {"validation_fraction": 0.3, "seed": 1}))
    g["edges"] = [e for e in g["edges"] if not (e["to"]["node"] == "db_diag" and e["to"]["port"] == "table")]
    g["edges"] += [_e(90, ("moons", "table"), ("head", "table"), "table"), _e(91, ("head", "validation"), ("db_diag", "table"), "table")]
    gr = Graph.model_validate(g)
    store, status = run_inproc(gr, tmp_path)
    assert status == "failed" and "E_UNSUP_NO_PREDICT" in store.get_run("r1")["error"]


# ------------------------------------------------------------------------------------------------ PCA
def test_pca_matches_scikit_learn_explained_variance_loadings_and_reconstruction(tmp_path):
    store = run(cells_graph(), tmp_path)
    s = summary(store, "r1", "pca")
    z = table(store, "r1", "pca", "components")
    Xs = StandardScaler().fit_transform(CELLS[FEATS].to_numpy())
    ref = PCA(2, svd_solver="full").fit(Xs)
    assert s["explainedVarianceRatio"] == pytest.approx(ref.explained_variance_ratio_.tolist()) and z[["PC1", "PC2"]].to_numpy() == pytest.approx(ref.transform(Xs))
    assert s["loadings"][0]["weights"]["area_um2"] == pytest.approx(ref.components_[0][0])
    mse = ((Xs - ref.inverse_transform(ref.transform(Xs))) ** 2).mean()
    assert s["reconstruction"]["mseKept"] == pytest.approx(mse)
    by = {r["components"]: r["mse"] for r in s["reconstruction"]["byComponents"]}
    assert by[2] == pytest.approx(mse) and by[4] == pytest.approx(0, abs=1e-9) and by[0] == pytest.approx(1.0) and by[1] > by[2] > by[3] >= by[4]
    assert s["canTransformNewPoints"] is True and "true_group" in z.columns and "area_um2" not in z.columns
    assert s["cumulativeRatio"][-1] == pytest.approx(sum(s["explainedVarianceRatio"]))
    assert s["mean"]["area_um2"] == pytest.approx(0, abs=1e-9)    # centered (standardized) features


def test_pca_components_cannot_exceed_features():
    g = Graph.model_validate(set_(cells_graph(), "pca", n_components=9))
    assert "E_PCA_COMPONENTS" in codes(validate(g), "error")


# ------------------------------------------------------------------------------------------------ projection
def test_tsne_is_a_labelled_non_metric_projection_that_cannot_transform_new_points(tmp_path):
    g = cells_graph()
    gr = Graph.model_validate(g)
    assert "W_PROJECTION_NON_METRIC" in codes(validate(gr), "warning")
    store = run(g, tmp_path)
    s = summary(store, "r1", "tsne")
    e = table(store, "r1", "tsne", "embedding")
    assert list(e.columns[:2]) == ["x", "y"] and len(e) == 360 and s["canTransformNewPoints"] is False and "NON-METRIC" in s["projectionStatus"]
    assert 0.5 < s["trustworthiness5"] <= 1.0 and any("not evidence" in x or "not meaningful" in x for x in s["limits"]) and "no transform" in s["transformNote"]
    assert not any(k for k in s if "accuracy" in k.lower())


def test_perplexity_must_be_below_the_row_count():
    g = Graph.model_validate(set_(cells_graph(), "tsne", perplexity=400))
    # the row count of a CSV source is known statically, so this is caught before running
    assert "E_PROJECTION_PERPLEXITY" in codes(validate(g), "error")


# ------------------------------------------------------------------------------------------------ A55 diagnostics are method-appropriate
def test_a55_diagnostics_differ_by_method_and_are_not_a_universal_accuracy(tmp_path):
    store = run(cells_graph(), tmp_path)
    km, gm, pc = (summary(store, "r1", n) for n in ("km_diag", "gmm_diag", "pca_diag"))
    assert {"silhouette", "daviesBouldin", "calinskiHarabasz", "inertia"} <= km["values"].keys() and "bic" not in km["values"]
    assert {"bic", "aic", "logLikelihood", "silhouette"} <= gm["values"].keys() and "inertia" not in gm["values"]
    assert {"explainedVarianceRatioKept", "reconstructionMse", "componentsFor90"} <= pc["values"].keys() and "silhouette" not in pc["values"] and "inertia" not in pc["values"]
    for r in (km, gm, pc):
        assert "no universal unsupervised accuracy" in r["policy"] and "accuracy" not in r["values"]
    # every internal metric states what it assumes and which direction is better
    sil = km["internal"]["silhouette"]
    Xs = StandardScaler().fit_transform(CELLS[FEATS].to_numpy())
    lab = table(store, "r1", "kmeans", "assignments")["cluster"].to_numpy()
    assert sil["value"] == pytest.approx(skm.silhouette_score(Xs, lab)) and km["values"]["daviesBouldin"] == pytest.approx(skm.davies_bouldin_score(Xs, lab))
    assert "convex" in sil["assumption"] and sil["better"].startswith("higher")
    assert km["valueKinds"]["silhouette"] == "internal" and km["valueKinds"]["externalAdjustedRand"] == "external" and km["valueKinds"]["stabilityAriMean"] == "stability"


def test_elbow_and_bic_sweeps_refit_each_k(tmp_path):
    store = run(cells_graph(), tmp_path)
    km, gm = summary(store, "r1", "km_diag"), summary(store, "r1", "gmm_diag")
    rows = km["sweep"]["rows"]
    assert [r["k"] for r in rows] == list(range(2, 9)) and all(a["inertia"] >= b["inertia"] for a, b in zip(rows, rows[1:]))
    Xs = StandardScaler().fit_transform(CELLS[FEATS].to_numpy())
    assert rows[1]["inertia"] == pytest.approx(summary(store, "r1", "kmeans")["inertia"]) and "diminishing" in km["sweep"]["note"]
    grows = gm["sweep"]["rows"]
    assert [r["k"] for r in grows] == list(range(2, 7)) and min(grows, key=lambda r: r["bic"])["k"] in (2, 3, 4)
    assert grows[1]["bic"] == pytest.approx(summary(store, "r1", "gmm")["bic"])


def test_external_metrics_only_with_labels_and_are_labelled_external(tmp_path):
    g = cells_graph()
    store = run(g, tmp_path / "a")
    ext = summary(store, "r1", "km_diag")["external"]
    assert ext["kind"] == "external" and "NOT used for fitting" in ext["statement"] and ext["labelsColumn"] == "true_group"
    lab = table(store, "r1", "kmeans", "assignments")["cluster"]
    assert ext["adjustedRand"] == pytest.approx(skm.adjusted_rand_score(CELLS["true_group"], lab))
    store2 = run(set_(g, "km_diag", labels_column=""), tmp_path / "b")
    d = summary(store2, "r1", "km_diag")
    assert d["external"]["available"] is False and "No label column" in d["external"]["reason"] and not any(k.startswith("external") for k in d["values"])
    # a label column that is also a feature would make the agreement circular
    bad = Graph.model_validate(set_(set_(g, "kmeans", features=FEATS + ["cell_id"]), "km_diag", labels_column="cell_id"))
    assert "E_LABEL_IN_FEATURES" in codes(validate(bad), "error")


def test_moons_internal_metrics_can_prefer_the_wrong_partition_and_external_shows_it(tmp_path):
    store = run(moons_graph(), tmp_path)
    km, db = summary(store, "r1", "km_diag"), summary(store, "r1", "db_diag")
    assert km["values"]["silhouette"] > 0.4 and km["values"]["externalAdjustedRand"] < 0.5          # k-means: decent silhouette, wrong shapes
    assert db["values"]["externalAdjustedRand"] == pytest.approx(1.0) and db["values"]["nClusters"] == 2


def test_single_cluster_metrics_are_unavailable_with_a_reason(tmp_path):
    g = set_(moons_graph(), "dbscan", eps=5.0)
    d = summary(run(g, tmp_path), "r1", "db_diag")
    assert "silhouette" not in d["values"] and "at least" in d["unavailable"]["silhouette"].replace("needs", "at least") and d["internal"]["silhouette"]["applicable"] is False


# ------------------------------------------------------------------------------------------------ stability
def test_stability_is_high_for_separated_groups_and_low_for_structureless_data(tmp_path):
    store = run(cells_graph(), tmp_path / "a")
    st = summary(store, "r1", "km_diag")["stability"]
    assert st["seeds"]["pairwise"]["min"] >= 0.9 and st["resampling"]["mean"] >= 0.9 and len(st["resampling"]["ari"]) == 8 and "Stable does not mean correct" in st["caveat"]
    rng = np.random.default_rng(0)
    noise = pd.DataFrame({"a": rng.uniform(size=250), "b": rng.uniform(size=250)})
    path = tmp_path / "noise.csv"
    noise.to_csv(path, index=False)
    g = mk_graph([_n("src", "tabular.csv_source", {"path": str(path)}), _n("km", "sklearn.kmeans", {"n_clusters": 6, "n_init": 1, "seed": 0}),
                  _n("d", "sklearn.cluster_diagnostics", {"stability_runs": 10, "k_min": 2, "k_max": 3})],
                 [_e(0, ("src", "table"), ("km", "table"), "table"), _e(1, ("km", "model"), ("d", "model"), "unsup_model"), _e(2, ("src", "table"), ("d", "table"), "table")])
    d = summary(run(g, tmp_path / "b"), "r1", "d")
    assert d["stability"]["resampling"]["mean"] < 0.8 and d["stability"]["seeds"]["pairwise"]["mean"] < 0.9
    assert d["values"]["silhouette"] > 0.2          # a respectable-looking silhouette on data with no cluster structure: why stability is reported next to it


def test_bootstrap_resampling_and_off_switch(tmp_path):
    d = summary(run(set_(cells_graph(), "km_diag", resample="bootstrap", stability_runs=5), tmp_path / "a"), "r1", "km_diag")
    assert "bootstrap" in d["stability"]["resampling"]["scheme"] and len(d["stability"]["resampling"]["ari"]) == 5
    g = Graph.model_validate(set_(cells_graph(), "km_diag", stability_runs=0))
    assert "W_NO_STABILITY" in codes(validate(g), "warning")
    d0 = summary(run(set_(cells_graph(), "km_diag", stability_runs=0), tmp_path / "b"), "r1", "km_diag")
    assert "stability" not in d0 and "stabilityAriMean" not in d0["values"]


def test_pca_stability_compares_components_across_bootstraps(tmp_path):
    d = summary(run(cells_graph(), tmp_path), "r1", "pca_diag")
    pcs = d["stability"]["perComponent"]
    assert [p["pc"] for p in pcs] == ["PC1", "PC2"] and pcs[0]["meanAbsCosine"] > 0.95 and d["external"]["available"] is False
    assert d["explainedVariance"]["componentsNeeded"]["0.9"] == 2


def test_tsne_diagnostics_report_only_neighbour_preservation(tmp_path):
    g = cells_graph()
    g["nodes"].append(_n("tsne_diag", "sklearn.cluster_diagnostics", {"stability_runs": 0}))
    g["edges"] += [_e(80, ("tsne", "model"), ("tsne_diag", "model"), "unsup_model"), _e(81, ("cells", "table"), ("tsne_diag", "table"), "table")]
    d = summary(run(g, tmp_path), "r1", "tsne_diag")
    assert set(d["values"]) == {"trustworthiness5", "klDivergence"} and "No cluster-quality, accuracy or stability metric" in d["projection"]["statement"]
    assert "silhouette" not in d["values"] and "internal" not in d


# ------------------------------------------------------------------------------------------------ errors
def test_static_and_runtime_errors_have_stable_codes(tmp_path):
    g = Graph.model_validate(set_(cells_graph(), "kmeans", n_clusters=150))
    assert validate(g).ok
    g = Graph.model_validate(set_(cells_graph(), "kmeans", features=["true_group"]))
    assert "E_COLUMN_TYPE" in codes(validate(g), "error")
    g = Graph.model_validate(set_(cells_graph(), "kmeans", features=["nope"]))
    assert "E_COLUMN_NOT_FOUND" in codes(validate(g), "error")
    p = tmp_path / "m.csv"
    pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 5.0]}).to_csv(p, index=False)
    mg = mk_graph([_n("src", "tabular.csv_source", {"path": str(p)}), _n("km", "sklearn.kmeans", {"n_clusters": 2})], [_e(0, ("src", "table"), ("km", "table"), "table")])
    store, status = run_inproc(Graph.model_validate(mg), tmp_path / "w")
    assert status == "failed" and "E_MISSING_VALUES" in store.get_run("r1")["error"]
    mg["nodes"][1]["config"]["n_clusters"] = 9
    assert "E_UNSUP_TOO_FEW_ROWS" in codes(validate(Graph.model_validate(mg)), "error")


# ------------------------------------------------------------------------------------------------ API: run, inspect, study over k
@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with TestClient(create_app(tmp_path_factory.mktemp("wb"))) as c:
        yield c


def wait(c, rid, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        r = c.get(f"/api/runs/{rid}").json()
        if r["status"] in ("completed", "failed", "cancelled"):
            return r
        time.sleep(0.2)
    raise AssertionError(r)


def test_api_run_inspection_of_assignments_and_diagnostics_and_a_k_study(client):
    gj = cells_graph()
    v = client.post("/api/validate", json={"graph": gj}).json()
    assert v["ok"] and v["nodes"]["kmeans"]["outputShapes"]["assignments"]["columns"][-1]["name"] == "silhouette"
    ops = {o["type"]: o for o in client.get("/api/registry").json()["ops"]}
    assert ops["sklearn.cluster_diagnostics"]["outputKinds"] == {"report": "cluster_report"} and ops["sklearn.kmeans"]["summaryKind"] == "clustering"
    rid = client.post("/api/runs", json={"graph": gj, "config": {}}, headers={"Idempotency-Key": "unsup-1"}).json()["runId"]
    assert wait(client, rid)["status"] == "completed"
    t = client.post(f"/api/runs/{rid}/inspect", json={"kind": "table", "node": "kmeans", "port": "assignments", "limit": 5}).json()
    assert [c["name"] for c in t["columns"]][-4:] == ["cluster", "distance_to_centroid", "distance_to_second_centroid", "silhouette"] and len(t["rows"]) == 5 and t["total"] == 360
    d = client.post(f"/api/runs/{rid}/inspect", json={"kind": "summary", "node": "km_diag"}).json()
    assert d["available"] and d["summaryKind"] == "cluster_report" and d["data"]["values"]["silhouette"] > 0
    csv = client.get(f"/api/runs/{rid}/tables/tsne/embedding.csv")
    assert csv.status_code == 200 and csv.text.splitlines()[0].startswith("row_id,x,y")
    # a study over k with the silhouette of the diagnostics node as objective: each k is a trial
    body = {"name": "k sweep", "graph": gj, "run_config": {}, "objective": {"metric": {"name": "silhouette", "node": "km_diag"}, "direction": "maximize"},
            "search": {"method": "grid", "variables": [{"target": {"scope": "node", "node": "kmeans", "field": "n_clusters"}, "values": [2, 3, 5]}]},
            "repeats": {}, "limits": {"max_trials": 5}, "include_baseline": False}
    bad = {**body, "objective": {"metric": {"name": "bic", "node": "km_diag"}, "direction": "minimize"}}
    assert client.post("/api/studies/plan", json=bad).json()["detail"]["code"] == "objective_invalid"      # bic is not computed for k-means
    sid = client.post("/api/studies", json=body).json()["id"]
    end = time.time() + 180
    while time.time() < end:
        s = client.get(f"/api/studies/{sid}").json()
        if s["state"] not in ("running", "planned"):
            break
        time.sleep(0.5)
    assert s["state"] == "completed" and len(s["trials"]) == 3
    vals = {t["label"]: t["value"] for t in s["trials"]}
    assert all(0 < v < 1 for v in vals.values())
    assert all(t["metric"]["available"] and t["metric"]["aggregation"].startswith("single evaluation") for t in s["trials"])


def test_points_endpoint_returns_bounded_coloured_points_from_recorded_tables(client):
    rid = client.post("/api/runs", json={"graph": cells_graph(), "config": {}}, headers={"Idempotency-Key": "unsup-points"}).json()["runId"]
    assert wait(client, rid)["status"] == "completed"
    r = client.get(f"/api/runs/{rid}/points", params={"node": "tsne", "port": "embedding", "x": "x", "y": "y", "color_node": "kmeans", "color_port": "assignments", "color": "cluster"}).json()
    assert r["total"] == 360 == r["shown"] and not r["downsampled"] and r["colorBy"] == {"column": "cluster", "from": "kmeans.assignments"}
    assert {p["c"] for p in r["points"]} == {0, 1, 2}
    lab = client.get(f"/api/runs/{rid}/points", params={"node": "tsne", "port": "embedding", "x": "x", "y": "y", "color": "true_group"}).json()
    assert {p["c"] for p in lab["points"]} == {"A", "B", "C"}
    small = client.get(f"/api/runs/{rid}/points", params={"node": "pca", "port": "components", "x": "PC1", "y": "PC2", "limit": 100}).json()
    assert small["downsampled"] and small["shown"] <= 100 and small["total"] == 360
    assert client.get(f"/api/runs/{rid}/points", params={"node": "pca", "port": "components", "x": "nope", "y": "PC2"}).status_code == 422
