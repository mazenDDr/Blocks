"""Tabular graph kind: typed wires, schema inference, leakage policy (A05), fit state ownership, native-baseline parity."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from conftest import EXAMPLES
from graph_core.hashing import semantic_hash
from graph_core.schema import Graph
from graph_core.validate import ExecutionBlocked, require_executable, validate
from tabular_helpers import FIX, codes, example, rewire, run_inproc, set_cfg, summary, table
from worker.process import submit_run
from worker.tabular_run import TabularRunConfig


# ---------------------------------------------------------------------------------------------- schema + kinds
def test_regression_example_validates_cleanly_with_schema_and_row_counts():
    g = example("tabular_regression")
    r = validate(g)
    assert g.graphKind == "tabular" and r.ok and r.diagnostics == []
    src = r.output_types["housing"]["table"]
    assert [c["name"] for c in src.columns] == ["id", "area_m2", "rooms", "age_years", "neighborhood", "price_k"]
    assert [c["dtype"] for c in src.columns] == ["int", "float", "int", "float", "string", "float"]
    assert src.info["rows"] == 412 and src.info["sourceSha256"] == hashlib.sha256((FIX / "synthetic_housing.csv").read_bytes()).hexdigest()
    sel = r.output_types["select"]["table"]
    assert sel.col_names == ["area_m2", "rooms", "age_years", "neighborhood", "price_k"]
    assert sel.info["rows"] is None  # dedupe changes the row count in a data-dependent way: not claimed statically
    oh = r.output_types["oh_train"]["table"]
    assert not oh.complete and "neighborhood" not in oh.col_names and any("neighborhood" in p for p in oh.info["pending"])
    assert r.output_types["split"]["train"].partition == "train" and r.output_types["split"]["validation"].partition == "validation"


def test_split_row_estimate_matches_actual(tmp_path):
    g = example("tabular_regression")
    # remove the row-changing steps so the count is statically known
    g.nodes = [n for n in g.nodes if n.id != "dedupe"]
    g.edges = [e for e in g.edges if "dedupe" not in (e.from_.node, e.to.node)]
    rewire(g, "select", "table", "profile", "table")
    r = validate(g)
    est = (r.output_types["split"]["train"].info["rows"], r.output_types["split"]["validation"].info["rows"])
    assert est == (309, 103)
    store, status = run_inproc(g, tmp_path)
    assert status == "completed" and (len(table(store, "r1", "split", "train")), len(table(store, "r1", "split", "validation"))) == est


def test_tensor_and_tabular_wires_are_distinct_kinds():
    g = example("tabular_regression")
    rewire(g, "metrics", "table", "ols", "model")  # a model wire into a table port
    g.edges[-1].kind = "model"
    r = validate(g)
    d = [x for x in r.diagnostics if x.code == "E_PORT_TYPE"][0]
    assert d.nodeId == "metrics" and d.port == "table" and "'model'" in d.message and "'table'" in d.message
    g = example("tabular_regression")
    g.edges[0].kind = "tensor"
    assert "E_EDGE_KIND" in codes(validate(g))


def test_model_ops_cannot_live_in_tabular_graphs_and_vice_versa():
    g = example("tabular_regression")
    g.nodes.append(Graph.model_validate({"nodes": [{"id": "relu", "type": "pytorch.nn.relu"}]}).nodes[0])
    assert "E_OP_GRAPH_KIND" in codes(validate(g))
    m = Graph.model_validate({"nodes": [{"id": "src", "type": "tabular.csv_source", "config": {"path": "x.csv"}}]})
    assert "E_OP_GRAPH_KIND" in codes(validate(m))


def test_source_errors_are_diagnostics():
    g = example("tabular_regression")
    set_cfg(g, "housing", path="examples/fixtures/does_not_exist.csv")
    d = [x for x in validate(g).diagnostics if x.code == "E_SOURCE_NOT_FOUND"][0]
    assert d.nodeId == "housing"
    set_cfg(g, "housing", path="")
    assert "E_SOURCE_NOT_SET" in codes(validate(g))


def test_column_and_target_checks():
    g = example("tabular_regression")
    set_cfg(g, "ols", target="nope")
    assert "E_COLUMN_NOT_FOUND" in codes(validate(g))
    g = example("tabular_regression")
    set_cfg(g, "ols", features=["area_m2", "price_k"])
    assert "E_TARGET_IN_FEATURES" in codes(validate(g))
    g = example("tabular_regression")
    set_cfg(g, "ols", target="")
    assert "E_TARGET_NOT_SET" in codes(validate(g))
    g = example("tabular_regression")
    set_cfg(g, "sc_fit", columns=["area_m2", "bogus"])
    d = [x for x in validate(g).diagnostics if x.code == "E_COLUMN_NOT_FOUND"]
    assert d and d[0].nodeId == "sc_fit" and "bogus" in d[0].message
    g = example("tabular_regression")
    set_cfg(g, "ols", features=["area_m2", "neighborhood_central"])  # a one-hot column that will exist once the encoder is fitted
    assert validate(g).ok


# ---------------------------------------------------------------------------------------------- A05 leakage
def test_a05_fit_on_validation_partition_is_rejected_with_path_and_fix():
    g = example("tabular_regression")
    rewire(g, "imp_fit", "train", "split", "validation")  # deliberately invalid wiring
    r = validate(g)
    d = [x for x in r.diagnostics if x.code == "E_LEAKAGE_FIT_ON_HELDOUT"]
    assert len(d) == 1 and not r.ok
    d = d[0]
    assert d.severity == "error" and d.nodeId == "imp_fit" and d.port == "train" and d.path == "/nodes/imp_fit/ports/train"
    assert "validation" in d.message and "split" in d.message
    assert d.fixes and "split.train" in d.fixes[0].label
    # nothing downstream of the failure is re-reported as a second leakage error
    assert codes(r, "error") == ["E_LEAKAGE_FIT_ON_HELDOUT"]
    with pytest.raises(ExecutionBlocked):
        require_executable(g)


@pytest.mark.parametrize("fit_node", ["imp_fit", "oh_fit", "sc_fit"])
def test_a05_every_fit_node_enforces_the_policy(fit_node):
    g = example("tabular_regression")
    rewire(g, fit_node, "train", "split", "validation")
    assert any(d.code == "E_LEAKAGE_FIT_ON_HELDOUT" and d.nodeId == fit_node for d in validate(g).diagnostics)


def test_a05_estimator_fitted_on_validation_data_is_rejected():
    g = example("tabular_regression")
    rewire(g, "ols", "train", "sc_val", "table")
    d = [x for x in validate(g).diagnostics if x.code == "E_LEAKAGE_FIT_ON_HELDOUT"]
    assert d and d[0].nodeId == "ols" and d[0].port == "train"


def test_a05_fit_before_any_split_is_rejected():
    g = example("tabular_regression")
    rewire(g, "imp_fit", "train", "select", "table")
    d = [x for x in validate(g).diagnostics if x.code == "E_LEAKAGE_FIT_BEFORE_SPLIT"]
    assert d and d[0].nodeId == "imp_fit" and "partitioned" in d[0].message and d[0].fixes


def test_a05_blocked_graph_never_runs(tmp_path):
    g = example("tabular_regression")
    rewire(g, "sc_fit", "train", "split", "validation")
    store, status = run_inproc(g, tmp_path)
    assert status == "failed"
    ev = store.events("r1", -1, ("validation_error",))
    assert ev and ev[0]["data"]["code"] == "E_LEAKAGE_FIT_ON_HELDOUT" and ev[0]["node_id"] == "sc_fit"
    assert not store.artifacts("r1", "node_output")  # no data was touched
    with pytest.raises(ExecutionBlocked):
        submit_run(g, TabularRunConfig(), tmp_path / "wb2")


def test_metrics_on_training_partition_is_a_warning_not_an_error():
    g = example("tabular_regression")
    rewire(g, "metrics", "table", "sc_train", "table")
    r = validate(g)
    assert r.ok and "W_METRICS_ON_TRAIN" in codes(r, "warning")


# ---------------------------------------------------------------------------------------------- run + fit state ownership
@pytest.fixture(scope="module")
def regression_run(tmp_path_factory):
    store, status = run_inproc(example("tabular_regression"), tmp_path_factory.mktemp("reg"))
    assert status == "completed", [e["data"] for e in store.events("r1", -1, ("node_failed", "error"))]
    return store


def test_run_records_data_hash_split_seed_and_fitted_state_artifacts(regression_run):
    s = regression_run
    ev = {e["type"]: e for e in s.events("r1", -1, ("source_recorded", "split_recorded", "run_started", "run_finished"))}
    assert ev["source_recorded"]["data"]["sha256"] == hashlib.sha256((FIX / "synthetic_housing.csv").read_bytes()).hexdigest()
    assert ev["split_recorded"]["data"]["seed"] == 42 and ev["split_recorded"]["data"]["nTrain"] == 300
    assert ev["run_finished"]["data"]["status"] == "completed" and ev["run_started"]["data"]["libraries"]["scikit-learn"]
    kinds = {(a["meta"]["node"], a["meta"]["port"]): a["meta"]["valueKind"] for a in s.artifacts("r1", "node_output")}
    assert kinds[("imp_fit", "fit")] == kinds[("oh_fit", "fit")] == kinds[("sc_fit", "fit")] == "fit_state"
    assert kinds[("ols", "model")] == "model" and kinds[("predictions", "predictions")] == "table"


def test_fit_state_is_owned_by_the_training_partition(regression_run):
    s = regression_run
    split = summary(s, "r1", "split")
    train_ids, val_ids = set(split["trainRowIds"]), set(split["validationRowIds"])
    assert not train_ids & val_ids and len(train_ids) == 300 and len(val_ids) == 100
    # imputation medians come from training rows only
    sel = table(s, "r1", "select")
    imp = summary(s, "r1", "imp_fit")
    for c in ("area_m2", "age_years"):
        assert imp["statistics"][c] == pytest.approx(sel.loc[sorted(train_ids), c].median())
    assert imp["statistics"]["area_m2"] != pytest.approx(sel["area_m2"].median())  # not the all-rows statistic
    assert imp["fittedOn"]["partition"] == "train" and imp["fittedOn"]["rowIdsSha256"] == split["trainRowIdsSha256"]
    # standardization statistics = mean/std (ddof=0) of the imputed training rows
    sc = summary(s, "r1", "sc_fit")
    tr = table(s, "r1", "oh_train")
    for c in ("area_m2", "rooms", "age_years"):
        assert sc["mean"][c] == pytest.approx(tr[c].mean()) and sc["scale"][c] == pytest.approx(tr[c].std(ddof=0))
    assert sc["fittedOn"]["rowIdsSha256"] == split["trainRowIdsSha256"]
    # the transformed training columns are standardized; the validation ones generally are not
    st = table(s, "r1", "sc_train")
    assert st["area_m2"].mean() == pytest.approx(0, abs=1e-9) and st["area_m2"].std(ddof=0) == pytest.approx(1)
    sv = table(s, "r1", "sc_val")
    assert abs(sv["area_m2"].mean()) > 1e-6
    ap = summary(s, "r1", "sc_val")
    assert ap["appliedToPartition"] == "validation" and ap["fittedOn"]["partition"] == "train"
    # one-hot categories are the training categories
    oh = summary(s, "r1", "oh_fit")
    assert oh["categories"]["neighborhood"] == sorted(sel["neighborhood"].unique())


def test_exported_predictions_match_a_handwritten_scikit_learn_pipeline(regression_run):
    raw = pd.read_csv(FIX / "synthetic_housing.csv").drop_duplicates()
    d = raw[["area_m2", "rooms", "age_years", "neighborhood", "price_k"]].copy()
    tr, va = train_test_split(d, test_size=0.25, random_state=42)
    X = ["area_m2", "rooms", "age_years", "neighborhood"]
    pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), ["area_m2", "age_years"]),
                             ("rooms", StandardScaler(), ["rooms"]),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["neighborhood"])])
    native = Pipeline([("pre", pre), ("lr", LinearRegression())]).fit(tr[X], tr["price_k"])
    pred = pd.Series(native.predict(va[X]), index=va.index).sort_index()
    ours = table(regression_run, "r1", "predictions", "predictions")
    assert list(ours.index) == list(pred.index)  # same validation rows, identified by source row id
    np.testing.assert_allclose(ours["predicted"].to_numpy(), pred.to_numpy(), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(ours["observed"].to_numpy(), va["price_k"].sort_index().to_numpy())
    np.testing.assert_allclose(ours["residual"], ours["observed"] - ours["predicted"], atol=1e-9)
    # and the metrics block equals scikit-learn's functions on those predictions
    m = summary(regression_run, "r1", "metrics")["values"]
    y = va["price_k"].sort_index().to_numpy()
    assert m["mse"] == pytest.approx(mean_squared_error(y, pred), rel=1e-8) and m["rmse"] == pytest.approx(mean_squared_error(y, pred) ** 0.5, rel=1e-8)
    assert m["mae"] == pytest.approx(mean_absolute_error(y, pred), rel=1e-8) and m["r2"] == pytest.approx(r2_score(y, pred), rel=1e-8)
    # coefficients carry feature names and equal the native model's (in the same order of names)
    coefs = {c["feature"]: c["value"] for c in summary(regression_run, "r1", "ols")["coefficients"]}
    names = native.named_steps["pre"].get_feature_names_out()
    for nm, v in zip(names, native.named_steps["lr"].coef_):
        assert coefs[nm.split("__", 1)[1]] == pytest.approx(v, rel=1e-6, abs=1e-8)


def test_the_run_is_reproducible(tmp_path, regression_run):
    store, status = run_inproc(example("tabular_regression"), tmp_path)
    assert status == "completed"
    a = table(store, "r1", "predictions", "predictions")
    b = table(regression_run, "r1", "predictions", "predictions")
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------------------------- individual ops
def test_logistic_regression_and_classification_metrics_match_native(tmp_path):
    g = Graph.model_validate({"graphKind": "tabular", "backend": "python", "nodes": [
        {"id": "src", "type": "tabular.csv_source", "config": {"path": str(FIX / "synthetic_churn.csv")}},
        {"id": "sel", "type": "tabular.select_columns", "config": {"columns": [
            {"name": "tenure_months", "dtype": "float"}, {"name": "monthly_charge", "dtype": "float"}, {"name": "support_tickets", "dtype": "int"}, {"name": "churned", "dtype": "int"}]}},
        {"id": "split", "type": "tabular.train_validation_split", "config": {"seed": 3, "validation_fraction": 0.3, "stratify_by": "churned"}},
        {"id": "fit", "type": "tabular.fit_standardize", "config": {"columns": ["tenure_months", "monthly_charge"]}},
        {"id": "a_tr", "type": "tabular.apply_transform"}, {"id": "a_va", "type": "tabular.apply_transform"},
        {"id": "lr", "type": "sklearn.logistic_regression", "config": {"target": "churned", "C": 0.5, "max_iter": 200}},
        {"id": "m", "type": "sklearn.metrics"}],
        "edges": [{"id": "e1", "kind": "table", "from": {"node": "src", "port": "table"}, "to": {"node": "sel", "port": "table"}},
                  {"id": "e2", "kind": "table", "from": {"node": "sel", "port": "table"}, "to": {"node": "split", "port": "table"}},
                  {"id": "e3", "kind": "table", "from": {"node": "split", "port": "train"}, "to": {"node": "fit", "port": "train"}},
                  {"id": "e4", "kind": "table", "from": {"node": "split", "port": "train"}, "to": {"node": "a_tr", "port": "table"}},
                  {"id": "e5", "kind": "fit_state", "from": {"node": "fit", "port": "fit"}, "to": {"node": "a_tr", "port": "fit"}},
                  {"id": "e6", "kind": "table", "from": {"node": "split", "port": "validation"}, "to": {"node": "a_va", "port": "table"}},
                  {"id": "e7", "kind": "fit_state", "from": {"node": "fit", "port": "fit"}, "to": {"node": "a_va", "port": "fit"}},
                  {"id": "e8", "kind": "table", "from": {"node": "a_tr", "port": "table"}, "to": {"node": "lr", "port": "train"}},
                  {"id": "e9", "kind": "model", "from": {"node": "lr", "port": "model"}, "to": {"node": "m", "port": "model"}},
                  {"id": "e10", "kind": "table", "from": {"node": "a_va", "port": "table"}, "to": {"node": "m", "port": "table"}}]})
    assert validate(g).ok
    store, status = run_inproc(g, tmp_path)
    assert status == "completed", [e["data"] for e in store.events("r1", -1, ("node_failed",))]
    raw = pd.read_csv(FIX / "synthetic_churn.csv")[["tenure_months", "monthly_charge", "support_tickets", "churned"]]
    tr, va = train_test_split(raw, test_size=0.3, random_state=3, stratify=raw["churned"])
    sc = StandardScaler().fit(tr[["tenure_months", "monthly_charge"]])

    def feat(d):
        x = d.copy()
        x[["tenure_months", "monthly_charge"]] = sc.transform(d[["tenure_months", "monthly_charge"]])
        return x[["tenure_months", "monthly_charge", "support_tickets"]]

    est = LogisticRegression(C=0.5, max_iter=200).fit(feat(tr), tr["churned"])
    p = est.predict_proba(feat(va))[:, 1]
    m = summary(store, "r1", "m")
    assert m["values"]["accuracy"] == pytest.approx(accuracy_score(va["churned"], est.predict(feat(va))))
    assert m["values"]["log_loss"] == pytest.approx(log_loss(va["churned"], p, labels=est.classes_))
    assert m["values"]["roc_auc"] == pytest.approx(roc_auc_score(va["churned"], p))
    assert sum(map(sum, m["confusion"]["matrix"])) == len(va) and m["confusion"]["labels"] == [0, 1]
    coef = summary(store, "r1", "lr")
    assert coef["perClass"][0]["coefficients"][0]["value"] == pytest.approx(est.coef_[0][0], rel=1e-6)
    assert coef["converged"] is True and coef["classes"] == [0, 1]
    # classification metrics are rejected for a regression model and vice versa
    set_cfg(g, "m", metrics=["rmse"])
    assert "E_METRIC_TASK_MISMATCH" in codes(validate(g))


def test_duplicates_report_groups_and_conflicting_labels(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("k,x,y\n1,a,0\n1,a,0\n2,b,1\n2,b,0\n3,c,1\n")
    g = Graph.model_validate({"graphKind": "tabular", "backend": "python", "nodes": [
        {"id": "s", "type": "tabular.csv_source", "config": {"path": str(csv)}},
        {"id": "d", "type": "tabular.duplicates", "config": {"key_columns": ["k"], "keep": "first", "label_column": "y"}}],
        "edges": [{"id": "e", "kind": "table", "from": {"node": "s", "port": "table"}, "to": {"node": "d", "port": "table"}}]})
    store, status = run_inproc(g, tmp_path)
    assert status == "completed"
    rep = summary(store, "r1", "d")
    assert rep["duplicateGroups"] == 2 and rep["removed"] == 2 and rep["rowsAfter"] == 3 and rep["conflictingLabelGroups"] == 1
    assert list(table(store, "r1", "d").index) == [0, 2, 4]
    set_cfg(g, "d", keep="none")
    store2, _ = run_inproc(g, tmp_path / "b")
    assert list(table(store2, "r1", "d").index) == [4]


def test_typed_selection_reports_and_blocks_conversion_failures(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text("a,b\n1,x\n2,y\noops,z\n")
    g = Graph.model_validate({"graphKind": "tabular", "backend": "python", "nodes": [
        {"id": "s", "type": "tabular.csv_source", "config": {"path": str(csv)}},
        {"id": "c", "type": "tabular.select_columns", "config": {"columns": [{"name": "a", "dtype": "float"}], "on_cast_failure": "error"}}],
        "edges": [{"id": "e", "kind": "table", "from": {"node": "s", "port": "table"}, "to": {"node": "c", "port": "table"}}]})
    store, status = run_inproc(g, tmp_path)
    assert status == "failed"
    f = store.events("r1", -1, ("node_failed",))[0]
    assert f["node_id"] == "c" and f["data"]["code"] == "E_CAST_FAILED" and "oops" in f["data"]["message"]
    set_cfg(g, "c", on_cast_failure="set_missing")
    store2, status2 = run_inproc(g, tmp_path / "b")
    assert status2 == "completed" and summary(store2, "r1", "c")["casts"][0]["failures"] == 1
    assert table(store2, "r1", "c")["a"].isna().sum() == 1


def test_profile_counts_missing_duplicates_and_flags(tmp_path):
    store, _ = run_inproc(example("tabular_regression"), tmp_path)
    p = summary(store, "r1", "profile")
    raw = pd.read_csv(FIX / "synthetic_housing.csv")
    assert p["rows"] == 412 and p["exact"] is True and p["duplicateRows"] == int(raw.duplicated().sum()) == 12
    cols = {c["name"]: c for c in p["columns"]}
    assert cols["area_m2"]["missing"] == int(raw["area_m2"].isna().sum()) and cols["area_m2"]["stats"]["mean"] == pytest.approx(raw["area_m2"].mean())
    assert "unique_per_row (possible identifier)" in cols["id"]["flags"] or cols["id"]["unique"] < 412
    assert {t["value"] for t in cols["neighborhood"]["top"]} == {"north", "central", "south"}


def test_drop_missing_rows(tmp_path):
    g = example("tabular_regression")
    g.nodes.append(Graph.model_validate({"nodes": [{"id": "dm", "type": "tabular.drop_missing", "config": {"columns": ["area_m2", "age_years"]}}]}).nodes[0])
    rewire(g, "dm", "table", "dedupe", "table")
    rewire(g, "select", "table", "dm", "table")
    store, status = run_inproc(g, tmp_path)
    assert status == "completed"
    raw = pd.read_csv(FIX / "synthetic_housing.csv").drop_duplicates().dropna(subset=["area_m2", "age_years"])
    assert summary(store, "r1", "dm")["rowsAfter"] == len(raw)
    assert summary(store, "r1", "imp_fit")["missingInTraining"] == {"area_m2": 0, "age_years": 0}


def test_grouped_split_keeps_groups_together(tmp_path):
    g = example("tabular_regression")
    set_cfg(g, "split", group_by="neighborhood")
    set_cfg(g, "split", validation_fraction=0.4)
    store, status = run_inproc(g, tmp_path)
    assert status == "completed"
    assert summary(store, "r1", "split")["groupOverlap"] == 0
    set_cfg(g, "split", stratify_by="neighborhood")
    assert "E_SPLIT_CONFLICT" in codes(validate(g))


def test_graph_hash_ignores_layout_and_tracks_config():
    g = example("tabular_regression")
    h = semantic_hash(g)
    assert semantic_hash(example("tabular_regression")) == h
    set_cfg(g, "split", seed=43)
    assert semantic_hash(g) != h


def test_fixture_generator_is_deterministic_and_matches_the_committed_files(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_tabular_fixtures", EXAMPLES / "make_tabular_fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(tmp_path)
    for name in mod.FIXTURES:
        assert (tmp_path / name).read_bytes() == (FIX / name).read_bytes(), name
