"""Statistics blocks: Gamma teaching example (A25, A26) and two-group comparison against direct SciPy calls (A27)."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from graph_core.hashing import semantic_hash
from graph_core.validate import validate
from tabular_helpers import FIX, codes, example, run_inproc, set_cfg, summary


def run(g, tmp_path):
    store, status = run_inproc(g, tmp_path)
    return store, status


# ---------------------------------------------------------------------------------------------- Gamma teaching example
def test_a25_gamma_teaching_example_tail_probability(tmp_path):
    g = example("gamma_teaching")
    assert validate(g).ok
    store, status = run(g, tmp_path)
    assert status == "completed"
    tail = summary(store, "r1", "upper_tail")
    expected = math.exp(-5) * (1 + 5)
    assert tail["pValue"] == pytest.approx(0.0404276820, abs=5e-11)
    assert tail["pValue"] == pytest.approx(expected, rel=1e-12) == pytest.approx(stats.gamma.sf(5, a=2, loc=0, scale=1), rel=1e-12)
    assert tail["closedForm"]["expression"] == "exp(-5) × (1 + 5)" and tail["closedForm"]["matchesScipy"]
    assert tail["areaNumerical"] == pytest.approx(tail["pValue"], rel=1e-7)  # the p-value is the area under the density to the right of 5
    assert tail["densityAtObserved"] == pytest.approx(5 * math.exp(-5)) and tail["densityAtObserved"] != pytest.approx(tail["pValue"])
    assert tail["reference"] == {"family": "gamma", "shape": 2.0, "scale": 1.0, "rate": 1.0, "loc": 0.0}
    t = summary(store, "r1", "test")
    assert t["pValue"] == tail["pValue"] and t["alpha"] == 0.05 and t["decision"] == "reject_h0"
    assert t["decisionText"] == "reject H0 under this specified test" and t["decisionRule"] == "reject H0 when p < alpha"
    assert t["statistic"] == {"name": "T", "value": 5.0} and t["tail"] == "upper"
    assert t["null"].startswith("T ~ Gamma(shape=2, scale=1, loc=0)") and "large values" in t["alternative"]
    assert any("SPECIFIED" in a for a in t["assumptions"]) and t["teachingFixture"] is True and t["sample"] is None
    assert t["criticalValue"] == pytest.approx(stats.gamma.isf(0.05, a=2), rel=1e-12)
    assert t["plot"]["observed"] == 5 and len(t["plot"]["x"]) == len(t["plot"]["density"]) > 100
    x = np.array(t["plot"]["x"])
    assert np.allclose(t["plot"]["density"], stats.gamma.pdf(x, a=2))  # real computed values, not a drawing


def test_a26_changing_alpha_moves_the_decision_not_the_p_value(tmp_path):
    results = {}
    for alpha in (0.05, 0.04, 0.01):
        g = example("gamma_teaching")
        set_cfg(g, "test", alpha=alpha)
        store, status = run(g, tmp_path / str(alpha))
        assert status == "completed"
        results[alpha] = (summary(store, "r1", "upper_tail"), summary(store, "r1", "test"))
    p = {a: r[1]["pValue"] for a, r in results.items()}
    assert p[0.05] == p[0.04] == p[0.01]  # bit-identical: alpha is not an input of the tail computation
    assert results[0.05][0] == results[0.04][0]
    assert results[0.05][1]["decision"] == "reject_h0"
    assert results[0.04][1]["decision"] == "fail_to_reject_h0" and "not evidence" in results[0.04][1]["decisionText"]  # 0.0404 > 0.04
    assert results[0.01][1]["decision"] == "fail_to_reject_h0"
    crit = {a: r[1]["criticalValue"] for a, r in results.items()}
    assert crit[0.05] < crit[0.04] < crit[0.01]  # smaller alpha => boundary moves right
    for a in crit:
        assert crit[a] == pytest.approx(stats.gamma.isf(a, a=2), rel=1e-12)
    # moving the observation (explore) changes the tail area
    g = example("gamma_teaching")
    set_cfg(g, "upper_tail", observed=3.0)
    store, _ = run(g, tmp_path / "obs")
    assert summary(store, "r1", "upper_tail")["pValue"] == pytest.approx(stats.gamma.sf(3, a=2)) and summary(store, "r1", "test")["decision"] == "fail_to_reject_h0"


def test_gamma_scale_and_rate_parametrizations_are_explicit_and_equivalent(tmp_path):
    g = example("gamma_teaching")
    set_cfg(g, "null_distribution", parametrization="shape_rate", rate=0.5, shape=3.0, loc=1.0)
    store, _ = run(g, tmp_path)
    d = summary(store, "r1", "null_distribution")
    assert d["scale"] == 2.0 and d["rate"] == 0.5 and d["mean"] == pytest.approx(1 + 3 * 2) and d["variance"] == pytest.approx(3 * 4)
    assert summary(store, "r1", "upper_tail")["pValue"] == pytest.approx(stats.gamma.sf(5, a=3, loc=1, scale=2))
    assert "density" in d["note"].lower()


def test_lower_tail_and_non_integer_shape_have_no_closed_form(tmp_path):
    g = example("gamma_teaching")
    set_cfg(g, "null_distribution", shape=2.5)
    store, _ = run(g, tmp_path)
    assert summary(store, "r1", "upper_tail")["closedForm"] is None
    set_cfg(g, "upper_tail", tail="lower", observed=1.0)
    store, _ = run(g, tmp_path / "b")
    assert summary(store, "r1", "upper_tail")["pValue"] == pytest.approx(stats.gamma.cdf(1.0, a=2.5))
    t = summary(store, "r1", "test")
    assert t["criticalValue"] == pytest.approx(stats.gamma.ppf(0.05, a=2.5))


def test_gamma_function_block_is_separate_and_checks_its_domain(tmp_path):
    from graph_core.schema import Graph

    def graph(x):
        return Graph.model_validate({"graphKind": "tabular", "backend": "python", "nodes": [{"id": "gf", "type": "scipy.gamma_function", "config": {"x": x}}], "edges": []})

    store, status = run(graph(5.0), tmp_path)
    assert status == "completed" and summary(store, "r1", "gf")["value"] == 24.0
    assert "E_DOMAIN" in codes(validate(graph(-2.0)))
    assert validate(graph(-2.5)).ok


# ---------------------------------------------------------------------------------------------- A27
@pytest.fixture(scope="module")
def groups():
    df = pd.read_csv(FIX / "synthetic_enzyme_two_group.csv")
    a = df.loc[df.condition == "control", "activity_u_per_mg"].to_numpy()
    b = df.loc[df.condition == "inhibitor", "activity_u_per_mg"].to_numpy()
    return df, a, b


def hedges_j(df):
    return math.gamma(df / 2) / (math.sqrt(df / 2) * math.gamma((df - 1) / 2))


@pytest.mark.parametrize("method,equal_var", [("welch", False), ("student", True)])
@pytest.mark.parametrize("alt", ["two-sided", "greater", "less"])
def test_a27_independent_t_tests_match_scipy(tmp_path, groups, method, equal_var, alt):
    _, a, b = groups
    g = example("two_group_comparison")
    set_cfg(g, "compare", method=method, alternative=alt)
    store, status = run(g, tmp_path)
    assert status == "completed"
    r = summary(store, "r1", "compare")
    ref = stats.ttest_ind(a, b, equal_var=equal_var, alternative=alt)
    ci = ref.confidence_interval(0.95)
    assert r["statistic"] == {"name": "t", "value": pytest.approx(ref.statistic, rel=1e-12)}
    assert r["pValue"] == pytest.approx(ref.pvalue, rel=1e-12) and r["df"] == pytest.approx(ref.df, rel=1e-12)
    md = r["meanDifference"]
    assert md["value"] == pytest.approx(a.mean() - b.mean()) and md["definition"] == "mean(A) - mean(B)"
    if alt == "two-sided":
        assert (md["ciLow"], md["ciHigh"]) == (pytest.approx(ci.low), pytest.approx(ci.high))
        assert md["ciLow"] < md["value"] < md["ciHigh"]
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    n1, n2 = len(a), len(b)
    eff = {e["name"]: e["value"] for e in r["effects"]}
    if equal_var:
        sp = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        d = (a.mean() - b.mean()) / sp
    else:
        d = (a.mean() - b.mean()) / math.sqrt((s1**2 + s2**2) / 2)
    assert eff["cohens_d"] == pytest.approx(d) and eff["hedges_g"] == pytest.approx(d * hedges_j(n1 + n2 - 2))
    # design, unit, assumptions and decision are all present
    assert r["design"]["pairing"] == "independent groups" and r["design"]["sampleUnit"] == "one independent assay tube"
    assert r["design"]["independence"] == {"checked": True, "nUnits": 26, "repeatedUnits": 0}
    assert r["decision"] == ("reject_h0" if ref.pvalue < 0.05 else "fail_to_reject_h0") and r["alpha"] == 0.05
    assert ("EQUAL variance" in " ".join(r["assumptions"])) == equal_var
    assert [x["n"] for x in r["groups"]] == [n1, n2] and r["groups"][0]["mean"] == pytest.approx(a.mean())
    assert "practical importance" in r["caveat"]
    assert r["plot"]["family"] == "t" and r["plot"]["df"] == pytest.approx(ref.df)
    assert r["plot"]["tail"] == {"two-sided": "two_sided", "greater": "upper", "less": "lower"}[alt]
    assert "levene_equal_variance" in r["diagnostics"] and r["diagnostics"]["levene_equal_variance"]["pValue"] == pytest.approx(stats.levene(a, b).pvalue)


def test_a27_the_welch_result_is_not_the_student_result(tmp_path, groups):
    g1, g2 = example("two_group_comparison"), example("two_group_comparison")
    set_cfg(g2, "compare", method="student")
    r1 = summary(run(g1, tmp_path / "w")[0], "r1", "compare")
    r2 = summary(run(g2, tmp_path / "s")[0], "r1", "compare")
    assert r1["df"] != r2["df"] and r2["df"] == 24 and r1["pValue"] != r2["pValue"]  # unequal n and unequal variances: the choice matters


def test_a27_mann_whitney_matches_scipy(tmp_path, groups):
    _, a, b = groups
    g = example("two_group_comparison")
    set_cfg(g, "compare", method="mann_whitney")
    store, status = run(g, tmp_path)
    assert status == "completed"
    r = summary(store, "r1", "compare")
    ref = stats.mannwhitneyu(a, b, alternative="two-sided")
    assert r["statistic"] == {"name": "U", "value": pytest.approx(ref.statistic)} and r["pValue"] == pytest.approx(ref.pvalue, rel=1e-12)
    eff = {e["name"]: e["value"] for e in r["effects"]}
    cl = ref.statistic / (len(a) * len(b))
    assert eff["common_language"] == pytest.approx(cl) and eff["rank_biserial"] == pytest.approx(2 * cl - 1)
    assert r["meanDifference"] is None and "rank-based" in r["uncertainty"] and r["plot"] is None
    assert "stochastically" in r["alternative"] or "differ" in r["alternative"]


def test_a27_paired_t_test_matches_scipy_and_pairs_by_unit(tmp_path):
    from graph_core.schema import Graph

    g = Graph.model_validate({"graphKind": "tabular", "backend": "python", "nodes": [
        {"id": "src", "type": "tabular.csv_source", "config": {"path": str(FIX / "synthetic_enzyme_paired.csv")}},
        {"id": "cmp", "type": "scipy.two_group_comparison", "config": {
            "value_column": "activity_u_per_mg", "group_column": "timepoint", "group_a": "before", "group_b": "after", "method": "paired",
            "sample_unit": "one specimen (measured twice)", "unit_column": "specimen"}}],
        "edges": [{"id": "e", "kind": "table", "from": {"node": "src", "port": "table"}, "to": {"node": "cmp", "port": "table"}}]})
    store, status = run(g, tmp_path)
    assert status == "completed"
    r = summary(store, "r1", "cmp")
    df = pd.read_csv(FIX / "synthetic_enzyme_paired.csv").pivot(index="specimen", columns="timepoint", values="activity_u_per_mg")
    a, b = df["before"].to_numpy(), df["after"].to_numpy()
    ref = stats.ttest_rel(a, b)
    ci = ref.confidence_interval(0.95)
    d = a - b
    assert r["statistic"]["value"] == pytest.approx(ref.statistic) and r["pValue"] == pytest.approx(ref.pvalue) and r["df"] == 9
    assert (r["meanDifference"]["ciLow"], r["meanDifference"]["ciHigh"]) == (pytest.approx(ci.low), pytest.approx(ci.high))
    eff = {e["name"]: e["value"] for e in r["effects"]}
    assert eff["cohens_dz"] == pytest.approx(d.mean() / d.std(ddof=1)) and eff["hedges_g"] == pytest.approx(d.mean() / d.std(ddof=1) * hedges_j(9))
    assert r["design"]["pairing"] == "paired" and r["design"]["nPairs"] == 10 and r["design"]["unitColumn"] == "specimen"
    assert r["observations"]["units"][0] == "S01"
    # ignoring the pairing (independent test on the same numbers) gives a different answer, and is refused because units repeat
    set_cfg(g, "cmp", method="welch")
    store2, status2 = run(g, tmp_path / "b")
    assert status2 == "failed"
    f = store2.events("r1", -1, ("node_failed",))[0]
    assert f["data"]["code"] == "E_REPEATED_UNITS" and "paired" in f["data"]["message"]


def test_comparison_requires_declared_sample_unit_and_pairing_id():
    g = example("two_group_comparison")
    set_cfg(g, "compare", sample_unit="")
    d = [x for x in validate(g).diagnostics if x.code == "E_SAMPLE_UNIT_MISSING"]
    assert d and d[0].nodeId == "compare" and d[0].port == "table"
    g = example("two_group_comparison")
    set_cfg(g, "compare", method="paired", unit_column=None)
    assert "E_PAIRING_UNIT_MISSING" in codes(validate(g))
    g = example("two_group_comparison")
    set_cfg(g, "compare", group_a="control", group_b="control")
    assert "E_GROUPS_NOT_SET" in codes(validate(g))


def test_unknown_group_label_and_missing_handling(tmp_path):
    g = example("two_group_comparison")
    set_cfg(g, "compare", group_b="nope")
    store, status = run(g, tmp_path)
    f = store.events("r1", -1, ("node_failed",))[0]
    assert status == "failed" and f["data"]["code"] == "E_GROUP_NOT_FOUND" and "control" in f["data"]["message"]
    csv = tmp_path / "m.csv"
    csv.write_text("t,c,v\nT1,x,1.0\nT2,x,2.0\nT3,x,3.0\nT4,y,4.0\nT5,y,\nT6,y,6.5\nT7,y,5.0\n")
    g = example("two_group_comparison")
    set_cfg(g, "enzyme", path=str(csv))
    set_cfg(g, "compare", value_column="v", group_column="c", group_a="x", group_b="y", unit_column="t")
    store, status = run(g, tmp_path / "ok")
    r = summary(store, "r1", "compare")
    assert status == "completed" and r["design"]["missingDropped"] == 1 and r["design"]["missingPolicy"] == "drop" and r["groups"][1]["n"] == 3
    ref = stats.ttest_ind([1, 2, 3], [4, 6.5, 5], equal_var=False)
    assert r["pValue"] == pytest.approx(ref.pvalue)
    set_cfg(g, "compare", missing="error")
    store, status = run(g, tmp_path / "err")
    assert status == "failed" and store.events("r1", -1, ("node_failed",))[0]["data"]["code"] == "E_MISSING_VALUES"


def test_alpha_in_the_comparison_changes_only_the_decision(tmp_path):
    out = {}
    for alpha in (0.05, 0.001):
        g = example("two_group_comparison")
        set_cfg(g, "compare", alpha=alpha)
        out[alpha] = summary(run(g, tmp_path / str(alpha))[0], "r1", "compare")
    assert out[0.05]["pValue"] == out[0.001]["pValue"] and out[0.05]["meanDifference"] == out[0.001]["meanDifference"]
    assert out[0.05]["plot"]["criticalHigh"] != out[0.001]["plot"]["criticalHigh"]
