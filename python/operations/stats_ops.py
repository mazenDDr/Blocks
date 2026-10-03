"""scipy.* statistics blocks: Gamma function and distribution, tail probability, a structured hypothesis test and two-group comparison.

A distribution's density at a point is not a probability; a p-value is a tail area of the stated reference distribution, compared with
alpha (never with a density height). Changing alpha moves the decision boundary and the decision, never the already computed p-value."""
from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field
from scipy import integrate, special, stats

from graph_core.registry import register
from graph_core.types import Fix, OpError
from tabular.core import (DISTRIBUTION, NUMBER, TABLE, TAIL, TEST_RESULT, ExecutionError, Plain, Table, TabularOperation, VType, clean,
                          need_columns, need_numeric)

from ._common import StrictConfig

CAVEAT = ("A p-value is not the probability that the null hypothesis is true, and neither a small nor a large p-value measures practical importance "
          "or establishes causation. Read it together with the effect estimate and the study design.")


def curve(dist, observed: float | None, points: int = 300, extra: tuple[float, ...] = ()) -> dict[str, Any]:
    """Density/CDF/survival values on a grid (real computed values) covering the bulk of the distribution and the observation."""
    lo, hi = float(dist.ppf(0.0005)), float(dist.ppf(0.9995))
    span = hi - lo
    anchors = [v for v in ((observed,) if observed is not None else ()) + extra if np.isfinite(v)]
    if anchors:
        lo, hi = min(lo, min(anchors) - 0.15 * span), max(hi, max(anchors) + 0.15 * span)
    sup_lo = float(dist.support()[0])
    if np.isfinite(sup_lo):
        lo = max(lo, sup_lo)
    x = np.linspace(lo, hi, points)
    return {"x": x, "density": dist.pdf(x), "cdf": dist.cdf(x), "survival": dist.sf(x)}


# ================================================================================================ Gamma function
class GammaFunctionConfig(StrictConfig):
    x: float = 5.0


@register
class GammaFunction(TabularOperation):
    type = "scipy.gamma_function"
    backend = "scipy"
    inputs = ()
    outputs = ("value",)
    out_kinds = {"value": NUMBER}
    Config = GammaFunctionConfig
    summary_kind = "number"

    def infer(self, cfg, ins, node_id):
        if cfg.x <= 0 and float(cfg.x).is_integer():
            raise OpError("E_DOMAIN", f"Gamma(x) has poles at the non-positive integers; x = {cfg.x:g} is undefined.", None, [Fix("Choose x that is not 0, -1, -2, ...")])
        return {"value": VType(NUMBER, {"name": f"Gamma({cfg.x:g})"})}

    def execute(self, cfg, ins, ctx):
        v = float(special.gamma(cfg.x))
        d = {"name": "gamma_function", "x": cfg.x, "value": v, "logGamma": float(special.gammaln(cfg.x)),
             "identity": f"Gamma(n) = (n-1)! for positive integers n" + (f"; here (x-1)! = {math.factorial(int(cfg.x) - 1)}" if float(cfg.x).is_integer() and 0 < cfg.x < 21 else ""),
             "scipy": "scipy.special.gamma"}
        return {"value": Plain(NUMBER, d)}, clean(d)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "Gamma(x) = integral_0^inf t^(x-1) e^(-t) dt", "rule": "Special function, not a probability distribution; it has no scale or rate. scipy.special.gamma.",
                "note": "Domain: all reals except 0, -1, -2, ... (poles)."}


# ================================================================================================ Gamma distribution
class GammaDistributionConfig(StrictConfig):
    parametrization: Literal["shape_scale", "shape_rate"] = Field("shape_scale", description="Which of scale or rate is used. rate = 1/scale.")
    shape: float = Field(2.0, gt=0, title="shape k")
    scale: float = Field(1.0, gt=0, title="scale theta (used when parametrization = shape_scale)")
    rate: float = Field(1.0, gt=0, title="rate beta = 1/theta (used when parametrization = shape_rate)")
    loc: float = Field(0.0, title="location")


def gamma_dist(cfg: GammaDistributionConfig):
    scale = cfg.scale if cfg.parametrization == "shape_scale" else 1.0 / cfg.rate
    return stats.gamma(a=cfg.shape, loc=cfg.loc, scale=scale), scale


@register
class GammaDistribution(TabularOperation):
    type = "scipy.gamma_distribution"
    backend = "scipy"
    inputs = ()
    outputs = ("distribution",)
    out_kinds = {"distribution": DISTRIBUTION}
    Config = GammaDistributionConfig
    summary_kind = "distribution"

    def infer(self, cfg, ins, node_id):
        _, scale = gamma_dist(cfg)
        return {"distribution": VType(DISTRIBUTION, {"family": "gamma", "shape": cfg.shape, "scale": scale, "rate": 1.0 / scale, "loc": cfg.loc,
                                                       "parametrization": cfg.parametrization})}

    def execute(self, cfg, ins, ctx):
        dist, scale = gamma_dist(cfg)
        c = curve(dist, None)
        qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        d = {"family": "gamma", "parametrization": cfg.parametrization, "shape": cfg.shape, "scale": scale, "rate": 1.0 / scale, "loc": cfg.loc,
             "mean": float(dist.mean()), "variance": float(dist.var()), "std": float(dist.std()), "support": [cfg.loc, None],
             "quantiles": [{"p": q, "x": float(dist.ppf(q))} for q in qs], "curve": c, "scipy": f"scipy.stats.gamma(a={cfg.shape:g}, loc={cfg.loc:g}, scale={scale:g})",
             "note": "The curve shows the probability DENSITY. A density value is not a probability; probabilities are areas under it."}
        return {"distribution": Plain(DISTRIBUTION, clean(d), dist)}, clean(d)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "f(x) = (x-loc)^(k-1) exp(-(x-loc)/theta) / (Gamma(k) theta^k),  x >= loc;  rate = 1/scale",
                "rule": "scipy.stats.gamma(a=shape, loc=loc, scale=scale). The block states which of scale or rate you mean; the other is derived.",
                "note": "A Gamma fitted to observations is a different thing from a justified null distribution of a test statistic."}


# ================================================================================================ tail probability
class TailConfig(StrictConfig):
    observed: float = Field(5.0, title="observed value")
    tail: Literal["upper", "lower"] = Field("upper", title="tail (the direction that counts against the null)")


def _closed_form_upper(shape: float, scale: float, loc: float, obs: float) -> tuple[str, float] | None:
    if not float(shape).is_integer() or shape < 1 or shape > 50:
        return None
    z = (obs - loc) / scale
    if z < 0:
        return None
    k = int(shape)
    terms = [1.0] + [z**j / math.factorial(j) for j in range(1, k)]
    val = math.exp(-z) * sum(terms)
    parts = ["1"] + [f"{z:g}" if j == 1 else f"{z:g}^{j}/{j}!" for j in range(1, k)]
    return f"exp(-{z:g}) × ({' + '.join(parts)})", val


@register
class TailProbability(TabularOperation):
    type = "scipy.tail_probability"
    backend = "scipy"
    inputs = ("distribution",)
    outputs = ("tail",)
    in_kinds = {"distribution": DISTRIBUTION}
    out_kinds = {"tail": TAIL}
    Config = TailConfig
    summary_kind = "tail"

    def infer(self, cfg, ins, node_id):
        d = ins["distribution"]
        return {"tail": VType(TAIL, {"tail": cfg.tail, "observed": cfg.observed, "family": d.info.get("family")})}

    def execute(self, cfg, ins, ctx):
        pd_: Plain = ins["distribution"]
        dist = pd_.obj
        info = pd_.data
        obs = cfg.observed
        if cfg.tail == "upper":
            p = float(dist.sf(obs))
            formula = "P(T >= t) = SF(t) = 1 - CDF(t)"
        else:
            p = float(dist.cdf(obs))
            formula = "P(T <= t) = CDF(t)"
        sup_lo = float(dist.support()[0])
        a, b = (max(obs, sup_lo), np.inf) if cfg.tail == "upper" else (sup_lo, obs)
        area, err = integrate.quad(dist.pdf, a, b, limit=200) if b > a else (0.0, 0.0)
        cf = _closed_form_upper(info["shape"], info["scale"], info["loc"], obs) if info.get("family") == "gamma" and cfg.tail == "upper" else None
        d = {"family": info.get("family"), "reference": {k: info[k] for k in ("family", "shape", "scale", "rate", "loc") if k in info},
             "observed": obs, "tail": cfg.tail, "pValue": p, "formula": formula, "densityAtObserved": float(dist.pdf(obs)),
             "areaNumerical": float(area), "areaNumericalError": float(err),
             "closedForm": ({"expression": cf[0], "value": cf[1], "matchesScipy": bool(abs(cf[1] - p) < 1e-12)} if cf else None),
             "note": "p is a probability (an area). densityAtObserved is a height, not a probability, and is shown only for contrast.",
             "plot": clean(curve(dist, obs))}
        d = clean(d)
        return {"tail": Plain(TAIL, d, dist)}, d

    def explain(self, cfg, inputs, outputs):
        return {"equation": "upper: p = P(T >= t) = SF(t);  lower: p = P(T <= t) = CDF(t)",
                "rule": "Uses the distribution's survival function (SciPy sf), cross-checked by numerically integrating the density over the tail. "
                        "Only one-sided tails are offered here: a two-sided p-value needs the method's own extremeness rule, not a mechanical doubling.",
                "note": "The tail direction is part of the hypothesis, chosen before looking at the statistic."}


# ================================================================================================ hypothesis test
class HypothesisConfig(StrictConfig):
    statistic_name: str = Field("T", title="statistic symbol")
    null_statement: str = Field("", title="null hypothesis (blank = generated from the reference distribution)")
    alternative_statement: str = Field("", title="alternative (blank = generated from the tail)")
    alpha: float = Field(0.05, gt=0, lt=1, title="significance level alpha")
    assumptions: list[str] = Field(default_factory=list, title="additional assumptions")
    teaching_fixture: bool = Field(False, description="Mark this test as a teaching fixture: the statistic is specified by the example, not measured from data.")


def decide(p: float, alpha: float) -> tuple[str, str]:
    if p < alpha:
        return "reject_h0", "reject H0 under this specified test"
    return "fail_to_reject_h0", "fail to reject H0 (this is not evidence that H0 is true)"


@register
class HypothesisTest(TabularOperation):
    type = "scipy.hypothesis_test"
    backend = "scipy"
    inputs = ("tail",)
    outputs = ("result",)
    in_kinds = {"tail": TAIL}
    out_kinds = {"result": TEST_RESULT}
    Config = HypothesisConfig
    summary_kind = "test_result"

    def infer(self, cfg, ins, node_id):
        return {"result": VType(TEST_RESULT, {"method": "tail_probability_test", "alpha": cfg.alpha})}

    def execute(self, cfg, ins, ctx):
        tail: Plain = ins["tail"]
        t, dist = tail.data, tail.obj
        ref = t["reference"]
        alpha = cfg.alpha
        p = t["pValue"]
        name = cfg.statistic_name
        refs = ", ".join(f"{k}={ref[k]:g}" for k in ("shape", "scale", "loc") if k in ref)
        upper = t["tail"] == "upper"
        decision, text = decide(p, alpha)
        crit = float(dist.isf(alpha)) if upper else float(dist.ppf(alpha))
        plot = dict(t["plot"])
        plot.update({"tail": t["tail"], "observed": t["observed"], "criticalValue": crit, "family": ref.get("family"), "statisticName": name})
        res = {
            "type": "hypothesis_test", "method": "tail_probability_test",
            "null": cfg.null_statement or f"{name} ~ {str(ref.get('family', '')).capitalize()}({refs}) (the stated null reference distribution)",
            "alternative": cfg.alternative_statement or (f"large values of {name} count against the null" if upper else f"small values of {name} count against the null"),
            "statistic": {"name": name, "value": t["observed"]}, "reference": ref, "tail": t["tail"],
            "pValue": p, "pValueFormula": t["formula"], "closedForm": t.get("closedForm"),
            "alpha": alpha, "decision": decision, "decisionText": text,
            "decisionRule": "reject H0 when p < alpha", "criticalValue": crit,
            "criticalValueMeaning": f"smallest {name} with P({name} >= c) = alpha" if upper else f"largest {name} with P({name} <= c) = alpha",
            "assumptions": ["The null distribution of the statistic is SPECIFIED by the analyst (it is not inferred from the raw observations).",
                            "The tail direction was chosen before looking at the statistic.", *cfg.assumptions],
            "sample": None, "statisticSource": "specified in the graph (not computed from data)", "teachingFixture": cfg.teaching_fixture,
            "caveat": CAVEAT, "plot": plot,
        }
        res = clean(res)
        return {"result": Plain(TEST_RESULT, res)}, res

    def explain(self, cfg, inputs, outputs):
        return {"equation": "decision: reject H0 iff p < alpha, equivalently iff the statistic lies beyond the critical value c = SF^-1(alpha)",
                "rule": "p comes from the connected tail-probability block and does not depend on alpha. Changing alpha moves the critical value and may flip the decision.",
                "note": "Compare p with alpha - never a density height or a Gamma-function value."}


# ================================================================================================ two-group comparison
class TwoGroupConfig(StrictConfig):
    value_column: str = Field("", title="measurement column")
    group_column: str = Field("", title="group column")
    group_a: str = Field("", title="group A (first sample)")
    group_b: str = Field("", title="group B (second sample)")
    method: Literal["welch", "student", "paired", "mann_whitney"] = "welch"
    alternative: Literal["two-sided", "less", "greater"] = "two-sided"
    alpha: float = Field(0.05, gt=0, lt=1)
    confidence_level: float = Field(0.95, gt=0, lt=1)
    sample_unit: str = Field("", title="sample unit (what one independent observation is)")
    unit_column: str | None = Field(None, title="unit / subject / pair id column")
    missing: Literal["drop", "error"] = Field("drop", title="missing values")


def hedges_j(df: float) -> float:
    return float(np.exp(special.gammaln(df / 2) - special.gammaln((df - 1) / 2)) / np.sqrt(df / 2)) if df > 1 else float("nan")


def _desc(a: np.ndarray, label: str) -> dict[str, Any]:
    return {"label": label, "n": len(a), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None, "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max())}


METHOD_LABEL = {"welch": "Welch's t-test (unequal variances)", "student": "Student's t-test (equal variances)",
                "paired": "Paired t-test", "mann_whitney": "Mann-Whitney U test"}
ALT_SYMBOL = {"two-sided": "≠", "less": "<", "greater": ">"}


@register
class TwoGroupComparison(TabularOperation):
    type = "scipy.two_group_comparison"
    backend = "scipy"
    inputs = ("table",)
    outputs = ("result",)
    in_kinds = {"table": TABLE}
    out_kinds = {"result": TEST_RESULT}
    Config = TwoGroupConfig
    summary_kind = "test_result"

    def infer(self, cfg, ins, node_id):
        t = ins["table"]
        if not (cfg.value_column and cfg.group_column and cfg.group_a and cfg.group_b):
            raise OpError("E_GROUPS_NOT_SET", "Choose the measurement column, the group column and the two group labels.", "table")
        if cfg.group_a == cfg.group_b:
            raise OpError("E_GROUPS_NOT_SET", "Group A and group B must differ.", "table")
        need_columns(t, [cfg.value_column, cfg.group_column] + ([cfg.unit_column] if cfg.unit_column else []), "table", f"Comparison '{node_id}'")
        need_numeric(t, [cfg.value_column], "table", f"Comparison '{node_id}'")
        if not cfg.sample_unit.strip():
            raise OpError("E_SAMPLE_UNIT_MISSING", "Declare the sample unit (what one independent observation is, e.g. 'one independent culture tube'). "
                          "Repeated measurements of one unit are not independent subjects.", "table", [Fix("Fill in 'sample_unit'")])
        if cfg.method == "paired" and not cfg.unit_column:
            raise OpError("E_PAIRING_UNIT_MISSING", "A paired test needs the column that identifies which observations belong together (subject/pair id).", "table",
                          [Fix("Set 'unit_column'")])
        return {"result": VType(TEST_RESULT, {"method": cfg.method, "alpha": cfg.alpha})}

    def execute(self, cfg, ins, ctx):
        t: Table = ins["table"]
        df = t.df
        for c in [cfg.value_column, cfg.group_column] + ([cfg.unit_column] if cfg.unit_column else []):
            if c not in df.columns:
                raise ExecutionError("E_COLUMN_NOT_FOUND", f"Column '{c}' not found.")
        val = pd.to_numeric(df[cfg.value_column], errors="coerce")
        grp = df[cfg.group_column].astype(str)
        in_a, in_b = grp == cfg.group_a, grp == cfg.group_b
        levels = sorted(grp.dropna().unique().tolist())
        if not in_a.any() or not in_b.any():
            raise ExecutionError("E_GROUP_NOT_FOUND", f"Group labels must be among {levels}; got A='{cfg.group_a}', B='{cfg.group_b}'.")
        sel = df[in_a | in_b].copy()
        sel["_v"], sel["_g"] = val[sel.index], np.where(in_a[sel.index], "A", "B")
        n_missing = int(sel["_v"].isna().sum())
        if n_missing and cfg.missing == "error":
            raise ExecutionError("E_MISSING_VALUES", f"{n_missing} missing measurement(s) in the compared groups; choose missing = 'drop' to analyse complete observations.")
        excluded_other = int(len(df) - len(sel))
        sel = sel.dropna(subset=["_v"])
        units = df.loc[sel.index, cfg.unit_column] if cfg.unit_column else None
        design: dict[str, Any] = {"sampleUnit": cfg.sample_unit, "unitColumn": cfg.unit_column, "missingPolicy": cfg.missing,
                                  "missingDropped": n_missing, "rowsInOtherGroups": excluded_other}
        paired = cfg.method == "paired"
        if paired:
            sel = sel.assign(_u=units.astype(str))
            if sel.duplicated(["_u", "_g"]).any():
                dups = sel[sel.duplicated(["_u", "_g"], keep=False)]["_u"].unique()[:5].tolist()
                raise ExecutionError("E_PAIRING_AMBIGUOUS", f"Unit(s) {dups} have several observations in one group, so pairs are ambiguous. Aggregate repeated measurements per unit first.")
            w = sel.pivot(index="_u", columns="_g", values="_v")
            unpaired = int(w.isna().any(axis=1).sum())
            if unpaired and cfg.missing == "error":
                raise ExecutionError("E_UNPAIRED", f"{unpaired} unit(s) lack an observation in one of the two groups.")
            w = w.dropna()
            a, b = w["A"].to_numpy(), w["B"].to_numpy()
            design.update({"pairing": "paired", "nPairs": len(w), "unpairedUnitsDropped": unpaired, "independence": "pairs are matched by unit; differences are assumed independent across units"})
            obs = {"A": a.tolist()[:500], "B": b.tolist()[:500], "units": w.index.tolist()[:500]}
        else:
            a, b = sel.loc[sel["_g"] == "A", "_v"].to_numpy(), sel.loc[sel["_g"] == "B", "_v"].to_numpy()
            design["pairing"] = "independent groups"
            if cfg.unit_column:
                u = units.astype(str)
                rep = u[u.duplicated(keep=False)]
                design["independence"] = {"checked": True, "nUnits": int(u.nunique()), "repeatedUnits": int(rep.nunique())}
                if len(rep):
                    raise ExecutionError("E_REPEATED_UNITS", f"{int(rep.nunique())} unit(s) (e.g. {rep.unique()[:5].tolist()}) contribute more than one observation. "
                                         f"An independent-groups {METHOD_LABEL[cfg.method]} would treat them as independent; use the paired test, or aggregate per unit first.")
            else:
                design["independence"] = {"checked": False, "note": f"No unit column given: every row is ASSUMED to be a separate '{cfg.sample_unit}'."}
            obs = {"A": a.tolist()[:500], "B": b.tolist()[:500]}
        obs["truncated"] = len(a) > 500 or len(b) > 500
        if len(a) < 2 or len(b) < 2:
            raise ExecutionError("E_GROUP_TOO_SMALL", f"Each group needs at least 2 observations (A has {len(a)}, B has {len(b)}).")
        alt, alpha, cl = cfg.alternative, cfg.alpha, cfg.confidence_level
        ma, mb = float(a.mean()), float(b.mean())
        sa, sb = float(a.std(ddof=1)), float(b.std(ddof=1))
        na, nb = len(a), len(b)
        effects: list[dict[str, Any]] = []
        md: dict[str, Any] | None = None
        plot = None
        if cfg.method in ("welch", "student"):
            eq = cfg.method == "student"
            res = stats.ttest_ind(a, b, equal_var=eq, alternative=alt)
            ci = res.confidence_interval(confidence_level=cl)
            stat, dof, p = float(res.statistic), float(res.df), float(res.pvalue)
            diff = ma - mb
            md = {"value": diff, "ciLow": float(ci.low), "ciHigh": float(ci.high), "level": cl, "alternative": alt, "definition": "mean(A) - mean(B)",
                  "se": float(np.sqrt(sa**2 / na + sb**2 / nb)) if not eq else float(np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2) * (1 / na + 1 / nb)))}
            if eq:
                sp = float(np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2)))
                d = diff / sp
                effects += [{"name": "cohens_d", "label": "Cohen's d (pooled SD)", "value": d}, {"name": "hedges_g", "label": "Hedges' g (bias-corrected d)", "value": d * hedges_j(na + nb - 2)}]
            else:
                dav = diff / float(np.sqrt((sa**2 + sb**2) / 2))
                effects += [{"name": "cohens_d", "label": "Cohen's d (root-mean-square of the two SDs; no equal-variance assumption)", "value": dav},
                            {"name": "hedges_g", "label": "Hedges' g (small-sample factor for df = nA+nB-2 applied to that d; approximate)", "value": dav * hedges_j(na + nb - 2)}]
            call = f"scipy.stats.ttest_ind(a, b, equal_var={eq}, alternative='{alt}')"
            stat_name = "t"
            ref = stats.t(dof)
        elif paired:
            diffs = a - b
            res = stats.ttest_rel(a, b, alternative=alt)
            ci = res.confidence_interval(confidence_level=cl)
            stat, dof, p = float(res.statistic), float(res.df), float(res.pvalue)
            sd_d = float(diffs.std(ddof=1))
            md = {"value": float(diffs.mean()), "ciLow": float(ci.low), "ciHigh": float(ci.high), "level": cl, "alternative": alt,
                  "definition": "mean of within-unit differences (A - B)", "se": sd_d / np.sqrt(len(diffs))}
            dz = float(diffs.mean()) / sd_d
            effects += [{"name": "cohens_dz", "label": "Cohen's d_z (mean difference / SD of differences)", "value": dz},
                        {"name": "hedges_g", "label": "Hedges' g_z (small-sample corrected d_z, df = n-1)", "value": dz * hedges_j(len(diffs) - 1)}]
            call = f"scipy.stats.ttest_rel(a, b, alternative='{alt}')"
            stat_name = "t"
            ref = stats.t(dof)
            ma, mb = float(a.mean()), float(b.mean())
        else:
            res = stats.mannwhitneyu(a, b, alternative=alt)
            stat, dof, p = float(res.statistic), None, float(res.pvalue)
            cles = stat / (na * nb)
            effects += [{"name": "common_language", "label": "P(A > B) + 0.5 P(A = B)  (= U_A / (nA nB))", "value": cles},
                        {"name": "rank_biserial", "label": "rank-biserial correlation (2 x common-language - 1)", "value": 2 * cles - 1}]
            md = None
            call = f"scipy.stats.mannwhitneyu(a, b, alternative='{alt}')  # method='auto'"
            stat_name = "U"
            ref = None
        A, B = cfg.group_a, cfg.group_b
        diag: dict[str, Any] = {}
        for lab, arr in (("A", a), ("B", b)):
            if 3 <= len(arr) <= 5000:
                sw = stats.shapiro(arr if not paired else (a - b))
                diag[f"shapiro_{'differences' if paired else lab}"] = {"statistic": float(sw.statistic), "pValue": float(sw.pvalue)}
                if paired:
                    break
        if not paired:
            lv = stats.levene(a, b)
            diag["levene_equal_variance"] = {"statistic": float(lv.statistic), "pValue": float(lv.pvalue), "note": "diagnostic only; it does not choose the test for you"}
        if ref is not None:
            crit = ({"low": float(ref.ppf(alpha / 2)), "high": float(ref.isf(alpha / 2))} if alt == "two-sided" else
                    {"high": float(ref.isf(alpha))} if alt == "greater" else {"low": float(ref.ppf(alpha))})
            c = curve(ref, stat, extra=tuple(crit.values()) + ((-abs(stat), abs(stat)) if alt == "two-sided" else ()))
            plot = {**c, "tail": {"two-sided": "two_sided", "greater": "upper", "less": "lower"}[alt], "observed": stat, "criticalLow": crit.get("low"),
                    "criticalHigh": crit.get("high"), "family": "t", "df": dof, "statisticName": "t", "twoSidedRule": "P(|T| >= |t|) = 2 x SF(|t|); the t distribution is symmetric"}
        if cfg.method == "mann_whitney":
            n_ = f"The distributions of {A} and {B} are equal"
            alt_s = {"two-sided": f"the distributions of {A} and {B} differ", "greater": f"values in {A} are stochastically greater than in {B}",
                     "less": f"values in {A} are stochastically smaller than in {B}"}[alt]
        elif paired:
            n_ = f"The mean within-unit difference ({A} - {B}) is 0"
            alt_s = f"mean within-unit difference ({A} - {B}) {ALT_SYMBOL[alt]} 0"
        else:
            n_ = f"mean({A}) - mean({B}) = 0"
            alt_s = f"mean({A}) - mean({B}) {ALT_SYMBOL[alt]} 0"
        assumptions = {
            "welch": ["Observations are independent: each row is a separate " + cfg.sample_unit, "Each group is roughly normal, or n is large enough for the mean to be roughly normal",
                      "Variances are NOT assumed equal (Welch-Satterthwaite degrees of freedom)"],
            "student": ["Observations are independent: each row is a separate " + cfg.sample_unit, "Each group is roughly normal, or n is large enough for the mean to be roughly normal",
                        "The two populations have EQUAL variances"],
            "paired": [f"Observations are matched by '{cfg.unit_column}' and the within-unit differences are independent across units", "The differences are roughly normal"],
            "mann_whitney": ["Observations are independent: each row is a separate " + cfg.sample_unit, "The response is at least ordinal",
                             "Reading the result as a location shift additionally requires the two distributions to have the same shape"],
        }[cfg.method]
        decision, text = decide(p, alpha)
        out = {
            "type": "two_group", "method": cfg.method, "methodLabel": METHOD_LABEL[cfg.method], "scipy": call,
            "null": n_, "alternative": alt_s, "alternativeSetting": alt,
            "statistic": {"name": stat_name, "value": stat}, "df": dof, "pValue": p, "alpha": alpha, "decision": decision, "decisionText": text,
            "decisionRule": "reject H0 when p < alpha", "criticalValue": (plot or {}).get("criticalHigh") if plot else None,
            "assumptions": assumptions, "meanDifference": md, "effects": effects,
            "groups": [_desc(a, A), _desc(b, B)], "design": design, "diagnostics": diag, "observations": obs,
            "statisticSource": "computed from the connected table", "teachingFixture": False, "caveat": CAVEAT, "plot": plot,
            "sample": {"rows": int(len(df)), "nA": na, "nB": nb, "sampleUnit": cfg.sample_unit, "inputPartition": t.partition,
                       "sourceSha256": (t.lineage.get("source") or {}).get("sha256")},
        }
        if md is None:
            out["uncertainty"] = "No confidence interval for the mean difference: this rank-based method does not test a mean."
        out = clean(out)
        return {"result": Plain(TEST_RESULT, out)}, out

    def explain(self, cfg, inputs, outputs):
        eqs = {"welch": "t = (mean_A - mean_B) / sqrt(s_A^2/n_A + s_B^2/n_B);  df by Welch-Satterthwaite",
               "student": "t = (mean_A - mean_B) / (s_p sqrt(1/n_A + 1/n_B));  df = n_A + n_B - 2",
               "paired": "t = mean(d) / (s_d / sqrt(n)),  d_i = A_i - B_i;  df = n - 1",
               "mann_whitney": "U_A = sum over pairs [A_i > B_j] + 0.5 [A_i = B_j]; p from scipy's exact/asymptotic distribution (method='auto')"}
        return {"equation": eqs[cfg.method], "rule": f"{METHOD_LABEL[cfg.method]} via SciPy; alternative '{cfg.alternative}'. The method is chosen by you, not by the software; diagnostics are shown beside it.",
                "assumptions": [], "note": "Effect size and its uncertainty are shown beside the decision. Declare the sample unit and keep repeated measurements out of independent tests."}
