"""scikit-learn estimators (ordinary least squares, logistic regression) and validation metrics.

Controls mirror each algorithm: LinearRegression is a closed-form least-squares solve, so there is no learning rate or epoch
control; LogisticRegression is iterative (lbfgs) and exposes C, tolerance and max_iter."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from pydantic import Field
from sklearn import metrics as skm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression

from graph_core.registry import register
from graph_core.types import Fix, OpError
from tabular.core import MODEL, METRICS, TABLE, ExecutionError, FittedModel, Plain, Table, TabularOperation, VType, clean, dtype_name, leakage_check, need_columns, need_numeric, row_ids_sha256

from ._common import StrictConfig
from .tabular_ops import NUMERIC, _fit_info, _owner, _require_train, predict_frame, sample_rows

REGRESSION_METRICS = ("mse", "rmse", "mae", "r2")
CLASSIFICATION_METRICS = ("accuracy", "log_loss", "roc_auc", "confusion_matrix")


class EstimatorConfigBase(StrictConfig):
    target: str = Field("", description="Column to predict")
    features: list[str] = Field(default_factory=list, title="feature columns (empty = every other column)")


def _resolve_features(t: VType, cfg, node_id: str, classification: bool) -> list[str] | None:
    what = f"Estimator '{node_id}'"
    if not cfg.target:
        raise OpError("E_TARGET_NOT_SET", "No target column chosen.", "train", [Fix("Set 'target' to the column to predict")])
    need_columns(t, [cfg.target] + cfg.features, "train", what)
    if cfg.target in cfg.features:
        raise OpError("E_TARGET_IN_FEATURES", f"The target '{cfg.target}' is also listed as a feature, which would leak the answer.", "train",
                      [Fix("Remove the target from 'features'")])
    td = t.dtype_of(cfg.target)
    if classification and td == "float":
        raise OpError("E_COLUMN_TYPE", f"Classification target '{cfg.target}' is float; cast it to int, bool or string in a column selection.", "train")
    if not classification and td not in (None, "int", "float"):
        raise OpError("E_COLUMN_TYPE", f"Regression target '{cfg.target}' is {td}, not numeric.", "train")
    if cfg.features:
        need_numeric(t, cfg.features, "train", what)
        return cfg.features
    if not t.complete:
        return None
    feats = [c["name"] for c in t.columns if c["name"] != cfg.target]
    non = [c["name"] for c in t.columns if c["name"] != cfg.target and c["dtype"] not in NUMERIC]
    if non:
        raise OpError("E_COLUMN_TYPE", f"Feature column(s) {non} are not numeric. One-hot encode them with a fit/apply pair, or leave them out of the column selection.", "train",
                      [Fix("Encode categorical columns with 'Fit one-hot encoding' + 'Apply fitted transform'")])
    if not feats:
        raise OpError("E_NO_COLUMNS", "There are no feature columns.", "train")
    return feats


def _runtime_xy(t: Table, cfg, what: str, classification: bool):
    df = t.df
    if cfg.target not in df.columns:
        raise ExecutionError("E_COLUMN_NOT_FOUND", f"Target '{cfg.target}' not found; the table has {list(df.columns)}.")
    feats = cfg.features or [str(c) for c in df.columns if c != cfg.target]
    miss = [f for f in feats if f not in df.columns]
    if miss:
        raise ExecutionError("E_COLUMN_NOT_FOUND", f"Feature columns {miss} not found.")
    nonnum = [f for f in feats if dtype_name(df[f]) not in ("int", "float", "bool")]
    if nonnum:
        raise ExecutionError("E_COLUMN_TYPE", f"{what}: feature column(s) {nonnum} are not numeric; encode them first.")
    X = df[feats]
    nan = {c: int(X[c].isna().sum()) for c in feats if X[c].isna().any()}
    if nan or df[cfg.target].isna().any():
        raise ExecutionError("E_MISSING_VALUES", f"{what}: missing values remain (features {nan}, target {int(df[cfg.target].isna().sum())}). "
                             "Impute them with a train-fitted imputer or drop those rows before fitting.")
    return feats, X.to_numpy(dtype="float64"), df[cfg.target].to_numpy()


class LinearRegressionConfig(EstimatorConfigBase):
    fit_intercept: bool = True
    positive: bool = Field(False, title="force non-negative coefficients")


@register
class LinearRegressionOp(TabularOperation):
    type = "sklearn.linear_regression"
    backend = "scikit-learn"
    inputs = ("train",)
    outputs = ("model",)
    in_kinds = {"train": TABLE}
    out_kinds = {"model": MODEL}
    Config = LinearRegressionConfig
    summary_kind = "coefficients"

    def infer(self, cfg, ins, node_id):
        t = ins["train"]
        leakage_check(t, "train", f"Linear regression '{node_id}'")
        feats = _resolve_features(t, cfg, node_id, False)
        return {"model": VType(MODEL, {"task": "regression", "estimator": "LinearRegression", "features": feats, "target": cfg.target,
                                       "fittedOn": _fit_info(t)})}

    def param_count(self, cfg, ins):
        f = ins["train"]
        if cfg.features:
            n = len(cfg.features)
        elif f.complete:
            n = len([c for c in f.col_names if c != cfg.target])
        else:
            return 0  # feature list not known until the one-hot categories are fitted
        return n + (1 if cfg.fit_intercept else 0)

    def execute(self, cfg, ins, ctx):
        t: Table = ins["train"]
        _require_train(t, f"Linear regression '{ctx.node_id}'")
        feats, X, y = _runtime_xy(t, cfg, f"Linear regression '{ctx.node_id}'", False)
        est = LinearRegression(fit_intercept=cfg.fit_intercept, positive=cfg.positive).fit(X, y.astype("float64"))
        owner = _owner(t, ctx.node_id)
        details = {"task": "regression", "estimator": "sklearn.linear_model.LinearRegression", "params": est.get_params(), "features": feats,
                   "target": cfg.target, "fittedOn": owner, "nTrain": len(X), "sklearn": sklearn.__version__,
                   "coefficients": [{"feature": f, "value": float(c)} for f, c in zip(feats, est.coef_)],
                   "intercept": float(est.intercept_), "rank": int(est.rank_), "singularValues": est.singular_.tolist() if hasattr(est, "singular_") else None,
                   "preprocessingContext": "Coefficients are in the units of the (transformed) columns that reached this block; standardized inputs give per-standard-deviation effects."}
        return {"model": FittedModel("regression", est, feats, cfg.target, owner, clean(details))}, clean(details)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "y_hat = X beta + beta_0,   beta = argmin ||y - X beta||^2  (ordinary least squares, closed form)",
                "rule": "sklearn.linear_model.LinearRegression: no learning rate, no epochs. rank_ and singular values expose collinearity.",
                "note": "Fitted on the training partition only."}


class LogisticRegressionConfig(EstimatorConfigBase):
    C: float = Field(1.0, gt=0, title="C (inverse L2 regularization strength)")
    max_iter: int = Field(100, ge=1)
    tol: float = Field(1e-4, gt=0)
    fit_intercept: bool = True


@register
class LogisticRegressionOp(TabularOperation):
    type = "sklearn.logistic_regression"
    backend = "scikit-learn"
    inputs = ("train",)
    outputs = ("model",)
    in_kinds = {"train": TABLE}
    out_kinds = {"model": MODEL}
    Config = LogisticRegressionConfig
    summary_kind = "coefficients"

    def infer(self, cfg, ins, node_id):
        t = ins["train"]
        leakage_check(t, "train", f"Logistic regression '{node_id}'")
        feats = _resolve_features(t, cfg, node_id, True)
        return {"model": VType(MODEL, {"task": "classification", "estimator": "LogisticRegression", "features": feats, "target": cfg.target,
                                       "fittedOn": _fit_info(t)})}

    def execute(self, cfg, ins, ctx):
        t: Table = ins["train"]
        _require_train(t, f"Logistic regression '{ctx.node_id}'")
        feats, X, y = _runtime_xy(t, cfg, f"Logistic regression '{ctx.node_id}'", True)
        if len(np.unique(y)) < 2:
            raise ExecutionError("E_ONE_CLASS", "The training partition contains a single class; logistic regression needs at least two.")
        est = LogisticRegression(C=cfg.C, max_iter=cfg.max_iter, tol=cfg.tol, fit_intercept=cfg.fit_intercept)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            est.fit(X, y)
        converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
        classes = [clean(c) for c in est.classes_]
        coef = est.coef_
        rows = ([{"class": classes[1], "note": "log-odds of this class vs the other", "coefficients": [{"feature": f, "value": float(v)} for f, v in zip(feats, coef[0])],
                  "intercept": float(est.intercept_[0])}] if len(classes) == 2 else
                [{"class": classes[i], "coefficients": [{"feature": f, "value": float(v)} for f, v in zip(feats, coef[i])], "intercept": float(est.intercept_[i])}
                 for i in range(len(classes))])
        owner = _owner(t, ctx.node_id)
        details = {"task": "classification", "estimator": "sklearn.linear_model.LogisticRegression", "params": est.get_params(), "features": feats, "target": cfg.target,
                   "fittedOn": owner, "nTrain": len(X), "classes": classes, "perClass": rows, "nIter": est.n_iter_.tolist(), "converged": converged,
                   "warnings": [str(x.message) for x in w], "sklearn": sklearn.__version__}
        return {"model": FittedModel("classification", est, feats, cfg.target, owner, clean(details))}, clean(details)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "p(class=1|x) = sigmoid(w.x + b);  minimize  0.5 ||w||^2 + C * sum_i log_loss_i",
                "rule": "sklearn.linear_model.LogisticRegression (lbfgs, L2). Larger C means weaker regularization. Not converged => raise max_iter or scale features.",
                "note": "Fitted on the training partition only."}


# ================================================================================================ metrics
class MetricsConfig(StrictConfig):
    metrics: list[str] = Field(default_factory=list, title="metrics (empty = all that apply)")


@register
class ValidationMetrics(TabularOperation):
    type = "sklearn.metrics"
    backend = "scikit-learn"
    inputs = ("model", "table")
    outputs = ("metrics",)
    in_kinds = {"model": MODEL, "table": TABLE}
    out_kinds = {"metrics": METRICS}
    Config = MetricsConfig
    summary_kind = "metrics"

    def infer(self, cfg, ins, node_id):
        m, t = ins["model"], ins["table"]
        allowed = REGRESSION_METRICS if m.info["task"] == "regression" else CLASSIFICATION_METRICS
        bad = [x for x in cfg.metrics if x not in allowed]
        if bad:
            raise OpError("E_METRIC_TASK_MISMATCH", f"Metric(s) {bad} do not apply to a {m.info['task']} model. Available: {list(allowed)}.", "model",
                          [Fix(f"Use metrics from {list(allowed)}", node_id, "metrics", [])])
        feats = m.info.get("features")
        need_columns(t, ([m.info["target"]] + (feats or [])), "table", f"Metrics '{node_id}'")
        return {"metrics": VType(METRICS, {"task": m.info["task"], "metrics": list(cfg.metrics or allowed), "evaluatedPartition": t.partition})}

    def warnings(self, cfg, ins, node_id):
        t = ins["table"]
        if t.partition == "train":
            return [("W_METRICS_ON_TRAIN", "Metrics are computed on the training partition the model was fitted on; they are optimistic. Connect the validation partition for an honest estimate.", "table")]
        if t.partition == "full":
            return [("W_METRICS_ON_UNSPLIT", "Metrics are computed on an unpartitioned table; they say nothing about held-out performance.", "table")]
        return []

    def execute(self, cfg, ins, ctx):
        m: FittedModel = ins["model"]
        t: Table = ins["table"]
        pf = predict_frame(m, t, False, True)
        y, p = pf["observed"].to_numpy(), pf["predicted"].to_numpy()
        n = len(pf)
        wanted = cfg.metrics or list(REGRESSION_METRICS if m.task == "regression" else CLASSIFICATION_METRICS)
        vals: dict[str, Any] = {}
        unavailable: dict[str, str] = {}
        extra: dict[str, Any] = {}
        if m.task == "regression":
            y = y.astype("float64")
            fn = {"mse": lambda: skm.mean_squared_error(y, p), "rmse": lambda: skm.root_mean_squared_error(y, p),
                  "mae": lambda: skm.mean_absolute_error(y, p), "r2": lambda: skm.r2_score(y, p)}
            for k in wanted:
                try:
                    vals[k] = float(fn[k]())
                except ValueError as e:
                    unavailable[k] = str(e)
            idx = np.linspace(0, n - 1, min(n, 400)).astype(int) if n else []
            extra["points"] = [{"rowId": int(pf.index[i]), "observed": float(y[i]), "predicted": float(p[i])} for i in idx]
            res = y - p
            extra["residuals"] = {"mean": float(res.mean()), "std": float(res.std(ddof=1)) if n > 1 else None, "min": float(res.min()), "max": float(res.max())}
        else:
            classes = list(m.estimator.classes_)
            proba = pf[[f"proba_{c}" for c in classes]].to_numpy()
            for k in wanted:
                try:
                    if k == "accuracy":
                        vals[k] = float(skm.accuracy_score(y, p))
                    elif k == "log_loss":
                        vals[k] = float(skm.log_loss(y, proba, labels=classes))
                    elif k == "roc_auc":
                        if len(np.unique(y)) < 2:
                            unavailable[k] = "ROC-AUC is undefined: only one class is present in the evaluated partition."
                        elif len(classes) == 2:
                            vals[k] = float(skm.roc_auc_score(y, proba[:, 1]))
                        else:
                            vals[k] = float(skm.roc_auc_score(y, proba, multi_class="ovr", labels=classes))
                    elif k == "confusion_matrix":
                        extra["confusion"] = {"labels": [clean(c) for c in classes], "matrix": skm.confusion_matrix(y, p, labels=classes).tolist(),
                                              "axes": "rows = observed, columns = predicted"}
                except ValueError as e:
                    unavailable[k] = str(e)
            if len(classes) == 2:
                extra["positiveClass"] = clean(classes[1])
        res = {"task": m.task, "values": vals, "unavailable": unavailable, "n": n, "evaluatedPartition": t.partition,
               "evaluatedRowIdsSha256": row_ids_sha256(pf.index),
               "model": {"estimator": m.details.get("estimator"), "fittedOn": m.fitted_on}, **extra}
        if t.partition == "train":
            res["warning"] = "Evaluated on the training partition: optimistic."
        res = clean(res)
        return {"metrics": Plain(METRICS, res)}, res

    def explain(self, cfg, inputs, outputs):
        return {"equation": "regression: MSE = mean((y-yhat)^2), RMSE = sqrt(MSE), MAE = mean|y-yhat|, R2 = 1 - SS_res/SS_tot;  classification: accuracy, log-loss, ROC-AUC, confusion matrix",
                "rule": "scikit-learn metric functions applied to the table connected to 'table' (use the validation partition). ROC-AUC uses P(second class); multiclass uses one-vs-rest.",
                "note": "Metrics are values from this run, never from a stored claim."}
