"""Unsupervised and representation-learning operations of the `tabular` graph kind (VISION 9.6): k-means, Gaussian mixture, DBSCAN, PCA, a t-SNE projection
(labelled non-metric) and method-appropriate diagnostics with stability. Every estimator is a native scikit-learn object; there is no target, so
fitted state is not owned by a training partition. Cluster membership and density / anomaly scores are different columns with different names."""
from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
import sklearn
from pydantic import Field
from sklearn import metrics as skm
from sklearn.preprocessing import StandardScaler

from graph_core.registry import register
from graph_core.types import Fix, OpError
from tabular.core import (CLUSTER_REPORT, TABLE, UNSUP_MODEL, ExecutionError, Plain, Table, TabularOperation, UnsupModel, VType, clean, dtype_name, need_columns, need_numeric,
                          row_ids_sha256, table_type)
from unsup import methods as M

from ._common import StrictConfig

NUMERIC = ("int", "float")
POLICY = ("There is no universal unsupervised accuracy. Each metric below has an assumption about what a good partition looks like; a value is evidence about that assumption on THIS data, "
          "not a measure of correctness. External metrics appear only when labels are declared and are labelled external.")


class UnsupConfig(StrictConfig):
    features: list[str] = Field(default_factory=list, title="feature columns (empty = every numeric column not excluded)")
    exclude: list[str] = Field(default_factory=list, title="columns to leave out (ids, labels)")
    scale: Literal["standardize", "none"] = Field("standardize", title="feature scaling (fitted on the same rows as the estimator)")


# ------------------------------------------------------------------------------------------------ shared helpers
def _features(t: VType, cfg: UnsupConfig, what: str) -> list[str] | None:
    if cfg.features:
        need_columns(t, cfg.features + cfg.exclude, "table", what)
        need_numeric(t, cfg.features, "table", what)
        both = set(cfg.features) & set(cfg.exclude)
        if both:
            raise OpError("E_FEATURE_EXCLUDED", f"{what}: column(s) {sorted(both)} are both selected and excluded.", "table")
        return list(cfg.features)
    need_columns(t, cfg.exclude, "table", what)
    if not t.complete:
        return None
    feats = [c["name"] for c in t.columns if c["dtype"] in NUMERIC and c["name"] not in cfg.exclude]
    if not feats:
        raise OpError("E_NO_COLUMNS", f"{what}: the table has no numeric feature columns (columns: {t.col_names}).", "table",
                      [Fix("Select numeric feature columns, or cast columns to numbers first")])
    return feats


def _rows_ok(t: VType, need: int, what: str, why: str) -> None:
    n = t.info.get("rows")
    if isinstance(n, int) and n < need:
        raise OpError("E_UNSUP_TOO_FEW_ROWS", f"{what}: {why} needs at least {need} rows; the table has {n}.", "table")


def _xy(table: Table, cfg: UnsupConfig, what: str, scaler: StandardScaler | None | str = "fit") -> tuple[list[str], np.ndarray, np.ndarray, StandardScaler | None]:
    df = table.df
    feats = cfg.features or [str(c) for c in df.columns if dtype_name(df[c]) in NUMERIC and c not in cfg.exclude]
    miss = [f for f in feats if f not in df.columns]
    if miss:
        raise ExecutionError("E_COLUMN_NOT_FOUND", f"{what}: feature columns {miss} not found; the table has {list(df.columns)}.")
    bad = [f for f in feats if dtype_name(df[f]) not in ("int", "float", "bool")]
    if bad:
        raise ExecutionError("E_COLUMN_TYPE", f"{what}: feature column(s) {bad} are not numeric.")
    if not feats:
        raise ExecutionError("E_NO_COLUMNS", f"{what}: no feature columns.")
    X = df[feats].to_numpy(dtype="float64")
    nan = {f: int(df[f].isna().sum()) for f in feats if df[f].isna().any()}
    if nan:
        raise ExecutionError("E_MISSING_VALUES", f"{what}: missing values remain in {nan}. Impute or drop them first (k-means, DBSCAN and PCA cannot use missing values).")
    if scaler == "fit":
        scaler = StandardScaler().fit(X) if cfg.scale == "standardize" else None
    Xs = scaler.transform(X) if scaler is not None else X
    return feats, X, Xs, scaler  # type: ignore[return-value]


def _scaler_json(sc: StandardScaler | None, feats: list[str]) -> dict[str, Any] | None:
    return None if sc is None else {"mean": dict(zip(feats, sc.mean_.tolist())), "scale": dict(zip(feats, sc.scale_.tolist())), "zeroVarianceColumns": [f for f, s in zip(feats, sc.var_) if s == 0]}


def _fitted_on(t: Table, node: str) -> dict[str, Any]:
    return {"rows": len(t.df), "rowIdsSha256": row_ids_sha256(t.df.index), "partition": t.partition, "fittedBy": node, "split": t.lineage.get("split", {}).get("node")}


def _with_columns(t: Table, add: dict[str, Any], drop: list[str] | None = None) -> pd.DataFrame:
    df = t.df.drop(columns=[c for c in (drop or []) if c in t.df.columns]).copy()
    for k, v in add.items():
        df[k] = v
    return df


def _assign_cols(ins: VType, feats: list[str] | None, extra: list[tuple[str, str]], keep_all: bool = True, drop: list[str] | None = None) -> list[dict[str, str]]:
    base = [c for c in ins.columns if (keep_all or (feats and c["name"] in feats)) and c["name"] not in (drop or [])]
    return base + [{"name": n, "dtype": d} for n, d in extra]


def _common_warnings(t: VType, cfg: UnsupConfig, node: str) -> list[tuple[str, str, str | None]]:
    w = []
    if not cfg.features and t.complete and cfg.scale == "none":
        w.append(("W_UNSUP_UNSCALED", "Features are not scaled: a column with a large range dominates every Euclidean distance.", "table"))
    if t.partition == "validation":
        w.append(("W_UNSUP_FIT_ON_VALIDATION", "This estimator is fitted on the validation partition. Clustering has no target, so this is not leakage, but the fit then says nothing about the training data.", "table"))
    return w


class _UnsupOp(TabularOperation):
    backend = "scikit-learn"
    inputs = ("table",)
    in_kinds = {"table": TABLE}
    summary_kind = "clustering"
    method = ""

    def warnings(self, cfg, ins, node_id):
        return _common_warnings(ins["table"], cfg, node_id)

    def param_count(self, cfg, ins):
        return 0


# ================================================================================================ k-means
class KMeansConfig(UnsupConfig):
    n_clusters: int = Field(3, ge=1, le=200, title="k (number of clusters)")
    init: Literal["k-means++", "random"] = "k-means++"
    n_init: int = Field(5, ge=1, le=50, title="independent initialisations (the lowest inertia wins)")
    max_iter: int = Field(300, ge=1)
    tol: float = Field(1e-4, ge=0)
    seed: int = 0


@register
class KMeansOp(_UnsupOp):
    type = "sklearn.kmeans"
    outputs = ("model", "assignments")
    out_kinds = {"model": UNSUP_MODEL, "assignments": TABLE}
    Config = KMeansConfig
    method = "kmeans"

    def infer(self, cfg, ins, node_id):
        t = ins["table"]
        what = f"K-means '{node_id}'"
        feats = _features(t, cfg, what)
        _rows_ok(t, cfg.n_clusters, what, f"k={cfg.n_clusters} clusters")
        cols = _assign_cols(t, feats, [("cluster", "int"), ("distance_to_centroid", "float"), ("distance_to_second_centroid", "float"), ("silhouette", "float")])
        return {"model": VType(UNSUP_MODEL, {"method": "kmeans", "k": cfg.n_clusters, "features": feats, "canPredict": True}),
                "assignments": table_type(cols, t.info.get("rows"), t.partition, t.complete, t.info.get("pending"), **{k: v for k, v in t.info.items() if k in ("splitNode", "splitSeed")})}

    def execute(self, cfg, ins, ctx):
        t: Table = ins["table"]
        what = f"K-means '{ctx.node_id}'"
        feats, X, Xs, sc = _xy(t, cfg, what)
        if len(Xs) < cfg.n_clusters:
            raise ExecutionError("E_UNSUP_TOO_FEW_ROWS", f"{what}: k={cfg.n_clusters} but only {len(Xs)} rows.")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = M.fit_kmeans(Xs, cfg.n_clusters, cfg.init, cfg.n_init, cfg.max_iter, cfg.tol, cfg.seed)
        km = r["estimator"]
        g = M.kmeans_geometry(km, Xs)
        lab = g["labels"]
        sil = np.full(len(Xs), np.nan)
        if 2 <= cfg.n_clusters <= len(Xs) - 1 and len(Xs) <= M.MAX_SILHOUETTE and len(set(lab.tolist())) >= 2:
            sil = skm.silhouette_samples(Xs, lab)
        centers_o = sc.inverse_transform(km.cluster_centers_) if sc is not None else km.cluster_centers_
        sizes = np.bincount(lab, minlength=cfg.n_clusters)
        clusters = [{"cluster": c, "size": int(sizes[c]), "fraction": float(sizes[c] / len(lab)), "inertiaContribution": g["contrib"][c],
                     "inertiaShare": g["contrib"][c] / float(km.inertia_) if km.inertia_ else None,
                     "center": dict(zip(feats, centers_o[c].tolist())), "centerScaled": dict(zip(feats, km.cluster_centers_[c].tolist()))} for c in range(cfg.n_clusters)]
        details = {"method": "kmeans", "estimator": "sklearn.cluster.KMeans", "sklearn": sklearn.__version__, "features": feats, "scale": cfg.scale, "scaler": _scaler_json(sc, feats),
                   "params": {"n_clusters": cfg.n_clusters, "init": cfg.init, "n_init": cfg.n_init, "max_iter": cfg.max_iter, "tol": cfg.tol, "seed": cfg.seed, "algorithm": "lloyd"},
                   "nRows": len(Xs), "inertia": float(km.inertia_), "nIter": int(km.n_iter_), "converged": bool(km.n_iter_ < cfg.max_iter), "selectedInit": int(r["initIndex"]),
                   "clusters": clusters, "iterations": r["iterations"], "emptyClusters": [c["cluster"] for c in clusters if c["size"] == 0],
                   "iterationNote": "Objective progression is a REPLAY of the selected initialisation with max_iter = 1, 2, ... using the same scikit-learn Lloyd implementation (scikit-learn does not expose per-iteration state).",
                   "emptyClusterNote": "scikit-learn relocates an empty cluster to a far point instead of leaving it empty; any cluster with size 0 would be listed in emptyClusters.",
                   "objective": "inertia = sum_i || x_i - mu_{c(i)} ||^2 (in scaled feature units)", "warnings": [str(x.message) for x in w], "canPredict": True,
                   "assignmentColumns": ["cluster", "distance_to_centroid", "distance_to_second_centroid", "silhouette"], "fittedOn": _fitted_on(t, ctx.node_id),
                   "limits": "Cluster numbers are arbitrary labels. k-means finds a local optimum of inertia for convex, similarly scaled groups; it returns k clusters whether or not the data has them."}
        df = _with_columns(t, {"cluster": lab.astype("int64"), "distance_to_centroid": g["distance"], "distance_to_second_centroid": g["second"], "silhouette": sil})
        model = UnsupModel("kmeans", km, sc, feats, _fitted_on(t, ctx.node_id), clean(details), {"labels": lab})
        return {"model": model, "assignments": t.derive(df)}, clean(details)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "minimize  J = sum_i || x_i - mu_{c(i)} ||^2 ;  repeat: c(i) = argmin_k || x_i - mu_k ||,  mu_k = mean of its members",
                "rule": f"sklearn.cluster.KMeans (Lloyd), k={cfg.n_clusters}, {cfg.init} initialisation, {cfg.n_init} initialisations (lowest inertia wins), seed {cfg.seed}.",
                "note": "Inertia always decreases as k grows; compare k with the diagnostics node, not with inertia alone."}


# ================================================================================================ Gaussian mixture
class GmmConfig(UnsupConfig):
    n_components: int = Field(3, ge=1, le=100)
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"
    n_init: int = Field(3, ge=1, le=50)
    max_iter: int = Field(200, ge=1)
    seed: int = 0


@register
class GmmOp(_UnsupOp):
    type = "sklearn.gaussian_mixture"
    outputs = ("model", "assignments")
    out_kinds = {"model": UNSUP_MODEL, "assignments": TABLE}
    Config = GmmConfig
    method = "gmm"

    def infer(self, cfg, ins, node_id):
        t = ins["table"]
        what = f"Gaussian mixture '{node_id}'"
        feats = _features(t, cfg, what)
        _rows_ok(t, cfg.n_components, what, f"{cfg.n_components} components")
        cols = _assign_cols(t, feats, [("cluster", "int"), ("max_responsibility", "float"), ("log_density", "float")])
        return {"model": VType(UNSUP_MODEL, {"method": "gmm", "k": cfg.n_components, "features": feats, "canPredict": True}),
                "assignments": table_type(cols, t.info.get("rows"), t.partition, t.complete, t.info.get("pending"), **{k: v for k, v in t.info.items() if k in ("splitNode", "splitSeed")})}

    def execute(self, cfg, ins, ctx):
        t: Table = ins["table"]
        what = f"Gaussian mixture '{ctx.node_id}'"
        feats, X, Xs, sc = _xy(t, cfg, what)
        if len(Xs) < cfg.n_components:
            raise ExecutionError("E_UNSUP_TOO_FEW_ROWS", f"{what}: {cfg.n_components} components but only {len(Xs)} rows.")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                gm = M.fit_gmm(Xs, cfg.n_components, cfg.covariance_type, cfg.n_init, cfg.max_iter, cfg.seed)
            except ValueError as e:
                raise ExecutionError("E_UNSUP_FIT", f"{what}: {e}") from e
        resp = gm.predict_proba(Xs)
        lab = resp.argmax(axis=1)
        dens = gm.score_samples(Xs)
        means_o = sc.inverse_transform(gm.means_) if sc is not None else gm.means_
        sizes = np.bincount(lab, minlength=cfg.n_components)
        ll = float(gm.score(Xs) * len(Xs))
        details = {"method": "gmm", "estimator": "sklearn.mixture.GaussianMixture", "sklearn": sklearn.__version__, "features": feats, "scale": cfg.scale, "scaler": _scaler_json(sc, feats),
                   "params": {"n_components": cfg.n_components, "covariance_type": cfg.covariance_type, "n_init": cfg.n_init, "max_iter": cfg.max_iter, "seed": cfg.seed},
                   "nRows": len(Xs), "converged": bool(gm.converged_), "nIter": int(gm.n_iter_), "lowerBound": float(gm.lower_bound_), "logLikelihood": ll, "bic": float(gm.bic(Xs)), "aic": float(gm.aic(Xs)),
                   "nParameters": int(gm._n_parameters()),
                   "components": [{"component": c, "weight": float(gm.weights_[c]), "size": int(sizes[c]), "mean": dict(zip(feats, means_o[c].tolist())),
                                   "meanScaled": dict(zip(feats, gm.means_[c].tolist())),
                                   "covarianceDiagonalScaled": dict(zip(feats, (np.diag(gm.covariances_[c]) if cfg.covariance_type == "full" else np.diag(gm.covariances_) if cfg.covariance_type == "tied"
                                                                                 else (gm.covariances_[c] if cfg.covariance_type == "diag" else np.full(len(feats), gm.covariances_[c]))).tolist()))} for c in range(cfg.n_components)],
                   "assignmentRule": "cluster = argmax_k responsibility_k (a soft assignment made hard); max_responsibility shows how soft it was",
                   "densityNote": "log_density is the log of the fitted mixture density at the point: a density SCORE (low = unusual under this model), not a cluster membership and not a calibrated anomaly probability.",
                   "warnings": [str(x.message) for x in w], "canPredict": True, "assignmentColumns": ["cluster", "max_responsibility", "log_density"], "fittedOn": _fitted_on(t, ctx.node_id),
                   "limits": "BIC / AIC compare models of different sizes on the same data under the Gaussian assumption; they do not say the data are Gaussian."}
        df = _with_columns(t, {"cluster": lab.astype("int64"), "max_responsibility": resp.max(axis=1), "log_density": dens})
        model = UnsupModel("gmm", gm, sc, feats, _fitted_on(t, ctx.node_id), clean(details), {"labels": lab})
        return {"model": model, "assignments": t.derive(df)}, clean(details)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "p(x) = sum_k pi_k N(x | mu_k, Sigma_k);  responsibility r_ik = pi_k N(x_i | mu_k, Sigma_k) / p(x_i);  EM maximizes sum_i log p(x_i)",
                "rule": f"sklearn.mixture.GaussianMixture, {cfg.n_components} components, covariance '{cfg.covariance_type}', {cfg.n_init} initialisations, seed {cfg.seed}.",
                "note": "Membership (cluster, max_responsibility) and the density score (log_density) are separate columns."}


# ================================================================================================ DBSCAN
class DbscanConfig(UnsupConfig):
    eps: float = Field(0.5, gt=0, title="eps (neighbourhood radius, in scaled feature units)")
    min_samples: int = Field(5, ge=1, title="min_samples (neighbours, including the point itself, that make a core point)")


@register
class DbscanOp(_UnsupOp):
    type = "sklearn.dbscan"
    outputs = ("model", "assignments")
    out_kinds = {"model": UNSUP_MODEL, "assignments": TABLE}
    Config = DbscanConfig
    method = "dbscan"

    def infer(self, cfg, ins, node_id):
        t = ins["table"]
        feats = _features(t, cfg, f"DBSCAN '{node_id}'")
        cols = _assign_cols(t, feats, [("cluster", "int"), ("point_status", "string"), ("neighbors_within_eps", "int")])
        return {"model": VType(UNSUP_MODEL, {"method": "dbscan", "features": feats, "canPredict": False}),
                "assignments": table_type(cols, t.info.get("rows"), t.partition, t.complete, t.info.get("pending"), **{k: v for k, v in t.info.items() if k in ("splitNode", "splitSeed")})}

    def execute(self, cfg, ins, ctx):
        t: Table = ins["table"]
        what = f"DBSCAN '{ctx.node_id}'"
        feats, X, Xs, sc = _xy(t, cfg, what)
        r = M.fit_dbscan(Xs, cfg.eps, cfg.min_samples)
        db = r["estimator"]
        lab = db.labels_
        n_cl = len(set(lab.tolist()) - {-1})
        sizes = {int(c): int((lab == c).sum()) for c in sorted(set(lab.tolist()) - {-1})}
        st = r["status"]
        details = {"method": "dbscan", "estimator": "sklearn.cluster.DBSCAN", "sklearn": sklearn.__version__, "features": feats, "scale": cfg.scale, "scaler": _scaler_json(sc, feats),
                   "params": {"eps": cfg.eps, "min_samples": cfg.min_samples, "metric": "euclidean"}, "nRows": len(Xs), "nClusters": n_cl, "nNoise": int((lab == -1).sum()),
                   "noiseFraction": float((lab == -1).mean()), "nCore": int((st == "core").sum()), "nBorder": int((st == "border").sum()), "clusterSizes": sizes,
                   "kDistance": {"k": cfg.min_samples, "sortedDescending": M.k_distance(Xs, cfg.min_samples)[:400], "note": "distance to the min_samples-th nearest neighbour (self included), sorted; an elbow suggests an eps"},
                   "statusRule": "core: at least min_samples points (itself included) within eps; border: not core but within eps of a core point (takes that core point's cluster); noise: neither (label -1)",
                   "connectivity": "A cluster is a maximal set of density-connected points: core points within eps of each other chain together; border points attach to one adjacent cluster (the first reached, so border assignment can depend on row order).",
                   "noiseNote": "Noise (-1) is not a cluster. Metrics computed on clusters exclude it and say so.", "canPredict": False,
                   "predictNote": "scikit-learn's DBSCAN has no predict: new points cannot be assigned from this fit.", "assignmentColumns": ["cluster", "point_status", "neighbors_within_eps"],
                   "fittedOn": _fitted_on(t, ctx.node_id), "limits": "A single eps / min_samples describes one density level; clusters of very different density are split or merged."}
        df = _with_columns(t, {"cluster": lab.astype("int64"), "point_status": st, "neighbors_within_eps": r["neighbors"].astype("int64")})
        model = UnsupModel("dbscan", db, sc, feats, _fitted_on(t, ctx.node_id), clean(details), {"labels": lab})
        return {"model": model, "assignments": t.derive(df)}, clean(details)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "core(x): |N_eps(x)| >= min_samples;  clusters = connected components of core points under eps-reachability, plus their border points",
                "rule": f"sklearn.cluster.DBSCAN, eps={cfg.eps} (scaled units), min_samples={cfg.min_samples}.", "note": "Deterministic up to border-point order; there is no seed."}


# ================================================================================================ PCA
class PcaConfig(UnsupConfig):
    n_components: int | None = Field(None, ge=1, title="components kept (empty = all)")
    whiten: bool = False


@register
class PcaOp(_UnsupOp):
    type = "sklearn.pca"
    outputs = ("model", "components")
    out_kinds = {"model": UNSUP_MODEL, "components": TABLE}
    Config = PcaConfig
    summary_kind = "pca"
    method = "pca"

    def infer(self, cfg, ins, node_id):
        t = ins["table"]
        what = f"PCA '{node_id}'"
        feats = _features(t, cfg, what)
        if feats is not None and cfg.n_components is not None and cfg.n_components > len(feats):
            raise OpError("E_PCA_COMPONENTS", f"{what}: n_components={cfg.n_components} exceeds the {len(feats)} features.", "table", [Fix(f"Set n_components to {len(feats)} or fewer", node_id, "n_components", len(feats))])
        _rows_ok(t, 2, what, "PCA")
        k = cfg.n_components or (len(feats) if feats else None)
        pcs = [{"name": f"PC{i + 1}", "dtype": "float"} for i in range(k)] if k else []
        others = [c for c in t.columns if c["name"] not in (feats or [])]
        return {"model": VType(UNSUP_MODEL, {"method": "pca", "k": k, "features": feats, "canPredict": True}),
                "components": table_type(pcs + others, t.info.get("rows"), t.partition, t.complete and k is not None, t.info.get("pending"), **{k_: v for k_, v in t.info.items() if k_ in ("splitNode", "splitSeed")})}

    def execute(self, cfg, ins, ctx):
        t: Table = ins["table"]
        what = f"PCA '{ctx.node_id}'"
        feats, X, Xs, sc = _xy(t, cfg, what)
        n, p = Xs.shape
        if cfg.n_components is not None and cfg.n_components > min(n, p):
            raise ExecutionError("E_PCA_COMPONENTS", f"{what}: n_components={cfg.n_components} exceeds min(rows={n}, features={p}).")
        pca = M.fit_pca(Xs, cfg.n_components, cfg.whiten)
        Z = pca.transform(Xs)
        k = pca.n_components_
        full = M.fit_pca(Xs, None, False)
        total_sq = float((Xs - Xs.mean(axis=0)).__pow__(2).sum())
        mse_by_k = [{"components": j, "mse": float(max(0.0, total_sq - float((full.singular_values_[:j] ** 2).sum())) / (n * p))} for j in range(0, full.n_components_ + 1)]
        recon = pca.inverse_transform(Z) if not cfg.whiten else None
        rec_mse = float(((Xs - recon) ** 2).mean()) if recon is not None else None
        ratio = full.explained_variance_ratio_
        details = {"method": "pca", "estimator": "sklearn.decomposition.PCA", "sklearn": sklearn.__version__, "features": feats, "scale": cfg.scale, "scaler": _scaler_json(sc, feats),
                   "params": {"n_components": cfg.n_components, "whiten": cfg.whiten, "svd_solver": "full"}, "nRows": n, "nComponentsKept": int(k),
                   "mean": dict(zip(feats, pca.mean_.tolist())), "explainedVariance": pca.explained_variance_.tolist(), "explainedVarianceRatio": pca.explained_variance_ratio_.tolist(),
                   "cumulativeRatio": np.cumsum(pca.explained_variance_ratio_).tolist(), "singularValues": pca.singular_values_.tolist(),
                   "scree": {"ratio": ratio.tolist(), "cumulative": np.cumsum(ratio).tolist()},
                   "componentsNeeded": {str(q): int(np.searchsorted(np.cumsum(ratio), q - 1e-12) + 1) for q in (0.8, 0.9, 0.95, 0.99)},
                   "loadings": [{"pc": f"PC{i + 1}", "ratio": float(pca.explained_variance_ratio_[i]), "weights": dict(zip(feats, pca.components_[i].tolist()))} for i in range(k)],
                   "reconstruction": {"mseKept": rec_mse, "byComponents": mse_by_k, "unit": "mean squared error per entry, in scaled feature units",
                                      "note": "reconstruction = PC scores x components + mean; with whitening the reconstruction is not computed here" if cfg.whiten else
                                      "reconstruction = PC scores x components + mean (the exact PCA inverse transform)"},
                   "centeringNote": "Features are centered (and standardized if scale='standardize') on these rows before the decomposition; PCA directions depend on the scaling choice.",
                   "canPredict": True, "canTransformNewPoints": True, "assignmentColumns": [f"PC{i + 1}" for i in range(k)], "fittedOn": _fitted_on(t, ctx.node_id),
                   "limits": "Components are directions of maximal linear variance, not necessarily directions that separate groups."}
        others = [c for c in t.df.columns if c not in feats]
        out = pd.DataFrame(Z, index=t.df.index, columns=[f"PC{i + 1}" for i in range(k)])
        for c in others:
            out[c] = t.df[c]
        model = UnsupModel("pca", pca, sc, feats, _fitted_on(t, ctx.node_id), clean(details), {"labels": None})
        return {"model": model, "components": t.derive(out)}, clean(details)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "X_c = X - mean;  X_c = U S V^T;  scores Z = X_c V_k;  reconstruction X_hat = Z V_k^T + mean;  explained variance ratio_j = s_j^2 / sum s^2",
                "rule": f"sklearn.decomposition.PCA(svd_solver='full'), keeping {cfg.n_components or 'all'} components, {'standardized' if cfg.scale == 'standardize' else 'unscaled'} features.",
                "note": "Centered features, component directions, explained variance, projection and reconstruction are all recorded."}


# ================================================================================================ projection (t-SNE)
class ProjectionConfig(UnsupConfig):
    perplexity: float = Field(30.0, gt=0)
    max_iter: int = Field(1000, ge=250)
    seed: int = 0


@register
class ProjectionOp(_UnsupOp):
    type = "sklearn.projection"
    outputs = ("model", "embedding")
    out_kinds = {"model": UNSUP_MODEL, "embedding": TABLE}
    Config = ProjectionConfig
    summary_kind = "projection"
    method = "tsne"

    def infer(self, cfg, ins, node_id):
        t = ins["table"]
        what = f"t-SNE projection '{node_id}'"
        feats = _features(t, cfg, what)
        n = t.info.get("rows")
        if isinstance(n, int) and cfg.perplexity >= n:
            raise OpError("E_PROJECTION_PERPLEXITY", f"{what}: perplexity={cfg.perplexity} must be smaller than the number of rows ({n}).", "table", [Fix(f"Use a perplexity below {n}", node_id, "perplexity", max(2.0, n / 4))])
        others = [c for c in t.columns if c["name"] not in (feats or [])]
        return {"model": VType(UNSUP_MODEL, {"method": "tsne", "features": feats, "canPredict": False}),
                "embedding": table_type([{"name": "x", "dtype": "float"}, {"name": "y", "dtype": "float"}] + others, n, t.partition, t.complete, t.info.get("pending"))}

    def warnings(self, cfg, ins, node_id):
        return super().warnings(cfg, ins, node_id) + [("W_PROJECTION_NON_METRIC", "t-SNE is a non-metric projection for viewing only: distances, cluster sizes and gaps between groups in the embedding are not evidence of structure in the original space.", None)]

    def execute(self, cfg, ins, ctx):
        t: Table = ins["table"]
        what = f"t-SNE projection '{ctx.node_id}'"
        feats, X, Xs, sc = _xy(t, cfg, what)
        if cfg.perplexity >= len(Xs):
            raise ExecutionError("E_PROJECTION_PERPLEXITY", f"{what}: perplexity={cfg.perplexity} must be smaller than the number of rows ({len(Xs)}).")
        ts = M.fit_tsne(Xs, cfg.perplexity, cfg.seed, cfg.max_iter)
        emb = ts.embedding_
        tw = M.trust(Xs, emb, 5)
        details = {"method": "tsne", "estimator": "sklearn.manifold.TSNE", "sklearn": sklearn.__version__, "features": feats, "scale": cfg.scale, "scaler": _scaler_json(sc, feats),
                   "params": {"n_components": 2, "perplexity": cfg.perplexity, "init": "pca", "learning_rate": "auto", "max_iter": cfg.max_iter, "seed": cfg.seed}, "nRows": len(Xs),
                   "klDivergence": float(ts.kl_divergence_), "nIter": int(ts.n_iter_), "trustworthiness5": tw,
                   "trustworthinessNote": "trustworthiness@5 in [0,1]: how many of each point's 5 nearest neighbours in the embedding were also close in the original space. High values say local neighbourhoods are kept; they say nothing about global distances or clusters.",
                   "projectionStatus": "NON-METRIC PROJECTION (for viewing only)", "canTransformNewPoints": False, "canPredict": False,
                   "transformNote": "scikit-learn's TSNE has no transform: points that were not in this fit cannot be placed in this embedding without refitting.",
                   "limits": ["Distances between far-apart groups, relative cluster sizes and densities in the plot are not meaningful.",
                              "Apparent visual separation is not proof that clusters exist in the original space; the embedding changes with perplexity and seed.",
                              "Fitted on exactly these rows; nothing here is an accuracy."],
                   "assignmentColumns": ["x", "y"], "fittedOn": _fitted_on(t, ctx.node_id)}
        others = [c for c in t.df.columns if c not in feats]
        out = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1]}, index=t.df.index)
        for c in others:
            out[c] = t.df[c]
        model = UnsupModel("tsne", ts, sc, feats, _fitted_on(t, ctx.node_id), clean(details), {"labels": None})
        return {"model": model, "embedding": t.derive(out)}, clean(details)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "minimize KL(P || Q) over 2-D points y_i, with P from high-dimensional neighbour probabilities (perplexity) and Q a Student-t kernel in the plane",
                "rule": f"sklearn.manifold.TSNE, perplexity {cfg.perplexity}, PCA initialisation, seed {cfg.seed}. NON-METRIC projection: for viewing, not measuring.", "note": "Cannot transform new points."}


# ================================================================================================ diagnostics
class DiagConfig(StrictConfig):
    labels_column: str = Field("", title="label column (optional; enables EXTERNAL metrics)")
    k_min: int = Field(2, ge=1, le=50, title="sweep: smallest k")
    k_max: int = Field(8, ge=1, le=50, title="sweep: largest k")
    stability_runs: int = Field(8, ge=0, le=30, title="resampling runs for stability (0 = off)")
    resample: Literal["subsample", "bootstrap"] = Field("subsample", title="resampling scheme")
    subsample_fraction: float = Field(0.8, gt=0.1, lt=1.0)
    seed: int = 0


def _refit(method: str, params: dict[str, Any], Xs: np.ndarray, seed: int):
    if method == "kmeans":
        return M.fit_kmeans(Xs, params["n_clusters"], params["init"], params["n_init"], params["max_iter"], params["tol"], seed)["estimator"]
    if method == "gmm":
        return M.fit_gmm(Xs, params["n_components"], params["covariance_type"], params["n_init"], params["max_iter"], seed)
    if method == "dbscan":
        return M.fit_dbscan(Xs, params["eps"], params["min_samples"])["estimator"]
    raise ValueError(method)


@register
class ClusterDiagnosticsOp(TabularOperation):
    type = "sklearn.cluster_diagnostics"
    backend = "scikit-learn"
    inputs = ("model", "table")
    outputs = ("report",)
    in_kinds = {"model": UNSUP_MODEL, "table": TABLE}
    out_kinds = {"report": CLUSTER_REPORT}
    Config = DiagConfig
    summary_kind = "cluster_report"
    METRICS = {"kmeans": ["silhouette", "daviesBouldin", "calinskiHarabasz", "inertia", "stabilityAriMean", "stabilityAriMin"],
               "gmm": ["silhouette", "daviesBouldin", "calinskiHarabasz", "logLikelihood", "bic", "aic", "stabilityAriMean", "stabilityAriMin"],
               "dbscan": ["silhouette", "daviesBouldin", "calinskiHarabasz", "noiseFraction", "nClusters", "stabilityAriMean", "stabilityAriMin"],
               "pca": ["explainedVarianceRatioKept", "reconstructionMse", "componentsFor90", "stabilityCosMean", "stabilityCosMin"], "tsne": ["trustworthiness5", "klDivergence"]}
    EXTERNAL = ["externalAdjustedRand", "externalNormalizedMutualInfo", "externalHomogeneity", "externalCompleteness", "externalVMeasure"]

    def infer(self, cfg, ins, node_id):
        m, t = ins["model"], ins["table"]
        if cfg.k_max < cfg.k_min:
            raise OpError("E_DIAG_K_RANGE", f"Diagnostics '{node_id}': k_max ({cfg.k_max}) is below k_min ({cfg.k_min}).", None)
        if cfg.labels_column:
            need_columns(t, [cfg.labels_column], "table", f"Diagnostics '{node_id}'")
            if cfg.labels_column in (m.info.get("features") or []):
                raise OpError("E_LABEL_IN_FEATURES", f"The label column '{cfg.labels_column}' is also a feature of the clustering, so agreement with it would be circular.", "table",
                              [Fix("Exclude the label column in the clustering node's 'exclude' list")])
        feats = m.info.get("features")
        if feats:
            need_columns(t, feats, "table", f"Diagnostics '{node_id}'")
        names = list(self.METRICS[m.info["method"]]) + (self.EXTERNAL if cfg.labels_column and m.info["method"] in ("kmeans", "gmm", "dbscan") else [])
        return {"report": VType(CLUSTER_REPORT, {"method": m.info["method"], "metrics": names, "evaluatedPartition": t.partition})}

    def warnings(self, cfg, ins, node_id):
        out = []
        if ins["model"].info["method"] == "tsne":
            out.append(("W_PROJECTION_NON_METRIC", "A t-SNE projection has no cluster-quality metric: only neighbour preservation (trustworthiness) is reported, and it is not a measure of cluster validity.", "model"))
        if cfg.stability_runs == 0:
            out.append(("W_NO_STABILITY", "Stability is off: one fit says nothing about how sensitive the result is to seeds or resampling.", None))
        return out

    def execute(self, cfg, ins, ctx):
        m: UnsupModel = ins["model"]
        t: Table = ins["table"]
        what = f"Diagnostics '{ctx.node_id}'"
        df = t.df
        missing = [f for f in m.features if f not in df.columns]
        if missing:
            raise ExecutionError("E_MODEL_FEATURES_MISSING", f"{what}: the model was fitted on {missing}, which the table lacks.")
        X = df[m.features].to_numpy("float64")
        if np.isnan(X).any():
            raise ExecutionError("E_MISSING_VALUES", f"{what}: missing values in the feature columns.")
        Xs = m.scaler.transform(X) if m.scaler is not None else X
        same = row_ids_sha256(df.index) == m.fitted_on["rowIdsSha256"]
        method, params = m.method, m.details.get("params", {})
        rep: dict[str, Any] = {"method": method, "policy": POLICY, "nRows": len(Xs), "sklearn": sklearn.__version__, "features": m.features,
                               "evaluatedOn": {"partition": t.partition, "sameRowsAsFit": same, "rowIdsSha256": row_ids_sha256(df.index)}, "warnings": []}
        values: dict[str, float] = {}
        unavailable: dict[str, str] = {}
        if method == "pca":
            self._pca(cfg, m, Xs, rep, values)
        elif method == "tsne":
            emb = m.estimator.embedding_
            if not same:
                raise ExecutionError("E_UNSUP_NO_TRANSFORM", f"{what}: a t-SNE embedding cannot place new points; give the diagnostics the table it was fitted on.")
            values["trustworthiness5"] = M.trust(Xs, emb, 5)
            values["klDivergence"] = float(m.estimator.kl_divergence_)
            rep["projection"] = {"status": "NON-METRIC PROJECTION", "trustworthiness5": values["trustworthiness5"], "klDivergence": values["klDivergence"],
                                 "statement": "No cluster-quality, accuracy or stability metric is reported for a projection. Visual separation in it is not evidence of clusters."}
        else:
            if method == "dbscan" and not same:
                raise ExecutionError("E_UNSUP_NO_PREDICT", f"{what}: DBSCAN has no predict; give the diagnostics the table the clustering was fitted on.")
            lab = m.extra["labels"] if same else m.estimator.predict(Xs)
            rep["clusterSizes"] = {str(int(c)): int((lab == c).sum()) for c in sorted(set(lab.tolist()))}
            ig = method == "dbscan"
            im = M.internal_metrics(Xs, lab, ignore_noise=ig)
            rep["internal"] = {"kind": "internal", **im}
            for src, dst in (("silhouette", "silhouette"), ("daviesBouldin", "daviesBouldin"), ("calinskiHarabasz", "calinskiHarabasz")):
                e = im[src]
                if e["applicable"]:
                    values[dst] = e["value"]
                else:
                    unavailable[dst] = e["reason"]
            if method == "kmeans":
                values["inertia"] = float(m.details["inertia"] if same else ((Xs - m.estimator.cluster_centers_[lab]) ** 2).sum())
            if method == "gmm":
                values["logLikelihood"] = float(m.estimator.score(Xs) * len(Xs))
                values["bic"], values["aic"] = float(m.estimator.bic(Xs)), float(m.estimator.aic(Xs))
            if method == "dbscan":
                values["noiseFraction"] = float((lab == -1).mean())
                values["nClusters"] = float(len(set(lab.tolist()) - {-1}))
                rep["kDistance"] = m.details.get("kDistance")
            if method in ("kmeans", "gmm"):
                rep["sweep"] = self._sweep(cfg, method, params, Xs, rep)
            if cfg.stability_runs:
                self._stability(cfg, method, params, Xs, lab, rep, values, unavailable)
            if cfg.labels_column:
                if cfg.labels_column not in df.columns:
                    raise ExecutionError("E_COLUMN_NOT_FOUND", f"{what}: label column '{cfg.labels_column}' not found.")
                truth = df[cfg.labels_column].astype(str).to_numpy()
                ext = M.external_metrics(lab, truth)
                ext["labelsColumn"] = cfg.labels_column
                if ig:
                    ext["noteNoise"] = "DBSCAN noise (-1) is treated as one extra label here."
                rep["external"] = ext
                for k_, name in (("adjustedRand", "externalAdjustedRand"), ("normalizedMutualInfo", "externalNormalizedMutualInfo"), ("homogeneity", "externalHomogeneity"),
                                 ("completeness", "externalCompleteness"), ("vMeasure", "externalVMeasure")):
                    values[name] = ext[k_]
            else:
                rep["external"] = {"available": False, "reason": "No label column declared: no external metric is computed. Internal metrics above are the only evidence."}
        rep.update({"values": values, "unavailable": unavailable, "n": len(Xs), "evaluatedPartition": t.partition, "evaluatedRowIdsSha256": row_ids_sha256(df.index),
                    "valueKinds": {k: ("external" if k.startswith("external") else "internal" if k in ("silhouette", "daviesBouldin", "calinskiHarabasz") else "model-based" if k in ("bic", "aic", "logLikelihood", "inertia") else "stability" if k.startswith("stability") else "descriptive") for k in values}})
        rep = clean(rep)
        return {"report": Plain(CLUSTER_REPORT, rep)}, rep

    # ------------------------------------------------------------------------------------------ pieces
    def _sweep(self, cfg, method, params, Xs, rep):
        out = []
        n = len(Xs)
        ks = [k for k in range(max(2, cfg.k_min), cfg.k_max + 1) if k < n][:12]
        for k in ks:
            if method == "kmeans":
                est = M.fit_kmeans(Xs, k, params["init"], params["n_init"], params["max_iter"], params["tol"], params["seed"])["estimator"]
                lab = est.labels_
                row = {"k": k, "inertia": float(est.inertia_)}
            else:
                est = M.fit_gmm(Xs, k, params["covariance_type"], params["n_init"], params["max_iter"], params["seed"])
                lab = est.predict(Xs)
                row = {"k": k, "bic": float(est.bic(Xs)), "aic": float(est.aic(Xs)), "logLikelihood": float(est.score(Xs) * n)}
            im = M.internal_metrics(Xs, lab, False)
            row["silhouette"] = im["silhouette"].get("value")
            row["daviesBouldin"] = im["daviesBouldin"].get("value")
            out.append(row)
        return {"rows": out, "note": ("Elbow: inertia always falls as k grows, so look for diminishing returns, not a minimum. " if method == "kmeans" else
                                       "Lower BIC / AIC is better; BIC penalizes parameters more strongly and is not guaranteed to pick the 'true' k. ") +
                "Each k is refit with the node's own initialisation settings and seed.", "k": ks}

    def _stability(self, cfg, method, params, Xs, ref, rep, values, unavailable):
        rng = np.random.default_rng(cfg.seed)
        n = len(Xs)
        runs, seeds_info = [], None
        if method in ("kmeans", "gmm"):
            labs = []
            for j in range(min(cfg.stability_runs, 10)):
                est = _refit(method, params, Xs, params.get("seed", 0) + 1000 + j)
                labs.append(est.labels_ if method == "kmeans" else est.predict(Xs))
            vs_ref = [float(skm.adjusted_rand_score(ref, l)) for l in labs]
            pair = [float(skm.adjusted_rand_score(labs[a], labs[b])) for a in range(len(labs)) for b in range(a + 1, len(labs))]
            seeds_info = {"runs": len(labs), "versusReference": vs_ref, "pairwise": {"mean": float(np.mean(pair)) if pair else None, "min": float(np.min(pair)) if pair else None, "max": float(np.max(pair)) if pair else None},
                          "statement": "Adjusted Rand index between the reference partition and refits that differ ONLY in the random seed (same data, same settings). 1 = identical partitions, ~0 = chance agreement."}
        else:
            seeds_info = {"applicable": False, "reason": "DBSCAN has no random seed (deterministic up to border-point order); only resampling stability applies."}
        aris = []
        for j in range(cfg.stability_runs):
            if cfg.resample == "bootstrap" and method in ("kmeans", "gmm"):
                idx = rng.choice(n, size=n, replace=True)
                est = _refit(method, params, Xs[idx], params.get("seed", 0) + 2000 + j)
                lab_full = est.predict(Xs)
                aris.append(float(skm.adjusted_rand_score(ref, lab_full)))
            else:
                idx = np.sort(rng.choice(n, size=max(2, int(round(cfg.subsample_fraction * n))), replace=False))
                est = _refit(method, params, Xs[idx], params.get("seed", 0) + 2000 + j)
                lab_sub = est.labels_ if method in ("kmeans", "dbscan") else est.predict(Xs[idx])
                aris.append(float(skm.adjusted_rand_score(ref[idx], lab_sub)))
        scheme = ("bootstrap rows (with replacement); the refit predicts every original row" if cfg.resample == "bootstrap" and method in ("kmeans", "gmm") else
                  f"subsample {int(cfg.subsample_fraction * 100)}% of rows without replacement; compared on the subsampled rows only" + ("" if method != "dbscan" else " (DBSCAN cannot predict, so bootstrap is replaced by subsampling)"))
        rs = {"scheme": scheme, "runs": cfg.stability_runs, "ari": aris, "mean": float(np.mean(aris)), "min": float(np.min(aris)), "max": float(np.max(aris)),
              "statement": "Adjusted Rand index between the reference partition and the partition refit on a resample. Low values mean the partition depends on which rows were seen."}
        rep["stability"] = {"seeds": seeds_info, "resampling": rs, "caveat": "Stable does not mean correct: a poor partition can be reproducible. Unstable means a conclusion drawn from one fit is fragile."}
        values["stabilityAriMean"], values["stabilityAriMin"] = rs["mean"], rs["min"]

    def _pca(self, cfg, m: UnsupModel, Xs, rep, values):
        pca = m.estimator
        d = m.details
        full_ratio = d["scree"]["ratio"]
        rep["explainedVariance"] = {"ratio": d["explainedVarianceRatio"], "cumulative": d["cumulativeRatio"], "scree": d["scree"], "componentsNeeded": d["componentsNeeded"]}
        rep["reconstruction"] = d["reconstruction"]
        values["explainedVarianceRatioKept"] = float(sum(d["explainedVarianceRatio"]))
        values["componentsFor90"] = float(d["componentsNeeded"]["0.9"])
        if d["reconstruction"]["mseKept"] is not None:
            values["reconstructionMse"] = d["reconstruction"]["mseKept"]
        if cfg.stability_runs:
            rng = np.random.default_rng(cfg.seed)
            n = len(Xs)
            k = min(3, pca.n_components_)
            coss = [[] for _ in range(k)]
            for _ in range(cfg.stability_runs):
                idx = rng.choice(n, size=n, replace=True)
                b = M.fit_pca(Xs[idx], k, False)
                for i in range(k):
                    coss[i].append(M.subspace_cos(pca.components_[i], b.components_[i]))
            rep["stability"] = {"scheme": "bootstrap rows (with replacement)", "runs": cfg.stability_runs,
                                "perComponent": [{"pc": f"PC{i + 1}", "meanAbsCosine": float(np.mean(c)), "minAbsCosine": float(np.min(c))} for i, c in enumerate(coss)],
                                "statement": "|cosine| between each reference component and the bootstrap component with the same index (sign is arbitrary). Near 1 = a stable direction; similar eigenvalues make components swap or rotate."}
            values["stabilityCosMean"] = float(np.mean([np.mean(c) for c in coss]))
            values["stabilityCosMin"] = float(min(np.min(c) for c in coss))
        rep["external"] = {"available": False, "reason": "PCA is a representation, not a partition: external cluster metrics do not apply."}

    def explain(self, cfg, inputs, outputs):
        return {"equation": "internal: silhouette s = (b - a) / max(a, b), Davies-Bouldin, Calinski-Harabasz;  k-means: inertia;  GMM: BIC = p ln n - 2 ln L, AIC = 2p - 2 ln L;  PCA: explained variance ratio",
                "rule": "Which metrics appear depends on the fitted method. Stability: adjusted Rand index between the reference partition and seed refits / resamples. External metrics (ARI, NMI, homogeneity, completeness, V-measure) only when a label column is declared.",
                "note": POLICY}
