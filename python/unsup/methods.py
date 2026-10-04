"""Pure functions that fit scikit-learn estimators and describe the result. No graph or storage knowledge."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics as skm
from sklearn.cluster import DBSCAN, KMeans, kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

MAX_SILHOUETTE = 4000  # silhouette_samples is O(n^2): above this many rows per-sample silhouette is not computed (the aggregate uses a seeded sample)


def initial_centers(X: np.ndarray, k: int, init: str, seed: int) -> np.ndarray:
    if init == "k-means++":
        return kmeans_plusplus(X, k, random_state=seed)[0]
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), size=k, replace=False)]


def fit_kmeans(X: np.ndarray, k: int, init: str, n_init: int, max_iter: int, tol: float, seed: int) -> dict[str, Any]:
    """n_init independent initialisations, each a native `KMeans(init=<centres>, n_init=1)` fit; the lowest inertia wins (what scikit-learn does internally).
    Then the winner's iterations are REPLAYED with max_iter = 1, 2, ... from the same initial centres (same Lloyd implementation) to expose the objective progression."""
    best = None
    for i in range(n_init):
        c0 = initial_centers(X, k, init, seed + i)
        km = KMeans(n_clusters=k, init=c0, n_init=1, max_iter=max_iter, tol=tol, algorithm="lloyd").fit(X)
        if best is None or km.inertia_ < best[1].inertia_:
            best = (i, km, c0)
    i, km, c0 = best
    iters, prev = [], None
    prev_labels = None
    for it in range(1, min(km.n_iter_, 30) + 1):
        r = KMeans(n_clusters=k, init=c0, n_init=1, max_iter=it, tol=0.0, algorithm="lloyd").fit(X)
        moved = None if prev_labels is None else int((r.labels_ != prev_labels).sum())
        shift = None if prev is None else float(np.sqrt(((r.cluster_centers_ - prev) ** 2).sum(axis=1)).max())
        iters.append({"iteration": it, "inertia": float(r.inertia_), "maxCentroidShift": shift, "reassigned": moved})
        prev, prev_labels = r.cluster_centers_, r.labels_
    return {"estimator": km, "initIndex": i, "initialCenters": c0, "iterations": iters, "initInertias": None}


def kmeans_geometry(km: KMeans, X: np.ndarray) -> dict[str, Any]:
    d = km.transform(X)   # distances to every centre
    lab = km.labels_
    own = d[np.arange(len(X)), lab]
    d2 = d.copy()
    d2[np.arange(len(X)), lab] = np.inf
    return {"labels": lab, "distance": own, "second": d2.min(axis=1), "contrib": [float((own[lab == c] ** 2).sum()) for c in range(km.n_clusters)]}


def fit_gmm(X: np.ndarray, k: int, covariance_type: str, n_init: int, max_iter: int, seed: int) -> GaussianMixture:
    return GaussianMixture(n_components=k, covariance_type=covariance_type, n_init=n_init, max_iter=max_iter, random_state=seed).fit(X)


def fit_dbscan(X: np.ndarray, eps: float, min_samples: int) -> dict[str, Any]:
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    nn = NearestNeighbors(radius=eps).fit(X)
    counts = np.array([len(ix) for ix in nn.radius_neighbors(X, return_distance=False)])   # includes the point itself, as DBSCAN counts it
    core = np.zeros(len(X), bool)
    core[db.core_sample_indices_] = True
    status = np.where(core, "core", np.where(db.labels_ >= 0, "border", "noise"))
    return {"estimator": db, "neighbors": counts, "status": status}


def fit_pca(X: np.ndarray, n_components: int | None, whiten: bool) -> PCA:
    return PCA(n_components=n_components, whiten=whiten, svd_solver="full").fit(X)


def fit_tsne(X: np.ndarray, perplexity: float, seed: int, max_iter: int) -> TSNE:
    t = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed, max_iter=max_iter)
    t.embedding_ = t.fit_transform(X)
    return t


def internal_metrics(X: np.ndarray, labels: np.ndarray, ignore_noise: bool) -> dict[str, Any]:
    """Silhouette, Davies-Bouldin and Calinski-Harabasz on a hard partition. Each is `applicable: False` with the reason when its precondition fails."""
    lab = np.asarray(labels)
    notes = []
    if ignore_noise:
        keep = lab >= 0
        X, lab = X[keep], lab[keep]
        notes.append(f"noise points (label -1) excluded: {int((~keep).sum())}")
    n_clusters = len(set(lab.tolist()))
    out: dict[str, Any] = {"nUsed": int(len(lab)), "nClusters": n_clusters, "notes": notes}
    ok = 2 <= n_clusters <= len(lab) - 1
    why = f"needs 2 <= clusters <= samples - 1 (have {n_clusters} clusters, {len(lab)} samples)"
    def entry(name, fn, better, assumption):
        if not ok:
            return {"applicable": False, "reason": why, "assumption": assumption, "better": better}
        return {"applicable": True, "value": float(fn()), "better": better, "assumption": assumption}
    sample = None if len(lab) <= MAX_SILHOUETTE else MAX_SILHOUETTE
    out["silhouette"] = entry("silhouette", lambda: skm.silhouette_score(X, lab, sample_size=sample, random_state=0), "higher (max 1)",
                              "mean over samples of (b - a) / max(a, b) with Euclidean distances: favours compact, convex, similarly sized clusters" + (f"; computed on a seeded sample of {MAX_SILHOUETTE} rows" if sample else ""))
    out["daviesBouldin"] = entry("db", lambda: skm.davies_bouldin_score(X, lab), "lower (min 0)", "average similarity of each cluster to its most similar one (scatter / centroid distance); favours compact, well-separated, roughly spherical clusters")
    out["calinskiHarabasz"] = entry("ch", lambda: skm.calinski_harabasz_score(X, lab), "higher", "ratio of between- to within-cluster dispersion; favours convex clusters and grows with k for many data sets")
    return out


def external_metrics(labels: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    return {"kind": "external", "statement": "These compare the cluster labels with labels that were NOT used for fitting. They measure agreement with that labelling, not the quality of the clusters.",
            "adjustedRand": float(skm.adjusted_rand_score(truth, labels)), "normalizedMutualInfo": float(skm.normalized_mutual_info_score(truth, labels)),
            "homogeneity": float(skm.homogeneity_score(truth, labels)), "completeness": float(skm.completeness_score(truth, labels)), "vMeasure": float(skm.v_measure_score(truth, labels)),
            "contingency": skm.cluster.contingency_matrix(truth, labels).tolist(), "truthClasses": [str(c) for c in np.unique(truth)], "clusterLabels": [int(c) for c in np.unique(labels)]}


def k_distance(X: np.ndarray, k: int) -> list[float]:
    d, _ = NearestNeighbors(n_neighbors=min(k, len(X))).fit(X).kneighbors(X)
    return np.sort(d[:, -1])[::-1].tolist()


def subspace_cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def trust(X: np.ndarray, emb: np.ndarray, k: int = 5) -> float:
    return float(trustworthiness(X, emb, n_neighbors=min(k, max(1, len(X) // 2 - 1))))
