"""Observed request window, input/prediction drift and separately labelled quality."""
from __future__ import annotations

import io
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from tabular.core import clean


def distribution_compare(reference, current):
    a, b = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return {"available": False, "reason": "No finite observations in one window."}
    ks = ks_2samp(a, b)
    return {"available": True, "referenceN": len(a), "currentN": len(b), "referenceMean": float(a.mean()), "currentMean": float(b.mean()),
            "meanChange": float(b.mean()-a.mean()), "ksStatistic": float(ks.statistic), "ksPValue": float(ks.pvalue),
            "method": "scipy.stats.ks_2samp; descriptive evidence, no automatic release/retraining decision"}


def categorical_compare(reference, current):
    a, b = pd.Series(reference).astype(str), pd.Series(current).astype(str)
    keys = sorted(set(a) | set(b))
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    return {"available": bool(len(a) and len(b)), "referenceN": len(a), "currentN": len(b),
            "totalVariation": float(sum(abs(pa.get(k, 0)-pb.get(k, 0)) for k in keys)/2),
            "categories": [{"value": k, "referenceFraction": float(pa.get(k, 0)), "currentFraction": float(pb.get(k, 0))} for k in keys[:100]],
            "categoriesTruncated": len(keys)>100, "method": "total variation = 0.5 * sum absolute category-frequency differences"}


def monitoring(runtime, release_id, since=0):
    ps, store = runtime.ps, runtime.store
    release = ps.get("release", release_id)
    pipeline = runtime.pipeline(release["versionId"])
    m = pipeline.manifest
    traces = [t for t in ps.traces(release_id) if t["receivedAt"] >= since]
    ok = [t for t in traces if t["status"] == 200]
    baseline = pd.read_csv(io.BytesIO(store.read_artifact(m["referenceSha256"])))
    records = [r for t in ok if t["records"] is not None for r in t["records"]]
    current = pd.DataFrame(records)
    input_drift = {}
    for c in m["inputSchema"]:
        name = c["name"]
        if name not in current:
            input_drift[name] = {"available": False, "reason": "No captured successful inputs; enable capture on a new release to measure input drift."}
            continue
        numeric = c["dtype"] in ("int", "float")
        cmp = distribution_compare(baseline[name], current[name]) if numeric else categorical_compare(baseline[name], current[name])
        input_drift[name] = {**cmp, "referenceMissingFraction": float(baseline[name].isna().mean()), "currentMissingFraction": float(current[name].isna().mean())}
    raw = baseline.astype(object).where(baseline.notna(), None).to_dict("records")
    base_preds = pipeline.predict(raw)[0]["predictions"]
    preds = [p for t in ok for p in t["result"]["predictions"]]
    classification = m["outputSchema"]["task"] == "classification"
    prediction_drift = categorical_compare(base_preds, preds) if classification else distribution_compare(base_preds, preds)
    labels = {(r["user"], r["request"]): r for r in ps.query("SELECT * FROM labels")}
    truth, labelled_preds, delays, ids = [], [], [], []
    for t in ok:
        row = labels.get((t["user"], t["requestId"]))
        if row:
            truth.extend(json.loads(row["labels"]))
            labelled_preds.extend(t["result"]["predictions"])
            delays.append(row["ts"]-t["receivedAt"])
            ids.append({"user": t["user"], "requestId": t["requestId"], "traceSha256": t["traceSha256"]})
    quality = {"available": bool(truth), "labelledRows": len(truth), "labelledRequests": len(ids), "evidence": ids,
               "reason": None if truth else "Ground-truth labels have not been recorded; quality is not measured.",
               "meanLabelDelaySeconds": float(np.mean(delays)) if delays else None}
    if truth:
        quality["values"] = {"accuracy": float(accuracy_score(truth, labelled_preds))} if classification else {
            "mse": float(mean_squared_error(truth, labelled_preds)), "mae": float(mean_absolute_error(truth, labelled_preds))}
    lat = [t["totalMs"] for t in traces if t["totalMs"] is not None]
    failures = [t for t in traces if t["status"] != 200]
    return clean({"releaseId": release_id, "versionId": release["versionId"], "pipelineSha256": release["pipelineSha256"], "mode": "observed local requests",
                  "window": {"since": since, "until": time.time(), "maxRequests": 1000, "recordedRequests": len(traces), "order": "newest 1000 requests for this release"},
                  "health": {"requests": len(traces), "errors": len(failures), "errorFraction": len(failures)/len(traces) if traces else None,
                             "p50Ms": float(np.percentile(lat, 50)) if lat else None, "p95Ms": float(np.percentile(lat, 95)) if lat else None,
                             "schemaErrors": sum(t.get("error", {}).get("code") == "E_REQUEST_SCHEMA" for t in failures)},
                  "inputDrift": input_drift, "predictionDrift": prediction_drift, "labelBasedQuality": quality,
                  "reference": {"runId": m["runId"], "partition": m["referencePartition"], "sha256": m["referenceSha256"], "rows": len(baseline), "sampling": m["referenceSampling"]},
                  "alertEvidence": [{"requestId": t["requestId"], "user": t["user"], "status": t["status"], "traceSha256": t["traceSha256"]} for t in failures[:20]],
                  "interpretation": "Input/prediction changes do not establish an accuracy drop. Quality uses only separately supplied labels; no automatic retraining or rollback."})
