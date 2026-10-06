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


def domain_monitoring(runtime, release, version, pipeline, traces):
    from .domain_adapter import input_features, prediction_values, quality

    fam, m = version["family"], pipeline.manifest
    ok = [t for t in traces if t["status"] == 200]
    ref_records = pipeline.reference_records()
    ref_preds = pipeline.predict(ref_records)[0]["predictions"]
    captured = [r for t in ok if t["records"] is not None for r in t["records"]]
    ref_feats = [input_features(fam, r) for r in ref_records]
    cur_feats = [input_features(fam, r) for r in captured]
    input_drift = {}
    for name in (ref_feats[0] if ref_feats else {}):
        if not cur_feats:
            input_drift[name] = {"available": False, "reason": "No captured successful inputs; enable capture on a new release to measure input drift."}
        else:
            input_drift[name] = {**distribution_compare([f[name] for f in ref_feats], [f[name] for f in cur_feats]),
                                 "caution": f"reference is the source run's recorded held-out example ({len(ref_feats)} record); a tiny reference gives weak evidence"}
    cur_vals = [v for t in ok for p in t["result"]["predictions"] for v in prediction_values(fam, p)]
    ref_vals = [v for p in ref_preds for v in prediction_values(fam, p)]
    prediction_drift = categorical_compare(ref_vals, cur_vals) if cur_vals else {"available": False, "reason": "No successful predictions in this window."}
    prediction_drift["unit"] = {"nlp": "predicted word labels", "speech": "decoded characters", "vision": "predicted mask pixel classes"}[fam]
    labels = {(r["user"], r["request"]): r for r in runtime.ps.query("SELECT * FROM labels")}
    pairs, delays, ids = [], [], []
    for t in ok:
        row = labels.get((t["user"], t["requestId"]))
        if row:
            pairs += list(zip(t["result"]["predictions"], json.loads(row["labels"])))
            delays.append(row["ts"] - t["receivedAt"])
            ids.append({"user": t["user"], "requestId": t["requestId"], "traceSha256": t["traceSha256"]})
    classes = m["outputSchema"]["classes"] or []
    q = {"available": bool(pairs), "labelledRecords": len(pairs), "labelledRequests": len(ids), "evidence": ids,
         "reason": None if pairs else "Ground-truth labels have not been recorded; quality is not measured.",
         "meanLabelDelaySeconds": float(np.mean(delays)) if delays else None}
    if pairs:
        q["values"] = quality(fam, pairs, len(classes))
    lat = [t["totalMs"] for t in traces if t["totalMs"] is not None]
    failures = [t for t in traces if t["status"] != 200]
    return clean({"releaseId": release["id"], "versionId": release["versionId"], "pipelineSha256": release["pipelineSha256"], "family": fam,
                  "mode": "observed local requests", "window": {"recordedRequests": len(traces), "maxRequests": 1000, "order": "newest 1000 requests for this release"},
                  "health": {"requests": len(traces), "errors": len(failures), "errorFraction": len(failures) / len(traces) if traces else None,
                             "p50Ms": float(np.percentile(lat, 50)) if lat else None, "p95Ms": float(np.percentile(lat, 95)) if lat else None,
                             "schemaErrors": sum((t.get("error") or {}).get("code") == "E_REQUEST_SCHEMA" for t in failures)},
                  "inputDrift": input_drift, "predictionDrift": prediction_drift, "labelBasedQuality": q,
                  "reference": {"runId": m["runId"], "partition": "recorded held-out example (SYNTHETIC fixture)", "sha256": m["exampleSha256"], "rows": len(ref_records)},
                  "alertEvidence": [{"requestId": t["requestId"], "user": t["user"], "status": t["status"], "traceSha256": t["traceSha256"]} for t in failures[:20]],
                  "interpretation": "Input descriptors and prediction frequencies are descriptive; they do not establish a quality change. Quality uses only separately supplied labels; no automatic retraining or rollback."})


def model_monitoring(runtime, release, version, pipeline, traces):
    """Image classifier: descriptive image statistics and predicted-class frequencies against the frozen held-out reference."""
    import base64

    from .domain_adapter import input_features

    m = pipeline.manifest
    ok = [t for t in traces if t["status"] == 200]
    ref_records = pipeline.reference_records()
    ref_result = pipeline.predict(ref_records)[0] if ref_records else {"predictions": []}
    captured = [r for t in ok if t["records"] is not None for r in t["records"]]
    feats = lambda recs: [input_features("vision", r) for r in recs]
    rf, cf = feats(ref_records), feats(captured)
    input_drift = {name: ({**distribution_compare([f[name] for f in rf], [f[name] for f in cf])} if cf else
                          {"available": False, "reason": "No captured successful inputs; enable capture on a new release to measure input drift."})
                   for name in ("meanIntensity", "intensityStd")}
    preds = [p for t in ok for p in t["result"]["predictions"]]
    prediction_drift = categorical_compare(ref_result["predictions"], preds) if preds else {"available": False, "reason": "No successful predictions in this window."}
    labels = {(r["user"], r["request"]): r for r in runtime.ps.query("SELECT * FROM labels")}
    truth, got, delays, ids = [], [], [], []
    for t in ok:
        row = labels.get((t["user"], t["requestId"]))
        if row:
            truth += json.loads(row["labels"]); got += t["result"]["predictions"]
            delays.append(row["ts"] - t["receivedAt"]); ids.append({"user": t["user"], "requestId": t["requestId"], "traceSha256": t["traceSha256"]})
    q = {"available": bool(truth), "labelledRows": len(truth), "labelledRequests": len(ids), "evidence": ids,
         "reason": None if truth else "Ground-truth labels have not been recorded; quality is not measured.",
         "meanLabelDelaySeconds": float(np.mean(delays)) if delays else None}
    if truth:
        q["values"] = {"accuracy": float(accuracy_score(truth, got))}
    ref_truth = [r["label"] for r in pipeline.reference()]
    lat = [t["totalMs"] for t in traces if t["totalMs"] is not None]
    failures = [t for t in traces if t["status"] != 200]
    return clean({"releaseId": release["id"], "versionId": release["versionId"], "pipelineSha256": release["pipelineSha256"], "family": "image_classifier",
                  "mode": "observed local requests", "window": {"recordedRequests": len(traces), "maxRequests": 1000, "order": "newest 1000 requests for this release"},
                  "health": {"requests": len(traces), "errors": len(failures), "errorFraction": len(failures) / len(traces) if traces else None,
                             "p50Ms": float(np.percentile(lat, 50)) if lat else None, "p95Ms": float(np.percentile(lat, 95)) if lat else None,
                             "schemaErrors": sum((t.get("error") or {}).get("code") == "E_REQUEST_SCHEMA" for t in failures)},
                  "inputDrift": input_drift, "predictionDrift": prediction_drift, "labelBasedQuality": q,
                  "reference": {"runId": m["runId"], "partition": "held-out validation images frozen at registration", "sha256": m["referenceSha256"], "rows": len(ref_records),
                                "referenceAccuracy": float(accuracy_score(ref_truth, ref_result["predictions"])) if ref_truth else None,
                                "referenceAccuracyNote": "pinned model on its own frozen validation reference (not production traffic)"},
                  "alertEvidence": [{"requestId": t["requestId"], "user": t["user"], "status": t["status"], "traceSha256": t["traceSha256"]} for t in failures[:20]],
                  "interpretation": "Image statistics and predicted-class frequencies are descriptive; quality uses only separately supplied labels; no automatic retraining or rollback."})


def rl_monitoring(runtime, release, version, pipeline, traces):
    """Greedy policy: per-dimension observation drift and action frequencies against the frozen replay-buffer reference."""
    m = pipeline.manifest
    ok = [t for t in traces if t["status"] == 200]
    ref = np.asarray(pipeline.reference_observations(), dtype=float)
    ref_actions = pipeline.predict([{"observation": o} for o in ref.tolist()])[0]["predictions"] if len(ref) else []
    cur = np.asarray([r["observation"] for t in ok if t["records"] is not None for r in t["records"]], dtype=float)
    input_drift = {}
    for i in range(m["observationDim"]):
        input_drift[f"obs[{i}]"] = distribution_compare(ref[:, i], cur[:, i]) if len(cur) else {
            "available": False, "reason": "No captured successful inputs; enable capture on a new release to measure input drift."}
    actions = [a for t in ok for a in t["result"]["predictions"]]
    action_drift = categorical_compare(ref_actions, actions) if actions else {"available": False, "reason": "No successful predictions in this window."}
    action_drift["unit"] = "greedy actions"
    labels = {(r["user"], r["request"]): r for r in runtime.ps.query("SELECT * FROM labels")}
    truth, got, delays, ids = [], [], [], []
    for t in ok:
        row = labels.get((t["user"], t["requestId"]))
        if row:
            truth += json.loads(row["labels"]); got += t["result"]["predictions"]
            delays.append(row["ts"] - t["receivedAt"]); ids.append({"user": t["user"], "requestId": t["requestId"], "traceSha256": t["traceSha256"]})
    q = {"available": bool(truth), "labelledRows": len(truth), "labelledRequests": len(ids), "evidence": ids,
         "reason": None if truth else "No reference actions have been supplied; agreement is not measured.",
         "meanLabelDelaySeconds": float(np.mean(delays)) if delays else None}
    if truth:
        q["values"] = {"actionAgreement": float(np.mean([a == b for a, b in zip(truth, got)])),
                       "note": "agreement with actions a person supplied; not environment return (see the run's evaluation report)"}
    lat = [t["totalMs"] for t in traces if t["totalMs"] is not None]
    failures = [t for t in traces if t["status"] != 200]
    return clean({"releaseId": release["id"], "versionId": release["versionId"], "pipelineSha256": release["pipelineSha256"], "family": "rl_policy",
                  "mode": "observed local requests", "window": {"recordedRequests": len(traces), "maxRequests": 1000, "order": "newest 1000 requests for this release"},
                  "health": {"requests": len(traces), "errors": len(failures), "errorFraction": len(failures) / len(traces) if traces else None,
                             "p50Ms": float(np.percentile(lat, 50)) if lat else None, "p95Ms": float(np.percentile(lat, 95)) if lat else None,
                             "schemaErrors": sum((t.get("error") or {}).get("code") == "E_REQUEST_SCHEMA" for t in failures)},
                  "inputDrift": input_drift, "predictionDrift": action_drift, "labelBasedQuality": q,
                  "reference": {"runId": m["runId"], "partition": "replay-buffer observations frozen at registration", "sha256": m["referenceSha256"], "rows": len(ref),
                                "evaluation": m.get("evaluation")},
                  "alertEvidence": [{"requestId": t["requestId"], "user": t["user"], "status": t["status"], "traceSha256": t["traceSha256"]} for t in failures[:20]],
                  "interpretation": "Observation and action-frequency changes are descriptive. Policy quality in the environment is the run's recorded evaluation; agreement uses only supplied actions."})


def unsup_monitoring(runtime, release, version, pipeline, traces):
    """k-means / GMM / PCA: feature drift against the fitted rows; cluster frequencies (or PC1 scores) against the pinned model's own
    assignments of those rows; external agreement (ARI/NMI) only from supplied labels; PCA reconstruction error needs no labels."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    m, method = pipeline.manifest, pipeline.manifest["method"]
    ok = [t for t in traces if t["status"] == 200]
    ref = pipeline.reference_frame()
    ref_out = pipeline.predict(json.loads(ref.to_json(orient="records")))[0]
    captured = pd.DataFrame([r for t in ok if t["records"] is not None for r in t["records"]])
    numeric = {c["name"]: c["dtype"] in ("int", "float") for c in m.get("inputSchema", [])}  # fitted pipelines take raw, possibly categorical columns
    input_drift = {f: ((distribution_compare if numeric.get(f, True) else categorical_compare)(ref[f], captured[f]) if f in captured else
                       {"available": False, "reason": "No captured successful inputs; enable capture on a new release to measure input drift."}) for f in m["features"]}
    if method == "pca":
        cur = [p[0] for t in ok for p in t["result"]["predictions"]]
        prediction_drift = {**distribution_compare([p[0] for p in ref_out["predictions"]], cur), "unit": "PC1 score"} if cur else {"available": False, "reason": "No successful predictions in this window."}
        errs = [e for t in ok for e in t["result"]["reconstructionError"]]
        quality = {"available": bool(errs), "labelFree": True, "reason": None if errs else "No successful predictions in this window.",
                   "values": {"meanReconstructionError": float(np.mean(errs)), "referenceMeanReconstructionError": float(np.mean(ref_out["reconstructionError"]))} if errs else None,
                   "note": "PCA has no labels; reconstruction error (scaled units) is compared with the fitted rows"}
    else:
        cur = [c for t in ok for c in t["result"]["predictions"]]
        prediction_drift = {**categorical_compare(ref_out["predictions"], cur), "unit": "cluster assignments"} if cur else {"available": False, "reason": "No successful predictions in this window."}
        labels = {(r["user"], r["request"]): r for r in runtime.ps.query("SELECT * FROM labels")}
        truth, got, ids = [], [], []
        for t in ok:
            row = labels.get((t["user"], t["requestId"]))
            if row:
                truth += json.loads(row["labels"]); got += t["result"]["predictions"]
                ids.append({"user": t["user"], "requestId": t["requestId"], "traceSha256": t["traceSha256"]})
        quality = {"available": bool(truth), "labelledRows": len(truth), "labelledRequests": len(ids), "evidence": ids,
                   "reason": None if truth else "No external labels supplied; clusters have no ground truth, so agreement is not measured.",
                   "values": {"adjustedRandIndex": float(adjusted_rand_score([str(v) for v in truth], got)),
                              "normalizedMutualInfo": float(normalized_mutual_info_score([str(v) for v in truth], got))} if truth else None,
                   "note": "external agreement with supplied labels (permutation-invariant); clusters are not presented as classes or accuracy"}
    lat = [t["totalMs"] for t in traces if t["totalMs"] is not None]
    failures = [t for t in traces if t["status"] != 200]
    return clean({"releaseId": release["id"], "versionId": release["versionId"], "pipelineSha256": release["pipelineSha256"], "family": method,
                  "mode": "observed local requests", "window": {"recordedRequests": len(traces), "maxRequests": 1000, "order": "newest 1000 requests for this release"},
                  "health": {"requests": len(traces), "errors": len(failures), "errorFraction": len(failures) / len(traces) if traces else None,
                             "p50Ms": float(np.percentile(lat, 50)) if lat else None, "p95Ms": float(np.percentile(lat, 95)) if lat else None,
                             "schemaErrors": sum((t.get("error") or {}).get("code") == "E_REQUEST_SCHEMA" for t in failures)},
                  "inputDrift": input_drift, "predictionDrift": prediction_drift, "labelBasedQuality": quality,
                  "reference": {"runId": m["runId"], "partition": "fitted rows (in-sample)", "sha256": m["referenceSha256"], "rows": len(ref)},
                  "alertEvidence": [{"requestId": t["requestId"], "user": t["user"], "status": t["status"], "traceSha256": t["traceSha256"]} for t in failures[:20]],
                  "interpretation": "Feature and assignment changes are descriptive; no automatic refit or rollback."})


def agent_monitoring(runtime, release, version, pipeline, traces):
    """Read recorded turns and the immutable warmup; monitoring never calls an LLM."""
    m = pipeline.manifest
    ok = [t for t in traces if t["status"] == 200]
    ref = pipeline.reference_records()[0]
    captured = [r for t in ok if t.get("records") for r in t["records"]]
    drift = {f"{k}.characters": distribution_compare([len(ref[k])], [len(r[k]) for r in captured]) if captured else
             {"available": False, "reason": "No captured successful input; enable capture in a new release."} for k in m["inputFields"]}
    from .pipeline import read_verified
    warm = json.loads(read_verified(runtime.store, release["compatibility"]["warmupResultSha256"]))
    preds = [p for t in ok for p in t["result"]["predictions"]]
    json_output = version.get("adapter") in ("agent_json", "conversation_json")
    canonical = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    output_size = lambda value: len(canonical(value).encode()) if json_output else len(value)
    output_drift = distribution_compare([output_size(p) for p in warm["predictions"]], [output_size(p) for p in preds])
    output_drift["unit"] = "canonical JSON UTF8bytes (descriptive, single warmup reference)" if json_output else "output text characters (descriptive, single warmup reference)"
    labels = {(r["user"], r["request"]): r for r in runtime.ps.query("SELECT * FROM labels")}
    pairs, ids, delays = [], [], []
    for t in ok:
        row = labels.get((t["user"], t["requestId"]))
        if row:
            pairs.extend(zip(t["result"]["predictions"], json.loads(row["labels"])))
            ids.append({"user": t["user"], "requestId": t["requestId"], "traceSha256": t["traceSha256"]})
            delays.append(row["ts"]-t["receivedAt"])
    q = {"available": bool(pairs), "labelledRecords": len(pairs), "labelledRequests": len(ids), "evidence": ids,
         "reason": None if pairs else "No reference objects supplied; agreement is not measured." if json_output else "No reference strings supplied; agreement is not measured.",
         "meanLabelDelaySeconds": float(np.mean(delays)) if delays else None,
         "values": {"exactJsonAgreement" if json_output else "exactStringAgreement": float(np.mean([canonical(p) == canonical(y) if json_output else p == y for p, y in pairs]))} if pairs else None,
         "method": "canonical JSON equality (sorted object keys; number representation retained) against supplied schema-valid objects; not semantic accuracy or an LLM judge" if json_output else "literal string equality against supplied reference text; not semantic accuracy or an LLM judge"}
    lat = [t["totalMs"] for t in traces if t.get("totalMs") is not None]
    fail = [t for t in traces if t["status"] != 200]
    calls = [c for t in ok for c in t["result"]["agent"]["contexts"]]
    provider = [c["usage"] for c in calls if c["usage"].get("source") == "provider"]
    return clean({"releaseId": release["id"], "versionId": version["id"], "pipelineSha256": version["pipelineSha256"], "family": m["family"],
                  "window": {"recordedRequests": len(traces), "maxRequests": 1000}, "mode": "recorded isolated native turns",
                  "health": {"requests": len(traces), "errors": len(fail), "errorFraction": len(fail)/len(traces) if traces else None,
                             "p50Ms": float(np.percentile(lat, 50)) if lat else None, "p95Ms": float(np.percentile(lat, 95)) if lat else None,
                             "schemaErrors": sum((t.get("error") or {}).get("code") == "E_REQUEST_SCHEMA" for t in fail)},
                  "inputDrift": drift, "predictionDrift": output_drift, "labelBasedQuality": q,
                  "usage": {"successfulTurnModelCalls": len(calls), "providerReportedCalls": len(provider),
                            "inputTokens": sum(u.get("inputTokens") or 0 for u in provider) if provider else None,
                            "outputTokens": sum(u.get("outputTokens") or 0 for u in provider) if provider else None,
                            "scope": "successful recorded turns only; failed/discarded calls and warmup/replay excluded", "cost": "not measured"},
                  "reference": {"runId": m["runId"], "partition": m["referencePartition"], "sha256": m["referenceSha256"], "rows": 1,
                                "warmupResultSha256": release["compatibility"]["warmupResultSha256"], "caution": "one source input/warmup; weak descriptive evidence"},
                  "alertEvidence": [{"requestId": t["requestId"], "user": t["user"], "status": t["status"], "traceSha256": t["traceSha256"]} for t in fail[:20]],
                  "interpretation": "Output size/usage do not measure semantic correctness; no model call, automatic retraining or rollback during monitoring." if json_output else "Character counts/usage do not measure answer correctness; no model call, automatic retraining or rollback during monitoring."})


def monitoring(runtime, release_id, since=0):
    ps, store = runtime.ps, runtime.store
    release = ps.get("release", release_id)
    pipeline = runtime.pipeline(release["versionId"])
    version = ps.get("version", release["versionId"])
    from .runtime import AGENT_ADAPTERS
    if version.get("adapter") in AGENT_ADAPTERS:
        return agent_monitoring(runtime, release, version, pipeline, [t for t in ps.traces(release_id) if t["receivedAt"] >= since])
    if version.get("adapter") == "unsup":
        return unsup_monitoring(runtime, release, version, pipeline, [t for t in ps.traces(release_id) if t["receivedAt"] >= since])
    if version.get("adapter") == "rl":
        return rl_monitoring(runtime, release, version, pipeline, [t for t in ps.traces(release_id) if t["receivedAt"] >= since])
    if version.get("adapter") in ("model", "model_keras", "model_jax"):
        return model_monitoring(runtime, release, version, pipeline, [t for t in ps.traces(release_id) if t["receivedAt"] >= since])
    if version.get("adapter") == "domain":
        return domain_monitoring(runtime, release, version, pipeline, [t for t in ps.traces(release_id) if t["receivedAt"] >= since])
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
