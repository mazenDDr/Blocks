"""Read recorded paused/END segments without re-executing a model."""
import numpy as np
from .monitor import agent_monitoring
from tabular.core import clean

def monitoring(runtime, release, version, pipeline, traces):
    completed = [t for t in traces if t["status"] == 200]
    pending = [t for t in traces if t["status"] == 202]
    errors = [t for t in traces if t["status"] not in (200, 202)]
    report = agent_monitoring(runtime, release, version, pipeline, completed)
    report["window"]["recordedRequests"] = len(traces)
    lat = [t["totalMs"] for t in traces if t.get("totalMs") is not None]
    report["health"].update(requests=len(traces), errors=len(errors), pendingSegments=len(pending),
                            errorFraction=len(errors)/len(traces) if traces else None,
                            p50Ms=float(np.percentile(lat,50)) if lat else None, p95Ms=float(np.percentile(lat,95)) if lat else None,
                            schemaErrors=sum((t.get("error") or {}).get("code") == "E_REQUEST_SCHEMA" for t in errors))
    calls = [c for t in completed+pending for c in t["result"]["agent"]["contexts"]]
    provider = [c["usage"] for c in calls if c["usage"].get("source") == "provider"]
    report["usage"].update(successfulTurnModelCalls=len(calls), providerReportedCalls=len(provider),
                           inputTokens=sum(u.get("inputTokens") or 0 for u in provider) if provider else None,
                           outputTokens=sum(u.get("outputTokens") or 0 for u in provider) if provider else None,
                           scope="committed paused/END segments; no double counting across resume; warmup/replay/failed calls excluded")
    report["predictionDrift"] = {"available": False, "reason": "Warmup pauses for review and has no final output reference."}
    report["alertEvidence"] = [{"requestId":t["requestId"], "user":t["user"], "status":t["status"], "traceSha256":t["traceSha256"]} for t in errors[:20]]
    return clean(report)
