"""Verified request-owned historical END checkpoints; no arbitrary SHA/import path."""
import json
import re

from .pipeline import ProductionError, read_verified


def inspect_history(runtime, release_id, user, session, request_id, *, pipeline=None):
    release = runtime.ps.get("release", release_id)
    if release["config"]["sessionMode"] != "conversation":
        raise ProductionError("E_RELEASE_CONFIG", "This release has no native conversations.")
    trace = runtime.ps.trace(user, request_id)
    recorded = runtime.ps.query("SELECT state,trace,release FROM requests WHERE user=? AND id=?", (user, request_id))
    if not recorded or recorded[0]["state"] != "completed" or trace.get("available") is False:
        raise ProductionError("E_SESSION_HISTORY_EMPTY", "Request has no successful committed historical checkpoint.", 409)
    if (recorded[0]["trace"], recorded[0]["release"], trace.get("requestId")) != (trace.get("traceSha256"), release_id, request_id):
        raise ProductionError("E_SESSION_HISTORY_SCOPE", "Historical request identity differs from this recorded scope.", 409)
    if (trace.get("user"), trace.get("session"), trace.get("releaseId"), trace.get("versionId")) != (user, session, release_id, release["versionId"]):
        if trace.get("available") is False:
            raise ProductionError("E_SESSION_HISTORY_EMPTY", "Request has no committed historical checkpoint.", 409)
        raise ProductionError("E_SESSION_HISTORY_SCOPE", "Historical request must belong to this release/user/session/version.", 409)
    source = trace.get("conversationState")
    if (trace.get("status") != 200 or not isinstance(source, dict) or type(source.get("revision")) is not int
            or source["revision"] < 1 or source.get("lastRequestId") != request_id
            or not isinstance(source.get("checkpointSha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", source["checkpointSha256"])):
        raise ProductionError("E_SESSION_HISTORY_EMPTY", "Select a successful native turn with a committed END checkpoint.", 409)
    sha = source["checkpointSha256"]
    envelope = json.loads(read_verified(runtime.store, sha))
    if not isinstance(envelope, dict) or not isinstance(source.get("threadId"), str) or not source["threadId"] or envelope.get("threadId") != source["threadId"]:
        raise ProductionError("E_SESSION_HISTORY_INTEGRITY", "Historical checkpoint thread provenance differs from its request.", 409)
    native = pipeline or runtime.pipeline(release["versionId"])
    state = native.checkpoint_state(sha)
    return {"releaseId": release_id, "versionId": release["versionId"], "user": user, "session": session,
            "sourceRequestId": request_id, "sourceTraceSha256": trace["traceSha256"], "sourceCheckpointSha256": sha,
            "sourceHead": {"revision": source["revision"], "checkpointSha256": sha, "lastRequestId": request_id},
            "sourceThreadId": envelope["threadId"], "state": state, "readOnly": True,
            "policy": "Verified historical END checkpoint from this request/scope; inspection makes no model call or state change. Restoration requires reviewed current and historical identities and preserves earlier evidence."}
