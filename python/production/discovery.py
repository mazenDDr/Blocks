"""Read-only bounded conversation discovery, independent of native inference pins."""
import json
import re

from tabular.core import dumps
from .pipeline import ProductionError

KEY = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
PREFIX = re.compile(r"[A-Za-z0-9_-]{0,64}\Z")
POLICY = "Metadata only; no native state loading or model call. Checkpoint integrity/state require explicit inspection. Caller-declared user keys are isolation scopes, not authenticated ownership. Pages reflect committed heads when read, not one historical snapshot."


def discover(ps, release_id, user, *, limit=25, after=None, prefix=""):
    if (not isinstance(user, str) or not KEY.fullmatch(user) or type(limit) is not int or not 1 <= limit <= 100
            or not isinstance(prefix, str) or not PREFIX.fullmatch(prefix)
            or after is not None and (not isinstance(after, str) or not KEY.fullmatch(after))):
        raise ProductionError("E_SESSION_QUERY", "Use valid user/session keys, a literal prefix and page size 1–100.")
    release = ps.get("release", release_id)
    if release["config"]["sessionMode"] != "conversation":
        raise ProductionError("E_RELEASE_CONFIG", "This release has no native conversations.")
    # Existing writers use canonical dumps([release,user,session]). PK ranges
    # avoid scanning or decoding all other users' sessions; '_' is literal.
    start = dumps([release_id, user])[:-1] + ', "' + prefix
    cursor = dumps([release_id, user, after]) if after is not None else ""
    rows = ps.query("SELECT scope,revision,checkpoint,last FROM agent_sessions WHERE scope>=? AND scope<? AND scope>? ORDER BY scope LIMIT ?",
                    (start, start + "\uffff", cursor, limit + 1))
    sessions = []
    for row in rows[:limit]:
        try:
            scope = json.loads(row["scope"])
        except ValueError:
            raise ProductionError("E_SESSION_INTEGRITY", "Stored session scope is invalid; no state was loaded.", 409) from None
        if (not isinstance(scope, list) or len(scope) != 3 or scope[:2] != [release_id, user]
                or not isinstance(scope[2], str) or not KEY.fullmatch(scope[2])
                or not scope[2].startswith(prefix) or row["scope"] != dumps(scope)
                or type(row["revision"]) is not int or row["revision"] < 1
                or row["checkpoint"] is not None and (not isinstance(row["checkpoint"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["checkpoint"]))):
            raise ProductionError("E_SESSION_INTEGRITY", "Stored session scope is invalid; no state was loaded.", 409)
        sessions.append({"session": scope[2], "status": "checkpoint" if row["checkpoint"] is not None else "reset",
                         "head": {"revision": row["revision"], "checkpointSha256": row["checkpoint"], "lastRequestId": row["last"]}})
    return {"releaseId": release_id, "versionId": release["versionId"], "user": user, "prefix": prefix, "limit": limit,
            "after": after, "sessions": sessions, "nextAfter": sessions[-1]["session"] if len(rows) > limit else None,
            "readOnly": True, "policy": POLICY}
