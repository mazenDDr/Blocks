"""Source-specific, recoverable errors with stable codes (A40).

Every connector failure is mapped to one of these codes so the editor can say *what* failed (credentials, permission,
network, missing resource, timeout, invalid query) and what to do, and so the project is never lost because a source failed."""
from __future__ import annotations

from typing import Any

# code -> (http status, default hint)
CODES: dict[str, tuple[int, str]] = {
    "E_SRC_CONNECTION_NOT_FOUND": (404, "Add the connection in the Connections workspace, or pick an existing one."),
    "E_SRC_SECRET_UNRESOLVED": (422, "The secret reference cannot be resolved here: set the environment variable or fix the secrets file path/key."),
    "E_SRC_SECRET_REF_INVALID": (422, "Use {kind:'env', name} or {kind:'file', path (absolute, outside the project), key}."),
    "E_SRC_SETTINGS_INVALID": (422, "Fix the connection settings; secrets are passed as references in `secrets`, never as settings."),
    "E_SRC_AUTH": (401, "The source rejected the credentials. Check the user/key and the secret reference."),
    "E_SRC_PERMISSION": (403, "The credentials are valid but lack permission for this resource. Ask for read access (SELECT / s3:GetObject / s3:ListBucket)."),
    "E_SRC_NETWORK": (502, "The source could not be reached (host, port, TLS, firewall, endpoint). Check the address and that the service is running."),
    "E_SRC_NOT_FOUND": (404, "The table, column, bucket, object, revision or path does not exist (or is not visible to these credentials)."),
    "E_SRC_QUERY_TIMEOUT": (408, "The statement exceeded the time bound. Narrow the query or raise the timeout (capped by the server)."),
    "E_SRC_QUERY_INVALID": (422, "The query is not valid for this source (syntax, types or semantics)."),
    "E_SRC_TYPE_MISMATCH": (422, "A typed comparison does not fit the column's type."),
    "E_SRC_READONLY_VIOLATION": (422, "Only read-only statements are allowed; this one tried to change data."),
    "E_SRC_BOUNDS": (422, "The request exceeds a configured bound (rows, bytes, objects)."),
    "E_SRC_UNSUPPORTED": (422, "This source or format is not supported by the connector."),
    "E_SRC_SNAPSHOT_UNAVAILABLE": (410, "The pinned snapshot can no longer be resolved (version deleted, remote lacks the data, extract missing)."),
    "E_SRC_SNAPSHOT_MISMATCH": (409, "The pinned snapshot does not belong to this node's query/resource; refusing to substitute data silently."),
    "E_SRC_SNAPSHOT_CORRUPT": (500, "The stored extract no longer matches its recorded hash."),
    "E_SRC_ERROR": (502, "The source reported an error."),
}


class SourceError(Exception):
    """A recoverable failure of an external source. `message` never contains secret values."""

    def __init__(self, code: str, message: str, *, resource: str | None = None, connection: str | None = None, hint: str | None = None,
                 detail: dict[str, Any] | None = None):
        super().__init__(message)
        assert code in CODES, code
        self.code, self.message, self.resource, self.connection = code, message, resource, connection
        self.hint = hint or CODES[code][1]
        self.detail = detail or {}

    @property
    def status(self) -> int:
        return CODES[self.code][0]

    def to_json(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "recoverable": True, "resource": self.resource, "connectionId": self.connection,
                "hint": self.hint, **({"detail": self.detail} if self.detail else {})}


def redact(text: str, secrets: list[str]) -> str:
    for s in secrets:
        if s and len(s) >= 3:
            text = text.replace(s, "***")
    return text
