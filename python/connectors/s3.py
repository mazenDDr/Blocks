"""Generic S3 connector: any S3 API endpoint (AWS or compatible) with credentials by reference."""
from __future__ import annotations

import hashlib
import io
import re
import time
from typing import Any

import pandas as pd

from .errors import SourceError, redact

MAX_LIST = 200
MAX_PREVIEW_BYTES = 1_048_576
MAX_OBJECT_BYTES = 200_000_000
MAX_LISTING_OBJECTS = 10_000


def classify(e: Exception, row: dict[str, Any] | None, resource: str | None, secrets: list[str]) -> SourceError:
    from botocore import exceptions as bex

    cid = row["id"] if row else None
    kw = dict(resource=resource, connection=cid)
    if isinstance(e, SourceError):
        return e
    if isinstance(e, bex.ClientError):
        err = e.response.get("Error", {})
        code, msg = err.get("Code", ""), redact(err.get("Message", "") or code, secrets)
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in ("InvalidAccessKeyId", "SignatureDoesNotMatch", "ExpiredToken", "InvalidToken", "TokenRefreshRequired", "AuthorizationHeaderMalformed") or status == 401:
            return SourceError("E_SRC_AUTH", f"S3 rejected the credentials ({code}): {msg}", **kw)
        if code in ("AccessDenied", "AllAccessDisabled", "Forbidden", "AccountProblem") or status == 403:
            return SourceError("E_SRC_PERMISSION", f"S3 denied access ({code}): {msg}", **kw)
        if code in ("NoSuchBucket", "NoSuchKey", "NoSuchVersion", "404", "NotFound") or status == 404:
            return SourceError("E_SRC_NOT_FOUND", f"S3 resource not found ({code}): {msg}", **kw)
        return SourceError("E_SRC_ERROR", f"S3 error ({code}): {msg}", **kw)
    if isinstance(e, (bex.EndpointConnectionError, bex.ConnectTimeoutError, bex.ReadTimeoutError, bex.ConnectionClosedError, bex.SSLError)):
        return SourceError("E_SRC_NETWORK", f"Cannot reach the S3 endpoint: {redact(str(e), secrets)}", **kw)
    if isinstance(e, bex.NoCredentialsError):
        return SourceError("E_SRC_AUTH", "No credentials: configure access_key_id / secret_access_key references (or rely on the ambient AWS chain only if you set neither).", **kw)
    if isinstance(e, bex.BotoCoreError):
        return SourceError("E_SRC_ERROR", f"S3 client error: {redact(str(e), secrets)}", **kw)
    if isinstance(e, OSError):
        return SourceError("E_SRC_NETWORK", f"Network error: {redact(str(e), secrets)}", **kw)
    return SourceError("E_SRC_ERROR", f"{type(e).__name__}: {redact(str(e), secrets)}", **kw)


class S3:
    def __init__(self, registry, connection_id: str):
        self.row = registry.require(connection_id)
        if self.row["type"] != "s3":
            raise SourceError("E_SRC_UNSUPPORTED", f"Connection '{connection_id}' is a {self.row['type']} connection, not s3.", connection=connection_id)
        self.secrets = registry.secret_values(self.row)
        s = self.row["settings"]
        self.bucket = s["bucket"]
        import boto3
        from botocore.config import Config

        kw: dict[str, Any] = {}
        if self.secrets.get("access_key_id") or self.secrets.get("secret_access_key"):
            kw.update(aws_access_key_id=self.secrets.get("access_key_id"), aws_secret_access_key=self.secrets.get("secret_access_key"),
                      aws_session_token=self.secrets.get("session_token"))
        self.client = boto3.client("s3", region_name=s["region"], endpoint_url=s["endpoint_url"] or None, verify=s["verify_tls"],
                                   config=Config(signature_version="s3v4", retries={"max_attempts": 1}, connect_timeout=5, read_timeout=15,
                                                 s3={"addressing_style": "path" if s["endpoint_url"] else "auto"}), **kw)

    def call(self, fn, resource: str, **kwargs):
        try:
            return fn(**kwargs)
        except Exception as e:  # noqa: BLE001
            raise classify(e, self.row, resource, list(self.secrets.values())) from None

    def versioning(self) -> str:
        r = self.call(self.client.get_bucket_versioning, f"s3://{self.bucket}", Bucket=self.bucket)
        return r.get("Status", "Disabled")  # Enabled | Suspended | Disabled(absent)


CAPABILITIES = {"discovery": True, "sampling": "ranged read of the first bytes", "filterProjectionPushdown": False, "streaming": False,
                "versionedReads": "object version id when the bucket is versioned", "incrementalReads": False, "writes": False,
                "cancellation": False, "privateNetwork": "via the machine running the worker"}


def test_connection(registry, connection_id: str) -> dict[str, Any]:
    t0 = time.time()
    try:
        c = S3(registry, connection_id)
        c.call(c.client.head_bucket, f"s3://{c.bucket}", Bucket=c.bucket)
        status = c.versioning()
        r = c.call(c.client.list_objects_v2, f"s3://{c.bucket}", Bucket=c.bucket, MaxKeys=1)
        return {"ok": True, "latencyMs": round((time.time() - t0) * 1000), "bucket": c.bucket, "versioning": status,
                "versioned": status == "Enabled", "listable": True, "hasObjects": r.get("KeyCount", 0) > 0, "capabilities": CAPABILITIES,
                "note": None if status == "Enabled" else "Bucket versioning is not enabled: reads cannot be pinned to an object version; runs will materialize a copy and flag limited reproducibility."}
    except SourceError as e:
        return {"ok": False, "latencyMs": round((time.time() - t0) * 1000), "error": e.to_json(), "capabilities": CAPABILITIES}


def fmt_of(key: str) -> str:
    k = key.lower()
    for ext, f in ((".csv", "csv"), (".tsv", "tsv"), (".parquet", "parquet"), (".json", "json"), (".jsonl", "jsonl"), (".png", "png"), (".jpg", "jpeg"), (".jpeg", "jpeg"), (".npy", "npy")):
        if k.endswith(ext):
            return f
    return "unknown"


def browse(registry, connection_id: str, prefix: str = "", token: str | None = None, max_keys: int = 100) -> dict[str, Any]:
    c = S3(registry, connection_id)
    max_keys = max(1, min(max_keys, MAX_LIST))
    versioned = c.versioning() == "Enabled"
    res = f"s3://{c.bucket}/{prefix}"
    if versioned:
        kw: dict[str, Any] = {"Bucket": c.bucket, "Prefix": prefix, "Delimiter": "/", "MaxKeys": max_keys}
        if token:
            kw["KeyMarker"] = token
        r = c.call(c.client.list_object_versions, res, **kw)
        by_key: dict[str, dict[str, Any]] = {}
        for v in r.get("Versions", []):
            d = by_key.setdefault(v["Key"], {"key": v["Key"], "versionCount": 0})
            d["versionCount"] += 1
            if v["IsLatest"]:
                d.update(size=v["Size"], etag=v["ETag"].strip('"'), lastModified=v["LastModified"].isoformat(), versionId=v["VersionId"])
        for m in r.get("DeleteMarkers", []):
            d = by_key.setdefault(m["Key"], {"key": m["Key"], "versionCount": 0})
            if m["IsLatest"]:
                d["deleted"] = True
        objs = [d for d in by_key.values() if not d.get("deleted")]
        nxt = r.get("NextKeyMarker") if r.get("IsTruncated") else None
    else:
        kw = {"Bucket": c.bucket, "Prefix": prefix, "Delimiter": "/", "MaxKeys": max_keys}
        if token:
            kw["ContinuationToken"] = token
        r = c.call(c.client.list_objects_v2, res, **kw)
        objs = [{"key": o["Key"], "size": o["Size"], "etag": o["ETag"].strip('"'), "lastModified": o["LastModified"].isoformat(), "versionId": None, "versionCount": None}
                for o in r.get("Contents", [])]
        nxt = r.get("NextContinuationToken") if r.get("IsTruncated") else None
    for o in objs:
        o["format"] = fmt_of(o["key"])
    return {"connectionId": connection_id, "bucket": c.bucket, "prefix": prefix, "versioning": "Enabled" if versioned else "not enabled",
            "prefixes": [p["Prefix"] for p in r.get("CommonPrefixes", [])], "objects": objs, "nextToken": nxt,
            "bounds": {"maxKeys": max_keys}, "note": None if versioned else "not versioned: object versions cannot be pinned"}


def versions(registry, connection_id: str, key: str, limit: int = 50) -> dict[str, Any]:
    c = S3(registry, connection_id)
    r = c.call(c.client.list_object_versions, f"s3://{c.bucket}/{key}", Bucket=c.bucket, Prefix=key, MaxKeys=min(limit, 100))
    vs = [{"versionId": v["VersionId"], "isLatest": v["IsLatest"], "size": v["Size"], "etag": v["ETag"].strip('"'), "lastModified": v["LastModified"].isoformat()}
          for v in r.get("Versions", []) if v["Key"] == key]
    return {"connectionId": connection_id, "key": key, "versions": vs, "versioned": c.versioning() == "Enabled"}


def preview_object(registry, connection_id: str, key: str, version_id: str | None = None, rows: int = 20, max_bytes: int = 65_536) -> dict[str, Any]:
    """Ranged read of the first bytes (bounded); CSV/TSV are parsed, other formats show metadata only."""
    c = S3(registry, connection_id)
    max_bytes = max(1024, min(max_bytes, MAX_PREVIEW_BYTES))
    rows = max(1, min(rows, 200))
    kw = {"Bucket": c.bucket, "Key": key, "Range": f"bytes=0-{max_bytes - 1}"}
    if version_id:
        kw["VersionId"] = version_id
    head = {"Bucket": c.bucket, "Key": key, **({"VersionId": version_id} if version_id else {})}
    h = c.call(c.client.head_object, f"s3://{c.bucket}/{key}", **head)
    fmt = fmt_of(key)
    out: dict[str, Any] = {"connectionId": connection_id, "key": key, "format": fmt, "size": h["ContentLength"], "etag": h["ETag"].strip('"'),
                           "versionId": h.get("VersionId"), "lastModified": h["LastModified"].isoformat(),
                           "bounds": {"maxBytes": max_bytes, "rows": rows, "bytesRead": 0}}
    if fmt not in ("csv", "tsv"):
        out.update(columns=None, rows=None, note=f"No tabular preview for format '{fmt}'; metadata only.")
        return out
    body = c.call(c.client.get_object, f"s3://{c.bucket}/{key}", **kw)["Body"].read()
    out["bounds"]["bytesRead"] = len(body)
    partial = h["ContentLength"] > len(body)
    text = body.decode("utf-8", errors="replace")
    if partial and "\n" in text:
        text = text[: text.rfind("\n") + 1]  # drop the cut-off last line
    try:
        df = pd.read_csv(io.StringIO(text), sep="\t" if fmt == "tsv" else ",", nrows=rows)
    except Exception as e:  # noqa: BLE001
        raise SourceError("E_SRC_QUERY_INVALID", f"Cannot parse the object as {fmt}: {e}", resource=f"s3://{c.bucket}/{key}", connection=connection_id)
    from tabular.core import clean, schema_of

    out.update(columns=schema_of(df), rows=[[clean(v) for v in r] for r in df.itertuples(index=False, name=None)], truncated=partial,
               note="schema is inferred from the sampled prefix only" if partial else None)
    return out


def read_object(c: S3, key: str, version_id: str | None, max_bytes: int = MAX_OBJECT_BYTES) -> tuple[bytes, dict[str, Any]]:
    kw = {"Bucket": c.bucket, "Key": key, **({"VersionId": version_id} if version_id else {})}
    res = f"s3://{c.bucket}/{key}"
    h = c.call(c.client.head_object, res, **kw)
    if h["ContentLength"] > max_bytes:
        raise SourceError("E_SRC_BOUNDS", f"Object is {h['ContentLength']} bytes; the limit is {max_bytes}.", resource=res, connection=c.row["id"])
    r = c.call(c.client.get_object, res, **kw)
    try:
        body = r["Body"].read()
    except Exception as e:  # noqa: BLE001
        raise classify(e, c.row, res, list(c.secrets.values())) from None
    meta = {"bucket": c.bucket, "key": key, "versionId": h.get("VersionId") if h.get("VersionId") not in (None, "null") else None, "etag": h["ETag"].strip('"'),
            "size": h["ContentLength"], "lastModified": h["LastModified"].isoformat(), "sha256": hashlib.sha256(body).hexdigest(), "endpoint": c.row["settings"]["endpoint_url"]}
    return body, meta


def list_all(c: S3, prefix: str, max_objects: int) -> tuple[list[dict[str, Any]], bool]:
    out, token = [], None
    while True:
        kw: dict[str, Any] = {"Bucket": c.bucket, "Prefix": prefix, "MaxKeys": min(1000, max_objects + 1 - len(out))}
        if token:
            kw["ContinuationToken"] = token
        r = c.call(c.client.list_objects_v2, f"s3://{c.bucket}/{prefix}", **kw)
        out += [{"key": o["Key"], "size": o["Size"], "etag": o["ETag"].strip('"'), "last_modified": o["LastModified"].isoformat()} for o in r.get("Contents", [])]
        token = r.get("NextContinuationToken")
        if len(out) > max_objects:
            return out[:max_objects], True
        if not token:
            return out, False


KEY_ID_RE_MAX = 200


def compile_key_regex(pattern: str) -> re.Pattern:
    if len(pattern) > KEY_ID_RE_MAX:
        raise SourceError("E_SRC_QUERY_INVALID", "key regex is too long")
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise SourceError("E_SRC_QUERY_INVALID", f"Invalid key regex: {e}")
    if "id" not in rx.groupindex:
        raise SourceError("E_SRC_QUERY_INVALID", "The key regex must contain a named group (?P<id>...).")
    return rx
