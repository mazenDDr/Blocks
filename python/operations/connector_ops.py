"""Connector source nodes and the cross-source join (tabular graph kind).

postgres.query (visual builder or raw read-only SQL), s3.csv_source, s3.object_listing, dvc.csv_source, tabular.join.
Each source resolves to a recorded identity (a snapshot manifest in the artifact store) on every run; a node or a run can pin a
snapshot id to repeat from it. Static validation reads the schema from the pinned manifest or probes the source (LIMIT 0 / ranged read)."""
from __future__ import annotations

import io
import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from artifact_store import ArtifactStore
from connectors import context, snapshots
from connectors import dvc as dvc_mod
from connectors import postgres as pg
from connectors import s3 as s3_mod
from connectors.errors import SourceError
from connectors.pg_query import MAX_LIMIT, QuerySpec
from graph_core.registry import register
from graph_core.types import Fix, OpError
from tabular.core import (TABLE, ExecCtx, ExecutionError, Table, TabularOperation, VType, clean, need_columns, row_ids_sha256, schema_of, table_type)

from ._common import StrictConfig
from .tabular_ops import sample_rows

_SRC_FIX = [Fix("Open the Connections workspace, test the connection, then pick a resource")]


def _opx(e: SourceError, port: str | None = None) -> OpError:
    return OpError(e.code, f"{e.message} ({e.hint})", port, list(_SRC_FIX))


def _exx(e: SourceError) -> ExecutionError:
    return ExecutionError(e.code, f"{e.message} ({e.hint})")


def _registry():
    return context.registry()


def _store() -> ArtifactStore:
    return ArtifactStore(context.workbench())


def _dtype_cols(schema: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"name": c["name"], "dtype": c["dtype"]} for c in schema]


def _csv_frame(raw: bytes, delimiter: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=delimiter)
    except Exception as e:  # noqa: BLE001
        raise ExecutionError("E_SRC_QUERY_INVALID", f"Cannot parse the data as CSV: {e}")
    df.index = pd.RangeIndex(len(df), name="row_id")
    return df


def _lineage(node: str, kind: str, uri: str, sha: str | None, sid: str, mode: str) -> dict[str, Any]:
    return {"source": {"node": node, "kind": kind, "path": uri, "sha256": sha, "snapshotId": sid, "mode": mode}}


def _same(a: Any, b: Any) -> bool:
    import json

    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


_probe_cache: dict[tuple, tuple[float, Any]] = {}


def _cached(key: tuple, fn):
    hit = _probe_cache.get(key)
    if hit and time.time() - hit[0] < 30:
        return hit[1]
    v = fn()
    _probe_cache[key] = (time.time(), v)
    return v


# ================================================================================================ PostgreSQL
class SqlParam(StrictConfig):
    name: str = Field(..., pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")
    type: Literal["int", "float", "string", "bool", "date", "timestamp"]
    value: Any = None


class PostgresQueryConfig(StrictConfig):
    connection: str = Field("", description="Connection id from the Connections workspace")
    mode: Literal["visual", "sql"] = "visual"
    query: QuerySpec | None = Field(None, description="Visual query (compiled to parameterized SQL)")
    sql: str = Field("", description="Raw SQL for mode 'sql': one read-only SELECT; values via %(name)s parameters")
    params: list[SqlParam] = Field(default_factory=list)
    limit: int = Field(10_000, ge=1, le=MAX_LIMIT, description="Row bound for raw SQL (visual queries carry their own limit)")
    timeout_ms: int = Field(10_000, ge=100, le=30_000)
    pin: str | None = Field(None, description="Snapshot id: read this recorded extract instead of the live database")


def _pg_identity(cfg: PostgresQueryConfig):
    params = [p.model_dump() for p in cfg.params]
    return pg.query_identity(cfg.mode, cfg.query, cfg.sql, params, cfg.limit)


@register
class PostgresQuery(TabularOperation):
    type = "postgres.query"
    backend = "postgresql"
    inputs = ()
    outputs = ("table",)
    out_kinds = {"table": TABLE}
    Config = PostgresQueryConfig
    summary_kind = "connector_source"

    def infer(self, cfg: PostgresQueryConfig, ins, node_id):
        if not cfg.connection:
            raise OpError("E_SRC_CONNECTION_NOT_FOUND", "No connection chosen.", None, list(_SRC_FIX))
        try:
            text, pnames, comp, run_sql, run_params = _pg_identity(cfg)
            warn = comp.warnings if comp else []
            if cfg.pin:
                m, _ = snapshots.load(_store(), cfg.pin, want_extract=False)
                self._check_match(m, text, pnames, cfg.pin)
                cols, rows = m["result"]["columns"], m["result"]["rows"]
            else:
                cols = pg.probe_schema(_registry(), cfg.connection, comp if comp else text, None if comp else (run_params if isinstance(run_params, dict) else None))
                rows = None
        except SourceError as e:
            raise _opx(e)
        return {"table": table_type(_dtype_cols(cols), rows, "full", True, rowsExact=rows is not None, sourceNode=node_id, sourceKind="postgres",
                                     connectionId=cfg.connection, pinned=cfg.pin, queryWarnings=warn)}

    @staticmethod
    def _check_match(m: dict[str, Any], text: str, pnames: list, sid: str) -> None:
        if m.get("kind") != "postgres" or m["query"]["text"] != text or not _same(m["query"]["params"], pnames):
            raise SourceError("E_SRC_SNAPSHOT_MISMATCH", f"Pinned snapshot {sid[:12]}... was taken from a different query than this node's.")

    def execute(self, cfg: PostgresQueryConfig, ins, ctx: ExecCtx):
        try:
            text, pnames, comp, run_sql, run_params = _pg_identity(cfg)
            pin = snapshots.pinned_for(cfg.pin, ctx)
            if pin:
                m, raw = snapshots.load(ctx.store, pin)
                self._check_match(m, text, pnames, pin)
                df = pg.read_extract(raw, m["result"]["columns"])  # type: ignore[arg-type]
                snapshots.reference(ctx, pin)
                mode, sid, schema, reports, res_sha = "pinned", pin, m["result"]["columns"], m.get("joinReports", []), m["result"]["sha256"]
                manifest = m
            else:
                if not cfg.connection:
                    raise SourceError("E_SRC_CONNECTION_NOT_FOUND", "No connection chosen.")
                r = pg.run_query(_registry(), cfg.connection, mode=cfg.mode, spec=cfg.query, sql=cfg.sql, params=[p.model_dump() for p in cfg.params],
                                 limit=cfg.limit, timeout_ms=cfg.timeout_ms, with_join_reports=True)
                df, schema, reports, res_sha = r["df"], r["schema"], r["joinReports"], r["resultSha256"]
                manifest = {"kind": "postgres", "connectionId": cfg.connection,
                            "server": {"host": r["server"]["host"], "port": r["server"]["port"], "database": r["server"]["database"], "version": r["server"]["version"]},
                            "query": r["query"], "snapshotToken": r["server"]["snapshotToken"], "takenAt": snapshots.now_iso(),
                            "transaction": f"{r['server']['isolation']}, read only={r['server']['readOnly']}",
                            "result": {"rows": len(df), "columns": schema, "sha256": res_sha, "bytes": len(r["extract"]), "ordered": r["ordered"],
                                       "reachedLimit": r["reachedLimit"], "truncated": r["truncated"]},
                            "joinReports": reports,
                            "reproducibility": {"level": "materialized_extract", "limited": False,
                                                "statement": "The result is stored as an extract; repeating from this snapshot reads the stored extract, not the live database. "
                                                             "The live source may have changed since; use the drift check to compare."
                                                             + ("" if r["ordered"] else " The query has no ORDER BY, so a fresh execution may return rows in a different order.")}}
                sid = snapshots.record(ctx, manifest, r["extract"])
                manifest, _ = snapshots.load(ctx.store, sid, want_extract=False)  # the stored form (includes the extract reference)
                mode = "live"
        except SourceError as e:
            raise _exx(e)
        conn_label = cfg.connection or manifest.get("connectionId")
        t = Table(df, "full", _lineage(ctx.node_id, "postgres", f"postgres://{conn_label}", res_sha, sid, mode))
        summary = {"connector": "postgres", "mode": mode, "snapshotId": sid, "snapshot": manifest, "sql": manifest["query"]["text"], "params": manifest["query"]["params"],
                   "queryMode": manifest["query"]["mode"], "joinReports": reports, "rows": len(df), "schema": schema, "resultSha256": res_sha,
                   "sha256": res_sha, "path": f"postgres://{conn_label}", "bytes": manifest["result"]["bytes"],
                   "reproducibility": manifest["reproducibility"], "preview": sample_rows(df),
                   "transfer": {"from": f"PostgreSQL ({manifest['server']['host']}:{manifest['server']['port']}/{manifest['server']['database']})" if mode == "live" else "stored extract (no source access)",
                                "to": "worker process memory", "rows": len(df), "bytes": manifest["result"]["bytes"]}}
        return {"table": t}, clean(summary)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "table = psycopg.execute(compiled SQL, bound parameters) in a READ ONLY REPEATABLE READ transaction",
                "rule": "Visual queries compile to parameterized SQL (identifiers quoted, values bound with explicit casts). Raw SQL runs inside SELECT * FROM (...) LIMIT n in a read-only transaction with a statement timeout.",
                "note": "Each run stores the query text, parameters, server version, snapshot token and a content hash of the result as a snapshot, plus the extract. Pin a snapshot to repeat from it."}


# ================================================================================================ S3 CSV object
class S3CsvConfig(StrictConfig):
    connection: str = ""
    key: str = Field("", description="Object key")
    version_id: str | None = Field(None, description="Read this exact object version (versioned buckets)")
    delimiter: str = Field(",", min_length=1, max_length=1)
    max_bytes: int = Field(50_000_000, ge=1024, le=s3_mod.MAX_OBJECT_BYTES)
    pin: str | None = None


@register
class S3CsvSource(TabularOperation):
    type = "s3.csv_source"
    backend = "s3"
    inputs = ()
    outputs = ("table",)
    out_kinds = {"table": TABLE}
    Config = S3CsvConfig
    summary_kind = "connector_source"

    def infer(self, cfg: S3CsvConfig, ins, node_id):
        if not cfg.connection:
            raise OpError("E_SRC_CONNECTION_NOT_FOUND", "No connection chosen.", None, list(_SRC_FIX))
        if not cfg.key:
            raise OpError("E_SOURCE_NOT_SET", "No object key chosen.", None, list(_SRC_FIX))
        try:
            if cfg.pin:
                m, _ = snapshots.load(_store(), cfg.pin, want_extract=False)
                if m.get("kind") != "s3_object" or m["object"]["key"] != cfg.key:
                    raise SourceError("E_SRC_SNAPSHOT_MISMATCH", f"Pinned snapshot {cfg.pin[:12]}... is for a different object.")
                cols, rows = m["result"]["columns"], m["result"]["rows"]
            else:
                def probe():
                    p = s3_mod.preview_object(_registry(), cfg.connection, cfg.key, cfg.version_id, rows=200)
                    if p.get("columns") is None:
                        raise SourceError("E_SRC_UNSUPPORTED", f"'{cfg.key}' is not a CSV/TSV object (format: {p['format']}).")
                    return p["columns"]
                cols, rows = _cached(("s3", cfg.connection, cfg.key, cfg.version_id), probe), None
        except SourceError as e:
            raise _opx(e)
        return {"table": table_type(cols, rows, "full", True, rowsExact=rows is not None, sourceNode=node_id, sourceKind="s3", connectionId=cfg.connection,
                                    schemaFromSample=rows is None)}

    def execute(self, cfg: S3CsvConfig, ins, ctx: ExecCtx):
        try:
            pin = snapshots.pinned_for(cfg.pin, ctx)
            if pin:
                m, raw = snapshots.load(ctx.store, pin)
                if m.get("kind") != "s3_object" or m["object"]["key"] != cfg.key:
                    raise SourceError("E_SRC_SNAPSHOT_MISMATCH", f"Pinned snapshot {pin[:12]}... is for a different object.")
                if raw is None:  # versioned: re-read the exact version and verify the content hash
                    o = m["object"]
                    raw, meta = s3_mod.read_object(s3_mod.S3(_registry(), m["connectionId"]), o["key"], o["versionId"], cfg.max_bytes)
                    if meta["sha256"] != o["sha256"]:
                        raise SourceError("E_SRC_SNAPSHOT_MISMATCH", "The pinned object version no longer has the recorded content hash.")
                snapshots.reference(ctx, pin)
                manifest, sid, mode = m, pin, "pinned"
            else:
                if not cfg.connection or not cfg.key:
                    raise SourceError("E_SRC_CONNECTION_NOT_FOUND", "Connection and key are required.")
                raw, meta = s3_mod.read_object(s3_mod.S3(_registry(), cfg.connection), cfg.key, cfg.version_id, cfg.max_bytes)
                df0 = _csv_frame(raw, cfg.delimiter)
                versioned = meta["versionId"] is not None
                manifest = {"kind": "s3_object", "connectionId": cfg.connection, "object": meta, "delimiter": cfg.delimiter,
                            "takenAt": snapshots.now_iso(),
                            "result": {"rows": len(df0), "columns": schema_of(df0), "sha256": meta["sha256"]},
                            "reproducibility": ({"level": "pinned_object_version", "limited": False,
                                                 "statement": "Pinned to bucket/key/versionId; repeating re-reads exactly that version and verifies the content hash."}
                                                if versioned else
                                                {"level": "materialized_copy_unversioned", "limited": True,
                                                 "statement": "NOT VERSIONED - REPRODUCIBILITY LIMITED: the bucket gives no version id, so the object at this key can change or disappear. "
                                                              "A copy was stored with the snapshot; repeating reads the stored copy, not the bucket."})}
                sid = snapshots.record(ctx, manifest, None if versioned else raw)
                manifest, _ = snapshots.load(ctx.store, sid, want_extract=False)  # the stored form (includes the extract reference)
                mode = "live"
        except SourceError as e:
            raise _exx(e)
        df = _csv_frame(raw, manifest.get("delimiter", cfg.delimiter))
        o = manifest["object"]
        uri = f"s3://{o['bucket']}/{o['key']}"
        t = Table(df, "full", _lineage(ctx.node_id, "s3_object", uri, o["sha256"], sid, mode))
        summary = {"connector": "s3", "mode": mode, "snapshotId": sid, "snapshot": manifest, "object": o, "rows": len(df), "schema": schema_of(df), "sha256": o["sha256"],
                   "path": uri, "bytes": o["size"], "reproducibility": manifest["reproducibility"], "preview": sample_rows(df),
                   "transfer": {"from": f"S3 {o['endpoint'] or 'AWS'} / {o['bucket']}" if mode == "live" else ("stored copy (no source access)" if manifest.get("extract") else "S3 (pinned version)"),
                                "to": "worker process memory", "rows": len(df), "bytes": o["size"]}}
        return {"table": t}, clean(summary)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "table = pandas.read_csv(s3.get_object(bucket, key, VersionId))",
                "rule": "The object's bucket/key/versionId (or ETag, flagged as limited reproducibility when the bucket is not versioned) and its SHA-256 are recorded with every run.",
                "note": "Unversioned objects are materialized into the snapshot so the run can still be repeated."}


# ================================================================================================ S3 object listing
class S3ListingConfig(StrictConfig):
    connection: str = ""
    prefix: str = ""
    max_objects: int = Field(1000, ge=1, le=s3_mod.MAX_LISTING_OBJECTS)
    key_regex: str = Field("", description="Optional regex with a named group (?P<id>...) to extract an identifier from each key into column key_id")
    pin: str | None = None


LISTING_COLS = [{"name": "key", "dtype": "string"}, {"name": "size", "dtype": "int"}, {"name": "etag", "dtype": "string"}, {"name": "last_modified", "dtype": "string"}]


@register
class S3ObjectListing(TabularOperation):
    type = "s3.object_listing"
    backend = "s3"
    inputs = ()
    outputs = ("table",)
    out_kinds = {"table": TABLE}
    Config = S3ListingConfig
    summary_kind = "connector_source"

    def infer(self, cfg: S3ListingConfig, ins, node_id):
        if not cfg.connection:
            raise OpError("E_SRC_CONNECTION_NOT_FOUND", "No connection chosen.", None, list(_SRC_FIX))
        try:
            if cfg.key_regex:
                s3_mod.compile_key_regex(cfg.key_regex)
            rows = None
            if cfg.pin:
                m, _ = snapshots.load(_store(), cfg.pin, want_extract=False)
                if m.get("kind") != "s3_listing" or m["listing"]["prefix"] != cfg.prefix:
                    raise SourceError("E_SRC_SNAPSHOT_MISMATCH", f"Pinned snapshot {cfg.pin[:12]}... is for a different prefix.")
                rows = m["result"]["rows"]
        except SourceError as e:
            raise _opx(e)
        cols = LISTING_COLS + ([{"name": "key_id", "dtype": "string"}] if cfg.key_regex else [])
        return {"table": table_type(cols, rows, "full", True, rowsExact=rows is not None, sourceNode=node_id, sourceKind="s3", connectionId=cfg.connection)}

    def execute(self, cfg: S3ListingConfig, ins, ctx: ExecCtx):
        try:
            pin = snapshots.pinned_for(cfg.pin, ctx)
            if pin:
                m, raw = snapshots.load(ctx.store, pin)
                if m.get("kind") != "s3_listing" or m["listing"]["prefix"] != cfg.prefix:
                    raise SourceError("E_SRC_SNAPSHOT_MISMATCH", f"Pinned snapshot {pin[:12]}... is for a different prefix.")
                snapshots.reference(ctx, pin)
                manifest, sid, mode = m, pin, "pinned"
            else:
                c = s3_mod.S3(_registry(), cfg.connection)
                objs, truncated = s3_mod.list_all(c, cfg.prefix, cfg.max_objects)
                df0 = pd.DataFrame(objs, columns=["key", "size", "etag", "last_modified"])
                if cfg.key_regex:
                    rx = s3_mod.compile_key_regex(cfg.key_regex)
                    df0["key_id"] = [(mm.group("id") if (mm := rx.search(k)) else None) for k in df0["key"]]
                raw = df0.to_csv(index=False, lineterminator="\n").encode()
                import hashlib

                sha = hashlib.sha256(raw).hexdigest()
                manifest = {"kind": "s3_listing", "connectionId": cfg.connection, "listing": {"bucket": c.bucket, "prefix": cfg.prefix, "maxObjects": cfg.max_objects,
                                                                                               "truncated": truncated, "keyRegex": cfg.key_regex or None,
                                                                                               "endpoint": c.row["settings"]["endpoint_url"]},
                            "takenAt": snapshots.now_iso(), "result": {"rows": len(df0), "columns": schema_of(df0.set_index(pd.RangeIndex(len(df0)))), "sha256": sha},
                            "reproducibility": {"level": "materialized_listing", "limited": True,
                                                "statement": "A prefix listing is a point-in-time view of mutable storage. The listing is stored with the snapshot and repeated runs read the stored copy; "
                                                             "the bucket's contents may differ now."}}
                sid = snapshots.record(ctx, manifest, raw)
                manifest, _ = snapshots.load(ctx.store, sid, want_extract=False)  # the stored form (includes the extract reference)
                mode = "live"
        except SourceError as e:
            raise _exx(e)
        df = pd.read_csv(io.BytesIO(raw), dtype={"key": "str", "etag": "str", "last_modified": "str", **({"key_id": "str"} if "key_id" in schema_of(pd.read_csv(io.BytesIO(raw), nrows=0)) else {})})
        df.index = pd.RangeIndex(len(df), name="row_id")
        lst = manifest["listing"]
        uri = f"s3://{lst['bucket']}/{lst['prefix']}"
        t = Table(df, "full", _lineage(ctx.node_id, "s3_listing", uri, manifest["result"]["sha256"], sid, mode))
        summary = {"connector": "s3", "mode": mode, "snapshotId": sid, "snapshot": manifest, "rows": len(df), "schema": schema_of(df), "sha256": manifest["result"]["sha256"],
                   "path": uri, "bytes": int(df["size"].sum()) if len(df) else 0, "reproducibility": manifest["reproducibility"], "preview": sample_rows(df),
                   "transfer": {"from": "S3 listing (metadata only; object bodies are not read)" if mode == "live" else "stored listing (no source access)",
                                "to": "worker process memory", "rows": len(df), "bytes": len(raw)},
                   "warnings": (["Listing truncated at max_objects"] if lst["truncated"] else [])}
        return {"table": t}, clean(summary)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "table = list_objects_v2(prefix) -> (key, size, etag, last_modified[, key_id])",
                "rule": "Object metadata only; bodies are not read. key_id is extracted by the named group (?P<id>...) of the regex, when given.",
                "note": "The listing is stored with the snapshot (a prefix is mutable storage) and flagged as limited reproducibility."}


# ================================================================================================ DVC
class DvcCsvConfig(StrictConfig):
    connection: str = ""
    path: str = Field("", description="Path of the DVC-tracked file inside the repository")
    rev: str | None = Field(None, description="Branch, tag or commit (resolved to a commit at run time and recorded)")
    delimiter: str = Field(",", min_length=1, max_length=1)
    pin: str | None = None


@register
class DvcCsvSource(TabularOperation):
    type = "dvc.csv_source"
    backend = "dvc"
    inputs = ()
    outputs = ("table",)
    out_kinds = {"table": TABLE}
    Config = DvcCsvConfig
    summary_kind = "connector_source"

    def infer(self, cfg: DvcCsvConfig, ins, node_id):
        if not cfg.connection:
            raise OpError("E_SRC_CONNECTION_NOT_FOUND", "No connection chosen.", None, list(_SRC_FIX))
        if not cfg.path:
            raise OpError("E_SOURCE_NOT_SET", "No dataset path chosen.", None, list(_SRC_FIX))
        try:
            if cfg.pin:
                m, _ = snapshots.load(_store(), cfg.pin, want_extract=False)
                if m.get("kind") != "dvc_file" or m["file"]["path"] != cfg.path:
                    raise SourceError("E_SRC_SNAPSHOT_MISMATCH", f"Pinned snapshot {cfg.pin[:12]}... is for a different path.")
                cols, rows = m["result"]["columns"], m["result"]["rows"]
            else:
                def probe():
                    raw, meta = dvc_mod.read_file(_registry(), cfg.connection, cfg.rev, cfg.path, head_bytes=65536)
                    text = raw.decode("utf-8", errors="replace")
                    if meta["size"] and meta["size"] > len(raw) and "\n" in text:
                        text = text[: text.rfind("\n") + 1]
                    return schema_of(pd.read_csv(io.StringIO(text), sep=cfg.delimiter, nrows=200)), meta["commit"]
                (cols, _c), rows = _cached(("dvc", cfg.connection, cfg.rev, cfg.path), probe), None
        except SourceError as e:
            raise _opx(e)
        except Exception as e:  # noqa: BLE001
            raise OpError("E_SRC_QUERY_INVALID", f"Cannot parse '{cfg.path}' as CSV: {e}")
        return {"table": table_type(cols, rows, "full", True, rowsExact=rows is not None, sourceNode=node_id, sourceKind="dvc", connectionId=cfg.connection)}

    def execute(self, cfg: DvcCsvConfig, ins, ctx: ExecCtx):
        try:
            pin = snapshots.pinned_for(cfg.pin, ctx)
            if pin:
                m, _ = snapshots.load(ctx.store, pin, want_extract=False)
                f = m["file"]
                if m.get("kind") != "dvc_file" or f["path"] != cfg.path:
                    raise SourceError("E_SRC_SNAPSHOT_MISMATCH", f"Pinned snapshot {pin[:12]}... is for a different path.")
                raw, meta = dvc_mod.read_file(_registry(), m["connectionId"], f["commit"], f["path"])
                if meta["dvcMd5"] != f["dvcMd5"] or meta["sha256"] != f["sha256"]:
                    raise SourceError("E_SRC_SNAPSHOT_MISMATCH", "The data at the pinned commit no longer has the recorded md5/content hash.")
                snapshots.reference(ctx, pin)
                manifest, sid, mode = m, pin, "pinned"
            else:
                if not cfg.connection or not cfg.path:
                    raise SourceError("E_SRC_CONNECTION_NOT_FOUND", "Connection and path are required.")
                raw, meta = dvc_mod.read_file(_registry(), cfg.connection, cfg.rev, cfg.path)
                df0 = _csv_frame(raw, cfg.delimiter)
                manifest = {"kind": "dvc_file", "connectionId": cfg.connection, "file": meta, "delimiter": cfg.delimiter, "takenAt": snapshots.now_iso(),
                            "result": {"rows": len(df0), "columns": schema_of(df0), "sha256": meta["sha256"]},
                            "reproducibility": {"level": "pinned_git_commit_and_dvc_md5", "limited": False,
                                                "statement": "Pinned to the resolved Git commit and the DVC md5 of the file; repeating re-materializes that exact content from the DVC cache or remote "
                                                             "and verifies md5 and SHA-256. Hashes verify identity; they do not keep the data if the remote loses it."}}
                sid = snapshots.record(ctx, manifest, None)
                manifest, _ = snapshots.load(ctx.store, sid, want_extract=False)  # the stored form (includes the extract reference)
                mode = "live"
        except SourceError as e:
            raise _exx(e)
        df = _csv_frame(raw, manifest.get("delimiter", cfg.delimiter))
        f = manifest["file"]
        uri = f"dvc://{f['repo']}@{f['commit'][:12]}/{f['path']}"
        t = Table(df, "full", _lineage(ctx.node_id, "dvc_file", uri, f["sha256"], sid, mode))
        summary = {"connector": "dvc", "mode": mode, "snapshotId": sid, "snapshot": manifest, "file": f, "rows": len(df), "schema": schema_of(df), "sha256": f["sha256"],
                   "path": uri, "bytes": f["size"], "reproducibility": manifest["reproducibility"], "preview": sample_rows(df),
                   "transfer": {"from": f"DVC cache/remote of {f['repo']}", "to": "worker process memory", "rows": len(df), "bytes": f["size"]}}
        return {"table": t}, clean(summary)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "table = pandas.read_csv(DVCFileSystem(repo, rev=commit).open(path))",
                "rule": "A branch or tag is resolved to a commit once per run. The repository, commit, DVC md5 of the file and the content SHA-256 are recorded.",
                "note": "A .dvc file is a pointer; the content is materialized through DVC (cache or remote) and verified."}


# ================================================================================================ cross-source join
class JoinConfig(StrictConfig):
    left_on: list[str] = Field(default_factory=list, min_length=0)
    right_on: list[str] = Field(default_factory=list)
    how: Literal["inner", "left", "right", "outer"] = "inner"
    expect: Literal["any", "one_to_one", "one_to_many", "many_to_one"] = Field("any", title="expected key cardinality (error if violated)")
    suffix_left: str = "_left"
    suffix_right: str = "_right"
    unmatched_sample: int = Field(10, ge=0, le=100)


def _src_of(t: Table | VType) -> list[dict[str, Any]]:
    if isinstance(t, Table):
        l = t.lineage
        return l.get("sources") or ([l["source"]] if l.get("source") else [])
    return []


@register
class Join(TabularOperation):
    type = "tabular.join"
    backend = "pandas"
    inputs = ("left", "right")
    outputs = ("table",)
    in_kinds = {"left": TABLE, "right": TABLE}
    out_kinds = {"table": TABLE}
    Config = JoinConfig
    summary_kind = "join"

    def infer(self, cfg: JoinConfig, ins, node_id):
        l, r = ins["left"], ins["right"]
        if not cfg.left_on or len(cfg.left_on) != len(cfg.right_on):
            raise OpError("E_JOIN_KEYS", "Give the same number (at least one) of left and right key columns.", None,
                          [Fix("Set left_on and right_on to matching lists of columns")])
        need_columns(l, cfg.left_on, "left", f"Join '{node_id}' (left)")
        need_columns(r, cfg.right_on, "right", f"Join '{node_id}' (right)")
        for a, b in zip(cfg.left_on, cfg.right_on):
            da, db = l.dtype_of(a), r.dtype_of(b)
            num = ("int", "float")
            if da != db and not (da in num and db in num):
                raise OpError("E_JOIN_KEY_TYPE", f"Key '{a}' is {da} on the left but '{b}' is {db} on the right; nothing is coerced silently. Cast one side with a typed column selection.",
                              "right", [Fix("Cast the key column on one side to the same type")])
        lc, rc = l.columns, r.columns
        right_keys_dropped = {b for a, b in zip(cfg.left_on, cfg.right_on) if a == b}
        lnames = {c["name"] for c in lc}
        rnames = {c["name"] for c in rc if c["name"] not in right_keys_dropped}
        out = []
        for c in lc:
            out.append({"name": c["name"] + (cfg.suffix_left if c["name"] in rnames else ""), "dtype": c["dtype"]})
        for c in rc:
            if c["name"] in right_keys_dropped:
                continue
            out.append({"name": c["name"] + (cfg.suffix_right if c["name"] in lnames else ""), "dtype": c["dtype"]})
        names = [c["name"] for c in out]
        if len(set(names)) != len(names):
            raise OpError("E_JOIN_COLUMN_CLASH", "Suffixes do not make the output column names unique.", None, [Fix("Change suffix_left / suffix_right")])
        return {"table": table_type(out, None, "full", l.complete and r.complete, rowsExact=False, sourceNode=None)}

    def execute(self, cfg: JoinConfig, ins, ctx: ExecCtx):
        L: Table = ins["left"]
        R: Table = ins["right"]
        ldf, rdf = L.df.reset_index(drop=True), R.df.reset_index(drop=True)
        lk, rk = cfg.left_on, cfg.right_on

        def stats(df, keys):
            nn = df[keys].notna().all(axis=1)
            sub = df.loc[nn, keys]
            dist = len(sub.drop_duplicates())
            return {"rows": len(df), "nullKeyRows": int((~nn).sum()), "distinctKeys": dist, "duplicateKeyRows": int(len(sub) - dist), "unique": len(sub) == dist}

        ls, rs = stats(ldf, lk), stats(rdf, rk)
        card = {(True, True): "one_to_one", (True, False): "one_to_many", (False, True): "many_to_one", (False, False): "many_to_many"}[(ls["unique"], rs["unique"])]
        if cfg.expect != "any" and card != cfg.expect:
            raise ExecutionError("E_JOIN_CARDINALITY", f"Expected a {cfg.expect} join but the keys are {card} (left: {ls['duplicateKeyRows']} duplicate-key rows, "
                                                       f"right: {rs['duplicateKeyRows']} duplicate-key rows). Fix the data or set 'expect' to 'any'.")
        lkeys = pd.MultiIndex.from_frame(ldf[lk]) if len(lk) > 1 else pd.Index(ldf[lk[0]])
        rkeys = pd.MultiIndex.from_frame(rdf[rk]) if len(rk) > 1 else pd.Index(rdf[rk[0]])
        l_nn, r_nn = ldf[lk].notna().all(axis=1).to_numpy(), rdf[rk].notna().all(axis=1).to_numpy()
        l_un = l_nn & ~np.asarray(lkeys.isin(rkeys))
        r_un = r_nn & ~np.asarray(rkeys.isin(lkeys))
        # pandas would match NaN keys to each other; SQL semantics (and this op's contract) say NULL never matches. Match on non-null keys only,
        # then append the null-key rows of the preserved side(s) after the matched rows.
        kw = dict(left_on=lk, right_on=rk, suffixes=(cfg.suffix_left, cfg.suffix_right))
        lnn, lnul, rnn, rnul = ldf[l_nn], ldf[~l_nn], rdf[r_nn], rdf[~r_nn]
        parts = [lnn.merge(rnn, how=cfg.how, **kw)]
        if cfg.how in ("left", "outer") and len(lnul):
            parts.append(lnul.merge(rnn.iloc[0:0], how="left", **kw))
        if cfg.how in ("right", "outer") and len(rnul):
            parts.append(lnn.iloc[0:0].merge(rnul, how="right", **kw))
        merged = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0].reset_index(drop=True)
        merged.index = pd.RangeIndex(len(merged), name="row_id")
        sample_n = cfg.unmatched_sample
        summary = {
            "how": cfg.how, "leftKeys": lk, "rightKeys": rk, "cardinality": card, "expected": cfg.expect,
            "left": {**ls, "unmatchedRows": int(l_un.sum()), "unmatchedSample": [{"rowId": int(L.df.index[i]), "key": clean(ldf.loc[i, lk].tolist())} for i in np.flatnonzero(l_un)[:sample_n]]},
            "right": {**rs, "unmatchedRows": int(r_un.sum()), "unmatchedSample": [{"rowId": int(R.df.index[i]), "key": clean(rdf.loc[i, rk].tolist())} for i in np.flatnonzero(r_un)[:sample_n]]},
            "resultRows": len(merged), "rowMultiplication": (len(merged) / ls["rows"]) if ls["rows"] else None,
            "rowsMultiplied": cfg.how in ("inner", "left") and len(merged) > ls["rows"],
            "execution": {"where": "worker process (pandas.merge on in-memory tables)", "pushdown": False,
                          "dataMoved": [{"input": "left", "sources": _src_of(L), "rows": len(ldf)}, {"input": "right", "sources": _src_of(R), "rows": len(rdf)}],
                          "note": "Both inputs were read into the worker; the join did not run inside any source system."},
            "lineage": {"left": _src_of(L), "right": _src_of(R)},
            "schema": schema_of(merged), "preview": sample_rows(merged),
            "semantics": "keys compare with ==; NULL/NaN keys never match; no case or whitespace normalization; no type coercion",
        }
        sources = _src_of(L) + _src_of(R)
        lin = {"sources": sources, "join": {"node": ctx.node_id, "how": cfg.how, "cardinality": card, "resultRows": len(merged),
                                            "unmatchedLeft": summary["left"]["unmatchedRows"], "unmatchedRight": summary["right"]["unmatchedRows"]}}
        return {"table": Table(merged, "full", lin)}, clean(summary)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "left.merge(right, how, left_on, right_on)",
                "rule": "Reports key uniqueness and cardinality on both sides, rows with no counterpart, and whether rows were multiplied. NULL keys never match; there is no silent type coercion.",
                "note": "Runs in the worker on tables already moved there; says where it executed and which sources were read."}
