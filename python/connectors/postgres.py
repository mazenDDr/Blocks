"""PostgreSQL connector: connection test, discovery, bounded read-only execution, snapshot manifests, join reports."""
from __future__ import annotations

import hashlib
import io
import time
from datetime import date, datetime
from typing import Any

import pandas as pd
import psycopg
from psycopg import errors as pgerr

from .errors import SourceError, redact
from .pg_query import Compiled, QuerySpec, compile_query, join_pairs, q, typed_value, Typed

MAX_PREVIEW_ROWS = 200
MAX_TIMEOUT_MS = 30_000
DEFAULT_TIMEOUT_MS = 10_000
PROBE_TIMEOUT_MS = 5_000

INT_T = {"int2", "int4", "int8", "oid"}
FLOAT_T = {"float4", "float8", "numeric"}
TS_T = {"timestamp", "timestamptz"}


# ------------------------------------------------------------------------------------------------ errors
def classify(e: Exception, row: dict[str, Any] | None, resource: str | None = None, secrets: list[str] | None = None) -> SourceError:
    secrets = secrets or []
    cid = row["id"] if row else None
    if isinstance(e, SourceError):
        return e
    sqlstate = getattr(e, "sqlstate", None) or ""
    diag = getattr(e, "diag", None)
    msg = (getattr(diag, "message_primary", None) if diag else None) or str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    msg = redact(msg, secrets)
    kw = dict(resource=resource, connection=cid)
    if sqlstate.startswith("28"):
        return SourceError("E_SRC_AUTH", f"PostgreSQL rejected the login: {msg}", **kw)
    if sqlstate == "42501":
        return SourceError("E_SRC_PERMISSION", f"PostgreSQL denied access: {msg}", **kw)
    if sqlstate == "57014":
        return SourceError("E_SRC_QUERY_TIMEOUT", f"The statement was cancelled by the time bound: {msg}", **kw)
    if sqlstate in ("42P01", "42703", "3F000", "3D000", "42704"):
        return SourceError("E_SRC_NOT_FOUND", f"Not found: {msg}", **kw)
    if sqlstate == "25006":
        return SourceError("E_SRC_READONLY_VIOLATION", f"Read-only transaction: {msg}", **kw)
    if sqlstate in ("42883", "42804", "22P02", "22007", "22003"):
        return SourceError("E_SRC_TYPE_MISMATCH", f"Type problem: {msg}", **kw)
    if sqlstate.startswith("42") or sqlstate.startswith("22"):
        return SourceError("E_SRC_QUERY_INVALID", f"Invalid query: {msg}", **kw)
    low = msg.lower()
    if not sqlstate and "fatal:" in low:  # login-time failures carry no SQLSTATE in psycopg; the server's FATAL message is the only signal
        if "password authentication failed" in low or "does not exist" in low and "role" in low or "pg_hba.conf" in low or "authentication" in low:
            return SourceError("E_SRC_AUTH", f"PostgreSQL rejected the login: {msg}", **kw)
        if "database" in low and "does not exist" in low:
            return SourceError("E_SRC_NOT_FOUND", f"Not found: {msg}", **kw)
        if "permission denied" in low:
            return SourceError("E_SRC_PERMISSION", f"PostgreSQL denied access: {msg}", **kw)
    if isinstance(e, (psycopg.OperationalError, OSError)) and not sqlstate:
        return SourceError("E_SRC_NETWORK", f"Cannot reach PostgreSQL: {msg}", **kw)
    return SourceError("E_SRC_ERROR", f"PostgreSQL error ({type(e).__name__}): {msg}", **kw)


def connect(row: dict[str, Any], secret_values: dict[str, str]) -> psycopg.Connection:
    s = row["settings"]
    pw = secret_values.get("password")
    try:
        c = psycopg.connect(host=s["host"], port=s["port"], dbname=s["dbname"], user=s["user"], password=pw, sslmode=s["sslmode"],
                            connect_timeout=s["connect_timeout_s"], application_name="project-void", autocommit=False)
    except Exception as e:  # noqa: BLE001
        raise classify(e, row, f"postgres://{s['host']}:{s['port']}/{s['dbname']}", list(secret_values.values())) from None
    c.read_only = True
    c.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
    return c


class Session:
    """One read-only REPEATABLE READ transaction with a statement timeout; closes (rolls back) on exit."""

    def __init__(self, registry, connection_id: str, timeout_ms: int = DEFAULT_TIMEOUT_MS):
        self.row = registry.require(connection_id)
        if self.row["type"] != "postgres":
            raise SourceError("E_SRC_UNSUPPORTED", f"Connection '{connection_id}' is a {self.row['type']} connection, not postgres.", connection=connection_id)
        self.secrets = registry.secret_values(self.row)
        self.timeout_ms = max(100, min(int(timeout_ms), MAX_TIMEOUT_MS))
        self.conn: psycopg.Connection | None = None

    def __enter__(self) -> "Session":
        self.conn = connect(self.row, self.secrets)
        try:
            self.conn.execute(f"SET LOCAL statement_timeout = {int(self.timeout_ms)}")  # integer cast above: not a user-controlled string
        except Exception as e:  # noqa: BLE001
            raise classify(e, self.row, None, list(self.secrets.values())) from None
        return self

    def __exit__(self, *a) -> None:
        if self.conn is not None:
            try:
                self.conn.rollback()
            finally:
                self.conn.close()

    def execute(self, sql: str, params: list[Any] | None = None, resource: str | None = None):
        assert self.conn is not None
        try:
            cur = self.conn.cursor()
            cur.execute(sql, params)  # type: ignore[arg-type]
            return cur
        except Exception as e:  # noqa: BLE001
            raise classify(e, self.row, resource, list(self.secrets.values())) from None


# ------------------------------------------------------------------------------------------------ type mapping
def dtype_from_pg(name: str) -> str:
    if name in INT_T:
        return "int"
    if name in FLOAT_T:
        return "float"
    if name == "bool":
        return "bool"
    if name == "date" or name in TS_T:
        return "datetime"
    return "string"


def category_from_format_type(ft: str) -> str:
    f = ft.lower()
    if f in ("smallint", "integer", "bigint"):
        return "int"
    if f.startswith(("numeric", "decimal", "real", "double")):
        return "float"
    if f == "boolean":
        return "bool"
    if f == "date" or f.startswith("timestamp"):
        return "datetime"
    return "string"


def build_frame(session: Session, cur, rows: list[tuple]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    schema, data = [], {}
    types = session.conn.adapters.types  # type: ignore[union-attr]
    for i, col in enumerate(cur.description or []):
        ti = types.get(col.type_code)
        pgname = ti.name if ti else f"oid:{col.type_code}"
        dt = dtype_from_pg(pgname)
        vals = [r[i] for r in rows]
        if dt == "int":
            s = pd.Series(vals, dtype="float64" if any(v is None for v in vals) else "int64")
        elif dt == "float":
            s = pd.Series([float("nan") if v is None else float(v) for v in vals], dtype="float64")
        elif dt == "bool":
            s = pd.Series(vals, dtype=object if any(v is None for v in vals) else bool)
        elif dt == "datetime":
            s = pd.to_datetime(pd.Series(vals, dtype=object), utc=(pgname == "timestamptz"))
            if pgname == "timestamptz":
                s = s.dt.tz_convert(None)
        else:
            s = pd.Series([None if v is None else str(v) for v in vals], dtype="str")
        entry = {"name": col.name, "dtype": dt, "pgType": pgname}
        if pgname == "numeric":
            entry["conversion"] = "numeric -> float64 (precision beyond ~15 digits can be lost)"
        elif pgname == "timestamptz":
            entry["conversion"] = "timestamptz -> UTC, timezone dropped"
        elif dt == "string" and pgname not in ("text", "varchar", "bpchar", "name", "uuid"):
            entry["conversion"] = f"{pgname} -> text via str()"
        schema.append(entry)
        data[col.name] = s
    df = pd.DataFrame(data) if data else pd.DataFrame()
    df.index = pd.RangeIndex(len(df), name="row_id")
    return df, schema


def extract_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, lineterminator="\n").encode()


def read_extract(raw: bytes, schema: list[dict[str, Any]]) -> pd.DataFrame:
    dt = {c["name"]: ("int64" if c["dtype"] == "int" else "float64" if c["dtype"] == "float" else "str") for c in schema
          if c["dtype"] in ("int", "float", "string")}
    boolcols = [c["name"] for c in schema if c["dtype"] == "bool"]
    df = pd.read_csv(io.BytesIO(raw), dtype=dt, parse_dates=[c["name"] for c in schema if c["dtype"] == "datetime"])
    for b in boolcols:
        df[b] = df[b].astype(bool) if not df[b].isna().any() else df[b]
    df.index = pd.RangeIndex(len(df), name="row_id")
    return df


# ------------------------------------------------------------------------------------------------ operations
def server_info(s: Session) -> dict[str, Any]:
    cur = s.execute("SELECT version(), current_database(), pg_current_snapshot()::text, now(), "
                    "current_setting('transaction_read_only'), current_setting('transaction_isolation')")
    v, db, snap, now, ro, iso = cur.fetchone()
    return {"version": v, "database": db, "snapshotToken": snap, "serverTime": now.isoformat(), "readOnly": ro == "on", "isolation": iso}


def test_connection(registry, connection_id: str) -> dict[str, Any]:
    t0 = time.time()
    try:
        with Session(registry, connection_id, 5000) as s:
            info = server_info(s)
            cur = s.execute("SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.relkind IN ('r','v','m','p','f') "
                            "AND n.nspname NOT IN ('pg_catalog','information_schema') AND has_table_privilege(c.oid,'SELECT')")
            readable = cur.fetchone()[0]
        return {"ok": True, "latencyMs": round((time.time() - t0) * 1000), "serverVersion": info["version"], "database": info["database"],
                "readOnlyEnforced": info["readOnly"], "readableTables": readable,
                "capabilities": CAPABILITIES}
    except SourceError as e:
        return {"ok": False, "latencyMs": round((time.time() - t0) * 1000), "error": e.to_json(), "capabilities": CAPABILITIES}


CAPABILITIES = {"discovery": True, "sampling": True, "filterProjectionPushdown": True, "streaming": False, "versionedReads": "snapshot token + materialized extract",
                "incrementalReads": False, "writes": False, "cancellation": "statement timeout", "privateNetwork": "via the machine running the worker"}


def discover(registry, connection_id: str, schema: str | None = None, table: str | None = None) -> dict[str, Any]:
    with Session(registry, connection_id, 8000) as s:
        if table:
            if not schema:
                raise SourceError("E_SRC_QUERY_INVALID", "'schema' is required with 'table'.")
            cur = s.execute(
                "SELECT a.attname, format_type(a.atttypid, a.atttypmod), NOT a.attnotnull, has_column_privilege(c.oid, a.attnum, 'SELECT'), "
                "EXISTS (SELECT 1 FROM pg_index i WHERE i.indrelid = c.oid AND i.indisprimary AND a.attnum = ANY(i.indkey)) "
                "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = %s AND c.relname = %s AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attnum LIMIT 501", [schema, table], f"{schema}.{table}")
            cols = [{"name": n, "type": ft, "dtype": category_from_format_type(ft), "nullable": nl, "selectable": sel, "primaryKey": pk}
                    for n, ft, nl, sel, pk in cur.fetchall()]
            if not cols:
                raise SourceError("E_SRC_NOT_FOUND", f"Table {schema}.{table} does not exist or is not visible.", resource=f"{schema}.{table}", connection=connection_id)
            return {"connectionId": connection_id, "schema": schema, "table": table, "columns": cols[:500], "truncated": len(cols) > 500}
        cur = s.execute(
            "SELECT n.nspname, c.relname, c.relkind::text, c.reltuples::bigint, has_table_privilege(c.oid, 'SELECT') "
            "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE c.relkind IN ('r','v','m','p','f') AND n.nspname NOT IN ('pg_catalog','information_schema') AND n.nspname NOT LIKE 'pg\\_toast%%' "
            "AND has_schema_privilege(n.oid, 'USAGE') AND (%s::text IS NULL OR n.nspname = %s) ORDER BY 1, 2 LIMIT 501", [schema, schema])
        kinds = {"r": "table", "v": "view", "m": "materialized view", "p": "partitioned table", "f": "foreign table"}
        tables = [{"schema": n, "name": t, "kind": kinds.get(k, k), "estimatedRows": (r if r >= 0 else None), "selectable": sel}
                  for n, t, k, r, sel in cur.fetchall()]
        return {"connectionId": connection_id, "tables": tables[:500], "truncated": len(tables) > 500,
                "note": "estimatedRows come from planner statistics (reltuples), not an exact count; only schemas/tables visible to these credentials are listed"}


_probe_cache: dict[tuple, tuple[float, Any]] = {}


def probe_schema(registry, connection_id: str, c: Compiled | str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    """Result column names/types of a query without reading rows (`LIMIT 0`), cached for 30 s. Used by static validation."""
    sql, ps = (c.sql, c.params) if isinstance(c, Compiled) else (c, params or [])
    row = registry.require(connection_id)
    key = (connection_id, row["updated_at"], sql, repr(ps))
    hit = _probe_cache.get(key)
    if hit and time.time() - hit[0] < 30:
        return hit[1]
    wrapped = f"SELECT * FROM ({_strip(sql)}) AS q LIMIT 0" if not isinstance(c, Compiled) else c.sql.rsplit("LIMIT %s", 1)[0] + "LIMIT 0"
    wrapped_params = ps if not isinstance(c, Compiled) else ps[:-1]
    with Session(registry, connection_id, PROBE_TIMEOUT_MS) as s:
        cur = s.execute(wrapped, wrapped_params)
        types = s.conn.adapters.types  # type: ignore[union-attr]
        out = []
        for col in cur.description or []:
            ti = types.get(col.type_code)
            name = ti.name if ti else f"oid:{col.type_code}"
            out.append({"name": col.name, "dtype": dtype_from_pg(name), "pgType": name})
    _probe_cache[key] = (time.time(), out)
    return out


def _strip(sql: str) -> str:
    return sql.strip().rstrip(";").strip()


def sql_params(params: list[dict[str, Any]] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    out, desc = {}, []
    for p in params or []:
        v = typed_value(Typed(type=p["type"], value=p["value"]))
        out[p["name"]] = v
        desc.append({"name": p["name"], "type": p["type"], "value": v.isoformat() if isinstance(v, (date, datetime)) else v})
    return out, desc


def query_identity(mode: str, spec: QuerySpec | None, sql: str | None, params: list[dict[str, Any]] | None, limit: int):
    """(query text, parameter list as recorded, compiled-or-None, text to execute, parameters to bind). Pure: no connection."""
    if mode == "visual":
        if spec is None:
            raise SourceError("E_SRC_QUERY_INVALID", "A visual query needs a query specification.")
        comp = compile_query(spec)
        shown = comp.to_json()["params"]
        pn = [{"position": i + 1, "type": t, "value": v} for i, (t, v) in enumerate(zip(comp.param_types, shown))]
        return comp.sql, pn, comp, comp.sql, comp.params
    if not sql or not sql.strip():
        raise SourceError("E_SRC_QUERY_INVALID", "SQL text is empty.")
    text = _strip(sql)
    named, pn = sql_params(params)
    return text, pn, None, f"SELECT * FROM ({text}) AS q LIMIT {int(limit) + 1}", (named or None)


def run_query(registry, connection_id: str, *, mode: str, spec: QuerySpec | None, sql: str | None, params: list[dict[str, Any]] | None,
              limit: int, timeout_ms: int = DEFAULT_TIMEOUT_MS, session: Session | None = None, with_join_reports: bool = False) -> dict[str, Any]:
    """Execute once in a read-only snapshot transaction. Returns frame + schema + the exact query identity (text, params, token)."""
    text, pnames, comp, run_sql, run_params = query_identity(mode, spec, sql, params, limit)
    ordered = comp.ordered if comp else " order by " in f" {text.lower()} "
    limit_used = spec.limit if (mode == "visual" and spec) else limit
    reach_target = limit_used
    own = session is None
    s = session or Session(registry, connection_id, timeout_ms)
    if own:
        s.__enter__()
    try:
        info = server_info(s)
        t0 = time.time()
        cur = s.execute(run_sql, run_params, "query")
        rows = cur.fetchall()
        elapsed = round((time.time() - t0) * 1000)
        truncated = False
        if mode != "visual" and len(rows) > limit:
            rows, truncated = rows[:limit], True
        df, schema = build_frame(s, cur, rows)
        srv_settings = s.row["settings"]
        reports = join_reports(registry, connection_id, spec, s) if (with_join_reports and mode == "visual" and spec and spec.joins) else []
    finally:
        if own:
            s.__exit__(None, None, None)
    raw = extract_bytes(df)
    return {"df": df, "schema": schema, "extract": raw, "elapsedMs": elapsed, "server": {**info, "host": srv_settings["host"], "port": srv_settings["port"]},
            "query": {"mode": mode, "text": text, "params": pnames, "limit": limit_used, "spec": spec.model_dump(mode="json", by_alias=True) if (mode == "visual" and spec) else None},
            "ordered": ordered, "truncated": truncated, "joinReports": reports, "reachedLimit": len(df) >= reach_target, "resultSha256": hashlib.sha256(raw).hexdigest()}


def preview(registry, connection_id: str, *, mode: str, spec: QuerySpec | None, sql: str | None, params: list[dict[str, Any]] | None,
            rows: int = 50, timeout_ms: int = 5000) -> dict[str, Any]:
    """Bounded preview: row limit enforced in SQL (the source), time limit by statement_timeout, read-only transaction."""
    rows = max(1, min(int(rows), MAX_PREVIEW_ROWS))
    timeout_ms = max(100, min(int(timeout_ms), MAX_TIMEOUT_MS))
    if mode == "visual":
        assert spec is not None
        spec = spec.model_copy(update={"limit": min(spec.limit, rows)})
    r = run_query(registry, connection_id, mode=mode, spec=spec, sql=sql, params=params, limit=rows, timeout_ms=timeout_ms)
    df = r["df"]
    from tabular.core import clean

    return {"connectionId": connection_id, "columns": r["schema"], "rows": [[clean(v if not isinstance(v, (pd.Timestamp, datetime, date)) else str(v)) for v in row]
                                                                           for row in df.itertuples(index=False, name=None)],
            "rowCount": len(df), "truncated": r["truncated"] or r["reachedLimit"], "elapsedMs": r["elapsedMs"],
            "bounds": {"rowLimit": rows, "timeoutMs": timeout_ms, "readOnly": True, "isolation": "REPEATABLE READ"},
            "query": r["query"], "serverVersion": r["server"]["version"]}


def join_reports(registry, connection_id: str, spec: QuerySpec, session: Session | None = None) -> list[dict[str, Any]]:
    """Key cardinality and unmatched records for each join, computed in the database on the unfiltered tables."""
    reports = []
    own = session is None
    s = session or Session(registry, connection_id, 20_000)
    if own:
        s.__enter__()
    try:
        for jp in join_pairs(spec):
            if not jp["available"]:
                reports.append(jp)
                continue
            L, R = jp["left"], jp["right"]
            lt, rt = f"{q(L.schema_)}.{q(L.name)}", f"{q(R.schema_)}.{q(R.name)}"
            lk, rk = jp["leftKeys"], jp["rightKeys"]
            nn = lambda cols, a: " AND ".join(f"{a}.{q(c)} IS NOT NULL" for c in cols)  # noqa: E731
            anyn = lambda cols, a: " OR ".join(f"{a}.{q(c)} IS NULL" for c in cols)  # noqa: E731
            eq = " AND ".join(f"r.{q(b)} = l.{q(a)}" for a, b in zip(lk, rk))
            eq2 = " AND ".join(f"l.{q(a)} = r.{q(b)}" for a, b in zip(lk, rk))
            sql = (f"SELECT (SELECT count(*) FROM {lt} l), (SELECT count(*) FROM {lt} l WHERE {anyn(lk, 'l')}), "
                   f"(SELECT count(*) FROM (SELECT DISTINCT {', '.join(f'l.{q(c)}' for c in lk)} FROM {lt} l WHERE {nn(lk, 'l')}) d), "
                   f"(SELECT count(*) FROM {rt} r), (SELECT count(*) FROM {rt} r WHERE {anyn(rk, 'r')}), "
                   f"(SELECT count(*) FROM (SELECT DISTINCT {', '.join(f'r.{q(c)}' for c in rk)} FROM {rt} r WHERE {nn(rk, 'r')}) d), "
                   f"(SELECT count(*) FROM {lt} l WHERE {nn(lk, 'l')} AND NOT EXISTS (SELECT 1 FROM {rt} r WHERE {eq})), "
                   f"(SELECT count(*) FROM {rt} r WHERE {nn(rk, 'r')} AND NOT EXISTS (SELECT 1 FROM {lt} l WHERE {eq2}))")
            ln, lnull, ldist, rn, rnull, rdist, lun, run_ = s.execute(sql, None, f"join {jp['alias']}").fetchone()
            lnn, rnn = ln - lnull, rn - rnull
            lu, ru = lnn == ldist, rnn == rdist
            card = {(True, True): "one_to_one", (True, False): "one_to_many", (False, True): "many_to_one", (False, False): "many_to_many"}[(lu, ru)]
            reports.append({"available": True, "alias": jp["alias"], "joinType": jp["type"], "left": f"{L.schema_}.{L.name}", "right": f"{R.schema_}.{R.name}",
                            "leftKeys": lk, "rightKeys": rk, "cardinality": card, "leftRows": ln, "leftNullKeys": lnull, "leftDistinctKeys": ldist,
                            "leftDuplicateKeyRows": lnn - ldist, "rightRows": rn, "rightNullKeys": rnull, "rightDistinctKeys": rdist,
                            "rightDuplicateKeyRows": rnn - rdist, "unmatchedLeftRows": lun, "unmatchedRightRows": run_,
                            "rowMultiplicationPossible": not ru,
                            "executedIn": "PostgreSQL (counts over the unfiltered tables" + (", same snapshot transaction as the extract)" if not own else ")")})
    finally:
        if own:
            s.__exit__(None, None, None)
    return reports
