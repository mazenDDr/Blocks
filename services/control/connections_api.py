"""Connections, data explorer (discover / preview), snapshots, compiled queries and the project bundle export.

Secrets are references: this module never returns, logs or stores a resolved secret value."""
from __future__ import annotations

import json
import time
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from connectors import dvc as dvc_mod
from connectors import postgres as pg
from connectors import s3 as s3_mod
from connectors import snapshots
from connectors.errors import SourceError
from connectors.pg_query import QuerySpec, compile_query, join_pairs
from graph_core.hashing import semantic_hash
from graph_core.project_io import load_project

from . import app as app_mod


class ConnectionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str = Field(..., min_length=1, max_length=120)
    type: Literal["postgres", "s3", "dvc"]
    settings: dict[str, Any] = Field(default_factory=dict)
    secrets: dict[str, Any] = Field(default_factory=dict, description="secret field -> reference ({kind:'env',name} | {kind:'file',path,key}); never a value")


class ConnectionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    settings: dict[str, Any] | None = None
    secrets: dict[str, Any] | None = None


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # postgres
    mode: Literal["visual", "sql"] = "visual"
    query: dict[str, Any] | None = None
    sql: str | None = None
    params: list[dict[str, Any]] = Field(default_factory=list)
    # s3 / dvc
    key: str | None = None
    versionId: str | None = None
    path: str | None = None
    rev: str | None = None
    delimiter: str = ","
    rows: int = Field(50, ge=1)
    timeoutMs: int = Field(5000, ge=100)
    maxBytes: int = Field(65536, ge=1024)


class CompileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: dict[str, Any]


def _spec(d: dict[str, Any] | None) -> QuerySpec:
    if d is None:
        raise SourceError("E_SRC_QUERY_INVALID", "A visual query needs 'query'.")
    try:
        return QuerySpec.model_validate(d)
    except ValidationError as e:
        raise SourceError("E_SRC_QUERY_INVALID", "Invalid query: " + "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()))


def _prov(cid: str, what: str) -> dict[str, Any]:
    return {"connectionId": cid, "source": what, "fetchedAt": time.time()}


def register(app: FastAPI, sv: "app_mod.Services") -> None:
    reg = sv.connections

    def need(cid: str, type_: str) -> dict[str, Any]:
        row = reg.require(cid)
        if row["type"] != type_:
            raise SourceError("E_SRC_UNSUPPORTED", f"This route is for {type_} connections; '{cid}' is {row['type']}.", connection=cid)
        return row

    # ------------------------------------------------------------------ registry
    @app.get("/api/connections")
    def list_connections():
        return {"connections": [reg.public(i) for i in reg.list_ids()],
                "note": "Secrets are stored as references (environment variable or a secrets file outside the project); values are never returned."}

    @app.post("/api/connections", status_code=201)
    def create_connection(req: ConnectionCreate):
        return reg.create(req.id, req.name, req.type, req.settings, req.secrets)

    @app.get("/api/connections/{cid}")
    def get_connection(cid: str):
        reg.require(cid)
        return reg.public(cid)

    @app.put("/api/connections/{cid}")
    def update_connection(cid: str, req: ConnectionUpdate):
        return reg.update(cid, req.name, req.settings, req.secrets)

    @app.delete("/api/connections/{cid}")
    def delete_connection(cid: str):
        users = _projects_using(cid)
        reg.delete(cid)
        return {"deleted": cid, "stillReferencedBy": users, "note": "Projects keep their connection id; they will report E_SRC_CONNECTION_NOT_FOUND until it is re-added." if users else None}

    def _projects_using(cid: str) -> list[str]:
        out = []
        for p in sorted(sv.projects.glob("*.project.json")):
            try:
                g = json.loads(p.read_text())
                if any(n.get("config", {}).get("connection") == cid for n in g.get("nodes", [])):
                    out.append(p.name[: -len(".project.json")])
            except (OSError, ValueError):
                continue
        return out

    @app.post("/api/connections/{cid}/test")
    def test_connection(cid: str):
        row = reg.require(cid)
        try:
            fn = {"postgres": pg.test_connection, "s3": s3_mod.test_connection, "dvc": dvc_mod.test_connection}[row["type"]]
            res = fn(reg, cid)
        except SourceError as e:  # e.g. an unresolved secret reference
            res = {"ok": False, "error": e.to_json()}
        res["testedAt"] = time.time()
        reg.record_test(cid, res)
        return {"connectionId": cid, "type": row["type"], **res}

    # ------------------------------------------------------------------ explorer
    @app.get("/api/connections/{cid}/discover")
    def discover(cid: str, schema: str | None = None, table: str | None = None):
        need(cid, "postgres")
        return {**pg.discover(reg, cid, schema, table), "provenance": _prov(cid, "PostgreSQL catalog")}

    @app.get("/api/connections/{cid}/objects")
    def objects(cid: str, prefix: str = "", token: str | None = None, maxKeys: int = Query(100, ge=1)):
        need(cid, "s3")
        return {**s3_mod.browse(reg, cid, prefix, token, maxKeys), "provenance": _prov(cid, "S3 listing")}

    @app.get("/api/connections/{cid}/object-versions")
    def object_versions(cid: str, key: str):
        need(cid, "s3")
        return {**s3_mod.versions(reg, cid, key), "provenance": _prov(cid, "S3 object versions")}

    @app.get("/api/connections/{cid}/revisions")
    def revisions(cid: str, limit: int = Query(30, ge=1)):
        need(cid, "dvc")
        return {**dvc_mod.revisions(reg, cid, limit), "provenance": _prov(cid, "git log")}

    @app.get("/api/connections/{cid}/tree")
    def tree(cid: str, rev: str | None = None, path: str = ""):
        need(cid, "dvc")
        return {**dvc_mod.ls(reg, cid, rev, path), "provenance": _prov(cid, "DVC file system at the resolved commit")}

    @app.post("/api/connections/{cid}/preview")
    def preview(cid: str, req: PreviewRequest):
        row = reg.require(cid)
        t = row["type"]
        if t == "postgres":
            r = pg.preview(reg, cid, mode=req.mode, spec=_spec(req.query) if req.mode == "visual" else None, sql=req.sql, params=req.params, rows=req.rows,
                           timeout_ms=req.timeoutMs)
        elif t == "s3":
            if not req.key:
                raise SourceError("E_SRC_QUERY_INVALID", "'key' is required for an S3 preview.")
            r = s3_mod.preview_object(reg, cid, req.key, req.versionId, req.rows, req.maxBytes)
        else:
            if not req.path:
                raise SourceError("E_SRC_QUERY_INVALID", "'path' is required for a DVC preview.")
            import io

            import pandas as pd

            from tabular.core import clean, schema_of

            rows = max(1, min(req.rows, 200))
            max_bytes = max(1024, min(req.maxBytes, s3_mod.MAX_PREVIEW_BYTES))
            raw, meta = dvc_mod.read_file(reg, cid, req.rev, req.path, head_bytes=max_bytes)
            text = raw.decode("utf-8", errors="replace")
            partial = bool(meta["size"] and meta["size"] > len(raw))
            if partial and "\n" in text:
                text = text[: text.rfind("\n") + 1]
            try:
                df = pd.read_csv(io.StringIO(text), sep=req.delimiter, nrows=rows)
            except Exception as e:  # noqa: BLE001
                raise SourceError("E_SRC_QUERY_INVALID", f"Cannot parse as CSV: {e}", resource=req.path, connection=cid)
            r = {"connectionId": cid, "path": req.path, "commit": meta["commit"], "dvcMd5": meta["dvcMd5"], "size": meta["size"], "columns": schema_of(df),
                 "rows": [[clean(v) for v in x] for x in df.itertuples(index=False, name=None)], "truncated": partial or len(df) >= rows,
                 "bounds": {"rows": rows, "maxBytes": max_bytes, "bytesRead": len(raw)}}
        return {**r, "provenance": _prov(cid, "bounded preview; not a recorded snapshot")}

    @app.post("/api/queries/compile")
    def compile_only(req: CompileRequest):
        """Compile a visual query to parameterized SQL without touching any database."""
        spec = _spec(req.query)
        comp = compile_query(spec)
        return {**comp.to_json(), "joins": [{"alias": j["alias"], "available": j["available"]} for j in join_pairs(spec)],
                "note": "Shown read-only: exactly this text is executed, with the listed parameters bound by the driver."}

    @app.post("/api/connections/{cid}/join-report")
    def join_report(cid: str, req: CompileRequest):
        need(cid, "postgres")
        spec = _spec(req.query)
        return {"connectionId": cid, "reports": pg.join_reports(reg, cid, spec), "provenance": _prov(cid, "counts computed in PostgreSQL")}

    # ------------------------------------------------------------------ snapshots
    @app.get("/api/snapshots/{sid}")
    def get_snapshot(sid: str):
        m, _ = snapshots.load(sv.store, sid, want_extract=False)
        return {"id": sid, "manifest": m, "provenance": {"source": "stored snapshot manifest", "sha256": sid}}

    @app.post("/api/snapshots/{sid}/check")
    def check_snapshot(sid: str):
        """Compare the recorded identity with the source as it is now. Never changes the snapshot."""
        m, _ = snapshots.load(sv.store, sid, want_extract=False)
        cid = m["connectionId"]
        kind = m["kind"]
        res: dict[str, Any] = {"snapshotId": sid, "kind": kind, "checkedAt": time.time()}
        if kind == "postgres":
            q = m["query"]
            if q["mode"] == "visual":
                r = pg.run_query(reg, cid, mode="visual", spec=_spec(q["spec"]), sql=None, params=None, limit=q["limit"])
            else:
                r = pg.run_query(reg, cid, mode="sql", spec=None, sql=q["text"], params=[{k: p[k] for k in ("name", "type", "value")} for p in q["params"]], limit=q["limit"])
            res.update(identical=r["resultSha256"] == m["result"]["sha256"], recordedSha256=m["result"]["sha256"], currentSha256=r["resultSha256"],
                       recordedRows=m["result"]["rows"], currentRows=len(r["df"]),
                       note="Same content hash means the live query currently returns the recorded extract." + ("" if m["result"].get("ordered") else " The query is unordered, so a different hash can also mean only a different row order."))
        elif kind == "s3_object":
            o = m["object"]
            c = s3_mod.S3(reg, cid)
            h = c.call(c.client.head_object, f"s3://{o['bucket']}/{o['key']}", Bucket=o["bucket"], Key=o["key"])
            cur_ver = h.get("VersionId") if h.get("VersionId") not in (None, "null") else None
            res.update(identical=(h["ETag"].strip('"') == o["etag"] and cur_ver == o["versionId"]), recordedEtag=o["etag"], currentEtag=h["ETag"].strip('"'),
                       recordedVersionId=o["versionId"], currentVersionId=cur_ver,
                       note="A newer object version exists; the pinned version is still readable." if (o["versionId"] and cur_ver != o["versionId"]) else None)
        elif kind == "dvc_file":
            f = m["file"]
            d = dvc_mod.Dvc(reg, cid)
            commit, info = dvc_mod.resolve_file(d, f["requestedRev"], f["path"])
            res.update(identical=(commit == f["commit"] and info["md5"] == f["dvcMd5"]), recordedCommit=f["commit"], currentCommit=commit, recordedMd5=f["dvcMd5"],
                       currentMd5=info["md5"], requestedRev=f["requestedRev"],
                       note="The requested revision now resolves to a different commit: this is a new source revision (the pinned one is unchanged)." if commit != f["commit"] else None)
        else:
            raise SourceError("E_SRC_UNSUPPORTED", f"Cannot check a {kind} snapshot against its source.")
        return res

    # ------------------------------------------------------------------ project bundle (A17)
    @app.get("/api/projects/{pid}/export/bundle")
    def export_bundle(pid: str):
        path = sv.project_path(pid)
        if not path.exists():
            raise HTTPException(404, {"code": "not_found", "message": f"no project '{pid}'"})
        p = load_project(path)
        conns, pins, files = {}, [], []
        for n in p.graph.nodes:
            cid = n.config.get("connection")
            if cid and cid not in conns:
                row = reg.get_row(cid)
                conns[cid] = ({"id": cid, "type": row["type"], "name": row["name"], "settings": row["settings"], "secretsRequired": sorted(row["secret_refs"]),
                               "status": "defined here; the recipient must define the same connection and supply its own secrets"} if row else
                              {"id": cid, "status": "not defined in this workbench"})
            if n.config.get("pin"):
                pins.append({"node": n.id, "snapshotId": n.config["pin"]})
            if n.type == "tabular.csv_source":
                files.append({"node": n.id, "path": n.config.get("path"), "status": "local file; not packaged - supply it or its identity"})
        return {"format": "project-void-bundle/1", "projectId": pid, "graphHash": semantic_hash(p.graph), "graph": p.graph.to_json(), "ui": p.ui.to_json() if p.ui else None,
                "requirements": {"connections": list(conns.values()), "pinnedSnapshots": pins, "localFiles": files},
                "secrets": "none: secret values and secret references (environment names, secrets-file paths) are never exported"}
