"""Repository source browsing and pinned code import (A44). Browsing never executes repository code; see python/repos/core.py.

POST /api/repos/resolve            {url, rev}                      fetch into a bare mirror, resolve rev -> commit
GET  /api/repos/{id}/tree          ?commit=                         classified entries, declared dependencies, license
GET  /api/repos/{id}/file          ?commit=&path=                   raw blob text (bounded), blob id and sha256
GET  /api/repos/{id}/python        ?commit=&path=                   ast-only: functions, imports, wrappability
POST /api/repos/{id}/compare       {base, head, paths}              changed paths and patch between two pinned commits
POST /api/repos/{id}/import        {commit, path, function, interface, pins}  immutable import record + code block definition
GET  /api/repos/imports            list; GET /api/repos/imports/{importId} one record (integrity checked)
POST /api/repos/origin-status      {block}                          is a block still the unmodified pinned import?"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from repos import RepoError, Repos, origin_status


class ResolveReq(BaseModel):
    model_config = {"extra": "forbid"}
    url: str
    rev: str = "HEAD"


class CompareReq(BaseModel):
    model_config = {"extra": "forbid"}
    base: str
    head: str
    paths: list[str] = Field(default_factory=list)


class ImportReq(BaseModel):
    model_config = {"extra": "forbid"}
    commit: str
    path: str
    function: str
    interface: dict[str, Any]
    pins: list[str] = Field(default_factory=list)


class BlockReq(BaseModel):
    block: dict[str, Any]


def register(app: FastAPI, sv: Any) -> None:
    repos = Repos(sv.store.root)

    def call(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except RepoError as e:
            raise HTTPException(e.status, {"code": e.code, "message": e.message}) from e

    @app.post("/api/repos/resolve")
    def resolve(req: ResolveReq):
        return call(repos.resolve, req.url, req.rev)

    @app.get("/api/repos/imports")
    def imports():
        return {"imports": repos.list_imports()}

    @app.get("/api/repos/imports/{import_id}")
    def get_import(import_id: str):
        return call(repos.get_import, import_id)

    @app.post("/api/repos/origin-status")
    def status(req: BlockReq):
        return {"origin": origin_status(req.block)}

    @app.get("/api/repos/{repo_id}/tree")
    def tree(repo_id: str, commit: str):
        return call(repos.tree, repo_id, commit)

    @app.get("/api/repos/{repo_id}/file")
    def file(repo_id: str, commit: str, path: str):
        return call(repos.read, repo_id, commit, path)

    @app.get("/api/repos/{repo_id}/python")
    def python(repo_id: str, commit: str, path: str):
        return call(repos.inspect_python, repo_id, commit, path)

    @app.post("/api/repos/{repo_id}/compare")
    def compare(repo_id: str, req: CompareReq):
        return call(repos.compare, repo_id, req.base, req.head, req.paths)

    @app.post("/api/repos/{repo_id}/import", status_code=201)
    def do_import(repo_id: str, req: ImportReq):
        return call(repos.import_function, repo_id, req.commit, req.path, req.function, req.interface, req.pins)
