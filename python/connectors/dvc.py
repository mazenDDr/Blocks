"""DVC connector: resolve a revision of a Git+DVC repository to a commit, read the DVC-tracked file through DVC (cache or remote)."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .errors import SourceError

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_BYTES = 200_000_000
CAPABILITIES = {"discovery": True, "sampling": True, "filterProjectionPushdown": False, "streaming": False,
                "versionedReads": "git commit + DVC md5", "incrementalReads": False, "writes": False, "cancellation": False,
                "privateNetwork": "via the machine running the worker"}


def _git(args: list[str], cwd: str | None = None, timeout: int = 20) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1"}
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SourceError("E_SRC_NETWORK", f"git failed: {type(e).__name__}")
    if r.returncode != 0:
        raise SourceError("E_SRC_NOT_FOUND", (r.stderr.strip().splitlines() or ["git failed"])[0][:300])
    return r.stdout


def is_local(repo: str) -> bool:
    return not re.match(r"^[a-z][a-z0-9+.-]*://|^git@", repo)


class Dvc:
    def __init__(self, registry, connection_id: str):
        self.row = registry.require(connection_id)
        if self.row["type"] != "dvc":
            raise SourceError("E_SRC_UNSUPPORTED", f"Connection '{connection_id}' is a {self.row['type']} connection, not dvc.", connection=connection_id)
        s = self.row["settings"]
        self.repo = str(Path(s["repo"]).expanduser().resolve()) if is_local(s["repo"]) else s["repo"]
        self.remote = s["remote"]
        self.default_rev = s["default_rev"]
        if is_local(s["repo"]) and not (Path(self.repo) / ".git").exists():
            raise SourceError("E_SRC_NOT_FOUND", f"'{s['repo']}' is not a Git repository.", resource=s["repo"], connection=connection_id)

    def resolve(self, rev: str | None) -> str:
        rev = rev or self.default_rev or "HEAD"
        if SHA_RE.match(rev):
            return rev
        if is_local(self.repo):
            return _git(["rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"], self.repo).strip()
        out = _git(["ls-remote", self.repo, rev, f"{rev}^{{}}"], timeout=30).strip().splitlines()
        if not out:
            raise SourceError("E_SRC_NOT_FOUND", f"Revision '{rev}' not found in {self.repo}", connection=self.row["id"])
        return out[-1].split()[0]

    def fs(self, commit: str):
        from dvc.api import DVCFileSystem

        cfg = {"core": {"remote": self.remote}} if self.remote else None
        try:
            return DVCFileSystem(url=self.repo, rev=commit, config=cfg)
        except Exception as e:  # noqa: BLE001
            raise SourceError("E_SRC_ERROR", f"DVC cannot open the repository at {commit[:10]}: {type(e).__name__}: {str(e)[:200]}", connection=self.row["id"])


def test_connection(registry, connection_id: str) -> dict[str, Any]:
    t0 = time.time()
    try:
        d = Dvc(registry, connection_id)
        commit = d.resolve(None)
        d.fs(commit).ls("/", detail=False)
        return {"ok": True, "latencyMs": round((time.time() - t0) * 1000), "repo": d.repo, "head": commit, "capabilities": CAPABILITIES}
    except SourceError as e:
        return {"ok": False, "latencyMs": round((time.time() - t0) * 1000), "error": e.to_json(), "capabilities": CAPABILITIES}
    except Exception as e:  # noqa: BLE001
        err = SourceError("E_SRC_ERROR", f"{type(e).__name__}: {str(e)[:200]}", connection=connection_id)
        return {"ok": False, "latencyMs": round((time.time() - t0) * 1000), "error": err.to_json(), "capabilities": CAPABILITIES}


def revisions(registry, connection_id: str, limit: int = 30) -> dict[str, Any]:
    d = Dvc(registry, connection_id)
    if not is_local(d.repo):
        raise SourceError("E_SRC_UNSUPPORTED", "Listing revisions needs a local checkout; give the revision (branch, tag or commit) explicitly for remote repositories.", connection=connection_id)
    limit = max(1, min(limit, 100))
    log = _git(["log", f"-n{limit}", "--format=%H%x1f%an%x1f%aI%x1f%s", "--all"], d.repo).strip().splitlines()
    refs = _git(["for-each-ref", "--format=%(refname:short)%1f%(objecttype)%1f%(*objectname)%1f%(objectname)", "refs/heads", "refs/tags"], d.repo).strip().splitlines()
    names: dict[str, list[str]] = {}
    for line in refs:
        n, _, peeled, obj = line.split("\x1f")
        names.setdefault(peeled or obj, []).append(n)
    commits = []
    for line in log:
        h, a, dt, s = line.split("\x1f")
        commits.append({"commit": h, "author": a, "date": dt, "subject": s[:200], "refs": names.get(h, [])})
    return {"connectionId": connection_id, "repo": d.repo, "revisions": commits, "bounds": {"limit": limit}}


def ls(registry, connection_id: str, rev: str | None, path: str) -> dict[str, Any]:
    d = Dvc(registry, connection_id)
    commit = d.resolve(rev)
    fs = d.fs(commit)
    try:
        items = fs.ls("/" + path.strip("/"), detail=True)
    except FileNotFoundError:
        raise SourceError("E_SRC_NOT_FOUND", f"Path '{path}' does not exist at {commit[:10]}.", resource=path, connection=connection_id)
    out = []
    for it in sorted(items, key=lambda x: x["name"])[:200]:
        di = it.get("dvc_info") or {}
        out.append({"path": it["name"].lstrip("/"), "type": it.get("type"), "size": it.get("size"), "dvcTracked": bool(di.get("isdvc") or di.get("isout")),
                    "md5": di.get("md5")})
    return {"connectionId": connection_id, "rev": rev or d.default_rev or "HEAD", "commit": commit, "path": path, "entries": out, "bounds": {"maxEntries": 200}}


def resolve_file(d: Dvc, rev: str | None, path: str) -> tuple[str, dict[str, Any]]:
    commit = d.resolve(rev)
    fs = d.fs(commit)
    p = "/" + path.strip("/")
    try:
        info = fs.info(p)
    except FileNotFoundError:
        raise SourceError("E_SRC_NOT_FOUND", f"'{path}' does not exist at {commit[:10]}.", resource=path, connection=d.row["id"])
    if info.get("type") != "file":
        raise SourceError("E_SRC_UNSUPPORTED", f"'{path}' is a directory; choose a file.", resource=path, connection=d.row["id"])
    di = info.get("dvc_info") or {}
    return commit, {"md5": info.get("md5") or di.get("md5"), "size": info.get("size"), "dvcTracked": bool(di.get("isdvc") or di.get("isout")), "fs": fs}


def read_file(registry, connection_id: str, rev: str | None, path: str, max_bytes: int = MAX_BYTES, head_bytes: int | None = None) -> tuple[bytes, dict[str, Any]]:
    d = Dvc(registry, connection_id)
    commit, info = resolve_file(d, rev, path)
    if info["size"] and info["size"] > max_bytes:
        raise SourceError("E_SRC_BOUNDS", f"'{path}' is {info['size']} bytes; the limit is {max_bytes}.", resource=path, connection=connection_id)
    try:
        with info["fs"].open("/" + path.strip("/"), "rb") as f:
            raw = f.read(head_bytes) if head_bytes else f.read()
    except FileNotFoundError:
        raise SourceError("E_SRC_SNAPSHOT_UNAVAILABLE", f"The data for '{path}' at {commit[:10]} is neither in the DVC cache nor on the remote.", resource=path, connection=connection_id)
    except Exception as e:  # noqa: BLE001
        raise SourceError("E_SRC_ERROR", f"DVC could not materialize '{path}': {type(e).__name__}: {str(e)[:200]}", resource=path, connection=connection_id)
    meta = {"repo": d.repo, "requestedRev": rev or d.default_rev or "HEAD", "commit": commit, "path": path, "dvcMd5": info["md5"], "dvcTracked": info["dvcTracked"],
            "size": len(raw) if not head_bytes else info["size"], "sha256": hashlib.sha256(raw).hexdigest() if not head_bytes else None,
            "remote": d.remote}
    return raw, meta
