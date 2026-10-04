"""Packaged, versioned, immutable library items ("My modules", saved code blocks).

A published (id, version) is never overwritten: publishing different content under an existing version is refused, so a project that pins a
version always gets the same definition. A project embeds a copy of each definition it uses, which is what the semantic hash covers."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
VERSION_RE = re.compile(r"^\d{1,4}\.\d{1,4}\.\d{1,4}$")


class VersionConflict(Exception):
    pass


def canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(data: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(data).encode("utf-8")).hexdigest()


def version_key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


class LibraryStore:
    def __init__(self, workbench: str | Path, kind: str):
        self.kind = kind
        self.root = Path(workbench) / "library" / kind
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, item_id: str, version: str) -> Path:
        if not ID_RE.match(item_id) or not VERSION_RE.match(version):
            raise ValueError(f"invalid id/version: {item_id!r} {version!r} (id: letters, digits, underscore; version: MAJOR.MINOR.PATCH)")
        return self.root / item_id / f"{version}.json"

    def publish(self, definition: dict[str, Any], note: str = "") -> dict[str, Any]:
        item_id, version = definition["id"], definition.get("version", "1.0.0")
        p = self._path(item_id, version)
        h = content_hash(definition)
        if p.exists():
            existing = json.loads(p.read_text())
            if existing["contentHash"] == h:
                return {k: v for k, v in existing.items() if k != "definition"} | {"alreadyPublished": True}
            raise VersionConflict(f"{self.kind} '{item_id}' version {version} is already published with different content "
                                  f"({existing['contentHash'][:12]} vs {h[:12]}). Published versions are immutable: publish a new version.")
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"id": item_id, "version": version, "contentHash": h, "publishedAt": time.time(), "note": note, "definition": definition}
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        os.replace(tmp, p)
        return {k: v for k, v in rec.items() if k != "definition"} | {"alreadyPublished": False}

    def versions(self, item_id: str) -> list[str]:
        d = self.root / item_id
        return sorted((p.stem for p in d.glob("*.json")), key=version_key) if d.is_dir() else []

    def get(self, item_id: str, version: str | None = None) -> dict[str, Any] | None:
        version = version or (self.versions(item_id) or [None])[-1]
        if version is None:
            return None
        p = self._path(item_id, version)
        return json.loads(p.read_text()) if p.exists() else None

    def list(self) -> list[dict[str, Any]]:
        out = []
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for v in self.versions(d.name):
                rec = json.loads((d / f"{v}.json").read_text())
                df = rec["definition"]
                out.append({"id": rec["id"], "version": rec["version"], "contentHash": rec["contentHash"], "publishedAt": rec["publishedAt"], "note": rec.get("note", ""),
                            "description": df.get("description", ""), "inputs": [p["name"] for p in df.get("inputs", [])],
                            "outputs": [p["name"] for p in df.get("outputs", [])]})
        return out

    @staticmethod
    def next_version(v: str, bump: str = "minor") -> str:
        a, b, c = version_key(v)
        return {"major": f"{a + 1}.0.0", "minor": f"{a}.{b + 1}.0", "patch": f"{a}.{b}.{c + 1}"}[bump]
