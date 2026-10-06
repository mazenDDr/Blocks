"""Connection registry (SQLite `connections.db` in the workbench): non-secret settings + secret *references*."""
from __future__ import annotations

import json
import re
import sqlite3
from storage.schema import enable_wal, open_database
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import secrets as sec
from .errors import SourceError

ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PostgresSettings(_Strict):
    host: str = Field(..., min_length=1, description="Host name, or a directory for a Unix-domain socket")
    port: int = Field(5432, ge=1, le=65535)
    dbname: str = Field(..., min_length=1)
    user: str = Field(..., min_length=1)
    sslmode: Literal["disable", "prefer", "require", "verify-ca", "verify-full"] = "prefer"
    connect_timeout_s: int = Field(5, ge=1, le=30)


class S3Settings(_Strict):
    bucket: str = Field(..., min_length=3, max_length=63)
    region: str = Field("us-east-1", min_length=1)
    endpoint_url: str | None = Field(None, description="Empty = AWS. Set for S3-compatible stores.")
    verify_tls: bool = True


class DvcSettings(_Strict):
    repo: str = Field(..., min_length=1, description="Local Git+DVC repository path or Git URL")
    default_rev: str | None = None
    remote: str | None = Field(None, description="DVC remote name (default: the repository's default remote)")


SETTINGS = {"postgres": PostgresSettings, "s3": S3Settings, "dvc": DvcSettings}
# which secret fields each type accepts: name -> required?
SECRET_FIELDS = {"postgres": {"password": False}, "s3": {"access_key_id": False, "secret_access_key": False, "session_token": False}, "dvc": {}}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS connections (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, settings TEXT NOT NULL, secret_refs TEXT NOT NULL,
  created_at REAL NOT NULL, updated_at REAL NOT NULL, last_test TEXT);
"""


class ConnectionRegistry:
    def __init__(self, workbench: str | Path):
        self.workbench = Path(workbench)
        self.workbench.mkdir(parents=True, exist_ok=True)
        self.path = self.workbench / "connections.db"
        self._lock = threading.RLock()
        with self._lock, closing(self._db()) as db, db:
            enable_wal(db)
            # Explicit migration and downgrade checks run in the connection factory.

    def _db(self) -> sqlite3.Connection:
        db = open_database(self.path, "connections", _SCHEMA, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def _exec(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock, closing(self._db()) as db, db:
            return db.execute(sql, args).fetchall()

    # ---------------------------------------------------------------- validation
    def validate(self, type_: str, settings: dict[str, Any], secrets: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if type_ not in SETTINGS:
            raise SourceError("E_SRC_UNSUPPORTED", f"Connector type '{type_}' is not supported (supported: {sorted(SETTINGS)}).")
        try:
            s = SETTINGS[type_].model_validate(settings).model_dump(mode="json")
        except ValidationError as e:
            msg = "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors())
            raise SourceError("E_SRC_SETTINGS_INVALID", f"Invalid {type_} settings (secrets are passed as references in 'secrets', never as settings): {msg}")
        allowed = SECRET_FIELDS[type_]
        refs = {}
        for k, v in (secrets or {}).items():
            if k not in allowed:
                raise SourceError("E_SRC_SECRET_REF_INVALID", f"'{type_}' connections have no secret field '{k}' (have {sorted(allowed)}).")
            refs[k] = sec.normalize_ref(v, self.workbench, k)
        return s, refs

    # ---------------------------------------------------------------- CRUD
    def create(self, id_: str, name: str, type_: str, settings: dict[str, Any], secrets: dict[str, Any] | None = None) -> dict[str, Any]:
        if not ID_RE.match(id_):
            raise SourceError("E_SRC_SETTINGS_INVALID", "Connection id must be lowercase letters, digits, '-' or '_' (max 48), starting with a letter.")
        s, refs = self.validate(type_, settings, secrets or {})
        if self.get_row(id_):
            raise SourceError("E_SRC_SETTINGS_INVALID", f"Connection '{id_}' already exists.", hint="Choose another id, or update the existing connection.")
        now = time.time()
        self._exec("INSERT INTO connections VALUES (?,?,?,?,?,?,?,NULL)", (id_, name, type_, json.dumps(s), json.dumps(refs), now, now))
        return self.public(id_)

    def update(self, id_: str, name: str | None, settings: dict[str, Any] | None, secrets: dict[str, Any] | None) -> dict[str, Any]:
        row = self.require(id_)
        s, refs = self.validate(row["type"], settings if settings is not None else row["settings"], secrets if secrets is not None else row["secret_refs"])
        self._exec("UPDATE connections SET name=?, settings=?, secret_refs=?, updated_at=?, last_test=NULL WHERE id=?",
                   (name or row["name"], json.dumps(s), json.dumps(refs), time.time(), id_))
        return self.public(id_)

    def delete(self, id_: str) -> None:
        self.require(id_)
        self._exec("DELETE FROM connections WHERE id=?", (id_,))

    def record_test(self, id_: str, result: dict[str, Any]) -> None:
        self._exec("UPDATE connections SET last_test=? WHERE id=?", (json.dumps(result), id_))

    # ---------------------------------------------------------------- reads
    def get_row(self, id_: str) -> dict[str, Any] | None:
        rows = self._exec("SELECT * FROM connections WHERE id=?", (id_,))
        if not rows:
            return None
        d = dict(rows[0])
        d["settings"], d["secret_refs"] = json.loads(d["settings"]), json.loads(d["secret_refs"])
        d["last_test"] = json.loads(d["last_test"]) if d["last_test"] else None
        return d

    def require(self, id_: str) -> dict[str, Any]:
        row = self.get_row(id_)
        if row is None:
            raise SourceError("E_SRC_CONNECTION_NOT_FOUND", f"Connection '{id_}' does not exist.", connection=id_)
        return row

    def list_ids(self) -> list[str]:
        return [r["id"] for r in self._exec("SELECT id FROM connections ORDER BY id")]

    def public(self, id_: str) -> dict[str, Any]:
        """The only representation that leaves the process: settings, secret reference *status*, last test. No secret values."""
        r = self.require(id_)
        return {"id": r["id"], "name": r["name"], "type": r["type"], "settings": r["settings"],
                "secrets": {k: sec.describe(v) for k, v in r["secret_refs"].items()},
                "lastTest": r["last_test"], "createdAt": r["created_at"], "updatedAt": r["updated_at"]}

    def secret_values(self, row: dict[str, Any]) -> dict[str, str]:
        return {k: sec.resolve(ref, k, row["id"]) for k, ref in row["secret_refs"].items()}
