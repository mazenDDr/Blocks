"""Secrets by reference (A17, VISION 21): the registry stores *where* a secret lives, never the secret.

A reference is `{"kind": "env", "name": "PGPASSWORD_LAB"}` or `{"kind": "file", "path": "/abs/secrets.json", "key": "lab_db"}`
(a JSON object of name -> value that lives outside the project and the workbench). Values are resolved at execution time, in
the process that needs them, and are never written to the registry, API responses, events, artifacts, manifests or exports."""
from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from .errors import SourceError

ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
REPO = Path(__file__).resolve().parents[2]


def normalize_ref(ref: Any, workbench: Path | None = None, field: str = "secret") -> dict[str, str]:
    if not isinstance(ref, dict):
        raise SourceError("E_SRC_SECRET_REF_INVALID", f"'{field}' must be a reference object, never a literal value.")
    kind = ref.get("kind")
    if kind == "env":
        name = ref.get("name")
        if not isinstance(name, str) or not ENV_NAME.match(name) or set(ref) - {"kind", "name"}:
            raise SourceError("E_SRC_SECRET_REF_INVALID", f"'{field}': an env reference needs only a valid environment variable name.")
        return {"kind": "env", "name": name}
    if kind == "file":
        path, key = ref.get("path"), ref.get("key")
        if not isinstance(path, str) or not isinstance(key, str) or not key or set(ref) - {"kind", "path", "key"}:
            raise SourceError("E_SRC_SECRET_REF_INVALID", f"'{field}': a file reference needs only an absolute 'path' and a 'key'.")
        p = Path(path).expanduser()
        if not p.is_absolute():
            raise SourceError("E_SRC_SECRET_REF_INVALID", f"'{field}': the secrets file path must be absolute.")
        rp = p.resolve()
        banned = [REPO] + ([workbench.resolve()] if workbench else [])
        for b in banned:
            if rp == b or b in rp.parents:
                raise SourceError("E_SRC_SECRET_REF_INVALID", f"'{field}': the secrets file must live outside the project and the workbench ({b}).")
        return {"kind": "file", "path": str(rp), "key": key}
    raise SourceError("E_SRC_SECRET_REF_INVALID", f"'{field}': reference kind must be 'env' or 'file'.")


def resolve(ref: dict[str, str], field: str = "secret", connection: str | None = None) -> str:
    try:
        if ref["kind"] == "env":
            v = os.environ.get(ref["name"])
            if v is None or v == "":
                raise SourceError("E_SRC_SECRET_UNRESOLVED", f"Environment variable {ref['name']} (for '{field}') is not set in the process that runs this.",
                                  connection=connection)
            return v
        p = Path(ref["path"])
        if not p.is_file():
            raise SourceError("E_SRC_SECRET_UNRESOLVED", f"Secrets file for '{field}' does not exist.", connection=connection)
        data = json.loads(p.read_text())
        v = data.get(ref["key"]) if isinstance(data, dict) else None
        if not isinstance(v, str) or v == "":
            raise SourceError("E_SRC_SECRET_UNRESOLVED", f"Key '{ref['key']}' (for '{field}') is missing in the secrets file.", connection=connection)
        return v
    except (OSError, ValueError) as e:
        raise SourceError("E_SRC_SECRET_UNRESOLVED", f"Secrets file for '{field}' cannot be read ({type(e).__name__}).", connection=connection)


def describe(ref: dict[str, str]) -> dict[str, Any]:
    """What the API may show: the kind and whether it currently resolves. Never the value, never a file path."""
    ok, warn = True, None
    try:
        resolve(ref)
        if ref["kind"] == "file" and Path(ref["path"]).stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            warn = "the secrets file is readable by other users; chmod 600 it"
    except SourceError:
        ok = False
    d: dict[str, Any] = {"kind": ref["kind"], "resolvable": ok}
    if ref["kind"] == "env":
        d["name"] = ref["name"]
    else:
        d["key"] = ref["key"]
    if warn:
        d["warning"] = warn
    return d
