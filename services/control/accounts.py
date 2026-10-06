"""Optional named accounts with roles for the control service (ADR 0055).

With VOID_USERS_FILE (or create_app(users_file=...)), every request must carry `Authorization: Bearer <token>` for one account
in that file. The file stores only SHA-256 hashes of random tokens, so reading it does not reveal a token; tokens are shown once
by `python -m control.accounts add`. Roles:

* viewer   - read-only: GET/HEAD/OPTIONS only.
* operator - may change state, except the admin-only operations below; serving requests, traces, labels, cancellations and
             conversations are owned: the request's `user` must be the operator's own account name.
* admin    - everything, including other users' serving namespaces, release deploy/rollback, aliases, cache-retention
             policies, credential-bearing connection definitions and load tests (which send requests as synthetic users).

Every request that is not a read is appended to `<workbench>/audit/requests.jsonl` with account, role, method, path, status
and time (no bodies, no tokens). Without a users file the single shared VOID_API_TOKEN behaviour (ADR 0021) is unchanged.
This is local account control, not an identity provider: no passwords, sessions, expiry, rotation API or SSO.
"""
from __future__ import annotations

import argparse
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import sys
import threading
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

ROLES = ("viewer", "operator", "admin")
NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
HEX = re.compile(r"^[0-9a-f]{64}$")
READ_METHODS = {"GET", "HEAD", "OPTIONS"}
ADMIN_ONLY = [re.compile(p) for p in (
    r"^/api/production/releases/[^/]+/(deploy|rollback)$",
    r"^/api/production/aliases/[^/]+$",
    r"^/api/cache/nodes/policies/[^/]+$",
    r"^/api/connections(/[^/]+)?$",
    r"^/api/production/traffic(/[^/]+/cancel)?$",  # load tests act as synthetic users
)]


@dataclass(frozen=True)
class Account:
    name: str
    role: str


CURRENT: ContextVar[Account | None] = ContextVar("void_account", default=None)


class AccountError(Exception):
    def __init__(self, code, message, status=403):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def load(path) -> dict[str, Account]:
    """Strictly parse a users file into {tokenSha256: Account}. Refuses group/other-readable files."""
    p = Path(path).expanduser()
    if p.stat().st_mode & 0o077:
        raise ValueError("Users file must not be readable by group or others (chmod 600).")
    data = json.loads(p.read_text())
    if not isinstance(data, dict) or set(data) != {"format", "users"} or data["format"] != "void-users-v1" or not isinstance(data["users"], list):
        raise ValueError("Users file must be {\"format\": \"void-users-v1\", \"users\": [...]}.")
    out, names = {}, set()
    for u in data["users"]:
        if not isinstance(u, dict) or set(u) != {"name", "role", "tokenSha256"} or not NAME.fullmatch(str(u["name"])) \
                or u["role"] not in ROLES or not HEX.fullmatch(str(u["tokenSha256"])):
            raise ValueError("Each user needs exactly name (1-64 of A-Za-z0-9_-), role (viewer|operator|admin) and tokenSha256.")
        if u["name"] in names or u["tokenSha256"] in out:
            raise ValueError("User names and tokens must be unique.")
        names.add(u["name"])
        out[u["tokenSha256"]] = Account(u["name"], u["role"])
    if not out:
        raise ValueError("Users file has no users.")
    return out


def current() -> Account | None:
    return CURRENT.get()


def ensure_owner(user: str) -> None:
    """Serving data belongs to its declared user; only admins act for another user. No-op without accounts."""
    acc = CURRENT.get()
    if acc is not None and acc.role != "admin" and user != acc.name:
        raise AccountError("E_OWNER", f"Account '{acc.name}' may only act for user '{acc.name}', not '{user}'.")


class AccountAuth(BaseHTTPMiddleware):
    def __init__(self, app, accounts: dict[str, Account], audit_dir: Path):
        super().__init__(app)
        self.accounts, self.audit = accounts, Path(audit_dir) / "requests.jsonl"
        self.lock = threading.Lock()

    def _record(self, acc, request, status):
        line = json.dumps({"time": time.time(), "account": acc.name if acc else None, "role": acc.role if acc else None,
                           "method": request.method, "path": request.url.path, "status": status}, sort_keys=True)
        with self.lock:
            self.audit.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.audit, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a") as f:
                f.write(line + "\n")

    async def dispatch(self, request: Request, call_next):
        header = request.headers.get("authorization", "")
        token = header[7:] if header.startswith("Bearer ") else ""
        h = digest(token) if token else ""
        acc = next((a for k, a in self.accounts.items() if hmac.compare_digest(k, h)), None) if h else None
        mutating = request.method not in READ_METHODS
        if acc is None:
            if mutating:
                self._record(None, request, 401)
            return JSONResponse({"detail": {"code": "unauthorized", "message": "This control service requires an account token."}},
                                status_code=401, headers={"WWW-Authenticate": "Bearer"})
        refusal = None
        if mutating and acc.role == "viewer":
            refusal = ("E_ROLE", f"Account '{acc.name}' is a viewer; only reads are allowed.")
        elif mutating and acc.role != "admin" and any(p.match(request.url.path) for p in ADMIN_ONLY):
            refusal = ("E_ROLE", f"'{request.method} {request.url.path}' requires an admin account.")
        if refusal:
            self._record(acc, request, 403)
            return JSONResponse({"detail": {"code": refusal[0], "message": refusal[1]}}, status_code=403)
        token_ctx = CURRENT.set(acc)
        try:
            response = await call_next(request)
        finally:
            CURRENT.reset(token_ctx)
        if mutating:
            self._record(acc, request, response.status_code)
        return response


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m control.accounts", description="Manage the control service users file.")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("add", help="Add a user with a new random token; prints the token once.")
    a.add_argument("--file", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--role", choices=ROLES, required=True)
    r = sub.add_parser("remove", help="Remove a user; their token stops working on the next service start.")
    r.add_argument("--file", required=True)
    r.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    p = Path(args.file).expanduser()
    data = json.loads(p.read_text()) if p.exists() else {"format": "void-users-v1", "users": []}
    if p.exists():
        load(p)  # validate before changing anything
    if args.command == "add":
        if not NAME.fullmatch(args.name) or any(u["name"] == args.name for u in data["users"]):
            print(json.dumps({"error": "E_USER", "message": "Name must be 1-64 of A-Za-z0-9_- and not already present."}), file=sys.stderr)
            return 2
        token = secrets.token_urlsafe(32)
        data["users"].append({"name": args.name, "role": args.role, "tokenSha256": digest(token)})
    else:
        if not any(u["name"] == args.name for u in data["users"]):
            print(json.dumps({"error": "E_USER", "message": "No such user."}), file=sys.stderr)
            return 2
        data["users"] = [u for u in data["users"] if u["name"] != args.name]
        token = None
    tmp = p.with_name(p.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, p)
    print(json.dumps({"file": str(p.resolve()), "user": args.name, **({"role": args.role, "token": token,
                      "note": "Shown once; only its SHA-256 is stored."} if token else {"removed": True})}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
