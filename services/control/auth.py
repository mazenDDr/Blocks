"""Optional bearer-token boundary for the control service.

When VOID_API_TOKEN (or create_app(api_token=...)) is set, every request (the /api routes and FastAPI's /docs and /openapi.json) must carry `Authorization: Bearer <token>`. The comparison
is constant-time, and tokens shorter than 16 characters are refused at start-up. This is one shared secret for a local or single-team
deployment, not user accounts: there are no roles, per-user identities or audit attribution. The editor's Vite proxy adds the header on
the server side, so the token is never part of the browser bundle. The value is never logged or stored in the workbench."""
from __future__ import annotations

import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

MIN_LENGTH = 16


def check_token(token: str) -> str:
    if len(token) < MIN_LENGTH or token.strip() != token:
        raise ValueError(f"VOID_API_TOKEN must be at least {MIN_LENGTH} characters without surrounding whitespace")
    return token


class TokenAuth(BaseHTTPMiddleware):
    def __init__(self, app, token: str):
        super().__init__(app)
        self.expected = f"Bearer {check_token(token)}".encode()

    async def dispatch(self, request: Request, call_next):
        got = request.headers.get("authorization", "").encode()
        if not hmac.compare_digest(got, self.expected):
            return JSONResponse({"detail": {"code": "unauthorized", "message": "This control service requires its bearer token (VOID_API_TOKEN)."}},
                                status_code=401, headers={"WWW-Authenticate": "Bearer"})
        return await call_next(request)


def auth_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}
