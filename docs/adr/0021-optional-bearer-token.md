# ADR 0021 — Optional bearer-token boundary for the control service

Status: implemented, 2026-10-04. Bounded step toward VISION's authenticated team access; not user accounts.

- **Off by default.** Without `VOID_API_TOKEN` (or `create_app(api_token=...)`), behaviour is unchanged.
- **On.** Every request, including `/docs` and `/openapi.json`, must send `Authorization: Bearer <token>`, compared in constant time. Anything else gets 401 with `WWW-Authenticate: Bearer`. Tokens shorter than 16 characters or with surrounding whitespace are refused when the app is created. The token is never logged, stored in the workbench or echoed in responses.
- **Clients.** The editor's Vite dev proxy adds the header on the dev server, so the token is not in the browser bundle (checked in Chromium). EventSource streams and download links keep working because they go through the proxy. The production traffic generator's loopback calls to its own server send the token. The example journey scripts read `VOID_API_TOKEN` from the environment.
- **Scope.** One shared secret per control service: no users, roles, per-user audit or session identities. Caller-declared user/session fields in production serving remain isolation keys, not authentication (ADR 0012). Transport encryption is not provided; put TLS in front of the service before exposing it beyond the local machine. The remote-worker protocol keeps its own token (ADR 0013).
