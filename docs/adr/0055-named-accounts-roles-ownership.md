# ADR0055: named accounts, roles, serving ownership and TLS

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded separately in HANDOFF§52.

ADR0021 added one optional shared bearer token. Everyone holding it could do
everything, serving `user` fields were caller-declared labels, and nothing recorded
who changed what. The README told users to put a TLS proxy in front, but TLS was
never exercised.

Decision: optional named accounts with three roles, enforced ownership of serving
data, an audit trail, and verified TLS using uvicorn's own support.

- `VOID_USERS_FILE` (or `create_app(users_file=...)`) names a strict JSON file
  (`void-users-v1`) of users with `name`, `role` and `tokenSha256`. Only SHA-256
  hashes of random 256-bit tokens are stored; `python -m control.accounts add`
  prints a token once. The file must be mode 0600 and is parsed at start-up, so a
  malformed file stops the service. Shared-token and accounts modes are exclusive.
- Roles: `viewer` may only read (GET/HEAD/OPTIONS, so even read-like POSTs such as
  validation are refused); `operator` may change state except admin-only
  operations; `admin` may do everything. Admin-only: release deploy/rollback,
  aliases, cache-retention policy changes, creating/changing/deleting connections
  (which hold credentials), and load tests (they send requests as synthetic users
  and now run with the admin's own token).
- Ownership: predict (plain and streamed), trace reads, cancellation, labels,
  replay and all conversation reads/actions require the request's `user` to be the
  caller's account name unless the caller is an admin (E_OWNER). The request list
  shows non-admins only their own requests.
- Audit: every non-read request, including refusals and unauthenticated attempts,
  is appended to `<workbench>/audit/requests.jsonl` (mode 0600) with account,
  role, method, path, status and time. No bodies or tokens are written.
- `GET /api/whoami` reports the account and mode; the editor's Production
  workspace starts in the signed-in account's namespace.
- TLS: uvicorn `--ssl-certfile/--ssl-keyfile` with named accounts is exercised
  end to end; certificate issuance and renewal are the operator's responsibility.

Not provided: passwords, sessions or expiry, token rotation API (re-add a user),
SSO/identity providers, per-project or per-release permissions, row-level
filtering beyond serving data, rate limiting, audit tamper evidence or rotation,
and authorization of the separate worker/tracker services.

Verification: 7 pytest cases with the real control app: hashes-only file and CLI
refusals, 401s and whoami, viewer and admin-only refusals with admin pass-through,
ownership refusals on every user-scoped route with own-namespace and admin access,
filtered listing, audit content/permissions without tokens, unchanged shared-token
and open modes, and a real uvicorn TLS server with a SYNTHETIC self-signed
certificate (verified client succeeds, system trust store and cleartext HTTP
fail). A scratch owned-Chrome run with the editor proxy holding an operator token
showed the Production namespace defaulting to that account and E_OWNER for
another user's trace.
