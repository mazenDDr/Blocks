# ADR 0029 — Read-only native conversation discovery

Accepted, 2026-10-05. Persistent serving conversations previously required knowing
the exact session key. Add bounded discovery without loading every checkpoint or
calling/validating a provider, and without changing native model identities.

GET `/api/production/releases/{rid}/conversations` requires a native conversation
release and a caller-declared user key. Return session identifiers, committed
revision/checkpoint/producing-request metadata, release/version/user provenance,
and an explicit read-only/isolation policy. Do not return prompts, native state,
generated messages or guessed timestamps. Metadata discovery does not claim
checkpoint content integrity; explicit existing checkpoint inspection verifies
and loads the native state separately. Empty/reset heads remain listed as reset,
with no invented defaults. Warmup and failed turns do not create sessions.

Existing writers persist canonical JSON scope keys `[release,user,session]` in the
`agent_sessions` primary key. Use parameterized binary primary-key ranges and
`ORDER BY scope`, not a wildcard LIKE expression or an all-user scan. Literal
ASCII prefixes (including underscore) and the last session key support keyset
pagination. Default page25, maximum100; fetch at most limit+1 to determine the next
cursor. Stable lexicographic ordering is independent of insertion order; concurrent
new/reset heads can appear on later reads. This is not a frozen historical catalog,
and no total count/snapshot guarantee is fabricated. No DB schema/index migration.

Only metadata under the selected release/user range is read. Validate user, prefix,
cursor and page size; reject malformed in-scope records with a stable integrity
code. Shared-token protection applies when configured, but changing the declared
user key is still possible to a token holder; this is not authenticated ownership,
roles or multi-user security.

The Production Requests view provides explicit discovery, literal prefix input,
next/previous pages, recorded head provenance and per-session inspection. Release/
user changes remount scoped state; prefix edits clear old pages. Read failures
clear stale rows; invalid/busy controls cannot initiate discovery. Selecting a
session changes the actual Session input, invokes the existing checkpoint endpoint,
and makes its inspected state available to existing reviewed reset/fork controls.
No automatic action, model call, summary, native mutation or placeholder controls.
Fixed-layout wrapped tables keep inspect buttons inside their panel; rows use
headers, table caption, labels and native keyboard controls.

Native tests cover lexical pagination/prefixes, other-user/release exclusion,
actual successful/reset/forked/capture-off heads, restart persistence, query-plan
index use, metadata-only read after checkpoint corruption, no CAS/event/head changes,
token/query/stateless refusal and explicit inspection. Chrome seeds27 actual
model-free native turns to exercise25+2 pages/back, selected checkpoint and another
user's empty list after offline source deletion/recovery. No fake rows/state/provider.
CI uses the same expanded recovered journey. Source pin files/dependencies unchanged;
saved models and serving versions require no new retraining/re-registration.

Historical restore, physical deletion/privacy erasure, retention/GC/migration,
authenticated ownership/roles, retrieval/memory/tool/effect/streaming serving and
distributed replicas remain separate work. See HANDOFF §26 for actual outcomes.
