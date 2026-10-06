# ADR0066: owned SQLite schema versions and transactional legacy adoption

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded in HANDOFF§67.

HANDOFF§59 item6 introduces `storage/schema.py` and `python -m storage`. Nine
application databases: meta, production, connections, studies, integrations,
research, memory, embeddings, cache-retention. Native LangGraph checkpoints and
tracker/provider databases remain owned by their libraries; their user_version is
never repurposed. No new dependency, model training or new serving graph behavior.

Version1 is an explicit baseline migration, adopting unversioned databases only
when every existing non-internal schema object matches its known declaration.
Missing known tables/indexes from older additive bootstraps are created; foreign,
incompatible or altered current objects refuse. Versioned databases carry distinct
application_id ownership. Every managed store connection checks ownership, shape
and supported user_version before journal changes, reads or writes. Future/negative
versions refuse E_SCHEMA_VERSION without a downgrade, wrong owner E_SCHEMA_OWNER,
unknown/mismatching objects E_SCHEMA_SHAPE. No column inference or row rewrite.

Trusted application SQL steps run through individual execute calls inside BEGIN
IMMEDIATE; executescript is excluded because its implicit COMMIT breaks rollback.
Recheck after acquiring the write lock. SQL steps, version and owner commit together;
any failure rolls back. Migration engine supports ordered contiguous steps; shipping
stores currently declare only version1. Existing native rows, CAS identities, route
and checkpoint pointers remain intact. Transactions are per database, not whole
workbench or distributed. Older binaries that predate these checks cannot be made
to enforce them. Backup/restore remain version-preserving, not cross-version repair.

CLI inspect uses read-only connections and never creates absent optional files.
CLI migrate requires --offline, preflights every existing managed database first,
then migrates each independently. Online owning stores adopt at open using SQLite
locking. Erasure checks schema before its destructive SQL. Cache-retention now
closes its short-lived connection explicitly. Native checkpoint/provider schemas,
external credentials/services and arbitrary uploaded databases are outside scope.

Compatibility: source hooks change artifact_store/store.py, agent/memory.py and
agent/index.py. Base agent/conversation and approval manifests now additionally pin
storage/schema.py. **All eleven agent serving families need re-registration**:
agent, conversation, agent_json, conversation_json, agent_retrieval, agent_tools,
conversation_approval and the four context modes. Old release/checkpoint data is
not automatically rebound to new versions. Non-agent prediction implementation
sources (fitted tabular/unsupervised/model/RL/portable) remain unchanged. Original
6f63e09 source fixture is retained; a separate ADR0066 fixture explicitly records
five changed former pins plus new storage/schema.py. Its test still compares every
other prior source and asserts that each declared changed pin actually differs.

Acceptance must include actual legacy-row preservation through all nine owning
store constructors, current/future/foreign-shape/owner refusals, real two-process
opening, actual second-step constraint rollback, read-only/offline CLI and untouched
native-provider DB, then full native/live/all editor/integrated legacy recovery.
HANDOFF§65 records actual results, retained failures and remaining verification.

Control startup preflights all existing owned schema files before mutations.
Integrated recovery explicitly constructs compatible version0 headers on stopped
SYNTHETIC native seed outputs, then verifies exact native row hashes and version1
before restored browser actions. This is not arbitrary historical-binary coverage.
SQLite failure tests use native TEMP triggers per connection, preserving real
transaction rollback without adding forbidden persisted schema objects.

Receipt/reset/fork/history rollback fixtures likewise use real per-connection TEMP
SQLite triggers; all rollback/restart/competition assertions remain. The backup WAL
fixture records committed evidence in declared events rather than adding an
undeclared table to owned metadata; WAL presence and exact recovery checks remain.
Full native acceptance: fresh run 1386 passed, 1 skipped, 25 deselected, 1941 warnings in 516.62s (JUnit 1387 cases, 0 failures/errors; `/private/tmp/void-schema-native-final3.{log,xml}`).
