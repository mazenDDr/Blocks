# ADR 0027 — Offline whole-workbench backup and recovery

Accepted, 2026-10-05. The workbench spans several SQLite databases, CAS, projects,
library versions, indexes, repository imports/mirrors and tracker files. Copying
only metadata or raw live SQLite pages risks losing WAL commits or mismatching
checkpoint heads and artifacts. Recovery must preserve strict native identities.

Provide a local `python -m workbench_backup create|verify|restore` CLI and directory
format `void-offline-workbench-v1`: `manifest.json` plus `data/`. Creation requires
`--offline`: the caller must first stop **all** control, worker, tracker, repository
and other writers. No processes are automatically discovered/stopped. An idle
server/new DB/filesystem producer cannot be detected by SQLite locks; this is not
an online snapshot. Source tree/hash checks supplement the attestation and cannot
prove quiescence against arbitrary producers or ABA changes.

Discover native SQLite headers outside CAS, including nested tracker databases.
Hold BEGIN IMMEDIATE write reservations on all discovered DBs together. Refuse busy
writers/active run, request and study records. Through separate reader connections,
use native Connection.backup() to incorporate committed WAL pages. Only snapshot
copies are changed to DELETE journal mode; source databases are not migrated or
checkpointed. Transient sidecars are omitted; unrecognized sidecars are refused.
Preserve every other ordinary file's bytes and empty directory. CAS is opaque,
even when its bytes represent a SQLite database: normalizing it would change its
identity. Links/special files, including tracker latest-run links, are refused.

Record file SHA256/size, full inventory, directories, DB list, original root,
creation time and Python/platform/package versions. Check native SQLite integrity,
CAS filename/content hashes and direct metadata/cache/production references to
artifacts, traces, heads and action receipts. Verify rejects missing/extra/corrupt
files, malformed manifests, escaping paths and DB inventory mismatches. It does
not deserialize native state or recursively prove every application relationship.

Restore requires --trusted-local; native artifacts/codeblocks can execute code when
subsequently loaded. Checksums detect corruption, not authenticity; optional
--manifest-sha256 checks a separately retained manifest digest. This is an offline
trusted administrative filesystem operation, not a web upload API. Backup/verify
never resolve credentials, download, load models or perform forward passes. Data
is unredacted; directories are 0700 and files 0600, executable bits removed.

Create/restore build in owned sibling staging directories. Atomic mkdir reserves a
new destination; rename replaces only that own empty reservation. Existing paths
are never overwritten. Other actors/readers must not race publication. Caught I/O
failures remove staging. Uncatchable process death/power loss can leave staging or
reservations; full disk durability certification is unavailable. No encryption,
signing, archive extraction, compression, scheduling, retention or GC is provided.

Absolute paths and native identities remain unchanged. External data/secret files,
environment secrets, provider runtimes, source code and dependencies are not
bundled. Root-relative stores/CAS serving recover in the same pinned environment.
Source-dependent continuation, corpus/tracker/DVC paths and other references need
original paths/services or new reviewed configuration/runs. Immutable history is
never rewritten. Restore to the original root requires it to be absent and all
writers stopped. Native incompatibility checks still reject upgrades; migration
is unsupported. This release changes no pinned implementation files and itself
requires no retraining.

Tests use real WAL/SQLite locks and physical source deletion, identical fitted
scikit-learn and all-three-domain PyTorch predictions/lineage, whole conversation
state, fork/reset receipts, idempotent traces and a continued native research
thread. Failure cases include corruption, missing references, busy/active work,
changing files/manifests, partial I/O and destination collisions. Incompatible
environment after restore is still refused: upgrade safety, not migration. This refusal test injects a differing environment comparison; no dependency upgrade is installed or migrated.
Other resources have byte/row preservation evidence; complete native tracker,
FAISS and DVC recovery journeys remain separate work.

tools/recovery_smoke.py runs the existing real installed-Chrome native-state
journey, closes owned services, backs up/verifies, deletes only its generated
source workbench, restores to a new root, and launches a fresh editor/backend and
Chrome. It checks identical route/version/checkpoint identities, independent
branch/fresh continuations, old replay without head changes, and monitoring. CI
runs this two-stage journey, uploading logs/JSON/screenshots/JUnit only, never
backup/workbench/weights/credentials. Execution/evidence stay outside Git. User
port 8000 is untouched. Exact results and limitations are recorded in HANDOFF §24.

References: [Python native backup](https://docs.python.org/3.13/library/sqlite3.html#sqlite3.Connection.backup),
[SQLite backup API](https://www.sqlite.org/backup.html),
[SQLite transaction reservations](https://www.sqlite.org/lang_transaction.html).
