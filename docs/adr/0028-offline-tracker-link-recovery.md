# ADR 0028 — Explicit offline link and tracker recovery

Accepted, 2026-10-05. ADR0027's regular-file backup refuses actual offline W&B
exports: the SDK creates internal latest-run/debug links and an absolute link to
an external diagnostic log. Silently following or dropping those links would
misrepresent the snapshot and could bundle unrelated external bytes.

Keep default creation and existing backups as `void-offline-workbench-v1` with
links refused. Explicit `create --links internal` selects
`void-offline-workbench-v2`. Both formats remain readable. Backups themselves
contain only ordinary files/directories; links exist as inert manifest records
with original target text, lexically resolved member and file/directory kind.
Never traverse a symbolic link during inventory or verification. Admit only
relative targets whose every intermediate component is an inventoried ordinary
directory and whose final target is an ordinary file/directory. Refuse escaping,
absolute, dangling or chained links, database aliases, SQLite sidecar links and
CAS replacement. Recheck link text during offline creation. Restore copies and
validates all ordinary bytes first, then creates the declared internal links
inside its own stage, then publishes into a new destination. Failed link creation
cleans the stage. No link may be an ancestor of another manifest member.

Native W&B's `trackers/wandb/<export-sha>/wandb/offline-run-*/logs/debug-core.log`
may point outside the workbench. Only the separate explicit
`--omit-wandb-external-logs` option permits omitting that exact diagnostic-link
shape. Record target and reason in `omittedLinks`, returned by create, verify and
restore. Never read/stat/copy that external target and never recreate the omitted
link. All other external links are refused. Internal latest-run/debug links and
exact native SDK history/artifact record bytes survive new-root restore. This
is an offline export recovery path, not online W&B upload or live run resumption.

MLflow preserves native absolute artifact URIs. A relocated restore retains
run identity, parameters, tags, metric values and full history, but artifact
downloads fail when the original root is absent. Restoring to the absent original
root recovers the artifact and confirmed/lost-acknowledgement reconciliation.
Immutable native database history/CAS payloads are never rewritten to pretend
that a relocation is a supported migration. A runbook explains both paths.

Durable tests execute actual MLflow/W&B SDK exports and physical source deletion,
ordinary link roundtrips/refusals/tampering/failed publication, and native
FAISS/memory/effect-ledger recovery. Lexical hashing evidence is separate from a
new live actual Ollama semantic embedding test. The FAISS recovery assertions run
in a fresh subprocess, like native agent workers: initializing FAISS and torch
OpenMP runtimes in one Mac process in this collection order aborted; an isolated
FAISS process passes. No KMP duplicate-library override or fake embeddings.

The optional `tools/recovery_smoke.py --trackers` seeds actual synthetic sensors
training and native exports in a sanitized child process, closes all owned
services before backup, uses v2 with explicit external-log omission, deletes its
generated source, then verifies restored native conversations and confirmed
export mappings/reconnect through real Chrome. CI runs this version. Only
logs/JSON/screenshots/JUnit are uploaded; no snapshot, model, database or provider
credentials. Existing offline/trusted-local, same-environment and new-destination
requirements remain. Source/environment model pins are unchanged; no retraining.

This does not provide external file bundling, arbitrary symlink support, general
tracker relocation, cross-version migration, online snapshots, encryption,
retention, distributed or certified power-loss recovery. See HANDOFF §25 for
actual verification and remaining work.
