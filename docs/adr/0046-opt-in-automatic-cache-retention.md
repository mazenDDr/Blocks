# ADR0046: opt-in local automatic cache retention

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded separately in HANDOFF§43.

Existing tabular node-cache retention is explicit. Add optional per-saved-project
policies without editing native cache/store/execution sources or any saved serving
identity. The control process owns a lifecycle-managed scheduler; no external cron,
worker or user’s existing service is changed. Default is disabled with no policy
file. Only an explicit revision-guarded configuration creates cache-retention.sqlite.

Reuse native tabular.cache.prune: retain at least1/latest and at most1000variants
per node, delete only additional entries older than the declared age, and protect
bytes referenced by recorded artifacts/remaining cache entries. Policies permit
finite0–87600hours,5–604800second cadence, at most100projects. Dry-run preview
changes nothing. Configuration requires an existing saved project, strict fields,
shared-token protection when configured, and the exact previously read revision.

Scheduled checks serialize with native worker submission through the owning
Services.lock. They defer while any native research run is queued/preparing/running/
cancelling/paused, including an actual interrupt awaiting review; a later check
retries after completion. They do not restart, cancel or mutate those runs. One
owning control process is supported; external concurrent workbench writers are
outside this contract. Read-only monitoring observes actual scheduling metadata,
receipts and scheduler errors, never guessed values.

Each native prune publishes actual started/finished times, revision, removed-entry
counts/index bytes/bytes freed and entry provenance. Retain100receipts/project and
1000entry rows/receipt, with truncation and SHA256 of all native dropped-entry
metadata. Idle deferral has its own receipt rather than pretending pruning ran.
Policy metadata and CAS deletion are separate transactions: process death can
lose a receipt after a native prune; subsequent checks recompute candidates. This
is not an exactly-once or distributed scheduler. Recorded artifacts remain protected
by the unchanged native pruning logic. Shutdown joins the owner thread; stop every
writer before existing offline backup. Restore retains policy/revisions/receipts and
next-due timestamps, so enabled policies resume once their owning service starts.

Editor controls show recorded policy/revision/status and actual receipts. Edit
snapshots the source revision; preview identifies real candidates; explicit Save
sets future scheduling; disable cancels future eligible checks after the current
serialized transaction. Settings do not enter graph/UI/native/model/cache identities.

Verification requires actual cached native runs, literal recorded-artifact/shared
CAS protection, five-second scheduling/restart, real paused-agent deferral/resume,
stale/scope/auth/finite-bound refusals, owner shutdown, recomputation, actual browser
review/enable/prune/disable, curl, all original regressions, and offline source-deletion
recovery of policies/receipts/cache/old native conversations/JSON serving/trackers.
Default hosted CI adds real cache browser/recovery coverage; actual Ollama remains
Mac evidence. No GPU/cloud/distributed maintenance, cross-workbench sharing, physical
run/model erasure, encrypted/signed/online backup or retention for other graph kinds
is claimed. HANDOFF§43 records actual evolving evidence and final acceptance.

Final Mac acceptance:1239native tests pass/1skip/14deselected;14actualOllama
live tests and33Node tests pass;280module editor build/typecheck/coverage/source-pin
audit pass. Four native policy tests exercise real cached runs and paused/resumed
agent state. Actual Chrome preview/enable/prune/disable,13curlcases and all12original
editor journeys pass. Physical source-deletion137file/11DB recovery retains policies,
receipts, cache, native artifacts, old conversations, real JSON serving and local/offline
trackers. Existing legacy16-source and JSON22-source identities remain unchanged.
