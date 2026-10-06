# ADR0050: offline conservative CAS garbage collection

Status: accepted on Mac arm64 (2026-10-06); hosted verification recorded separately in HANDOFF§47.

The workbench content-addressed store (`artifacts/`) only grows. Fifteen
subsystems write to it (runs, node cache, snapshots, production releases/traces/
conversations, JSON agent releases, tracker exports, scale evidence, domain imports),
and their references live in different SQLite databases, project files and other
blobs, often nested inside JSON. Cache retention (ADR0046) prunes cache rows, not
CAS bytes. There was no way to reclaim unreferenced bytes.

Decision: an offline mark-and-sweep that over-approximates references, so an
unknown or future reference format errs towards keeping a blob.

- Mark: every 64-hex window (including windows inside longer hex runs) in every
  workbench file outside `artifacts/`, every relative path and every link target
  keeps the blob of that name. SQLite databases are read cell by cell through
  SQLite, because a raw scan can miss a reference split across overflow pages and
  cannot see WAL frames; other files are scanned as raw bytes in chunks with a
  63-byte carry. Kept blobs are scanned the same way, transitively.
- Before any decision, each database must pass the backup checks: integrity,
  no queued/running work, and every known CAS reference column pointing at an
  existing blob (E_GC_ACTIVE / E_GC_REFERENCE).
- Sweep: unreferenced blobs older than a grace period (default 24h) and stale
  `.tmp` partial writes are candidates. Dry-run is the default; deletion needs
  `--apply --offline`, the same writer-stopped attestation as backup
  (E_GC_OFFLINE). Unexpected files in the store refuse (E_GC_FILE_TYPE).
- The report gives counts, bytes, the first 1000 names and a SHA-256 of the full
  sorted list, so an applied run can be matched to its preview.

Known limit: references inside compressed or encrypted bytes are invisible. Current
writers store CAS references as plain hex text; the integrated check below is the
evidence that this holds for the real seeded workbench, not a proof for every
future writer. Deletion is physical and irreversible except from an earlier
backup; this is reclamation, not privacy erasure (other copies such as backups
and tracker exports are untouched). No online/concurrent collection, no
cross-process lock and no scheduling.

Verification: 7 pytest cases (dry run, offline refusal, transitive/name/link/
overflow/BLOB/embedded references kept, chunk boundary, active work and missing
reference refusal, unexpected store contents, CLI). Integrated
`tools/recovery_smoke.py --trackers --cache-retention --gc` injects an old orphan
into the real seeded workbench (native runs, comments, history, conversations,
research, cache policy, MLflow and offline W&B), collects with zero grace — the
most aggressive setting — then backs up, deletes the source, restores and passes
every restored journey. See HANDOFF§47 for counts.
