# Offline recovery runbook

Use the workbench-backup CLI only after every writer to that source is stopped:
control service, workers, trackers and repository/data import processes. Do not
use a running workbench as an offline source. The command does not stop or detect
idle services. Restore only backups whose provenance you trust, into an absent
destination. Separately retain the manifest SHA256 for corruption checks; it is
not a signature or authentication.

| Resource | New-root restore | Original-root restore | External requirements |
|---|---|---|---|
| Metadata, projects, CAS, internal native model versions | Bytes/identities retained; supported serving works | Same | Same pinned native environment; external data required for source-dependent training |
| Agent conversation checkpoints, forks/resets, request traces | Whole native state/receipts retained; continuation/replay verified | Same | Same pinned code/provider identity; no physical deletion or migration |
| FAISS index/chunks and embedding cache | Index reused while corpus remains; persisted search works after corpus deletion | Same | Actual query embedder still required; rebuilding/ensure needs original corpus |
| Memory records/audit/effect ledger | Versions/tombstones/claims retained | Same | An uncertain started effect remains uncertain; never automatically repeat it |
| Local MLflow | DB metadata/metrics/history readable; artifact downloads fail if old absolute root is absent | Native artifact download/hash and export reconciliation verified | Original artifact path required; no URI rewriting or portable migration |
| W&B offline exports | v2 retains SDK bytes/internal links; explicit external debug-log omission | Same | No online upload/live resume; confirmed directory provenance still names original root |
| Dataset/corpus/credential files outside workbench | Not bundled | Not bundled | Retain original paths/services or use new reviewed configurations/runs |
| DVC/other connector dependencies and repository/import/cache workflows | Ordinary stored bytes retained; full native recovery matrix unverified | Same | Original external roots, remotes, credentials and environments as applicable |

For ordinary trees the default v1 backup refuses links. For W&B trees opt in to
`--links internal --omit-wandb-external-logs`. v2 records links without creating
links in the backup; verify requires exact ordinary file inventory. Restore
creates internal relative links after native SQLite/CAS/hash checks. The external
W&B debug log is omitted and its target/reason is returned in `omittedLinks`.
Keep that report with the manifest. Other external/absolute links, dangling or
chained links and database/CAS aliases are refused. Do not work around refusal by
unreviewed link traversal or editing the manifest/history.

If a relocated MLflow restore cannot download artifacts, first inspect the
original `artifact_uri`. To recover the original native artifact behavior, stop
all writers and restore into the **absent original workbench root** recorded as
`sourceRoot`, retaining its external requirements. Never overwrite an existing
workbench. A separate new-root copy can remain for inspection; do not represent
it as a migrated tracker. Re-exporting new reviewed runs is a different operation
with new identities.

Backup contains unredacted user bytes (directories 0700, ordinary files 0600).
Native model/code artifacts may execute code when later loaded. Provider/device
incompatibilities still refuse serving; backup does not install dependencies or
bypass pins. No online/crash-consistency guarantee, automatic backup, encryption,
retention, garbage collection or version migration is implemented. The README
lists actual executed commands; HANDOFF §24–25 records measured environments and
bounded outcomes.
